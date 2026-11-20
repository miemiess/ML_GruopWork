"""用小型合成 CSV 检查 D 任务端到端输出。"""

import json

import numpy as np
import pandas as pd
import pytest

from emginsight.putemg_force import load_force_model
from emginsight.putemg_force.experiment import (
    CALIBRATION_SECONDS, FEATURE_SCALES, SMOOTHING, TARGETS, run, smooth, target_metrics,
)

CALIBRATION = int(CALIBRATION_SECONDS * 5120)


## @brief 写一份与作者表头一致的合成记录：开头静息，之后肌电幅度随力量变化，力量带有用户零点偏移。
# @param folder 输出目录。
# @param name 文件主名。
# @param rng 随机数生成器。
# @param offset 力量零点偏移。
# @param traj_at_start 为 True 时校准段内出现出力指令。
def write_record(folder, name, rng, offset, traj_at_start=False, n=CALIBRATION + 4000):
    force = np.zeros((n, 10))
    for start in range(CALIBRATION + 200, n - 300, 600):
        force[start:start + 300, rng.integers(10)] = rng.uniform(50, 300)
    values = {f"EMG_{c + 1}": rng.normal(0, 1, n) * (1 + force[:, c % 10] / 50) for c in range(24)}
    values.update({target: force[:, i] + offset for i, target in enumerate(TARGETS)})
    traj = (force.sum(axis=1) > 0).astype(float)
    if traj_at_start:
        traj[:100] = 1
    values.update(FORCE_MVC=np.full(n, 500.), TRAJ_1=traj, TRAJ_2=np.zeros(n))
    pd.DataFrame(values, index=np.arange(n) / 5120.).to_csv(folder / f"{name}.csv")


## @brief 验证用户隔离、校准段排除、异常记录留痕、指标可重算与缓存复用。
def test_force_experiment(tmp_path):
    rng = np.random.default_rng(0)
    root, output, cache = tmp_path / "raw", tmp_path / "result", tmp_path / "cache"
    for number in range(3, 8):
        subject = f"{number:02d}"
        folder = root / f"emg_force_{subject}"
        folder.mkdir(parents=True)
        for date in ("2018-04-04", "2018-04-23"):
            write_record(folder, f"emg_force-{subject}-repeats_long-{date}-10-00-00-000", rng, offset=number * 10)
        write_record(folder, f"emg_force-{subject}-sequential-2018-04-04-11-00-00-000", rng, offset=0)
    write_record(root / "emg_force_07", "emg_force-07-repeats_long-2018-05-01-10-00-00-000", rng, 0, traj_at_start=True)
    model_path = tmp_path / "model" / "force_model.joblib"
    run(root, output, cache, window_size=64, step_size=32, model_path=model_path)

    metrics = json.loads((output / "metrics.json").read_text(encoding="utf8"))
    users = [user for group in metrics["subjects"].values() for user in group]
    assert len(users) == len(set(users)) == 5
    assert [item["record"] for item in metrics["skipped"]] == ["emg_force-07-repeats_long-2018-05-01-10-00-00-000"]
    candidates = pd.read_csv(output / "validation.csv")
    assert set(candidates.scale) == set(FEATURE_SCALES) and set(candidates.smoothing) == set(SMOOTHING)
    assert len(candidates) == 3 * 2 * 4 * 3

    full = pd.read_csv(cache / "putemg_force_test_predictions.csv.gz", dtype={"subject": str})
    assert set(full.subject) == set(metrics["subjects"]["test"])
    assert set(full.record.str.split("-").str[2]) == {"repeats_long"}
    assert full.start_sample.min() >= CALIBRATION
    truth = full[[f"{name}_true" for name in TARGETS]].to_numpy()
    model = full[[f"{name}_pred" for name in TARGETS]].to_numpy()
    assert np.isclose(target_metrics(truth, model).rmse.mean(), metrics["test"]["model"]["rmse"], rtol=1e-3)
    # 去零点后静息窗口的实测力量应接近 0，而不是用户偏移。
    assert np.abs(np.median(truth, axis=0)).max() < 1
    assert pd.read_csv(output / "per_user.csv").shape[0] == metrics["test"]["users"]
    assert metrics["test"]["model"]["rmse"] < metrics["test"]["baseline"]["rmse"]
    for name in ("test_curves.png", "per_user_r2.png", "example_predictions.csv"):
        assert (output / name).stat().st_size > 0

    # 模型文件可重新读取，校验值与模型卡一致；模型卡记录获选配置与训练用户。
    card = json.loads((output / "model_card.json").read_text(encoding="utf8"))
    assert card["verified_test_predictions"] and card["max_abs_difference"] <= 1e-9
    assert card["scale"] == metrics["selected"]["scale"] and card["smoothing"] == metrics["selected"]["smoothing"]
    assert card["train_subjects"] == metrics["subjects"]["train"]
    model = load_force_model(model_path, card["sha256"])
    assert model.targets == TARGETS
    with pytest.raises(ValueError, match="SHA-256"):
        load_force_model(model_path, "0" * 64)

    cached = list((cache).rglob("*.npz"))
    assert len(cached) == 10
    run(root, tmp_path / "again", cache, window_size=64, step_size=32)
    again = json.loads((tmp_path / "again" / "metrics.json").read_text(encoding="utf8"))
    assert np.isclose(again["test"]["model"]["rmse"], metrics["test"]["model"]["rmse"])


## @brief 因果平滑只使用当前及之前的窗口，并且不跨记录。
def test_smooth_is_causal_per_record():
    pred = np.array([[0.], [3.], [6.], [10.], [20.]])
    out = smooth(pred, np.array(["a", "a", "a", "b", "b"]), 2)
    assert out.ravel().tolist() == [0., 1.5, 4.5, 10., 15.]
