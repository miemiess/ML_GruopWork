"""检查批处理预测入口：原始 CSV 输入、可选实测列、校准与采样率检查。"""

import json

import numpy as np
import pandas as pd
import pytest

from emginsight.putemg_force.experiment import run
from emginsight.putemg_force.preprocessing import CALIBRATION_SECONDS
from emginsight.putemg_force.targets import TARGETS
from emginsight.putemg_force.predict import drift_check, main

CALIBRATION = int(CALIBRATION_SECONDS * 5120)


## @brief 写合成记录：开头静息，之后肌电幅度随力量变化。
# @param path 输出路径。
# @param rng 随机数生成器。
# @param force_columns 是否写入实测力量和 TRAJ 列。
# @param rate 采样率。
# @param traj_at_start 校准段内是否出现出力指令。
def write_csv(path, rng, force_columns=True, rate=5120., traj_at_start=False, n=CALIBRATION + 4000):
    force = np.zeros((n, 10))
    for start in range(CALIBRATION + 200, n - 300, 600):
        force[start:start + 300, rng.integers(10)] = rng.uniform(50, 300)
    values = {f"EMG_{c + 1}": rng.normal(0, 1, n) * (1 + force[:, c % 10] / 50) for c in range(24)}
    if force_columns:
        values.update({name: force[:, i] + 20 for i, name in enumerate(TARGETS)})
        traj = (force.sum(axis=1) > 0).astype(float)
        if traj_at_start:
            traj[:100] = 1
        values["TRAJ_1"] = traj
    pd.DataFrame(values, index=pd.Index(np.arange(n) / rate, name="time")).to_csv(path)


## @brief 用小数据训练并保存模型，供预测测试读取。
@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("raw")
    rng = np.random.default_rng(1)
    for number in range(3, 8):
        folder = root / f"emg_force_{number:02d}"
        folder.mkdir()
        for date in ("2018-04-04", "2018-04-23"):
            write_csv(folder / f"emg_force-{number:02d}-repeats_long-{date}-10-00-00-000.csv", rng)
    output = tmp_path_factory.mktemp("result")
    model_path = output / "force_model.joblib"
    run(root, output, None, window_size=64, step_size=32, model_path=model_path)
    return output, model_path


## @brief 有实测列时给出对照指标；只有肌电时仍能预测并注明静息假设；坏文件不影响其他文件。
def test_predict_files(trained, tmp_path):
    output, model_path = trained
    rng = np.random.default_rng(2)
    inputs = tmp_path / "in"
    inputs.mkdir()
    write_csv(inputs / "with_force.csv", rng)
    write_csv(inputs / "emg_only.csv", rng, force_columns=False)
    write_csv(inputs / "wrong_rate.csv", rng, rate=1000.)
    write_csv(inputs / "moving_at_start.csv", rng, traj_at_start=True)
    out = tmp_path / "out"
    code = main(["--csv", str(inputs), "--model", str(model_path), "--model-card", str(output / "model_card.json"),
                 "--reference", str(output / "monitoring_reference.json"), "--level-model", str(tmp_path / "none.joblib"),
                 "--output-dir", str(out)])
    assert code == 1
    summary = json.loads((out / "summary.json").read_text(encoding="utf8"))
    assert summary["task"] == "putemg_force"
    assert summary["counts"] == {"total": 4, "succeeded": 2, "failed": 2}
    assert summary["model"]["sha256_checked"]
    files = {item["file"].split("\\")[-1].split("/")[-1]: item for item in summary["files"]}
    assert "evaluation" in files["with_force.csv"] and files["with_force.csv"]["evaluation"]["rmse"] > 0
    assert "evaluation" not in files["emg_only.csv"] and files["emg_only.csv"]["notes"]
    assert files["emg_only.csv"]["drift"]["available"]
    assert "采样率" in files["wrong_rate.csv"]["error"]
    assert files["emg_only.csv"]["status"] == "ok"
    assert files["wrong_rate.csv"]["status"] == "error"
    assert "TRAJ" in files["moving_at_start.csv"]["error"]
    pred = pd.read_csv(out / "emg_only_predictions.csv")
    assert list(pred.columns[2:]) == [f"{name}_pred" for name in TARGETS]
    assert pred.start_sample.min() >= CALIBRATION
    assert (out / "with_force_curves.png").stat().st_size > 0


## @brief 模型卡校验值不符时拒绝读取。
def test_predict_rejects_wrong_checksum(trained, tmp_path):
    output, model_path = trained
    card = json.loads((output / "model_card.json").read_text(encoding="utf8"))
    card["sha256"] = "0" * 64
    bad = tmp_path / "card.json"
    bad.write_text(json.dumps(card), encoding="utf8")
    with pytest.raises(ValueError, match="SHA-256"):
        main(["--input", str(tmp_path), "--model", str(model_path), "--model-card", str(bad),
              "--level-model", str(tmp_path / "none.joblib")])


## @brief 静息肌电明显偏离训练分布时告警。
def test_drift_check_flags_shift():
    reference = {"mean": [0.0] * 24, "std": [0.1] * 24}
    rest = np.ones(72)
    assert not drift_check(rest, reference)["warning"]
    rest[0::3] = np.e
    assert drift_check(rest, reference)["warning"]
