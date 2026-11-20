"""检查三档力量标签、分类实验与模型保存。"""

import json

import numpy as np
import pandas as pd

from emginsight.putemg_force.experiment import CALIBRATION_SECONDS, TARGETS
from emginsight.putemg_force import experiment
from emginsight.putemg_force.levels import LEVELS, level_labels, run
from emginsight.putemg_force.predict import main as predict_main
from emginsight.putemg_force.model import load_level_model

CALIBRATION = int(CALIBRATION_SECONDS * 5120)


## @brief 写合成记录：按段交替静息、轻（约 25% MVC）和重（约 50% MVC），两组传感器同时出力。
def write_record(path, rng, n=CALIBRATION + 6000):
    force, traj = np.zeros((n, 10)), np.zeros(n)
    for i, start in enumerate(range(CALIBRATION + 300, n - 400, 700)):
        level = (0.25, 0.5)[i % 2] * 500
        group = 2 * rng.integers(5)
        force[start:start + 350, group:group + 2] = level
        traj[start:start + 350] = level
    values = {f"EMG_{c + 1}": rng.normal(0, 1, n) * (1 + force[:, c % 10] / 50) for c in range(24)}
    values.update({name: force[:, i] + 30 for i, name in enumerate(TARGETS)})
    values.update(FORCE_MVC=np.full(n, 500.), TRAJ_1=traj)
    pd.DataFrame(values, index=np.arange(n) / 5120.).to_csv(path)


## @brief 单个传感器漂移不会被当成出力；两路同时出力按 MVC 比例分档。
def test_level_labels_ignore_single_sensor_drift():
    force = np.zeros((400, 10))
    force[:, 3] = np.linspace(0, 150, 400)        # 单路缓慢漂移到 30% MVC
    force[100:140, 0:2] = 250                      # 两路同时出力 50% MVC
    force[250:290, 4:6] = 125                      # 两路同时出力 25% MVC
    levels, relative = level_labels(force, 500., 0.1)
    assert (levels[:90] == 0).all() and (levels[300:] == 0).all()
    assert (levels[105:135] == 2).all() and (levels[255:285] == 1).all()


## @brief 端到端：用户隔离、三档都出现、模型文件可重新读取并复现测试预测。
def test_level_experiment(tmp_path):
    rng = np.random.default_rng(3)
    root, output = tmp_path / "raw", tmp_path / "levels"
    for number in range(3, 8):
        folder = root / f"emg_force_{number:02d}"
        folder.mkdir(parents=True)
        for date in ("2018-04-04", "2018-04-23"):
            write_record(folder / f"emg_force-{number:02d}-repeats_long-{date}-10-00-00-000.csv", rng)
    model_path = tmp_path / "level.joblib"
    run(root, output, tmp_path / "cache", model_path, None, window_size=64, step_size=32)

    metrics = json.loads((output / "metrics.json").read_text(encoding="utf8"))
    assert all(min(counts) > 0 for counts in metrics["label_share"].values())
    assert len(pd.read_csv(output / "validation.csv")) == 2 * 2 * 3 * 3
    assert metrics["test"]["model"]["macro_f1"] > metrics["test"]["majority"]["macro_f1"]
    assert 0 <= metrics["test"]["label_vs_instruction_agreement"] <= 1
    card = json.loads((output / "model_card.json").read_text(encoding="utf8"))
    model = load_level_model(model_path, card["sha256"])
    assert model.classes == LEVELS and card["verified_test_predictions"]
    example = pd.read_csv(output / "example_predictions.csv")
    assert np.allclose(example[[f"prob_{c}" for c in LEVELS]].sum(axis=1), 1, atol=1e-3)
    for name in ("confusion.png", "test_timeline.png", "per_user.csv", "per_class.csv"):
        assert (output / name).stat().st_size > 0

    # 批处理入口同时输出连续力量和三档；含 FORCE_MVC 时给出三档对照指标。
    force_output = tmp_path / "regression"
    experiment.run(root, force_output, tmp_path / "cache", window_size=64, step_size=32,
                   model_path=force_output / "force_model.joblib")
    record = next(root.glob("emg_force_03/*.csv"))
    out = tmp_path / "pred"
    code = predict_main(["--csv", str(record), "--model", str(force_output / "force_model.joblib"),
                         "--level-model", str(model_path), "--level-model-card", str(output / "model_card.json"),
                         "--output-dir", str(out)])
    assert code == 0
    summary = json.loads((out / "summary.json").read_text(encoding="utf8"))
    assert "level_evaluation" in summary["files"][0] and summary["level_model"]["levels"]["2"] == "heavy"
    pred = pd.read_csv(out / f"{record.stem}_predictions.csv")
    assert {"level_pred", "level_true", "level_prob_rest"} <= set(pred.columns)
    assert (out / f"{record.stem}_levels.png").stat().st_size > 0
