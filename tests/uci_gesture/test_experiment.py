"""用小型原始记录检查 C 任务端到端输出。"""

import json

import pandas as pd
from sklearn.metrics import f1_score

from emginsight.uci_gesture.experiment import LABELS, run


## @brief 验证用户隔离、第七类排除、坏记录留痕及输出指标可重算。
def test_gesture_experiment(tmp_path):
    root, output = tmp_path / "raw", tmp_path / "result"
    for number in range(1, 10):
        subject = root / f"{number:02d}"
        subject.mkdir(parents=True)
        rows = ["time " + " ".join(f"channel{i}" for i in range(1, 9)) + " class"]
        for sample in range(1400):
            label = 1 + sample // 200
            signal = " ".join(str(label * channel + number / 100) for channel in range(1, 9))
            rows.append(f"{sample} {signal} {label}")
        (subject / "1_raw_data_test.txt").write_text("\n".join(rows), encoding="utf8")
    (subject / "2_raw_data_invalid.txt").write_text("bad header", encoding="utf8")
    run(root, output)

    metrics = json.loads((output / "metrics.json").read_text(encoding="utf8"))
    candidates = pd.read_csv(output / "validation.csv")
    forest = candidates[candidates.model == "forest"]
    assert set(zip(forest.features, forest.parameter)) == {
        (features, parameter) for features in ("RMS+MAV", "RMS+MAV+WL") for parameter in ("trees=20", "trees=100")
    }
    users = [user for group in metrics["subjects"].values() for user in group]
    assert len(users) == len(set(users)) == 9
    assert len(metrics["skipped"]) == 1
    pred = pd.read_csv(output / "test_predictions.csv", dtype={"subject": str})
    assert set(pred.subject) == set(metrics["subjects"]["test"])
    assert set(pred.label) == set(LABELS)
    assert metrics["test"]["prediction"]["macro_f1"] == f1_score(pred.label, pred.prediction, average="macro")
    assert pd.read_csv(output / "confusion.csv", index_col=0).to_numpy().sum() == len(pred)
