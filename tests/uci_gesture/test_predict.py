"""检查手势模型的保存、读取与批处理识别入口。"""

import json

import numpy as np
import pandas as pd
import pytest

from emginsight.uci_gesture import load_uci_gesture_record, window_gesture_record
from emginsight.uci_gesture.experiment import LABELS, run
from emginsight.uci_gesture.model import load_gesture_model
from emginsight.uci_gesture.predict import main, window_starts


## @brief 写一份 UCI 格式的合成记录：标签按 200 点分块，信号幅度随标签变化。
# @param path 输出路径。
# @param number 用户编号，用于加入小的个体偏移。
def write_record(path, number):
    rows = ["time " + " ".join(f"channel{i}" for i in range(1, 9)) + " class"]
    for sample in range(1400):
        label = 1 + sample // 200
        signal = " ".join(str(label * channel + number / 100) for channel in range(1, 9))
        rows.append(f"{sample} {signal} {label}")
    path.write_text("\n".join(rows), encoding="utf8")


## @brief 训练并保存合成数据上的手势模型。
@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("raw")
    for number in range(1, 10):
        subject = root / f"{number:02d}"
        subject.mkdir()
        write_record(subject / "1_raw_data_test.txt", number)
    output = tmp_path_factory.mktemp("result")
    model_path = output / "gesture_model.joblib"
    run(root, output, model_path)
    return root, output, model_path


## @brief 模型卡与实验一致，重新读取的模型在同一窗口上给出与实验相同的预测。
def test_saved_model_matches_experiment(trained):
    root, output, model_path = trained
    card = json.loads((output / "model_card.json").read_text(encoding="utf8"))
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf8"))
    assert card["verified_test_predictions"] and card["classes"] == list(LABELS)
    assert card["train_subjects"] == metrics["subjects"]["train"]
    # 拒识阈值在验证集选出，写入模型；测试指标同时报告拒识前后。
    assert card["reject_threshold"] == metrics["rejection"]["threshold"]
    assert metrics["rejection"]["test_without_rejection"]["unknown_rejection"] == 0
    assert (output / "rejection_validation.csv").exists() and (output / "rejection_tradeoff.png").exists()
    model = load_gesture_model(model_path, card["sha256"])
    expected = pd.read_csv(output / "test_predictions.csv", dtype={"subject": str})
    subject = expected.subject.iloc[0]
    windows = window_gesture_record(load_uci_gesture_record(root / subject / "1_raw_data_test.txt"))
    keep = np.isin(windows.labels, LABELS)
    assert (model.predict(windows.signal[keep]) == expected[expected.subject == subject].prediction.to_numpy()).all()
    assert np.allclose(model.predict_proba(windows.signal[keep][:3]).sum(axis=1), 1)
    with pytest.raises(ValueError, match="SHA-256"):
        load_gesture_model(model_path, "0" * 64)


## @brief 识别全部窗口；有标注时按单标签窗口评估；无时间、无标签的逗号文件也可识别；错误采样间隔被拒绝。
def test_predict_files(trained, tmp_path):
    root, output, model_path = trained
    inputs = tmp_path / "in"
    inputs.mkdir()
    write_record(inputs / "labelled.txt", 5)
    frame = pd.read_csv(inputs / "labelled.txt", sep=r"\s+")
    frame[[f"channel{i}" for i in range(1, 9)]].to_csv(inputs / "channels_only.csv", index=False)
    slow = frame.copy()
    slow["time"] = slow["time"] * 5
    slow.to_csv(inputs / "slow.csv", index=False)
    out = tmp_path / "out"
    code = main(["--input", str(inputs), "--model", str(model_path),
                 "--model-card", str(output / "model_card.json"), "--output-dir", str(out)])
    assert code == 1
    summary = json.loads((out / "summary.json").read_text(encoding="utf8"))
    assert summary["task"] == "uci_gesture"
    assert summary["counts"] == {"total": 3, "succeeded": 2, "failed": 1}
    files = {item["file"].replace("\\", "/").split("/")[-1]: item for item in summary["files"]}
    assert files["labelled.txt"]["windows"] == 13
    assert files["labelled.txt"]["evaluation"]["windows"] == 6
    assert "evaluation" not in files["channels_only.csv"]
    assert "间隔" in files["slow.csv"]["error"]
    assert files["labelled.txt"]["status"] == "ok"
    assert files["slow.csv"]["status"] == "error"
    labelled = pd.read_csv(out / "labelled_predictions.csv")
    only = pd.read_csv(out / "channels_only_predictions.csv")
    assert (labelled.prediction == only.prediction).all()
    assert set(labelled.prediction) <= {0, *LABELS} and set(labelled.raw_prediction) <= set(LABELS)
    unknown = labelled.prediction == 0
    assert (labelled.raw_prediction[~unknown] == labelled.prediction[~unknown]).all()
    assert "rejection" in files["labelled.txt"] or not summary["reject_threshold"]
    # 输出保留 4 位小数，六类概率之和与 1 的差不超过舍入误差。
    assert np.allclose(labelled[[f"prob_{c}" for c in LABELS]].sum(axis=1), 1, atol=6e-4)
    assert (out / "labelled_gestures.png").stat().st_size > 0


## @brief 含时间断点的窗口被丢弃，与实验分窗规则一致。
def test_window_starts_drop_gaps():
    timestamps = np.r_[np.arange(300), np.arange(300) + 400.]
    starts = window_starts(np.zeros((600, 8)), timestamps, 200, 100, 20)
    assert starts.tolist() == [0, 100, 300, 400]


## @brief 阈值拒识与指标：低置信度改为未知，已知覆盖率与未知拒识率按定义计算。
def test_rejection_scores():
    from emginsight.uci_gesture.rejection import apply_threshold, choose_threshold, rejection_scores
    proba = np.array([[0.9, 0.1], [0.55, 0.45], [0.2, 0.8], [0.5, 0.5]])
    pred = apply_threshold(proba, (1, 2), 0.6)
    assert pred.tolist() == [1, 0, 2, 0]
    status = np.array([1, 0, 2, -1])
    result = rejection_scores(status, pred, (1, 2))
    assert result["known_coverage"] == 1 and result["unknown_rejection"] == 1
    table, threshold = choose_threshold(proba, status, (1, 2))
    assert threshold > 0.55 and table.threshold.iloc[0] == 0
