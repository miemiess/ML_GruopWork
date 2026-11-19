"""按用户留出的 UCI 六类手势识别实验。"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from matplotlib.figure import Figure
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score,
    precision_recall_fscore_support,
)

from emginsight.features import extract_time_features
from emginsight.models import KNNClassifier, LogisticRegression, MajorityClassifier, Standardizer
from emginsight.uci_gesture import load_uci_gesture_record, split_subject_ids, window_gesture_record

LABELS = (1, 2, 3, 4, 5, 6)
FEATURE_SETS = {"RMS+MAV": (0, 1), "RMS+MAV+WL": (0, 1, 2)}
SEED = 42


## @brief 先划分用户，再逐条记录提取特征并保留窗口来源。
# @param root 数据集根目录。
# @return 各组特征与窗口索引、用户名单和跳过记录的原因。
def prepare_data(root):
    paths = sorted(root.glob("[0-9]*/*raw_data*.txt"))
    if not paths:
        raise ValueError(f"找不到手势记录，请检查数据目录 {root}")
    groups = split_subject_ids([p.parent.name for p in paths], seed=SEED)
    owner = {subject: group for group, subjects in groups.items() for subject in subjects}
    features = {name: [] for name in groups}
    metadata = {name: [] for name in groups}
    skipped = []
    for path in paths:
        try:
            record = load_uci_gesture_record(path)
        except ValueError as error:
            skipped.append({"record": path.relative_to(root).as_posix(), "reason": str(error)})
            continue
        windows = window_gesture_record(record, window_size=200, step_size=100, max_gap_ms=20)
        keep = np.isin(windows.labels, LABELS)
        if not keep.any():
            continue
        group = owner[record.subject_id]
        features[group].append(extract_time_features(windows.signal[keep]))
        metadata[group].append(pd.DataFrame({
            "subject": windows.subject_ids[keep], "record": windows.record_ids[keep],
            "start_sample": windows.start_indices[keep], "label": windows.labels[keep],
        }))
    datasets = {}
    for group in groups:
        if not features[group]:
            raise ValueError(f"{group} 没有可用窗口。")
        index = pd.concat(metadata[group], ignore_index=True)
        if set(index.label) != set(LABELS):
            raise ValueError(f"{group} 未覆盖预设的六类，无法进行本实验。")
        datasets[group] = np.concatenate(features[group]), index
    return datasets, groups, skipped


## @brief 仅用训练用户拟合，按验证 macro-F1 选择特征、分类器及参数。
# @param train 训练特征矩阵与窗口索引。
# @param validation 验证特征矩阵与窗口索引。
# @return 候选结果表和获选模型、标准化器、特征列。
def select_model(train, validation):
    X, train_index = train
    X_val, val_index = validation
    rows = []
    best_score = -1
    for feature_name, offsets in FEATURE_SETS.items():
        columns = np.flatnonzero(np.isin(np.arange(X.shape[1]) % 3, offsets))
        scaler = Standardizer().fit(X[:, columns])
        scaled = scaler.transform(X[:, columns])
        val_scaled = scaler.transform(X_val[:, columns])
        candidates = [
            ("logistic", f"l2={l2}", LogisticRegression(l2=l2)) for l2 in (0.001, 0.01)
        ] + [
            ("knn", f"k={k}", KNNClassifier(k=k)) for k in (5, 15)
        ] + [
            ("forest", f"trees={n}", RandomForestClassifier(n_estimators=n, random_state=SEED))
            for n in (20, 100)
        ]
        for name, parameter, model in candidates:
            model.fit(scaled, train_index.label)
            pred = model.predict(val_scaled)
            score = f1_score(val_index.label, pred, labels=LABELS, average="macro", zero_division=0)
            rows.append(dict(features=feature_name, model=name, parameter=parameter,
                             macro_f1=score, accuracy=accuracy_score(val_index.label, pred)))
            if score > best_score:
                best_score = score
                best = model, scaler, columns
    return pd.DataFrame(rows), best


## @brief 用固定测试集评估获选模型和基线，输出逐窗预测、分类指标和混淆矩阵。
# @param root 已解压的数据集目录。
# @param output 保存结果的目录。
def run(root, output):
    data, groups, skipped = prepare_data(root)
    scores, (model, scaler, columns) = select_model(data["train"], data["validation"])
    X_test, predictions = data["test"]
    predictions = predictions.copy()
    predictions["prediction"] = model.predict(scaler.transform(X_test[:, columns]))
    X_train, train_index = data["train"]
    baseline = MajorityClassifier().fit(X_train, train_index.label)
    predictions["baseline"] = baseline.predict(X_test)
    summary = {
        "seed": SEED, "subjects": groups, "sklearn_version": sklearn.__version__,
        "windows": {group: len(index) for group, (_, index) in data.items()},
        "skipped": skipped, "selected": scores.loc[scores.macro_f1.idxmax()].to_dict(),
        "test": {},
    }
    for name in ("prediction", "baseline"):
        summary["test"][name] = {
            "accuracy": accuracy_score(predictions.label, predictions[name]),
            "macro_f1": f1_score(predictions.label, predictions[name], labels=LABELS, average="macro"),
        }
    precision, recall, f1, support = precision_recall_fscore_support(
        predictions.label, predictions.prediction, labels=LABELS, zero_division=0)
    per_class = pd.DataFrame(dict(label=LABELS, precision=precision, recall=recall, f1=f1, support=support))
    matrix = confusion_matrix(predictions.label, predictions.prediction, labels=LABELS)
    output.mkdir(parents=True, exist_ok=True)
    scores.to_csv(output / "validation.csv", index=False)
    predictions.to_csv(output / "test_predictions.csv", index=False)
    per_class.to_csv(output / "per_class.csv", index=False)
    pd.DataFrame(matrix, index=LABELS, columns=LABELS).to_csv(output / "confusion.csv", index_label="true_label")
    figure = Figure(figsize=(6, 5), layout="constrained")
    ax = figure.subplots()
    ConfusionMatrixDisplay(matrix, display_labels=LABELS).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("UCI gestures | held-out subjects")
    figure.savefig(output / "confusion.png", dpi=150)
    (output / "metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"获选配置 {summary['selected']}\n测试结果 {summary['test']}\n异常记录 {len(skipped)} 份，详见 {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/EMG_data_for_gestures-master"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/uci_gesture/classification"))
    args = parser.parse_args()
    run(args.data_dir, args.output_dir)
