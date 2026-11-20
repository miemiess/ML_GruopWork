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
from emginsight.common.splitting import split_subject_ids
from emginsight.gesture import load_uci_gesture_record, window_gesture_record
from emginsight.gesture.model import GestureModel, load_gesture_model, save_gesture_model
from emginsight.gesture.rejection import apply_threshold, choose_threshold, rejection_scores, status_windows

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
# @param model_path 获选模型文件路径；None 表示不保存。
def run(root, output, model_path=None):
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
    rejection_table, threshold = None, None
    if hasattr(model, "predict_proba"):
        # “未知”拒识：六类模型不变，只在验证用户上选概率阈值；未标注、第 7 类和混合标签窗口视为未知。
        status = {}
        for group in ("validation", "test"):
            proba, states = [], []
            for path in sorted(root.glob("[0-9]*/*raw_data*.txt")):
                if path.parent.name not in groups[group]:
                    continue
                try:
                    windows, state = status_windows(load_uci_gesture_record(path))
                except ValueError:
                    continue
                proba.append(model.predict_proba(scaler.transform(extract_time_features(windows)[:, columns])))
                states.append(state)
            status[group] = np.concatenate(proba), np.concatenate(states)
        rejection_table, threshold = choose_threshold(*status["validation"], LABELS)
        test_proba, test_status = status["test"]
        summary["rejection"] = {
            "threshold": threshold, "rule": "最大类别概率低于阈值时输出 0（未知）；阈值按验证用户七类 macro-F1 选择",
            "unknown_definition": "窗口内全为未标注 0、全为第 7 类手掌伸展，或含多个标签",
            "test_without_rejection": rejection_scores(test_status, apply_threshold(test_proba, LABELS, None), LABELS),
            "test_with_rejection": rejection_scores(test_status, apply_threshold(test_proba, LABELS, threshold), LABELS),
        }
    precision, recall, f1, support = precision_recall_fscore_support(
        predictions.label, predictions.prediction, labels=LABELS, zero_division=0)
    per_class = pd.DataFrame(dict(label=LABELS, precision=precision, recall=recall, f1=f1, support=support))
    matrix = confusion_matrix(predictions.label, predictions.prediction, labels=LABELS)
    output.mkdir(parents=True, exist_ok=True)
    if model_path is not None:
        # 保存获选模型，重新读取后复算测试窗口，确认与本次预测逐一相同。
        trained = GestureModel(
            model=model, scaler=scaler, columns=tuple(int(c) for c in columns),
            features=summary["selected"]["features"], classes=LABELS,
            preprocessing={"window_size": 200, "step_size": 100, "max_gap_ms": 20, "channels": 8,
                           "feature_order": "每通道 RMS、MAV、WL，按 columns 选列"},
            train_subjects=tuple(groups["train"]), metrics=summary["test"]["prediction"],
            reject_threshold=threshold)
        info = save_gesture_model(trained, model_path)
        reloaded = load_gesture_model(model_path, info["sha256"])
        again = reloaded.model.predict(reloaded.scaler.transform(X_test[:, list(reloaded.columns)]))
        if not np.array_equal(again, predictions.prediction.to_numpy()):
            raise RuntimeError("重新读取的手势模型预测与训练时不一致。")
        info.update(verified_test_predictions=True, **trained.describe())
        (output / "model_card.json").write_text(json.dumps(info, indent=2, ensure_ascii=False, default=str),
                                                encoding="utf-8")
        print(f"模型已保存 {info['path']}（{info['bytes'] / 2**20:.1f} MB）")
    scores.to_csv(output / "validation.csv", index=False)
    if rejection_table is not None:
        rejection_table.to_csv(output / "rejection_validation.csv", index=False)
        figure = Figure(figsize=(6.4, 4), layout="constrained")
        ax = figure.subplots()
        ax.plot(rejection_table.threshold, rejection_table.known_coverage, color="#2a78d6", lw=2, marker="o", ms=4,
                label="known gestures accepted")
        ax.plot(rejection_table.threshold, rejection_table.unknown_rejection, color="#eb6834", lw=2, marker="o", ms=4,
                label="unknown windows rejected")
        ax.plot(rejection_table.threshold, rejection_table.known_accuracy_accepted, color="#1baf7a", lw=2, marker="o",
                ms=4, label="accuracy of accepted gestures")
        ax.axvline(threshold, color="#3d3d3a", lw=0.8, ls="--")
        ax.text(threshold, 0.02, f" chosen {threshold:.2f}", fontsize=8, color="#3d3d3a")
        ax.set(xlabel="probability threshold", ylabel="share", ylim=(0, 1.02),
               title="Unknown rejection | validation users")
        ax.grid(axis="y", color="#e4e3dd", lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(frameon=False, fontsize=8, loc="center left")
        figure.savefig(output / "rejection_tradeoff.png", dpi=150)
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
    parser.add_argument("--output-dir", type=Path, default=Path("results/gesture/classification"))
    parser.add_argument("--model-path", type=Path, default=Path("results/runs/gesture/gesture_model.joblib"),
                        help="获选模型文件（Git 忽略）")
    args = parser.parse_args()
    run(args.data_dir, args.output_dir, args.model_path)
