"""固定随机森林配置，用验证用户选择通道并比较 1/2/4/8 通道。"""

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from matplotlib.figure import Figure
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from emginsight.gesture.experiment import LABELS, SEED, prepare_data
from emginsight.models import MajorityClassifier, Standardizer


def feature_columns(channels):
    """通道编号为 1..8；每个通道保留 RMS、MAV、WL 三列。"""
    channels = tuple(channels)
    if not channels or len(set(channels)) != len(channels) or any(
        not isinstance(c, (int, np.integer)) or c < 1 or c > 8 for c in channels
    ):
        raise ValueError("通道必须为不重复的 1..8 整数。")
    return np.array([3 * (c - 1) + offset for c in sorted(channels) for offset in range(3)])


def fit_subset(train, validation, channels):
    columns = feature_columns(channels)
    X, index = train
    X_val, val_index = validation
    scaler = Standardizer().fit(X[:, columns])
    model = RandomForestClassifier(n_estimators=100, random_state=SEED)
    model.fit(scaler.transform(X[:, columns]), index.label)
    pred = model.predict(scaler.transform(X_val[:, columns]))
    return model, scaler, columns, {
        "validation_macro_f1": f1_score(val_index.label, pred, labels=LABELS,
                                        average="macro", zero_division=0),
        "validation_accuracy": accuracy_score(val_index.label, pred),
    }


def run(root, output):
    """先完成全部验证选型，再访问测试特征；不应用未知拒识。"""
    root, output = Path(root), Path(output)
    data, groups, skipped = prepare_data(root)
    selected, candidates, snapshots = [], [], {}
    # 前向贪心：每步加入验证 macro-F1 最好的通道；并列时编号小者优先。
    # 固定一路嵌套子集，便于观察增加传感器的效果；不宣称全局最优。
    for count in range(1, 9):
        best = None
        for channel in range(1, 9):
            if channel in selected:
                continue
            channels = sorted(selected + [channel])
            model, scaler, columns, scores = fit_subset(data["train"], data["validation"], channels)
            candidates.append({"step": count, "added_channel": channel,
                               "channels": ",".join(map(str, channels)), **scores})
            if best is None or scores["validation_macro_f1"] > best[0]:
                best = scores["validation_macro_f1"], channels, model, scaler, columns, scores
        _, selected, model, scaler, columns, scores = best
        if count in (1, 2, 4, 8):
            snapshots[count] = selected.copy(), model, scaler, columns, scores
        print(f"验证选型 {count} 通道：{selected}，macro-F1={best[0]:.4f}", flush=True)

    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(candidates).to_csv(output / "validation.csv", index=False)
    X_test, test_index = data["test"]
    majority = MajorityClassifier().fit(data["train"][0], data["train"][1].label)
    baseline = majority.predict(X_test)
    baseline_f1 = f1_score(test_index.label, baseline, labels=LABELS, average="macro", zero_division=0)
    rows, user_rows, class_rows, predictions = [], [], [], []
    for count, (channels, model, scaler, columns, scores) in snapshots.items():
        pred = model.predict(scaler.transform(X_test[:, columns]))
        frame = test_index.copy()
        frame["channel_count"], frame["prediction"], frame["baseline"] = count, pred, baseline
        predictions.append(frame)
        rows.append({"channel_count": count, "channels": ",".join(map(str, channels)),
                     "feature_count": len(columns), **scores,
                     "test_macro_f1": f1_score(frame.label, pred, labels=LABELS, average="macro", zero_division=0),
                     "test_accuracy": accuracy_score(frame.label, pred)})
        for subject, part in frame.groupby("subject"):
            user_rows.append({"channel_count": count, "subject": subject, "windows": len(part),
                              "macro_f1": f1_score(part.label, part.prediction, labels=LABELS,
                                                   average="macro", zero_division=0),
                              "accuracy": accuracy_score(part.label, part.prediction)})
        for label in LABELS:
            class_rows.append({"channel_count": count, "label": label,
                               "f1": f1_score(frame.label, pred, labels=[label], average=None,
                                              zero_division=0)[0],
                               "support": int((frame.label == label).sum())})
        pd.DataFrame(confusion_matrix(frame.label, pred, labels=LABELS), index=LABELS,
                     columns=LABELS).to_csv(output / f"confusion_{count}.csv", index_label="true_label")
    results = pd.DataFrame(rows)
    reference = results.loc[results.channel_count == 8, "test_macro_f1"].iloc[0]
    results["f1_drop_from_8"] = reference - results.test_macro_f1
    results.to_csv(output / "comparison.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(output / "test_predictions.csv", index=False)
    pd.DataFrame(user_rows).to_csv(output / "per_user.csv", index=False)
    pd.DataFrame(class_rows).to_csv(output / "per_class.csv", index=False)
    manifest = [{"path": p.relative_to(root).as_posix(),
                 "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                for p in sorted(root.glob("[0-9]*/*raw_data*.txt"))]
    summary = {"seed": SEED, "sklearn_version": sklearn.__version__, "labels": LABELS,
               "python_version": platform.python_version(), "numpy_version": np.__version__,
               "pandas_version": pd.__version__,
               "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "dataset_url": "https://archive.ics.uci.edu/static/public/481/emg+data+for+gestures.zip",
               "subjects": groups, "skipped": skipped, "data_manifest": manifest,
               "windows": {name: len(index) for name, (_, index) in data.items()},
               "model": {"n_estimators": 100, "features": "RMS+MAV+WL"},
               "selection": "nested forward selection on validation macro-F1; ascending-channel tie break",
               "baseline_macro_f1": baseline_f1, "results": rows,
               "limitations": "Existing test users were previously evaluated in the main project; this is an exploratory extension, not a new untouched holdout."}
    (output / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    figure = Figure(figsize=(7, 4), layout="constrained")
    ax = figure.subplots()
    ax.plot(results.channel_count, results.test_macro_f1, "o-", label="Test macro-F1")
    ax.plot(results.channel_count, results.validation_macro_f1, "s--", label="Validation macro-F1")
    ax.axhline(baseline_f1, color="gray", linestyle=":", label="Majority baseline (test)")
    ax.set(xlabel="Number of EMG channels", ylabel="Macro-F1", xticks=[1, 2, 4, 8], ylim=(0, 1))
    ax.legend()
    figure.savefig(output / "channel_comparison.png", dpi=160)
    table = "|通道数|通道编号|验证 F1|测试 F1|测试准确率|较八路 F1 降幅|\n|---|---|---|---|---|---|\n"
    for row in results.itertuples():
        table += f"|{row.channel_count}|{row.channels}|{row.validation_macro_f1:.4f}|{row.test_macro_f1:.4f}|{row.test_accuracy:.4f}|{row.f1_drop_from_8:.4f}|\n"
    report = """# 手势通道减少实验

## 方法
复用六类手势任务、seed=42 用户划分、200 点窗口、100 点步长与 20 ms 断点阈值。
固定原实验获选的 100 棵随机森林及 RMS/MAV/WL；标准化仅在训练用户上拟合。
从零开始逐步加入验证集 macro-F1 最好的通道，并列时选编号较小者；报告嵌套的 1/2/4/8 路子集。
全部子集确定后才计算测试指标，不按测试成绩挑选通道。这里不应用未知拒识，准确率的分母是全部六类有效测试窗口。

## 结果
""" + table + f"\n训练多数类基线测试 macro-F1：{baseline_f1:.4f}。\n\n" + """![通道对比](channel_comparison.png)

## 解读与限制
表中降幅为八通道测试 F1 减去对应子集 F1，负数表示子集更高。更少通道是否值得使用，应结合实际降幅判断，不预设性能一定随通道数单调增长。
前向选择是贪心搜索，不保证所有同大小组合中的最优子集；同一验证集的多次比较也可能产生选择偏差。
各用户指标见 per_user.csv；不同用户窗口相关，不能把窗口数当独立样本数来声称统计显著。
该测试用户集合在原项目中已经使用过，本实验属于探索性扩展，不是新的独立未触碰测试集。
通道编号只对应本数据集传感器；未实际测量硬件成本、延迟或耗电，不能据此声称减少了多少成本。

## 复现
在仓库根目录安装 requirements.txt 并准备 UCI 原始数据后运行：

```bash
python -m emginsight.gesture.channel_ablation
```

输出目录默认 results/gesture/channel_ablation，与原分类报告分开。metrics.json 保存数据哈希、实际用户名单、参数与版本。
"""
    (output / "report.md").write_text(report, encoding="utf-8")
    print(results.to_string(index=False), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/EMG_data_for_gestures-master"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/gesture/channel_ablation"))
    args = parser.parse_args()
    run(args.data_dir, args.output_dir)
