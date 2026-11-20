"""用已训练的 C 任务模型识别原始 8 路肌电文件中的手势（批处理部署入口）。

用法：python -m emginsight.gesture.predict --input 记录.txt [更多文件或目录] --output-dir 输出目录
"""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from emginsight.common.batch import expand_inputs, run_batch

from .model import LABEL_NAMES_EN, UNKNOWN, load_gesture_model
from .rejection import rejection_scores

CHANNELS = tuple(f"channel{i}" for i in range(1, 9))
LOW_CONFIDENCE = 0.5
COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300")


## @brief 读取 UCI 格式的原始文件（空白或逗号分隔）；时间与 class 列可选。
# @param path 输入文件。
# @return tuple 信号 (N, 8)、毫秒时间 (N,)、标签 (N,) 或 None。
# @exception ValueError 缺少通道、非有限值、时间倒退或采样间隔与 UCI（约 1 ms）不符。
def read_input(path):
    path = Path(path)
    with path.open(encoding="utf-8-sig") as stream:
        first = stream.readline()
    frame = pd.read_csv(path, sep="," if "," in first else r"\s+", encoding="utf-8-sig")
    missing = [name for name in CHANNELS if name not in frame.columns]
    if missing:
        raise ValueError(f"缺少通道列 {missing}，需要 channel1 至 channel8。")
    signal = frame[list(CHANNELS)].to_numpy(dtype=float)
    if not len(signal) or not np.isfinite(signal).all():
        raise ValueError("文件为空，或通道包含缺失、非有限数值。")
    timestamps = frame["time"].to_numpy(dtype=float) if "time" in frame else np.arange(len(frame), dtype=float)
    if not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) < 0):
        raise ValueError("时间列必须为有限数值且不能倒退。")
    if len(timestamps) > 1:
        step = np.median(np.diff(timestamps))
        if not 0.8 <= step <= 1.25:
            raise ValueError(f"相邻采样的中位间隔为 {step:.3g} ms，模型只适用于约 1 kHz 的 UCI 信号。")
    labels = frame["class"].to_numpy() if "class" in frame else None
    if labels is not None and (not np.isfinite(labels).all() or np.any((labels < 0) | (labels > 7))):
        raise ValueError("class 列必须为 0–7 的整数。")
    return signal, timestamps, None if labels is None else labels.astype(int)


## @brief 按模型的窗长和步长切出全部完整窗口，丢弃内部有时间断点的窗口；不依赖标签。
# @param signal 原始信号。
# @param timestamps 毫秒时间。
# @param size 窗口点数。
# @param step 步长。
# @param max_gap 窗口内允许的最大相邻时间差（毫秒）。
# @return ndarray 保留窗口的起点索引。
def window_starts(signal, timestamps, size, step, max_gap):
    starts = np.arange(0, max(len(signal) - size + 1, 0), step)
    breaks = np.r_[0, np.cumsum(np.diff(timestamps) > max_gap)]
    return starts[breaks[starts + size - 1] == breaks[starts]] if len(starts) else starts


## @brief 绘制预测手势随时间的变化及六类概率；有标注时一并画出。
# @param frame 逐窗结果。
# @param classes 类别编码。
# @param path 输出 PNG。
# @param title 图标题。
def plot_prediction(frame, classes, path, title):
    from matplotlib.figure import Figure
    figure = Figure(figsize=(13, 6.5), layout="constrained")
    top, bottom = figure.subplots(2, 1, sharex=True, height_ratios=(1, 1.2))
    seconds = frame.time_ms / 1000
    if "label" in frame:
        labelled = frame.label.where(frame.label.isin(classes))
        top.step(seconds, labelled, where="mid", color="#b9b7ad", lw=9, solid_capstyle="butt",
                 label="annotated (single-label windows)", zorder=1)
    top.step(seconds, frame.prediction, where="mid", color="#2a78d6", lw=1.1, label="predicted")
    shown = [UNKNOWN, *classes] if (frame.prediction == UNKNOWN).any() else list(classes)
    top.set_yticks(shown, [LABEL_NAMES_EN[c] for c in shown], fontsize=8)
    top.legend(ncol=2, loc="lower left", bbox_to_anchor=(0, 1.02), frameon=False, fontsize=9)
    for c, color in zip(classes, COLORS):
        bottom.plot(seconds, frame[f"prob_{c}"], color=color, lw=1, label=LABEL_NAMES_EN[c])
    bottom.set_ylabel("probability")
    bottom.set_ylim(0, 1.02)
    bottom.legend(ncol=6, loc="lower left", bbox_to_anchor=(0, 1.0), frameon=False, fontsize=8)
    bottom.set_xlabel("time (s, window centre)")
    for ax in (top, bottom):
        ax.grid(axis="y", color="#e4e3dd", lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    figure.suptitle(title, fontsize=11)
    figure.savefig(path, dpi=100)


## @brief 识别一个文件：分窗、推理，写出逐窗结果与图。
# @param path 输入文件。
# @param model 已读取的 GestureModel。
# @param output 输出目录。
# @return dict 本文件的处理摘要。
def predict_file(path, model, output):
    started = time.perf_counter()
    signal, timestamps, labels = read_input(path)
    loaded = time.perf_counter()
    size, step = model.preprocessing["window_size"], model.preprocessing["step_size"]
    starts = window_starts(signal, timestamps, size, step, model.preprocessing["max_gap_ms"])
    if not len(starts):
        raise ValueError(f"记录不足一个 {size} 点的完整窗口。")
    windows = signal[starts[:, None] + np.arange(size)]
    prediction, raw, proba = model.predict_with_unknown(windows)
    finished = time.perf_counter()

    classes = tuple(model.classes)
    frame = pd.DataFrame({"start_sample": starts, "time_ms": timestamps[starts + size // 2], "prediction": prediction,
                          "prediction_name": [LABEL_NAMES_EN[int(c)] for c in prediction], "raw_prediction": raw})
    for i, c in enumerate(classes):
        frame[f"prob_{c}"] = proba[:, i]
    confidence = proba.max(axis=1)
    summary = {"file": str(path), "windows": len(frame), "duration_s": float((timestamps[-1] - timestamps[0]) / 1000),
               "seconds": {"read": loaded - started, "features_and_predict": finished - loaded},
               "ms_per_window": 1000 * (finished - loaded) / len(frame),
               "predicted_share": {LABEL_NAMES_EN[c]: float(np.mean(prediction == c)) for c in (UNKNOWN, *classes)},
               "reject_threshold": model.reject_threshold,
               "monitoring": {"mean_confidence": float(confidence.mean()),
                              "low_confidence_share": float(np.mean(confidence < LOW_CONFIDENCE)),
                              "rule": f"最大类别概率低于 {LOW_CONFIDENCE} 视为低置信度"}}
    if labels is not None:
        # 与实验口径一致：只评估窗口内标签统一且属于六类的窗口。
        rows = starts[:, None] + np.arange(size)
        uniform = (labels[rows] == labels[starts][:, None]).all(axis=1)
        frame["label"] = np.where(uniform, labels[starts], -1)
        keep = frame.label.isin(classes).to_numpy()
        if keep.any():
            summary["evaluation"] = {"windows": int(keep.sum()),
                                     "accuracy": float(accuracy_score(frame.label[keep], frame.raw_prediction[keep])),
                                     "macro_f1": float(f1_score(frame.label[keep], frame.raw_prediction[keep],
                                                                labels=list(classes), average="macro", zero_division=0))}
        if model.reject_threshold:
            # 拒识口径：单标签 1–6 为已知，未标注、第 7 类与混合标签为未知。
            summary["rejection"] = rejection_scores(frame.label.to_numpy(), prediction, classes)
    stem = Path(path).stem
    frame.round(4).to_csv(output / f"{stem}_predictions.csv", index=False)
    plot_prediction(frame, classes, output / f"{stem}_gestures.png", f"{stem} | UCI gesture model (6 classes)")
    return summary


## @brief 命令行入口：读取模型，逐个识别输入文件，写出汇总 JSON。
# @param argv 可选参数列表。
# @return int 进程退出码；任一文件失败时为 1。
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, nargs="+", required=True, help="一个或多个 .txt/.csv，或包含它们的目录")
    parser.add_argument("--model", type=Path, default=Path("results/runs/gesture/gesture_model.joblib"))
    parser.add_argument("--model-card", type=Path, default=None, help="默认读取模型同目录或结果目录的 model_card.json")
    parser.add_argument("--output-dir", type=Path, default=Path("predictions"))
    args = parser.parse_args(argv)

    card_path = args.model_card or next((p for p in (args.model.parent / "model_card.json",
                                                     Path("results/gesture/classification/model_card.json")) if p.exists()), None)
    sha256 = json.loads(card_path.read_text(encoding="utf-8"))["sha256"] if card_path else None
    model = load_gesture_model(args.model, sha256)
    files = expand_inputs(args.input, {".txt", ".csv"})
    if not files:
        parser.error("没有找到输入文件。")

    def predict(path):
        summary = predict_file(path, model, args.output_dir)
        details = "，未知 {:.1%}".format(summary["predicted_share"]["unknown"])
        if "evaluation" in summary:
            details += "，对照标注 六类准确率 {accuracy:.2%}、macro-F1 {macro_f1:.3f}".format(**summary["evaluation"])
        return summary, details

    report = {
        "model": {"path": str(args.model), "sha256_checked": sha256 is not None, **model.metrics},
        "classes": {str(c): LABEL_NAMES_EN[c] for c in model.classes},
        "reject_threshold": model.reject_threshold,
        "note": "最大类别概率低于 reject_threshold 时输出 0（unknown）；raw_prediction 保留六类结果。六类准确率按 raw_prediction 计算，与实验口径一致。",
    }
    return run_batch(files, args.output_dir, "uci_gesture", report, predict)


if __name__ == "__main__":
    sys.exit(main())
