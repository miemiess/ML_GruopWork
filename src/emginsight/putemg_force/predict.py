"""用已训练的 D 任务模型预测原始肌电 CSV 的十路手指力量（批处理部署入口）。

用法：python -m emginsight.putemg_force.predict --input 记录.csv [更多.csv 或目录] --output-dir 输出目录
"""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from emginsight._batch import expand_inputs, run_batch

from .model import LEVEL_NAMES_EN, load_force_model, load_level_model
from .preprocessing import BAND_HZ, CALIBRATION_SECONDS, FS, calibration_is_rest, record_features
from .reading import ForceRecord
from .targets import level_labels, target_metrics

EMG = tuple(f"EMG_{i}" for i in range(1, 25))
TIME_NAMES = ("Unnamed: 0", "Timestamp", "TIMESTAMP", "time")
DRIFT_Z = 3.0


## @brief 读取部署输入：时间列可选，24 路肌电必需，实测力量可选（仅用于对照评估）。
# @param path CSV 路径；文件名不要求符合 putEMG 命名。
# @param targets 模型输出的力量列名。
# @return tuple (ForceRecord, 是否含实测力量, 是否含 TRAJ 列, FORCE_MVC 或 None)。
# @exception ValueError 缺少肌电列、非有限值、时间不递增或采样率与模型不符。
def read_input(path, targets):
    path = Path(path)
    header = pd.read_csv(path, nrows=0).columns
    missing = [name for name in EMG if name not in header]
    if missing:
        raise ValueError(f"缺少肌电列 {missing[:3]}…，需要 EMG_1 至 EMG_24。")
    time_name = next((name for name in TIME_NAMES if name in header), None)
    has_force = all(name in header for name in targets)
    has_mvc = has_force and "FORCE_MVC" in header
    columns = (list(EMG) + (list(targets) if has_force else []) + (["FORCE_MVC"] if has_mvc else [])
               + ([time_name] if time_name else []))
    frame = pd.read_csv(path, usecols=columns, dtype=np.float64)
    values = frame.to_numpy()
    if not len(frame) or not np.isfinite(values).all():
        raise ValueError("文件为空，或包含缺失、非有限数值。")
    timestamps = frame[time_name].to_numpy() if time_name else np.arange(len(frame)) / FS
    delta = np.diff(timestamps)
    if np.any(delta <= 0):
        raise ValueError("时间必须严格递增。")
    rate = 1 / np.median(delta)
    if abs(rate - FS) / FS > 0.02:
        raise ValueError(f"估计采样率 {rate:.1f} Hz，模型只适用于约 {FS:.0f} Hz 的 putEMG 信号。")
    force = frame[list(targets)].to_numpy() if has_force else np.zeros((len(frame), len(targets)))
    record = ForceRecord(signal=frame[list(EMG)].to_numpy(), time=timestamps, force=force, subject_id="input",
                         session_id="input", record_id=path.stem, trial_type="input", target_names=tuple(targets))
    mvc = float(frame.FORCE_MVC.iloc[0]) if has_mvc else None
    return record, has_force, any(name.startswith("TRAJ_") for name in header), mvc


## @brief 比较本记录静息肌电与训练记录的分布，提示佩戴或设备漂移。
# @param rest 本记录静息特征 (72,)。
# @param reference 训练记录静息对数 RMS 的均值与标准差。
# @return dict 平均 |z|、超阈值通道和是否告警。
def drift_check(rest, reference):
    if reference is None:
        return {"available": False}
    log_rms = np.log(np.maximum(rest[0::3], 1e-6))
    z = (log_rms - np.asarray(reference["mean"])) / np.asarray(reference["std"])
    flagged = [f"EMG_{i + 1}" for i in np.flatnonzero(np.abs(z) > DRIFT_Z)]
    return {"available": True, "mean_abs_z": float(np.abs(z).mean()), "flagged_channels": flagged,
            "warning": bool(len(flagged) >= 6), "rule": f"静息对数 RMS 与训练记录相比 |z|>{DRIFT_Z} 的通道达到 6 路时告警"}


## @brief 绘制一条记录十路力量的预测曲线；有实测值时一并画出。
# @param frame 逐窗结果。
# @param targets 力量列名。
# @param path 输出 PNG。
# @param title 图标题（英文，容器中不依赖中文字体）。
def plot_prediction(frame, targets, path, title):
    from matplotlib.figure import Figure
    figure = Figure(figsize=(13, 17), layout="constrained")
    axes = figure.subplots(len(targets), 1, sharex=True)
    for ax, name in zip(axes, targets):
        if f"{name}_true" in frame:
            ax.plot(frame.time_s, frame[f"{name}_true"], color="#3d3d3a", lw=0.9, label="measured (rest-referenced)")
        ax.plot(frame.time_s, frame[f"{name}_pred"], color="#2a78d6", lw=0.9, label="predicted")
        ax.set_ylabel(name, rotation=0, ha="right", va="center", fontsize=9)
        ax.grid(axis="y", color="#e4e3dd", lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(labelsize=8)
    axes[0].legend(ncol=2, loc="lower left", bbox_to_anchor=(0, 1.02), frameon=False, fontsize=9)
    axes[-1].set_xlabel("time in record (s, window centre)")
    figure.suptitle(title, fontsize=11)
    figure.savefig(path, dpi=100)


## @brief 绘制三档力量的时间线与各档概率；有实测标签时一并画出。
# @param frame 逐窗结果。
# @param classes 档位编码。
# @param path 输出 PNG。
# @param title 图标题。
def plot_levels(frame, classes, path, title):
    from matplotlib.figure import Figure
    figure = Figure(figsize=(13, 5.5), layout="constrained")
    top, bottom = figure.subplots(2, 1, sharex=True)
    if "level_true" in frame:
        top.step(frame.time_s, frame.level_true, where="mid", color="#b9b7ad", lw=8, solid_capstyle="butt",
                 label="measured level (label)")
    top.step(frame.time_s, frame.level_pred, where="mid", color="#2a78d6", lw=1.1, label="predicted level")
    top.set_yticks(list(classes), [LEVEL_NAMES_EN[c] for c in classes])
    top.legend(ncol=2, loc="lower left", bbox_to_anchor=(0, 1.02), frameon=False, fontsize=9)
    for c, color in zip(classes, ("#9a9890", "#eda100", "#eb6834")):
        bottom.plot(frame.time_s, frame[f"level_prob_{LEVEL_NAMES_EN[c]}"], color=color, lw=1, label=LEVEL_NAMES_EN[c])
    bottom.set(ylim=(0, 1.02), ylabel="probability", xlabel="time in record (s, window centre)")
    bottom.legend(ncol=3, loc="lower left", bbox_to_anchor=(0, 1.0), frameon=False, fontsize=9)
    for ax in (top, bottom):
        ax.grid(axis="y", color="#e4e3dd", lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    figure.suptitle(title, fontsize=11)
    figure.savefig(path, dpi=100)


## @brief 预测一份 CSV：校准检查、特征提取、模型推理，并写出逐窗结果与曲线。
# @param path 输入 CSV。
# @param model 已读取的 ForceModel。
# @param output 输出目录。
# @param reference 漂移检查参考，可为 None。
# @param level_model 三档力量模型，可为 None。
# @return dict 本文件的处理摘要。
def predict_file(path, model, output, reference=None, level_model=None):
    started = time.perf_counter()
    record, has_force, has_traj, mvc = read_input(path, model.targets)
    loaded = time.perf_counter()
    window, step = model.preprocessing["window_size"], model.preprocessing["step_size"]
    calibration = int(model.preprocessing["calibration_seconds"] * FS)
    notes = []
    if has_traj:
        if not calibration_is_rest(path, calibration):
            raise ValueError(f"开头 {CALIBRATION_SECONDS} 秒内 TRAJ 有出力指令，不能作为静息校准段。")
    else:
        notes.append(f"无 TRAJ 列，按约定假设开头 {CALIBRATION_SECONDS} 秒为静息。")
    item = record_features(record, window, step, calibration)
    rest = np.broadcast_to(item["rest"], item["X"].shape)
    records = np.full(len(item["X"]), record.record_id)
    pred = model.predict(item["X"], rest, records)
    if level_model is not None:
        level_proba = level_model.predict_proba(item["X"], rest, records)
        level = np.asarray(level_model.classes)[level_proba.argmax(axis=1)]
    finished = time.perf_counter()

    frame = pd.DataFrame({"start_sample": item["starts"], "time_s": item["time_s"]})
    for i, name in enumerate(model.targets):
        frame[f"{name}_pred"] = pred[:, i]
    if level_model is not None:
        frame["level_pred"] = level
        frame["level_name"] = [LEVEL_NAMES_EN[int(c)] for c in level]
        for i, c in enumerate(level_model.classes):
            frame[f"level_prob_{LEVEL_NAMES_EN[c]}"] = level_proba[:, i]
    summary = {"file": str(path), "windows": len(frame), "duration_s": float(record.time[-1] - record.time[0]),
               "seconds": {"read": loaded - started, "features_and_predict": finished - loaded},
               "ms_per_window": 1000 * (finished - loaded) / len(frame), "notes": notes,
               "drift": drift_check(item["rest"], reference)}
    if has_force:
        truth = item["y"] - item["tare"]
        for i, name in enumerate(model.targets):
            frame[f"{name}_true"] = truth[:, i]
        scores = target_metrics(truth, pred)
        summary["evaluation"] = {k: float(scores[k].mean()) for k in ("mae", "rmse", "r2")}
        if level_model is not None and mvc:
            # 与三档实验相同的标签规则：第二强传感器去漂移后除以 FORCE_MVC。
            truth_level = level_labels(truth, mvc, step / FS)[0]
            frame["level_true"] = truth_level
            summary["level_evaluation"] = {
                "accuracy": float(accuracy_score(truth_level, level)),
                "macro_f1": float(f1_score(truth_level, level, labels=list(level_model.classes), average="macro",
                                           zero_division=0))}
    if level_model is not None:
        summary["level_share"] = {LEVEL_NAMES_EN[c]: float(np.mean(level == c)) for c in level_model.classes}
    stem = Path(path).stem
    frame.round(4).to_csv(output / f"{stem}_predictions.csv", index=False)
    if level_model is not None:
        plot_levels(frame, level_model.classes, output / f"{stem}_levels.png",
                    f"{stem} | whole-hand force level (rest / light ~25% MVC / heavy ~50% MVC)")
    plot_prediction(frame, model.targets, output / f"{stem}_curves.png",
                    f"{stem} | force relative to rest (raw putEMG units, not newtons)")
    return summary


## @brief 命令行入口：读取模型，逐个预测输入文件，写出汇总 JSON。
# @param argv 可选参数列表。
# @return int 进程退出码；任一文件失败时为 1。
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", "--csv", type=Path, nargs="+", required=True,
                        help="一个或多个 CSV，或包含 CSV 的目录")
    parser.add_argument("--model", type=Path, default=Path("results/runs/putemg_force/force_model.joblib"))
    parser.add_argument("--model-card", type=Path, default=None, help="默认读取模型同目录或结果目录的 model_card.json")
    parser.add_argument("--reference", type=Path, default=None, help="静息肌电漂移参考 JSON（可选）")
    parser.add_argument("--level-model", type=Path, default=Path("results/runs/putemg_force/force_level_model.joblib"),
                        help="三档力量模型；文件不存在时只输出连续力量")
    parser.add_argument("--level-model-card", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("predictions"))
    args = parser.parse_args(argv)

    card_path = args.model_card or next((p for p in (args.model.parent / "model_card.json",
                                                     Path("results/putemg_force/regression/model_card.json")) if p.exists()), None)
    sha256 = json.loads(card_path.read_text(encoding="utf-8"))["sha256"] if card_path else None
    model = load_force_model(args.model, sha256)
    ref_path = args.reference or next((p for p in (args.model.parent / "monitoring_reference.json",
                                                   Path("results/putemg_force/regression/monitoring_reference.json")) if p.exists()), None)
    reference = json.loads(ref_path.read_text(encoding="utf-8")) if ref_path else None
    level_model = None
    if args.level_model.exists():
        level_card = args.level_model_card or next(
            (p for p in (args.level_model.parent / "level_model_card.json",
                         Path("results/putemg_force/levels/model_card.json")) if p.exists()), None)
        level_sha = json.loads(level_card.read_text(encoding="utf-8"))["sha256"] if level_card else None
        level_model = load_level_model(args.level_model, level_sha)
        keys = ("window_size", "step_size", "calibration_seconds")
        if any(level_model.preprocessing[k] != model.preprocessing[k] for k in keys):
            parser.error("三档模型与回归模型的窗口或校准设置不一致，不能共用同一批特征。")
    files = expand_inputs(args.input, {".csv"})
    if not files:
        parser.error("没有找到输入 CSV。")

    def predict(path):
        summary = predict_file(path, model, args.output_dir, reference, level_model)
        details = ""
        if "evaluation" in summary:
            details += "，对照实测 RMSE {rmse:.2f}、R² {r2:.3f}".format(**summary["evaluation"])
        if "level_evaluation" in summary:
            details += "，力量三档 macro-F1 {macro_f1:.3f}".format(**summary["level_evaluation"])
        if summary["drift"].get("warning"):
            details += "，[漂移告警] 静息肌电与训练数据差异较大"
        return summary, details

    report = {
        "model": {"path": str(args.model), "sha256_checked": sha256 is not None, **model.metrics},
        "level_model": None if level_model is None else {
            "path": str(args.level_model), **level_model.metrics,
            "levels": {str(c): LEVEL_NAMES_EN[c] for c in level_model.classes},
        },
        "unit": "相对静息的 putEMG 原始力量数值，不是牛顿",
        "filter_hz": list(BAND_HZ),
    }
    return run_batch(files, args.output_dir, "putemg_force", report, predict)


if __name__ == "__main__":
    sys.exit(main())
