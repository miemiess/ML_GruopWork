"""右肱二头肌单路疲劳分类：固定用户留出、NumPy 模型及原始信号预测入口。"""

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import platform
import subprocess

import numpy as np
import pandas as pd
import scipy
from emginsight.common.signal import bandpass

from .reading import discover_fatigue_pairs, load_fatigue_record
from emginsight.common.splitting import split_subject_ids
from .windowing import window_fatigue_record


FS = 1259.0
WINDOW = 5036
STEP = 2518
FEATURE_NAMES = ("log_rms", "log_mav", "log_mean_waveform_length", "mean_frequency_hz", "median_frequency_hz")
LABEL_NAMES = {0: "无疲劳", 1: "向疲劳过渡", 2: "疲劳"}


## @brief 对连续单路信号作固定的因果 20～450 Hz 四阶带通滤波。
# @param signal 原始伏特信号，一维数组。
# @return ndarray 滤波后的信号，不使用未来采样，不估计任何测试用户归一化参数。
def filter_single_channel(signal):
    signal = np.asarray(signal, dtype=float)
    if signal.ndim != 1 or not np.isfinite(signal).all():
        raise ValueError("输入必须为有限的一维肌电信号。")
    return bandpass(signal, FS, (20, 450))


## @brief 从一批单路窗口提取幅度及频率特征，不包含时间或用户身份。
# @param windows 已滤波的窗口矩阵，形状为 (窗口数, 采样点数)。
# @param fs 已知采样率，默认协议的 1259 Hz。
# @return ndarray 五列为对数 RMS/MAV/平均波形长度、MNF、MDF。
def extract_single_channel_features(windows, fs=FS):
    windows = np.asarray(windows, dtype=float)
    if windows.ndim != 2 or windows.shape[1] < 2 or not np.isfinite(windows).all() or fs <= 0:
        raise ValueError("需要有限的二维窗口矩阵，每窗至少两点，采样率须为正。")
    centered = windows - windows.mean(axis=1, keepdims=True)
    rms = np.sqrt(np.mean(centered ** 2, axis=1))
    mav = np.mean(np.abs(centered), axis=1)
    wl = np.mean(np.abs(np.diff(centered, axis=1)), axis=1)
    power = np.abs(np.fft.rfft(centered * np.hanning(centered.shape[1]), axis=1)) ** 2
    frequencies = np.fft.rfftfreq(centered.shape[1], d=1 / fs)
    mask = (frequencies >= 20) & (frequencies <= 450)
    power, frequencies = power[:, mask], frequencies[mask]
    if not len(frequencies):
        raise ValueError("窗口太短或采样率不足，无法提取 20～450 Hz 频率特征。")
    total = power.sum(axis=1)
    mnf = np.divide(power @ frequencies, total, out=np.zeros(len(windows)), where=total > 0)
    mdf = frequencies[np.argmax(np.cumsum(power, axis=1) >= total[:, None] / 2, axis=1)]
    mdf[total == 0] = 0
    return np.column_stack([np.log10(np.maximum(rms, 1e-12)), np.log10(np.maximum(mav, 1e-12)),
                            np.log10(np.maximum(wl, 1e-12)), mnf, mdf])


class NumpyFatigueClassifier:
    """NumPy kNN / 类别平衡岭分类器；标准化统计仅在 fit 的数据上计算。"""

    ## @brief 创建分类器，固定特征列和超参数。
    # @param kind knn 或 ridge。
    # @param parameter 邻居数量或正则化强度。
    # @param columns 使用的特征列索引。
    # @return None。
    def __init__(self, kind="knn", parameter=15, columns=(0, 1, 2, 3, 4)):
        if kind not in {"knn", "ridge"} or not np.isfinite(parameter) or parameter <= 0:
            raise ValueError("分类器类型或超参数错误。")
        if kind == "knn" and int(parameter) != parameter:
            raise ValueError("kNN 邻居数必须为正整数。")
        self.kind, self.parameter, self.columns = kind, parameter, tuple(columns)

    ## @brief 用训练集拟合标准化及模型，不读取验证或测试集。
    # @param X 特征矩阵。
    # @param y 0/1/2 疲劳评分。
    # @return self 拟合后的模型。
    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float)[:, self.columns], np.asarray(y, dtype=int)
        if not len(X) or y.shape != (len(X),) or not np.isfinite(X).all() or not np.isin(y, [0, 1, 2]).all():
            raise ValueError("训练特征或标签不合法。")
        self.mean = X.mean(axis=0)
        self.scale = X.std(axis=0)
        self.scale[self.scale < 1e-12] = 1.
        self.X = (X - self.mean) / self.scale
        self.y = y.copy()
        self.weights = np.empty((0, 3))
        if self.kind == "ridge":
            design = np.column_stack([np.ones(len(X)), self.X])
            counts = np.bincount(y, minlength=3)
            weight = len(y) / (3 * np.maximum(counts[y], 1))
            regularizer = np.eye(design.shape[1]) * float(self.parameter)
            regularizer[0, 0] = 0
            self.weights = np.linalg.solve(design.T @ (design * weight[:, None]) + regularizer,
                                           design.T @ (np.eye(3)[y] * weight[:, None]))
        return self

    ## @brief 对新特征预测三类疲劳评分，使用训练期固定标准化。
    # @param X 与训练相同列顺序的特征矩阵。
    # @return ndarray 每行一个 0/1/2 整数，不输出未经校准的概率。
    def predict(self, X):
        X = np.asarray(X, dtype=float)[:, self.columns]
        if not np.isfinite(X).all():
            raise ValueError("预测特征含非有限值。")
        X = (X - self.mean) / self.scale
        if self.kind == "ridge":
            return np.argmax(np.column_stack([np.ones(len(X)), X]) @ self.weights, axis=1)
        output = []
        k = min(int(self.parameter), len(self.X))
        for batch in np.array_split(X, max(1, (len(X) + 255) // 256)):
            distance = np.maximum(np.sum(batch ** 2, axis=1)[:, None] + np.sum(self.X ** 2, axis=1)
                                  - 2 * batch @ self.X.T, 0)
            nearest = np.argsort(distance, axis=1, kind="stable")[:, :k]
            votes = np.stack([(self.y[nearest] == label).sum(axis=1) for label in range(3)], axis=1)
            output.extend(np.argmax(votes, axis=1).tolist())
        return np.asarray(output, dtype=np.int64)

    ## @brief 保存可复用模型及窗口配置，不包含原始肌电波形。
    # @param path 本地模型文件路径，建议位于被 Git 忽略的 results/runs/。
    # @return None。
    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, kind=self.kind, parameter=self.parameter, columns=self.columns,
                            mean=self.mean, scale=self.scale, X=self.X, y=self.y, weights=self.weights,
                            fs=FS, window_size=WINDOW, step_size=STEP, feature_names=FEATURE_NAMES)

    ## @brief 从本地 NumPy 文件恢复模型，禁用 pickle 加载。
    # @param path save 方法生成的模型文件。
    # @return NumpyFatigueClassifier 可直接预测的模型。
    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as bundle:
            if float(bundle["fs"]) != FS or int(bundle["window_size"]) != WINDOW or tuple(bundle["feature_names"].tolist()) != FEATURE_NAMES:
                raise ValueError("模型与当前预处理配置不一致。")
            model = cls(str(bundle["kind"]), float(bundle["parameter"]), bundle["columns"].tolist())
            for key in ("mean", "scale", "X", "y", "weights"):
                setattr(model, key, bundle[key].copy())
        return model


## @brief 计算固定三类混淆矩阵、准确率和宏平均 F1。
# @param truth 真实评分。
# @param prediction 预测评分。
# @return dict 行是真值、列是预测的混淆矩阵及逐类指标。
def classification_metrics(truth, prediction):
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    if not len(truth) or truth.shape != prediction.shape or not np.isin(truth, [0, 1, 2]).all() or not np.isin(prediction, [0, 1, 2]).all():
        raise ValueError("真实标签和预测必须等长、非空且为 0/1/2。")
    matrix = np.zeros((3, 3), dtype=int)
    np.add.at(matrix, (truth.astype(int), prediction.astype(int)), 1)
    per_class = []
    for label in range(3):
        tp = int(matrix[label, label])
        precision = tp / max(int(matrix[:, label].sum()), 1)
        recall = tp / max(int(matrix[label].sum()), 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        per_class.append(dict(label=label, name=LABEL_NAMES[label], precision=precision, recall=recall, f1=f1,
                              support=int(matrix[label].sum())))
    return dict(accuracy=float(np.mean(truth == prediction)), macro_f1=float(np.mean([v["f1"] for v in per_class])),
                confusion_matrix=matrix.tolist(), per_class=per_class)


## @brief 处理每位用户的右肱二头肌试验 5，滤波后分窗并提取单路特征。
# @param data_dir 疲劳数据集根目录。
# @return tuple 特征矩阵、样本元数据表、原始文件哈希和修正记录。
def prepare_biceps_data(data_dir):
    features, rows, sources = [], [], []
    for pair in discover_fatigue_pairs(data_dir):
        if pair.trial_id != 5:
            continue
        record = load_fatigue_record(pair.emg_path, pair.rating_path)
        if record.channel_names != ("R BICEPS BRACHII",):
            raise ValueError("必须只使用右肱二头肌单路。")
        filtered = replace(record, signal=filter_single_channel(record.signal[:, 0])[:, None])
        batch = window_fatigue_record(filtered)
        if not len(batch.labels):
            raise ValueError(f"{record.record_id}: 无可用窗口。")
        features.append(extract_single_channel_features(batch.signal[:, :, 0]))
        rows.extend(dict(subject_id=record.subject_id, record_id=record.record_id,
                         start_seconds=float(t[0]), end_seconds=float(t[-1]), label=int(label))
                    for t, label in zip(batch.time, batch.labels))
        sources.append(dict(subject_id=record.subject_id, emg_path=str(pair.emg_path), rating_path=str(pair.rating_path),
                            emg_sha256=hashlib.sha256(pair.emg_path.read_bytes()).hexdigest(),
                            rating_sha256=hashlib.sha256(pair.rating_path.read_bytes()).hexdigest(),
                            repairs=record.repairs, dropped=batch.dropped, windows=len(batch.labels)))
    if not features:
        raise ValueError("没有找到右肱二头肌 trial_5 数据。")
    return np.vstack(features), pd.DataFrame(rows), sources


## @brief 用已训练模型预测连续单路肌电，不需要疲劳评分文件。
# @param signal 原始单路伏特信号。
# @param time 对应秒时间轴，必须与训练的 1259 Hz 接近。
# @param model 已训练的分类器。
# @return DataFrame 完整窗口的时间、预测等级及中文名称；不声称是实时传感器测量。
def predict_fatigue_signal(signal, time, model):
    signal, time = np.asarray(signal, dtype=float), np.asarray(time, dtype=float)
    if signal.ndim != 1 or time.shape != signal.shape or len(time) < WINDOW or not np.isfinite(time).all():
        raise ValueError("需要至少一窗的同步一维信号和时间轴。")
    delta = np.diff(time)
    if np.any(delta <= 0) or not np.isclose(1 / np.median(delta), FS, rtol=.01):
        raise ValueError("时间顺序或采样率与模型不匹配；新硬件必须先验证/重采样，不直接套用模型。")
    filtered = filter_single_channel(signal)
    starts = np.arange(0, len(signal) - WINDOW + 1, STEP)
    starts = np.asarray([start for start in starts if np.all(delta[start:start + WINDOW - 1] <= .01)], dtype=int)
    if not len(starts):
        raise ValueError("没有连续的完整窗口。")
    values = filtered[starts[:, None] + np.arange(WINDOW)]
    prediction = model.predict(extract_single_channel_features(values))
    return pd.DataFrame(dict(start_seconds=time[starts], end_seconds=time[starts + WINDOW - 1],
                             predicted_label=prediction, predicted_name=[LABEL_NAMES[int(v)] for v in prediction]))


## @brief 执行单路离线训练评估，或对无评分 CSV 输出模型预测。
# @param argv 可选命令行参数。
# @return None: 写出验证候选表、保留用户测试结果、图表、报告及本地模型。
def main(argv=None):
    parser = argparse.ArgumentParser(description="右肱二头肌单路疲劳机器学习评估")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/EMG_data_for_ZenodoTired"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/fatigue/single_channel_biceps"))
    parser.add_argument("--model-file", type=Path, default=Path("results/runs/fatigue/single_channel_biceps/model.npz"))
    parser.add_argument("--predict-file", type=Path, help="仅预测指定 trial CSV，不读取评分、不重新训练")
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.predict_file:
        frame = pd.read_csv(args.predict_file)
        col = "R BICEPS BRACHII: EMG 1 [V]"
        if col not in frame:
            raise ValueError(f"缺少右肱二头肌列 {col}。")
        index = list(frame.columns).index(col)
        values = frame.iloc[:, [index - 1, index]].to_numpy(dtype=float)
        while len(values) and np.all(values[-1] == 0):
            values = values[:-1]
        result = predict_fatigue_signal(values[:, 1], values[:, 0], NumpyFatigueClassifier.load(args.model_file))
        result.to_csv(args.output_dir / "signal_predictions.csv", index=False, encoding="utf-8-sig")
        print(result.to_string(index=False))
        print("输出为模型估计的主观疲劳等级，不是诊断或疲劳百分比。")
        return

    X, metadata, sources = prepare_biceps_data(args.data_dir)
    groups = split_subject_ids(metadata.subject_id.tolist(), seed=42)
    masks = {name: metadata.subject_id.isin(users).to_numpy() for name, users in groups.items()}
    y = metadata.label.to_numpy(dtype=int)
    for name, mask in masks.items():
        if not all(np.any(y[mask] == label) for label in range(3)):
            raise ValueError(f"{name} 缺少某个等级，不能进行本次固定三类评估。")
    train, validation, test = (masks[name] for name in ("train", "validation", "test"))
    assert not np.any(train & validation | train & test | validation & test)
    candidates = []
    for subset, columns in (("amplitude_and_frequency", (0, 1, 2, 3, 4)), ("frequency_only", (3, 4))):
        for kind, parameters in (("knn", (5, 15, 31, 61)), ("ridge", (.1, 1., 10., 100.))):
            for parameter in parameters:
                model = NumpyFatigueClassifier(kind, parameter, columns).fit(X[train], y[train])
                score = classification_metrics(y[validation], model.predict(X[validation]))
                candidates.append(dict(kind=kind, parameter=parameter, subset=subset,
                                       columns=list(columns), validation_macro_f1=score["macro_f1"],
                                       validation_accuracy=score["accuracy"]))
    # 候选及并列规则预先固定；到此没有调用任何测试标签进行选择。
    best = max(candidates, key=lambda value: value["validation_macro_f1"])
    model = NumpyFatigueClassifier(best["kind"], best["parameter"], best["columns"]).fit(X[train], y[train])
    prediction = model.predict(X[test])
    model.save(args.model_file)
    assert np.array_equal(prediction, NumpyFatigueClassifier.load(args.model_file).predict(X[test]))
    majority = int(np.bincount(y[train], minlength=3).argmax())
    # 时间基线只用“已过去秒数”，不用文件总时长、评分边界或其他未来信息。
    elapsed = metadata[["end_seconds"]].to_numpy()
    time_choices = []
    for k in (5, 15, 31, 61):
        clock_model = NumpyFatigueClassifier("knn", k, (0,)).fit(elapsed[train], y[train])
        time_choices.append((classification_metrics(y[validation], clock_model.predict(elapsed[validation]))["macro_f1"], k))
    _, time_k = max(time_choices, key=lambda value: value[0])
    clock = NumpyFatigueClassifier("knn", time_k, (0,)).fit(elapsed[train], y[train])
    report = dict(muscle="R BICEPS BRACHII", trial=5, channel_count=1, seed=42,
                  fs_hz=FS, window_size=WINDOW, step_size=STEP, label_rule="纯 0/1/2 评分窗口",
                  filtering="每条记录固定 20-450Hz 四阶 Butterworth，sosfilt 因果滤波",
                  feature_names=FEATURE_NAMES, subject_groups=groups, selected_model=best,
                  split_counts={name: dict(subjects=list(users), windows=int(masks[name].sum()),
                                           label_counts=np.bincount(y[masks[name]], minlength=3).tolist())
                                for name, users in groups.items()},
                  validation=classification_metrics(y[validation], model.predict(X[validation])),
                  test=classification_metrics(y[test], prediction),
                  majority_baseline=classification_metrics(y[test], np.full(test.sum(), majority)),
                  elapsed_time_baseline=classification_metrics(y[test], clock.predict(elapsed[test])),
                  time_baseline_k=time_k, sources=sources,
                  environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__),
                  model_path=str(args.model_file))
    report["per_test_subject"] = {subject: classification_metrics(y[test & (metadata.subject_id == subject)],
                                                               model.predict(X[test & (metadata.subject_id == subject)]))
                                  for subject in groups["test"]}
    report["git_revision"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    report["experiment_date"] = "2026-11-18"
    report["code_sha256"] = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                              for name in ("single_channel.py", "reading.py", "alignment.py", "windowing.py")}
    report["note"] = "代码版本以源码 SHA256 与 Git 修订共同追溯；重叠窗口不独立，测试仅两位用户，非临床验证。"
    metadata["group"] = [next(name for name, users in groups.items() if user in users) for user in metadata.subject_id]
    for i, name in enumerate(FEATURE_NAMES):
        metadata[name] = X[:, i]
    metadata.to_csv(args.output_dir / "features.csv", index=False, encoding="utf-8-sig")
    output = metadata[test].copy()
    output["predicted_label"] = prediction
    output["predicted_name"] = [LABEL_NAMES[int(v)] for v in prediction]
    output["time_baseline_prediction"] = clock.predict(elapsed[test])
    output.to_csv(args.output_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(candidates).to_csv(args.output_dir / "validation_candidates.csv", index=False)
    (args.output_dir / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_plots_and_report(report, output, args.output_dir)
    print("右肱二头肌单路疲劳评估完成。")
    for name, values in report["split_counts"].items():
        print(f"{dict(train='训练集', validation='验证集', test='测试集')[name]} 用户 {values['subjects']}，"
              f"窗口 {values['windows']}，0/1/2 数量 {values['label_counts']}")
    print(f"验证集选择：{best}")
    for name in ("test", "majority_baseline", "elapsed_time_baseline"):
        title = {"test": "单路肌电模型（保留用户测试）", "majority_baseline": "训练集多数类基线",
                 "elapsed_time_baseline": "仅使用已运动秒数的基线"}[name]
        print(f"{title}：准确率 {report[name]['accuracy']:.2%}，macro-F1 {report[name]['macro_f1']:.4f}")
    print("本次仍属于离线评估，不是当前佩戴者的实时测量；输出为 0/1/2 主观疲劳等级。")
    print(f"模型：{args.model_file}；报告：{args.output_dir / 'report.md'}")


## @brief 保存独立的混淆矩阵和按保留用户分开的时间曲线及实验说明。
# @param report 实验指标与配置。
# @param output 保留用户的逐窗真值和预测表。
# @param folder 图表与报告输出目录。
# @return None。
def _write_plots_and_report(report, output, folder):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matrix = np.asarray(report["test"]["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(matrix, cmap="Blues")
    for row in range(3):
        for col in range(3):
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center")
    ax.set(xticks=range(3), yticks=range(3), xlabel="Predicted fatigue level", ylabel="Self-reported level",
           title="Single-channel right biceps: held-out users")
    fig.tight_layout(); fig.savefig(folder / "confusion_matrix.png", dpi=160); plt.close(fig)
    fig, axes = plt.subplots(len(report["subject_groups"]["test"]), 1, figsize=(10, 5), squeeze=False)
    for ax, subject in zip(axes[:, 0], report["subject_groups"]["test"]):
        values = output[output.subject_id == subject]
        ax.step(values.end_seconds, values.label, where="post", label="Self-reported")
        ax.scatter(values.end_seconds, values.predicted_label, s=12, color="tab:orange", label="EMG prediction")
        ax.set(title=f"Held-out subject {subject}", yticks=[0, 1, 2], ylabel="Fatigue level", xlabel="Elapsed seconds")
        ax.legend(loc="upper left")
    fig.tight_layout(); fig.savefig(folder / "test_predictions.png", dpi=160); plt.close(fig)
    lines = ["# 右肱二头肌单路疲劳分类评估", "", "输出是试验中的主观疲劳等级预测，不是百分比或临床诊断。", "",
             "只使用 trial_5 的右肱二头肌原始信号；每条记录因果滤波，约 4 秒窗口、2 秒步长。",
             "特征只来自肌电，不含评分边界、用户编号或试验总时长；固定用户划分，标准化仅用训练集拟合。", "",
             f"用户划分：`{report['subject_groups']}`。", "",
             f"验证集从 16 个预先固定候选中选择：`{report['selected_model']}`。测试集不参与选择。", "",
             "| 方法 | 测试准确率 | 测试 macro-F1 |", "| --- | --- | --- |"]
    for key, name in (("test", "单路肌电模型"), ("majority_baseline", "训练集多数类"), ("elapsed_time_baseline", "仅使用已过去秒数的 kNN")):
        lines.append(f"| {name} | {report[key]['accuracy']:.2%} | {report[key]['macro_f1']:.4f} |")
    lines += ["", "逐测试用户：", ""]
    for subject, metrics in report["per_test_subject"].items():
        lines.append(f"- 用户 {subject}：准确率 {metrics['accuracy']:.2%}，macro-F1 {metrics['macro_f1']:.4f}。")
    lines += ["", "## 本次实际判断", "",
              "肌电模型" + ("超过" if report["test"]["macro_f1"] > report["majority_baseline"]["macro_f1"] else "未超过") + "训练集多数类基线（以 macro-F1 比较）。",
              "肌电模型" + ("超过" if report["test"]["macro_f1"] > report["elapsed_time_baseline"]["macro_f1"] else "未超过") + "只使用已运动秒数的基线。",
              "仅凭本次结果不能认定单路设备已能可靠估计新用户的疲劳；不应将流程能输出等级误认为准确性已得到证明。"]
    lines += ["", "## 结论边界", "", "- 若模型不超过多数类，不能认为已经学到有效的疲劳区分。",
              "- 若不超过时间基线，则不能声称肌电提供了超出试验进程的识别价值。",
              "- 训练仅 8 人、验证 3 人、测试 2 人；窗口重叠，不能将窗口数当作独立人数。",
              "- 纯评分窗口排除了转折区间，真实在线输出也会遇到过渡窗口，不能直接套用该测试成绩。",
              "- 公共数据与廉价硬件的电极、带宽、增益和佩戴不同；尚不能证明新设备或新用户可用。",
              "- 建议后续做按用户交叉验证、跨天重贴测试及自采设备验证，不按当前测试结果反复调参。", "",
              "查看 metrics.json 获取完整逐类指标、源文件哈希、配置和修正记录。",
              "模型保存在被 Git 忽略的 results/runs/，重新训练可恢复。"]
    (folder / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
