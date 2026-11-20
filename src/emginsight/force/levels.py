"""整只手三档力量分类实验：静息 / 轻（约 25% MVC）/ 重（约 50% MVC），按用户留出。

运行：python -m emginsight.force.levels
"""

import argparse
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
import sklearn
from matplotlib.figure import Figure
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support,
)

from emginsight.common.plotting import use_chinese_font
from emginsight.models import LogisticRegression, MajorityClassifier, Standardizer
from emginsight.force.preprocessing import SEED, TRIAL, prepare_data
from emginsight.force.model import (
    FEATURE_SETS, LEVEL_NAMES, LEVEL_NAMES_EN, ForceLevelModel, load_force_model, load_level_model,
    save_level_model, smooth, transform,
)
from emginsight.force.preprocessing import FS
from emginsight.force.targets import BASELINE_PERCENTILE, BASELINE_SECONDS, THRESHOLDS, level_labels

LEVELS = (0, 1, 2)
SCALES = ("log", "log_rest")
SMOOTHING = (1, 3, 5)


## @brief 读取每条记录的 FORCE_MVC 和指令轨迹的窗口档位（后者只用于检查标签质量）。
# @param root 数据根目录。
# @param index 窗口索引表。
# @param window_size 每窗点数。
# @param with_instruction 是否读取 TRAJ 计算指令档位。
# @return tuple (每条记录的 MVC 字典, 逐窗指令档位或 None)。
def record_references(root, index, window_size, with_instruction):
    paths = {path.stem: path for path in root.rglob(f"emg_force-*-{TRIAL}-*.csv")}
    mvc, instruction = {}, np.full(len(index), -1)
    for record, rows in index.groupby("record", sort=False).indices.items():
        path = paths[record]
        mvc[record] = float(pd.read_csv(path, usecols=["FORCE_MVC"], nrows=1).FORCE_MVC.iloc[0])
        if with_instruction:
            traj = pd.read_csv(path, usecols=lambda name: name.startswith("TRAJ_")).to_numpy().max(axis=1) / mvc[record]
            starts = index.start_sample.to_numpy()[rows]
            cumulative = np.r_[0, np.cumsum(traj)]
            instruction[rows] = np.digitize((cumulative[starts + window_size] - cumulative[starts]) / window_size, THRESHOLDS)
    return mvc, instruction


## @brief 固定的分类候选：项目逻辑回归、随机森林及类别平衡随机森林。
# @return list (模型名, 参数说明, 模型工厂)。
def model_candidates():
    forest = dict(n_estimators=100, min_samples_leaf=10, max_samples=0.3, n_jobs=-1, random_state=SEED)
    return [("logistic", "l2=0.01", lambda: LogisticRegression(l2=0.01)),
            ("forest", "trees=100", lambda: RandomForestClassifier(**forest)),
            ("forest", "trees=100,balanced", lambda: RandomForestClassifier(class_weight="balanced", **forest))]


## @brief 训练并返回模型、标准化器及评估窗口的平滑前概率。
def fit_proba(inputs, labels, train, test, factory):
    scaler = Standardizer().fit(inputs[train])
    model = factory().fit(scaler.transform(inputs[train]), labels[train])
    return model.predict_proba(scaler.transform(inputs[test])), model, scaler


## @brief 汇总分类指标。
def scores(truth, pred):
    return {"accuracy": float(accuracy_score(truth, pred)),
            "macro_f1": float(f1_score(truth, pred, labels=list(LEVELS), average="macro", zero_division=0))}


## @brief 绘制一条测试记录的档位时间线与三档概率：上方两行色带分别是标签和预测。
def plot_timeline(frame, path, title):
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    colors = ("#d3d1c7", "#eda100", "#eb6834")
    figure = Figure(figsize=(13, 5.2), layout="constrained")
    top, bottom = figure.subplots(2, 1, sharex=True, height_ratios=(0.55, 1.2))
    t = frame.time_s.to_numpy()
    top.imshow(np.vstack([frame.label, frame.prediction]), aspect="auto", interpolation="nearest",
               cmap=ListedColormap(colors), vmin=-0.5, vmax=2.5, extent=(t[0], t[-1], 1.5, -0.5))
    top.set_yticks([0, 1], ["实测档位（标签）", "模型预测"])
    top.legend(handles=[Patch(color=c, label=LEVEL_NAMES[k]) for k, c in zip(LEVELS, colors)],
               ncol=3, loc="lower left", bbox_to_anchor=(0, 1.02), frameon=False, fontsize=9)
    for c, color in zip(LEVELS, ("#9a9890", "#eda100", "#eb6834")):
        bottom.plot(t, frame[f"prob_{c}"], color=color, lw=1, label=f"P({LEVEL_NAMES[c]})")
    bottom.set_ylim(0, 1.02)
    bottom.set_ylabel("概率")
    bottom.set_xlabel("记录内时间（秒，窗口中心）")
    bottom.legend(ncol=3, loc="lower left", bbox_to_anchor=(0, 1.0), frameon=False, fontsize=9)
    bottom.grid(axis="y", color="#e4e3dd", lw=0.6)
    for ax in (top, bottom):
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    figure.suptitle(title, fontsize=11)
    figure.savefig(path, dpi=110)


## @brief 选型、测试、保存模型并输出表格和图。
# @param root 数据根目录。
# @param output 结果目录。
# @param cache 特征缓存目录（与回归实验共用）。
# @param model_path 三档模型文件路径；None 表示不保存。
# @param regression_model 回归模型文件；存在时比较“回归结果换算成档位”。
# @param window_size 每窗点数。
# @param step_size 窗口步长。
def run(root, output, cache=None, model_path=None, regression_model=None, window_size=1024, step_size=512):
    use_chinese_font()
    X, y_raw, tare, rest, index, groups, skipped = prepare_data(root, cache, window_size, step_size)
    force = y_raw - tare
    test = (index.group == "test").to_numpy()
    mvc, instruction = record_references(root, index, window_size, with_instruction=True)
    labels, relative = np.empty(len(index), int), np.empty(len(index))
    records = index.record.to_numpy()
    for record, rows in index.groupby("record", sort=False).indices.items():
        labels[rows], relative[rows] = level_labels(force[rows], mvc[record], step_size / FS)

    train, val = (index.group == "train").to_numpy(), (index.group == "validation").to_numpy()
    rows, best = [], None
    for scale in SCALES:
        for features in FEATURE_SETS:
            inputs = transform(X, rest, features, scale)
            for name, parameter, factory in model_candidates():
                proba, *_ = fit_proba(inputs, labels, train, val, factory)
                for k in SMOOTHING:
                    pred = np.asarray(LEVELS)[smooth(proba, records[val], k).argmax(axis=1)]
                    row = dict(scale=scale, features=features, model=name, parameter=parameter, smoothing=k,
                               **scores(labels[val], pred))
                    rows.append(row)
                    if best is None or row["macro_f1"] > best[0]["macro_f1"]:
                        best = row, factory
                print(f"验证 {scale} {features} {name} {parameter}：macro-F1 {rows[-len(SMOOTHING)]['macro_f1']:.3f}", flush=True)
    table = pd.DataFrame(rows)
    choice, factory = best

    inputs = transform(X, rest, choice["features"], choice["scale"])
    raw_proba, model, scaler = fit_proba(inputs, labels, train, test, factory)
    proba = smooth(raw_proba, records[test], choice["smoothing"])
    pred = np.asarray(LEVELS)[proba.argmax(axis=1)]
    truth = labels[test]
    majority = MajorityClassifier().fit(inputs[train], labels[train]).predict(inputs[test])
    frame = index[test].reset_index(drop=True).drop(columns="group")
    frame["label"], frame["prediction"], frame["relative_force"] = truth, pred, relative[test]
    for i, c in enumerate(LEVELS):
        frame[f"prob_{c}"] = proba[:, i]
    frame["instruction"] = instruction[test]

    comparisons = {"model": scores(truth, pred), "majority": scores(truth, majority)}
    if regression_model is not None and Path(regression_model).exists():
        # 把回归预测的十路力量用同一规则分档，检验“先回归再分档”是否足够。
        regression = load_force_model(regression_model)
        force_pred = regression.predict(X[test], rest[test], records[test])
        converted = np.empty(len(force_pred), int)
        for record, rows_ in frame.groupby("record", sort=False).indices.items():
            converted[rows_] = level_labels(force_pred[rows_], mvc[record], step_size / FS)[0]
        comparisons["regression_to_levels"] = scores(truth, converted)
    comparisons["label_vs_instruction_agreement"] = float(np.mean(truth == frame.instruction))
    comparisons["model_vs_instruction"] = scores(frame.instruction, pred)

    precision, recall, f1, support = precision_recall_fscore_support(truth, pred, labels=list(LEVELS), zero_division=0)
    per_class = pd.DataFrame(dict(level=LEVELS, name=[LEVEL_NAMES[c] for c in LEVELS],
                                  precision=precision, recall=recall, f1=f1, support=support))
    matrix = confusion_matrix(truth, pred, labels=list(LEVELS))
    users = pd.DataFrame([dict(subject=s, windows=len(r), **scores(truth[r], pred[r]),
                               majority_accuracy=accuracy_score(truth[r], majority[r]))
                          for s, r in frame.groupby("subject").indices.items()]).sort_values("macro_f1").reset_index(drop=True)
    shown = users.subject.iloc[len(users) // 2]
    shown_record = frame[frame.subject == shown].record.iloc[0]

    output.mkdir(parents=True, exist_ok=True)
    model_info = None
    if model_path is not None:
        if hasattr(model, "n_jobs"):
            model.n_jobs = 1
        trained = ForceLevelModel(
            model=model, scaler=scaler, scale=choice["scale"], features=choice["features"],
            smoothing=int(choice["smoothing"]), classes=LEVELS, thresholds=THRESHOLDS,
            preprocessing={"window_size": window_size, "step_size": step_size, "nominal_fs_hz": FS,
                           "calibration_seconds": 4.0, "trial": TRIAL,
                           "label": "第二强传感器（减 10 秒滚动 10% 分位基线）÷ FORCE_MVC，阈值 12.5% / 37.5%"},
            train_subjects=tuple(groups["train"]), metrics=comparisons["model"])
        model_info = save_level_model(trained, model_path)
        again = load_level_model(model_path, model_info["sha256"]).predict(X[test], rest[test], records[test])
        if not np.array_equal(again, pred):
            raise RuntimeError("重新读取的三档模型预测与训练时不一致。")
        model_info.update(verified_test_predictions=True, **trained.describe())
        (output / "model_card.json").write_text(json.dumps(model_info, indent=2, ensure_ascii=False, default=str),
                                                encoding="utf-8")

    summary = {"seed": SEED, "subjects": groups, "trial": TRIAL, "python": platform.python_version(),
               "sklearn_version": sklearn.__version__, "levels": {str(k): v for k, v in LEVEL_NAMES.items()},
               "label_definition": {"thresholds_of_mvc": list(THRESHOLDS), "baseline_seconds": BASELINE_SECONDS,
                                    "baseline_percentile": BASELINE_PERCENTILE, "aggregate": "第二强传感器"},
               "label_share": {g: np.bincount(labels[(index.group == g).to_numpy()], minlength=3).tolist() for g in groups},
               "selected": choice, "test": comparisons, "skipped": skipped,
               "curve_example": {"subject": shown, "record": shown_record, "rule": "测试用户按 macro-F1 排序取中位数，仅用于展示"}}
    table.to_csv(output / "validation.csv", index=False)
    per_class.to_csv(output / "per_class.csv", index=False)
    users.to_csv(output / "per_user.csv", index=False)
    pd.DataFrame(matrix, index=LEVELS, columns=LEVELS).to_csv(output / "confusion.csv", index_label="true_level")
    frame[frame.record == shown_record].round(4).to_csv(output / "example_predictions.csv", index=False)
    figure = Figure(figsize=(5.5, 4.8), layout="constrained")
    ax = figure.subplots()
    ConfusionMatrixDisplay(matrix, display_labels=[LEVEL_NAMES_EN[c] for c in LEVELS]).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Force level | held-out users")
    figure.savefig(output / "confusion.png", dpi=150)
    label = f"{choice['model']} {choice['parameter']}，{choice['features']}，{choice['scale']}，平滑 {int(choice['smoothing'])} 窗"
    plot_timeline(frame[frame.record == shown_record], output / "test_timeline.png", f"未见用户 {shown} · {label}")
    (output / "metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    if model_info:
        print(f"模型已保存 {model_info['path']}（{model_info['bytes'] / 2**20:.1f} MB）")
    print(f"获选配置 {choice}\n测试结果 {comparisons}\n结果已写入 {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/EMG_data_for_putemgforce"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/force/levels"))
    parser.add_argument("--cache-dir", type=Path, default=Path("results/runs/force"))
    parser.add_argument("--model-path", type=Path, default=Path("results/runs/force/force_level_model.joblib"))
    parser.add_argument("--regression-model", type=Path, default=Path("results/runs/force/force_model.joblib"))
    args = parser.parse_args()
    run(args.data_dir, args.output_dir, args.cache_dir, args.model_path, args.regression_model)
