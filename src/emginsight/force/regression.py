"""按用户留出的 putEMG-Force 手指力量回归实验（44 名用户，repeats_long）。"""

import argparse
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
import sklearn
from matplotlib.figure import Figure
from sklearn.ensemble import RandomForestRegressor

from emginsight.common.plotting import use_chinese_font
from emginsight.models import MeanRegressor, RidgeRegressor, Standardizer
from emginsight.force.model import (
    FEATURE_SCALES, FEATURE_SETS, ForceModel, load_force_model, save_force_model, smooth, transform,
)
from emginsight.force.preprocessing import (
    BAND_HZ, CALIBRATION_SECONDS, FORCE_UNIT, FS, SEED, TRIAL, prepare_data,
)
from emginsight.force.targets import TARGETS, target_metrics

SMOOTHING = (1, 3, 5)


## @brief 统计训练记录静息段对数 RMS 的分布，供部署时检查佩戴或设备漂移。
# @param rest 每个窗口所属记录的静息特征。
# @param index 窗口索引表。
# @return dict 24 路通道的均值、标准差和训练记录数。
def rest_reference(rest, index):
    first = index[index.group == "train"].drop_duplicates("record").index.to_numpy()
    log_rms = np.log(np.maximum(rest[first][:, 0::3], 1e-6))
    return {"mean": log_rms.mean(axis=0).tolist(), "std": np.maximum(log_rms.std(axis=0, ddof=1), 1e-6).tolist(),
            "records": int(len(first)), "feature": "记录开头 4 秒静息段各通道 log(RMS)，滤波 20–500 Hz"}


## @brief 生成固定模型候选：岭回归三个正则化系数和一个随机森林。
# @return list (模型名, 参数说明, 模型工厂)。
def model_candidates():
    models = [("ridge", f"alpha={a}", lambda a=a: RidgeRegressor(alpha=a)) for a in (1.0, 10.0, 100.0)]
    models.append(("forest", "trees=100", lambda: RandomForestRegressor(
        n_estimators=100, min_samples_leaf=10, max_samples=0.3, n_jobs=-1, random_state=SEED)))
    return models


## @brief 在训练窗口拟合标准化与模型，返回评估窗口的预测和均值基线。
# @param inputs 模型输入矩阵。
# @param target 回归目标。
# @param train 训练掩码。
# @param test 评估掩码。
# @param factory 返回未训练模型的函数。
# @return tuple 模型预测、均值基线预测、已训练模型和标准化器。
def fit_predict(inputs, target, train, test, factory):
    scaler = Standardizer().fit(inputs[train])
    model = factory().fit(scaler.transform(inputs[train]), target[train])
    baseline = MeanRegressor().fit(inputs[train], target[train])
    return model.predict(scaler.transform(inputs[test])), baseline.predict(inputs[test]), model, scaler


## @brief 只用验证用户，按十个目标的平均 RMSE 选择特征尺度、组合、模型和平滑窗口。
# @param X 原始特征。
# @param target 去零点后的力量目标。
# @param rest 静息特征。
# @param index 窗口索引表。
# @return tuple 候选结果表和获选配置字典。
def select_model(X, target, rest, index):
    train, val = (index.group == "train").to_numpy(), (index.group == "validation").to_numpy()
    records = index.record.to_numpy()[val]
    rows = []
    for scale in FEATURE_SCALES:
        for features in FEATURE_SETS:
            inputs = transform(X, rest, features, scale)
            for name, parameter, factory in model_candidates():
                pred, *_ = fit_predict(inputs, target, train, val, factory)
                for k in SMOOTHING:
                    scores = target_metrics(target[val], smooth(pred, records, k))
                    rows.append(dict(scale=scale, features=features, model=name, parameter=parameter, smoothing=k,
                                     mean_rmse=scores.rmse.mean(), mean_mae=scores.mae.mean(), mean_r2=scores.r2.mean()))
                print(f"验证 {scale} {features} {name} {parameter}：平均 RMSE {rows[-len(SMOOTHING)]['mean_rmse']:.2f}",
                      flush=True)
    table = pd.DataFrame(rows)
    return table, table.loc[table.mean_rmse.idxmin()].to_dict()


## @brief 用获选配置在测试用户上预测指定目标。
# @param X 原始特征。
# @param target 力量目标。
# @param rest 静息特征。
# @param index 窗口索引表。
# @param choice 获选配置。
# @return tuple 测试掩码、模型预测、均值基线预测、已训练模型和标准化器。
def evaluate(X, target, rest, index, choice):
    train, test = (index.group == "train").to_numpy(), (index.group == "test").to_numpy()
    factory = {(n, p): f for n, p, f in model_candidates()}[(choice["model"], choice["parameter"])]
    pred, base, model, scaler = fit_predict(
        transform(X, rest, choice["features"], choice["scale"]), target, train, test, factory)
    return test, smooth(pred, index.record.to_numpy()[test], int(choice["smoothing"])), base, model, scaler


## @brief 每名测试用户分别计算平均误差和 R²。
# @param frame 测试窗口索引。
# @param truth 实测目标。
# @param pred 模型预测。
# @param base 基线预测。
# @return DataFrame 每名用户一行。
def per_user(frame, truth, pred, base):
    rows = []
    for subject, rows_index in frame.groupby("subject").indices.items():
        model, baseline = target_metrics(truth[rows_index], pred[rows_index]), target_metrics(truth[rows_index], base[rows_index])
        rows.append(dict(subject=subject, windows=len(rows_index), records=frame.record.iloc[rows_index].nunique(),
                         rmse_model=model.rmse.mean(), rmse_baseline=baseline.rmse.mean(),
                         mae_model=model.mae.mean(), mae_baseline=baseline.mae.mean(),
                         r2_model=model.r2.mean(), r2_baseline=baseline.r2.mean(), pearson_r_model=model.pearson_r.mean()))
    return pd.DataFrame(rows).sort_values("r2_model").reset_index(drop=True)


## @brief 按时间绘制一条测试记录每个目标的实测、模型预测和均值基线曲线。
# @param frame 单条记录的逐窗结果。
# @param path 输出 PNG 路径。
# @param title 图标题。
def plot_curves(frame, path, title):
    figure = Figure(figsize=(13, 17), layout="constrained")
    axes = figure.subplots(len(TARGETS), 1, sharex=True)
    for ax, name in zip(axes, TARGETS):
        ax.plot(frame.time_s, frame[f"{name}_true"], color="#3d3d3a", lw=0.9, label="实测（相对静息）")
        ax.plot(frame.time_s, frame[f"{name}_pred"], color="#2a78d6", lw=0.9, label="回归模型（未见用户）")
        ax.plot(frame.time_s, frame[f"{name}_baseline"], color="#9a9890", lw=1, ls="--", label="训练集均值基线")
        ax.set_ylabel(name, rotation=0, ha="right", va="center", fontsize=9)
        ax.grid(axis="y", color="#e4e3dd", lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(labelsize=8)
    axes[0].legend(ncol=3, loc="lower left", bbox_to_anchor=(0, 1.02), frameon=False, fontsize=9)
    axes[-1].set_xlabel("记录内时间（秒，窗口中心）")
    figure.suptitle(title, fontsize=11)
    figure.savefig(path, dpi=110)


## @brief 绘制每名测试用户的模型与基线平均 R²，展示用户间差异。
# @param users 每用户结果表。
# @param path 输出 PNG 路径。
def plot_users(users, path):
    figure = Figure(figsize=(8, 0.42 * len(users) + 1.4), layout="constrained")
    ax = figure.subplots()
    order = np.arange(len(users))
    ax.hlines(order, users.r2_baseline, users.r2_model, color="#d3d1c7", lw=2)
    ax.scatter(users.r2_baseline, order, s=50, color="#9a9890", label="训练集均值基线", zorder=3)
    ax.scatter(users.r2_model, order, s=50, color="#2a78d6", label="回归模型", zorder=3)
    ax.axvline(0, color="#3d3d3a", lw=0.8)
    ax.set_yticks(order, [f"用户 {s}" for s in users.subject])
    ax.set_xlabel("十个目标的平均 R²（相对静息力量）")
    ax.grid(axis="x", color="#e4e3dd", lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False)
    figure.savefig(path, dpi=130)


## @brief 选型后用测试用户评估获选模型与均值基线，输出表格、曲线和指标。
# @param root 已解压的数据目录。
# @param output 保存结果的目录。
# @param cache 特征缓存目录；None 表示不缓存。
# @param window_size 每窗点数，默认 1024（名义约 200 毫秒）。
# @param step_size 窗口步长，默认 512。
# @param model_path 获选模型文件路径；None 表示不保存。
def run(root, output, cache=None, window_size=1024, step_size=512, model_path=None):
    use_chinese_font()
    X, y_raw, tare, rest, index, groups, skipped = prepare_data(root, cache, window_size, step_size)
    y = y_raw - tare
    scores, choice = select_model(X, y, rest, index)
    test, pred, base, model, scaler = evaluate(X, y, rest, index, choice)
    _, pred_raw, base_raw, *_ = evaluate(X, y_raw, rest, index, choice)

    frame = index[test].reset_index(drop=True).drop(columns="group")
    truth = y[test]
    model_scores, base_scores = target_metrics(truth, pred), target_metrics(truth, base)
    per_target = model_scores.merge(base_scores, on="target", suffixes=("_model", "_baseline"))
    users = per_user(frame, truth, pred, base)
    raw_model, raw_base = target_metrics(y_raw[test], pred_raw), target_metrics(y_raw[test], base_raw)

    # 保存获选模型，并重新读取验证测试预测逐位一致，确保文件可复现主结果。
    model_info = None
    if model_path is not None:
        trained = ForceModel(
            model=model, scaler=scaler, scale=choice["scale"], features=choice["features"],
            smoothing=int(choice["smoothing"]), targets=TARGETS,
            preprocessing={"window_size": window_size, "step_size": step_size, "nominal_fs_hz": FS,
                           "filter_hz": list(BAND_HZ), "filter": "四阶 Butterworth 因果带通（sosfilt）",
                           "calibration_seconds": CALIBRATION_SECONDS, "trial": TRIAL,
                           "target": "窗口内实测力量均值减去记录开头静息段中位数"},
            train_subjects=tuple(groups["train"]),
            metrics={"test_mean_mae": float(model_scores.mae.mean()), "test_mean_rmse": float(model_scores.rmse.mean()),
                     "test_mean_r2": float(model_scores.r2.mean())})
        # 并行预测时各树结果的累加顺序不定，末位约有 1e-13 的差异；保存单线程版本使部署预测逐位可重复。
        if hasattr(model, "n_jobs"):
            model.n_jobs = 1
        model_info = save_force_model(trained, model_path)
        reloaded = load_force_model(model_path, model_info["sha256"])
        again = reloaded.predict(X[test], rest[test], index.record.to_numpy()[test])
        difference = float(np.abs(again - pred).max())
        if difference > 1e-9:
            raise RuntimeError(f"重新读取的模型预测与训练时不一致，最大差 {difference}。")
        model_info.update(verified_test_predictions=True, max_abs_difference=difference, **trained.describe())

    columns = {}
    for i, name in enumerate(TARGETS):
        columns.update({f"{name}_true": truth[:, i], f"{name}_pred": pred[:, i], f"{name}_baseline": base[:, i]})
    predictions = pd.concat([frame, pd.DataFrame(columns)], axis=1)
    shown = users.subject.iloc[len(users) // 2]
    shown_record = predictions[predictions.subject == shown].record.iloc[0]
    example = predictions[predictions.record == shown_record]

    mean = lambda table, keys=("mae", "rmse", "r2", "pearson_r"): {k: float(table[k].mean()) for k in keys}
    summary = {
        "seed": SEED, "subjects": groups, "trial": TRIAL, "force_unit": FORCE_UNIT, "targets": list(TARGETS),
        "python": platform.python_version(), "sklearn_version": sklearn.__version__,
        "window": {"size": window_size, "step": step_size, "nominal_fs_hz": FS, "filter_hz": list(BAND_HZ),
                   "target": "窗口内实测力量均值减去同记录开头静息段中位数"},
        "calibration_seconds": CALIBRATION_SECONDS,
        "windows": {g: int((index.group == g).sum()) for g in groups},
        "records": {g: int(index[index.group == g].record.nunique()) for g in groups},
        "skipped": skipped, "selected": choice,
        "test": {"windows": int(test.sum()), "model": mean(model_scores), "baseline": mean(base_scores, ("mae", "rmse", "r2")),
                 "users_model_better_rmse": int((users.rmse_model < users.rmse_baseline).sum()),
                 "users_model_r2_positive": int((users.r2_model > 0).sum()), "users": len(users)},
        "test_raw_force_without_tare": {"model": mean(raw_model), "baseline": mean(raw_base, ("mae", "rmse", "r2"))},
        "curve_example": {"subject": shown, "record": shown_record, "rule": "测试用户按平均 R² 排序取中位数，仅用于展示"},
    }
    output.mkdir(parents=True, exist_ok=True)
    scores.to_csv(output / "validation.csv", index=False)
    per_target.to_csv(output / "per_target.csv", index=False)
    users.to_csv(output / "per_user.csv", index=False)
    example.round(3).to_csv(output / "example_predictions.csv", index=False)
    if cache is not None:
        predictions.round(3).to_csv(cache / "putemg_force_test_predictions.csv.gz", index=False)
    label = f"{choice['model']} {choice['parameter']}，{choice['features']}，{choice['scale']}，平滑 {int(choice['smoothing'])} 窗"
    plot_curves(example, output / "test_curves.png", f"未见用户 {shown} · {shown_record} · {label}（原始力量数值，已减静息零点）")
    plot_users(users, output / "per_user_r2.png")
    for old in ("test_predictions.csv", "within_record_curves.png"):
        (output / old).unlink(missing_ok=True)
    (output / "metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    (output / "monitoring_reference.json").write_text(json.dumps(rest_reference(rest, index), indent=2), encoding="utf-8")
    if model_info is not None:
        (output / "model_card.json").write_text(json.dumps(model_info, indent=2, ensure_ascii=False, default=str),
                                                encoding="utf-8")
        print(f"模型已保存 {model_info['path']}（{model_info['bytes'] / 2**20:.1f} MB，SHA-256 {model_info['sha256'][:12]}…）")
    print(f"获选配置 {choice}\n测试结果 {summary['test']}\n未去零点 {summary['test_raw_force_without_tare']}\n结果已写入 {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/EMG_data_for_putemgforce"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/force/regression"))
    parser.add_argument("--cache-dir", type=Path, default=Path("results/runs/force"),
                        help="特征缓存与完整逐窗预测目录（Git 忽略）")
    parser.add_argument("--model-path", type=Path, default=Path("results/runs/force/force_model.joblib"),
                        help="获选模型文件（Git 忽略，较大时上传网盘）")
    args = parser.parse_args()
    run(args.data_dir, args.output_dir, args.cache_dir, model_path=args.model_path)
