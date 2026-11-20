"""跨日力量变化：第二阶段模型、动作保持段特征，以及用测力计标定十路指力预测的命令。"""

import argparse
from dataclasses import dataclass, field
from datetime import date
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from emginsight.common.artifacts import file_sha256, load_model_file, save_model_file
from .model import load_force_model
from .predict import inference_features, read_input
from .targets import TARGETS


RUNS = Path("results/runs/force")
RESULTS = Path("results/force/change")
FEATURES = tuple(f"{name}_pred" for name in TARGETS)
RAW_UNIT = "putEMG 原始刻度"


@dataclass
class ForceChangeModel:
    """已拟合的十路岭回归，权重已折算至原始特征尺度。"""

    weights: np.ndarray
    intercept: float
    stage1_sha256: str
    unit: str
    protocols: tuple[str, ...]
    patient: str | None = None
    fit_dates: tuple[str, ...] = ()
    validation: dict = field(default_factory=dict)
    sklearn_version: str = sklearn.__version__
    format_version: int = 1

    ## @brief 从每次动作的十路预测中位数计算一次标定后的动作力量。
    def predict(self, features):
        x = np.asarray(features, dtype=float)
        if x.ndim != 2 or x.shape[1] != 10 or not np.isfinite(x).all():
            raise ValueError("每次动作需要十个有限的指力特征。")
        return x @ self.weights + self.intercept

    ## @brief 核对训练协议，并核对牛顿标定的个人身份、日期及留出日误差。
    def check_visits(self, trials):
        if trials.protocol.iloc[0] not in self.protocols:
            raise ValueError(f"动作协议与模型不匹配，可用协议为 {self.protocols}")
        if self.patient is not None:
            if trials.patient.iloc[0] != self.patient or trials.date.min() <= max(self.fit_dates):
                raise ValueError("患者不匹配或比较日期早于个人标定完成日。")
            if not 0 <= self.validation["delta_mae_n"] < self.validation["no_change_mae_n"]:
                raise ValueError("个人标定的跨日误差未优于零变化基线。")


## @brief 保存第二阶段模型和可移植的配套校验卡。
def save_change_model(model, path, card, metadata):
    info = save_model_file(model, path)
    info.update(metadata, unit=model.unit, features=FEATURES, protocols=model.protocols,
                stage1_sha256=model.stage1_sha256, sklearn_version=model.sklearn_version,
                format_version=model.format_version)
    card.parent.mkdir(parents=True, exist_ok=True)
    card.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    return info


## @brief 读取并核对第二阶段文件的类型、版本和 SHA-256。
def load_change_model(path, card):
    info = json.loads(card.read_text(encoding="utf-8"))
    return load_model_file(path, ForceChangeModel, 1, info["sha256"])


## @brief 读取动作保持段清单，核对日期、身份、区间和参考力量。
# @param path 清单路径，记录路径相对于清单所在目录。
# @param fitting 标定时要求独立的牛顿参考值和 fit/check 标记。
# @return DataFrame 每行代表一次动作的稳定保持段。
def read_trials(path, fitting):
    trials = pd.read_csv(path, dtype=str, keep_default_na=False).apply(lambda col: col.str.strip())
    required = {"patient", "protocol", "date", "record", "start_s", "end_s"}
    if fitting:
        required.update({"reference_n", "split"})
    missing = required.difference(trials.columns)
    if missing:
        raise ValueError(f"清单缺少列 {sorted(missing)}")
    if trials.empty or (trials[sorted(required)] == "").any().any():
        raise ValueError("清单为空或必填项缺失。")
    if trials.patient.nunique() != 1 or trials.protocol.nunique() != 1:
        raise ValueError("每份清单须属于同一患者和同一动作协议。")
    trials["date"] = [date.fromisoformat(value).isoformat() for value in trials.date]
    trials["record"] = [(path.parent / value).resolve() for value in trials.record]
    for col in ["start_s", "end_s"] + (["reference_n"] if fitting else []):
        trials[col] = pd.to_numeric(trials[col])
        if not np.isfinite(trials[col]).all():
            raise ValueError(f"{col} 必须为有限数值。")
    if (trials.start_s < 0).any() or (trials.end_s <= trials.start_s).any():
        raise ValueError("保持段必须满足 0 ≤ start_s < end_s。")
    for _, rows in trials.sort_values("start_s").groupby("record"):
        if rows.date.nunique() != 1:
            raise ValueError("同一原始记录不能属于多个日期。")
        if (rows.start_s.to_numpy()[1:] < rows.end_s.to_numpy()[:-1]).any():
            raise ValueError("同一记录的保持段不能重复或重叠。")
    if fitting:
        if set(trials.split) != {"fit", "check"}:
            raise ValueError("标定清单须同时提供 fit 拟合日与 check 留出日。")
        fit, check = trials[trials.split == "fit"], trials[trials.split == "check"]
        if fit.date.max() >= check.date.min() or check.date.nunique() < 2:
            raise ValueError("拟合后至少留出两个不同日期，才能独立核对力量变化。")
        if fit.reference_n.nunique() < 2:
            raise ValueError("拟合段需要至少两个不同的测力计读数。")
    elif trials.date.nunique() != 2:
        raise ValueError("比较清单需要基线和复测两个不同日期。")
    return trials


## @brief 每次动作得到十个特征，仅使用采样窗及其平滑历史都落在保持段内的预测。
# @param trials 同一人同一协议的动作清单。
# @param model 已冻结的第一阶段模型。
# @return ndarray 每次动作各通道预测的中位数，顺序与清单一致。
def trial_features(trials, model):
    if tuple(model.targets) != TARGETS:
        raise ValueError("第一阶段模型必须按 FORCE_1 至 FORCE_10 顺序输出。")
    features = np.empty((len(trials), len(TARGETS)))
    for path, rows in trials.groupby("record", sort=False):
        record, _, has_traj, _ = read_input(path, model.targets)
        item, rest, records = inference_features(path, record, model, has_traj)
        pred = model.predict(item["X"], rest, records)
        seconds = record.time - record.time[0]
        starts = item["starts"]
        history = np.maximum(np.arange(len(starts)) + 1 - model.smoothing, 0)
        first = seconds[starts[history]]
        last = seconds[starts + model.preprocessing["window_size"] - 1]
        for row in rows.itertuples():
            if row.start_s < model.preprocessing["calibration_seconds"]:
                raise ValueError(f"{path.name} 第 {row.Index + 2} 行的保持段包含静息校准时间。")
            if row.end_s > seconds[-1]:
                raise ValueError(f"{path.name} 第 {row.Index + 2} 行的保持段超出记录。")
            use = (first >= row.start_s) & (last < row.end_s)
            if not use.any():
                raise ValueError(f"{path.name} 第 {row.Index + 2} 行没有完整的保持段预测窗。")
            features[row.Index] = np.median(pred[use], axis=0)
    if not np.isfinite(features).all():
        raise ValueError("第一阶段输出包含非有限值。")
    return features


## @brief 拟合标准化岭回归，并换算成可直接作用于原始十路预测的线性系数。
# @param X 每次动作的通道特征。
# @param y 同一次保持段的测力计参考值（N）。
# @return tuple 原始尺度上的系数和截距。
def fit_weights(X, y, alpha=1.0):
    scaler = StandardScaler().fit(X)
    reg = Ridge(alpha=alpha).fit(scaler.transform(X), y)
    weights = reg.coef_ / scaler.scale_
    return weights, float(reg.intercept_ - scaler.mean_ @ weights)


## @brief 按天取各次动作的中位数，再减去最早一天的值。
# @return ndarray 每个后续日期相对同一基线的变化。
def daily_changes(dates, values):
    daily = pd.Series(np.asarray(values)).groupby(np.asarray(dates), sort=True).median()
    return (daily.iloc[1:] - daily.iloc[0]).to_numpy()


## @brief 用配对读数标定，在完整留出日比较十路、单路和“无变化”基线。
# @return dict 个人标定参数、第一阶段校验值及以 N 为单位的验证误差。
def calibrate(trials, X, digest):
    fit = (trials.split == "fit").to_numpy()
    y = trials.reference_n.to_numpy()
    weights, intercept = fit_weights(X[fit], y[fit])
    if not np.any(weights):
        raise ValueError("拟合数据没有提供可用于估计力量的通道变化。")
    dates = trials.date[~fit]
    observed = daily_changes(dates, y[~fit])
    predicted = daily_changes(dates, X[~fit] @ weights + intercept)
    w8, b8 = fit_weights(X[fit, 7:8], y[fit])
    single = daily_changes(dates, X[~fit, 7:8] @ w8 + b8)
    return {
        "version": 1, "patient": trials.patient.iloc[0], "protocol": trials.protocol.iloc[0],
        "model_sha256": digest, "targets": list(TARGETS), "fit_dates": sorted(set(trials.date[fit])),
        "weights": weights.tolist(), "intercept_n": intercept,
        "validation": {
            "dates": sorted(set(dates)), "measured_changes_n": observed.tolist(),
            "estimated_changes_n": predicted.tolist(),
            "delta_mae_n": float(np.abs(predicted - observed).mean()),
            "force8_delta_mae_n": float(np.abs(single - observed).mean()),
            "no_change_mae_n": float(np.abs(observed).mean()),
        },
    }


## @brief 写入完整标定文件后原子替换，避免留下半份参数。
def save_calibration(path, calibration):
    content = json.dumps(calibration, ensure_ascii=False, indent=2, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        tmp = Path(stream.name)
    try:
        tmp.write_text(content, encoding="utf-8")
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


## @brief 命令行入口；change 成功时只打印一行患者可读的估计变化。
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["fit", "change"])
    parser.add_argument("--trials", required=True, type=Path, help="每行一次动作保持段的 CSV 清单")
    parser.add_argument("--calibration", required=True, type=Path, help="个人标定 .joblib（兼容旧版 JSON）")
    parser.add_argument("--model", type=Path, default=Path("results/runs/force/force_model.joblib"))
    args = parser.parse_args(argv)
    fitting = args.command == "fit"
    try:
        binary = args.calibration.suffix.lower() == ".joblib"
        if not binary and args.calibration.suffix.lower() != ".json":
            raise ValueError("个人标定文件应使用 .joblib 或 .json 扩展名。")
        card = args.calibration.with_suffix(".card.json")
        if args.calibration.resolve() == args.model.resolve():
            raise ValueError("标定输出路径与第一阶段模型相同。")
        trials = read_trials(args.trials, fitting)
        if fitting:
            digest = file_sha256(args.model)
            model = load_force_model(args.model, digest)
            calibration = calibrate(trials, trial_features(trials, model), digest)
            if binary:
                personal = ForceChangeModel(np.array(calibration["weights"]), calibration["intercept_n"],
                                            digest, "N", (calibration["protocol"],), calibration["patient"],
                                            tuple(calibration["fit_dates"]), calibration["validation"])
                save_change_model(personal, args.calibration, card,
                                  dict(estimator="StandardScaler + Ridge", alpha=1.0,
                                       fit_dates=personal.fit_dates, validation=personal.validation))
            else:
                save_calibration(args.calibration, calibration)
            print(f"标定已保存。留出日的力量变化平均绝对误差 {calibration['validation']['delta_mae_n']:.2f} N。")
        elif binary:
            personal = load_change_model(args.calibration, card)
            personal.check_visits(trials)
            base = load_force_model(args.model, personal.stage1_sha256)
            estimates = personal.predict(trial_features(trials, base))
            print(f"本次动作力量较基线的估计变化　{daily_changes(trials.date, estimates)[0]:+.1f} {personal.unit}")
        else:
            calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
            if calibration["version"] != 1 or calibration["targets"] != list(TARGETS):
                raise ValueError("标定文件版本或通道顺序不匹配。")
            if (trials.patient.iloc[0], trials.protocol.iloc[0]) != (calibration["patient"], calibration["protocol"]):
                raise ValueError("患者或动作协议与标定文件不符。")
            if trials.date.min() <= max(calibration["fit_dates"]):
                raise ValueError("基线和复测都应在标定拟合日之后。")
            validation = calibration["validation"]
            if not 0 <= validation["delta_mae_n"] < validation["no_change_mae_n"]:
                raise ValueError("留出日的十路变化误差未优于始终报零，本次标定暂不能出具个人结果。")
            model = load_force_model(args.model, calibration["model_sha256"])
            features = trial_features(trials, model)
            estimates = features @ np.asarray(calibration["weights"]) + calibration["intercept_n"]
            delta = daily_changes(trials.date, estimates)[0]
            if not np.isfinite(delta):
                raise ValueError("无法计算有限的力量变化值。")
            print(f"本次动作力量较基线的估计变化　{delta:+.1f} N")
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"无法完成力量标定或比较：{error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
