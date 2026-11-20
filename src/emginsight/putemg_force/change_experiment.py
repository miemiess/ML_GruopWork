"""从已冻结的十路回归器训练跨人指力变化模型，按用户验证、选型、测试并导出。"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from .change import fit_weights
from .change_model import FEATURES, RAW_UNIT, RESULTS, RUNS, ForceChangeModel, load_change_model, save_change_model
from .experiment import CACHE_VERSION
from .model import load_force_model
from .targets import TARGETS


## @brief 从实验缓存恢复第一阶段预测，以独立指令定位保持段，用实测力量构造原始刻度标签。
def collect(path, cache_dir, stage1):
    size = stage1.preprocessing["window_size"]
    with np.load(cache_dir / f"{path.stem}.npz") as cache:
        X, starts, rest = cache["X"], cache["starts"], cache["rest"]
        pred = stage1.predict(X, np.broadcast_to(rest, X.shape), np.repeat(path.stem, len(X)))
        tare = cache["tare"][6:]
    cues = [f"TRAJ_{i}" for i in range(1, 5)]
    times = pd.read_csv(path, nrows=0).columns[0]
    raw = pd.read_csv(path, usecols=[times, "FORCE_MVC", *cues, *TARGETS[6:]])
    seconds = (raw[times] - raw[times].iloc[0]).to_numpy()
    reference = (raw[list(TARGETS[6:])].to_numpy() - tare).mean(axis=1)
    isolated = (raw[cues[:3]] == 0).all(axis=1).to_numpy()
    first = starts[np.maximum(np.arange(len(starts)) + 1 - stage1.smoothing, 0)]
    fields = path.stem.split("-")
    rows = []
    for level in (20, 40):
        on = isolated & np.isclose(raw.TRAJ_4, raw.FORCE_MVC * level / 100, rtol=0, atol=0.04)
        edges = np.flatnonzero(np.diff(np.r_[False, on, False])).reshape(-1, 2)
        for a, b in edges:
            use = (first >= a) & (starts + size <= b)
            if not use.any():
                continue
            start = (seconds[max(a - 1, 0)] + seconds[a]) / 2
            end = (seconds[b - 1] + seconds[min(b, len(seconds)-1)]) / 2
            rows.append(dict(patient=fields[1], date="-".join(fields[3:6]), protocol=f"traj4-{level}",
                             record=path.name, start_s=start, end_s=end, reference_raw=np.median(reference[a:b]),
                             **dict(zip(FEATURES, np.median(pred[use], axis=0)))))
    return rows


## @brief 先聚合每人每天各档位的动作，再计算两日间的实测变化与预测变化。
def paired(frame):
    columns = ["reference_raw", "prediction", "force8"]
    rows = []
    for (patient, protocol), trials in frame.groupby(["patient", "protocol"]):
        daily = trials.groupby("date")[columns].median().sort_index()
        if len(daily) == 2:
            rows.append(dict(patient=patient, protocol=protocol, baseline=daily.index[0],
                             followup=daily.index[1], **(daily.iloc[1] - daily.iloc[0]).to_dict()))
    if not rows:
        raise ValueError("评估集没有同一用户同一协议的两日记录。")
    return pd.DataFrame(rows)


## @brief 使用拟合用户的样本估计融合系数，并对独立用户的每次动作推理。
def evaluate(fit, evaluate_on, alpha, digest):
    w, b = fit_weights(fit[list(FEATURES)], fit.reference_raw, alpha)
    model = ForceChangeModel(w, b, digest, RAW_UNIT, tuple(sorted(fit.protocol.unique())))
    w8, b8 = fit_weights(fit[["FORCE_8_pred"]], fit.reference_raw)
    rows = evaluate_on.assign(prediction=model.predict(evaluate_on[list(FEATURES)]),
                              force8=evaluate_on.FORCE_8_pred * w8[0] + b8)
    return model, rows, paired(rows)


## @brief 汇总每个“用户 × 动作档位”的变化误差、方向和零变化对照。
def scores(pairs):
    true = pairs.reference_raw
    result = {}
    for name, pred in (("ten", pairs.prediction), ("force8", pairs.force8), ("zero", np.zeros(len(true)))):
        error = pred - true
        result[name] = dict(mae=float(np.abs(error).mean()), rmse=float(np.sqrt(np.mean(error**2))),
                            direction_correct=int(np.sum(np.sign(true) == np.sign(pred))), pairs=len(true))
    return result


## @brief 按冻结模型的用户划分组织数据，验证日差 MAE 选型，保存并回读获选模型。
def run(root, runs, output):
    regression = output.parent / "regression"
    card = json.loads((regression / "model_card.json").read_text(encoding="utf-8"))
    groups = json.loads((regression / "metrics.json").read_text(encoding="utf-8"))["subjects"]
    digest = card["sha256"]
    stage1 = load_force_model(runs / "force_model.joblib", digest)
    # 第二阶段仅使用第一阶段训练用户之外的预测来拟合。保留原始九名测试用户。
    development = sorted(groups["validation"])
    cut = 2 * len(development) // 3
    splits = {"train": development[:cut], "validation": development[cut:], "test": groups["test"]}
    owners = {p: split for split, people in splits.items() for p in people}
    if len(owners) != sum(map(len, splits.values())) or set(owners).intersection(stage1.train_subjects):
        raise ValueError("第一阶段拟合用户或第二阶段划分存在重叠。")
    config = stage1.preprocessing
    cache_dir = runs / f"{CACHE_VERSION}-{config['window_size']}-{config['step_size']}"
    items = []
    for path in sorted(root.rglob("emg_force-*-repeats_long-*.csv")):
        if path.stem.split("-")[1] in owners:
            items.extend(collect(path, cache_dir, stage1))
            print(path.name, flush=True)
    frame = pd.DataFrame(items)
    if not items or set(frame.patient) != set(owners):
        raise ValueError("保持段未覆盖所需用户，请检查原始 CSV 和第一阶段实验缓存。")
    split = frame.patient.map(owners)
    train, val, test = [frame[split == name] for name in ("train", "validation", "test")]
    candidates = []
    for alpha in (1., 10., 100.):
        _, _, pairs = evaluate(train, val, alpha, digest)
        candidates.append(dict(alpha=alpha, **scores(pairs)["ten"]))
    validation = pd.DataFrame(candidates)
    alpha = float(validation.loc[validation.mae.idxmin(), "alpha"])
    model, trials, pairs = evaluate(train, test, alpha, digest)
    metrics = dict(subjects=splits, trials=split.value_counts().to_dict(), alpha=alpha, unit=RAW_UNIT,
                   test=scores(pairs), stage1_sha256=digest)
    output.mkdir(parents=True, exist_ok=True)
    model_path = runs / "force_change_model.joblib"
    card_path = output / "model_card.json"
    save_change_model(model, model_path, card_path, dict(estimator="StandardScaler + Ridge", alpha=alpha,
                      target="median of mean rest-referenced FORCE_7..10 during one hold", subjects=splits))
    restored = load_change_model(model_path, card_path)
    np.testing.assert_array_equal(restored.predict(test[list(FEATURES)]), trials.prediction)
    validation.to_csv(output / "validation.csv", index=False)
    trials.to_csv(output / "test_trials.csv", index=False)
    pairs.to_csv(output / "test_changes.csv", index=False)
    (output / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    example = trials[(trials.patient == sorted(test.patient.unique())[0]) & (trials.protocol == "traj4-40")]
    example[["patient", "protocol", "date", "record", "start_s", "end_s"]].to_csv(output / "visits.csv", index=False)
    fig = Figure(figsize=(6, 5), layout="constrained")
    ax = fig.subplots()
    for protocol, rows in pairs.groupby("protocol"):
        ax.scatter(rows.reference_raw, rows.prediction, label=protocol)
    limits = [pairs[["prediction", "reference_raw"]].min().min(), pairs[["prediction", "reference_raw"]].max().max()]
    ax.plot(limits, limits, "--", color="gray")
    ax.set(xlabel="Measured change (raw units)", ylabel="Estimated change (raw units)")
    ax.legend()
    fig.savefig(output / "test_changes.png")
    return metrics


## @brief 命令行训练入口，沿用第一阶段实验生成的数据和缓存目录。
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/raw/EMG_data_for_putemgforce"))
    parser.add_argument("--runs", type=Path, default=RUNS)
    parser.add_argument("--output", type=Path, default=RESULTS)
    args = parser.parse_args()
    print(json.dumps(run(args.root, args.runs, args.output)["test"], indent=2))


if __name__ == "__main__":
    main()
