"""学习曲线：固定获选配置，只改变参与训练的用户数，在验证用户上观察效果变化。

测试用户不参与。每个训练人数随机抽取若干组用户，结果用于说明"跨用户泛化主要受训练人数影响"。
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from emginsight.models import MajorityClassifier, Standardizer
from emginsight.putemg_force import experiment as force
from emginsight.putemg_force.model import smooth, transform
from emginsight.putemg_force.targets import target_metrics
from emginsight.uci_gesture import experiment as gesture

SEED = 42
REPEATS = 5


## @brief 为每个训练人数抽取若干组不重复的用户子集；全体用户只取一次。
# @param subjects 训练用户编号。
# @param sizes 训练人数列表。
# @param repeats 每个人数的抽样次数。
# @return list (人数, 抽样序号, 用户元组)。
def subsets(subjects, sizes, repeats=REPEATS):
    rng = np.random.default_rng(SEED)
    plans = []
    for size in sizes:
        seen = set()
        while len(seen) < (1 if size >= len(subjects) else repeats):
            users = tuple(sorted(rng.choice(subjects, size, replace=False)))
            if users not in seen:
                plans.append((size, len(seen), users))
                seen.add(users)
    return plans


## @brief 指力回归：获选的 log_rest + RMS/MAV/WL + 随机森林 + 五窗平滑。
# @param root putEMG-Force 原始数据目录。
# @param cache 特征缓存目录。
# @return DataFrame 每次抽样一行。
def force_curve(root, cache):
    X, y_raw, tare, rest, index, groups, _ = force.prepare_data(root, cache)
    target = y_raw - tare
    choice = json.loads(Path("results/putemg_force/regression/metrics.json").read_text(encoding="utf-8"))["selected"]
    inputs = transform(X, rest, choice["features"], choice["scale"])
    factory = {(n, p): f for n, p, f in force.model_candidates()}[(choice["model"], choice["parameter"])]
    val = (index.group == "validation").to_numpy()
    records = index.record.to_numpy()[val]
    rows = []
    for size, repeat, users in subsets(groups["train"], (1, 2, 4, 8, 16, len(groups["train"]))):
        train = index.subject.isin(users).to_numpy()
        pred, base, *_ = force.fit_predict(inputs, target, train, val, factory)
        model = target_metrics(target[val], smooth(pred, records, int(choice["smoothing"])))
        baseline = target_metrics(target[val], base)
        rows.append(dict(users=size, repeat=repeat, subjects=" ".join(users), windows=int(train.sum()),
                         rmse=model.rmse.mean(), r2=model.r2.mean(),
                         baseline_rmse=baseline.rmse.mean(), baseline_r2=baseline.r2.mean()))
        print(f"指力 {size:2d} 人 #{repeat}：RMSE {rows[-1]['rmse']:.2f}，基线 {rows[-1]['baseline_rmse']:.2f}", flush=True)
    return pd.DataFrame(rows)


## @brief 手势分类：获选的 RMS/MAV/WL + 100 棵树随机森林。
# @param root UCI 手势原始数据目录。
# @return DataFrame 每次抽样一行。
def gesture_curve(root):
    datasets, groups, _ = gesture.prepare_data(root)
    X, index = datasets["train"]
    X_val, val_index = datasets["validation"]
    rows = []
    for size, repeat, users in subsets(groups["train"], (1, 2, 4, 8, 16, len(groups["train"]))):
        train = index.subject.astype(str).isin(users).to_numpy()
        scaler = Standardizer().fit(X[train])
        model = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)
        model.fit(scaler.transform(X[train]), index.label[train])
        pred = model.predict(scaler.transform(X_val))
        majority = MajorityClassifier().fit(X[train], index.label[train]).predict(X_val)
        score = lambda p: f1_score(val_index.label, p, labels=gesture.LABELS, average="macro", zero_division=0)
        rows.append(dict(users=size, repeat=repeat, subjects=" ".join(users), windows=int(train.sum()),
                         macro_f1=score(pred), accuracy=accuracy_score(val_index.label, pred),
                         baseline_macro_f1=score(majority)))
        print(f"手势 {size:2d} 人 #{repeat}：macro-F1 {rows[-1]['macro_f1']:.3f}", flush=True)
    return pd.DataFrame(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force-data", type=Path, default=Path("data/raw/EMG_data_for_putemgforce"))
    parser.add_argument("--gesture-data", type=Path, default=Path("data/raw/EMG_data_for_gestures-master"))
    parser.add_argument("--cache-dir", type=Path, default=Path("results/runs/putemg_force"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/learning_curve"))
    parser.add_argument("--only", choices=("force", "gesture"))
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.only != "force":
        gesture_curve(args.gesture_data).to_csv(args.output_dir / "uci_gesture.csv", index=False)
    if args.only != "gesture":
        force_curve(args.force_data, args.cache_dir).to_csv(args.output_dir / "putemg_force.csv", index=False)


if __name__ == "__main__":
    main()
