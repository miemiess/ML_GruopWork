"""把五个模型的划分、验证候选、学习曲线和留出结果整理成"训练过程"视图的统一快照。

只读取 results/ 下已保存的实验表格，不重新训练模型。
"""

import json

import pandas as pd
from sklearn.metrics import f1_score

from snapshots import ROOT, file_sha256, write_snapshot

RESULTS = ROOT / "results"


def read_json(path):
    return json.loads((RESULTS / path).read_text(encoding="utf-8"))


def read_csv(path):
    return pd.read_csv(RESULTS / path, dtype={"subject": str, "patient": str})


## @brief 学习曲线：每个训练人数保留全部抽样结果，供网页画点和中位线。
def learning(path, score, baseline):
    if not (RESULTS / path).exists():
        return None
    table = read_csv(path)
    return [dict(users=int(r.users), score=float(r[score]), baseline=float(r[baseline])) for _, r in table.iterrows()]


def gesture():
    m = read_json("gesture/classification/metrics.json")
    names = {"logistic": "逻辑回归", "knn": "kNN", "forest": "随机森林"}
    candidates = [dict(label=f"{names[r.model]} · {r.parameter} · {r.features}", family=names[r.model],
                       score=r.macro_f1, group=r.features) for r in read_csv("gesture/classification/validation.csv").itertuples()]
    test = read_csv("gesture/classification/test_predictions.csv")
    labels = sorted(test.label.unique())
    per_user = [dict(subject=s, model=f1_score(g.label, g.prediction, labels=labels, average="macro", zero_division=0),
                     baseline=f1_score(g.label, g.baseline, labels=labels, average="macro", zero_division=0))
                for s, g in test.groupby("subject")]
    return dict(
        title="六类手势识别", metric=dict(label="macro-F1", better="high", digits=3),
        split=dict(subjects=m["subjects"], counts={k: f"{v:,} 窗" for k, v in m["windows"].items()}),
        families=list(names.values()), candidates=candidates,
        learning=learning("learning_curve/gesture.csv", "macro_f1", "baseline_macro_f1"),
        heldout=[dict(name="随机森林（获选）", value=m["test"]["prediction"]["macro_f1"], role="model"),
                 dict(name="多数类基线", value=m["test"]["baseline"]["macro_f1"], role="baseline")],
        per_user=dict(label="每名测试用户 macro-F1", rows=per_user),
        sources=["gesture/classification/metrics.json", "gesture/classification/validation.csv",
                 "gesture/classification/test_predictions.csv", "learning_curve/gesture.csv"],
    )


def continuous():
    m = read_json("force/regression/metrics.json")
    names = {"ridge": "岭回归", "forest": "随机森林"}
    candidates = [dict(label=f"{names[r.model]} · {r.parameter} · {r.scale} · {r.features} · 平滑 {r.smoothing} 窗",
                       family=names[r.model], score=r.mean_rmse, group=r.scale)
                  for r in read_csv("force/regression/validation.csv").itertuples()]
    users = read_csv("force/regression/per_user.csv")
    return dict(
        title="连续指力回归", metric=dict(label="平均 RMSE", better="low", digits=2),
        split=dict(subjects=m["subjects"],
                   counts={k: f"{m['records'][k]} 条记录 · {v:,} 窗" for k, v in m["windows"].items()}),
        families=list(names.values()), candidates=candidates,
        learning=learning("learning_curve/force.csv", "rmse", "baseline_rmse"),
        heldout=[dict(name="随机森林（获选）", value=m["test"]["model"]["rmse"], role="model"),
                 dict(name="训练均值基线", value=m["test"]["baseline"]["rmse"], role="baseline")],
        per_user=dict(label="每名测试用户 RMSE", rows=[dict(subject=r.subject, model=r.rmse_model,
                                                     baseline=r.rmse_baseline) for r in users.itertuples()]),
        sources=["force/regression/metrics.json", "force/regression/validation.csv",
                 "force/regression/per_user.csv", "learning_curve/force.csv"],
    )


def levels():
    m = read_json("force/levels/metrics.json")
    def family(r):
        return "逻辑回归" if r.model == "logistic" else "类别平衡森林" if "balanced" in r.parameter else "随机森林"
    candidates = [dict(label=f"{family(r)} · {r.scale} · {r.features} · 平滑 {r.smoothing} 窗",
                       family=family(r), score=r.macro_f1, group=r.scale)
                  for r in read_csv("force/levels/validation.csv").itertuples()]
    users = read_csv("force/levels/per_user.csv")
    return dict(
        title="力度三档分类", metric=dict(label="macro-F1", better="high", digits=3),
        split=dict(subjects=m["subjects"], counts={k: f"{sum(v):,} 窗" for k, v in m["label_share"].items()}),
        families=["逻辑回归", "随机森林", "类别平衡森林"], candidates=candidates, learning=None,
        heldout=[dict(name="三档分类器（获选）", value=m["test"]["model"]["macro_f1"], role="model"),
                 dict(name="连续回归后分档", value=m["test"]["regression_to_levels"]["macro_f1"], role="baseline"),
                 dict(name="多数类基线", value=m["test"]["majority"]["macro_f1"], role="baseline")],
        per_user=dict(label="每名测试用户准确率（对照：全判静息）", percent=True,
                      rows=[dict(subject=r.subject, model=r.accuracy, baseline=r.majority_accuracy)
                            for r in users.itertuples()]),
        sources=["force/levels/metrics.json", "force/levels/validation.csv",
                 "force/levels/per_user.csv"],
    )


def change():
    m = read_json("force/change/metrics.json")
    candidates = [dict(label=f"岭回归 · α = {r.alpha:g}", family="岭回归", score=r.mae)
                  for r in read_csv("force/change/validation.csv").itertuples()]
    pairs = read_csv("force/change/test_changes.csv")
    t = m["test"]
    return dict(
        title="两日力量变化", metric=dict(label="变化 MAE", better="low", digits=2),
        split=dict(subjects=m["subjects"], counts={k: f"{v} 次动作" for k, v in m["trials"].items()}),
        families=["岭回归"], candidates=candidates, learning=None,
        heldout=[dict(name="十路融合（获选）", value=t["ten"]["mae"], role="model"),
                 dict(name="仅 FORCE_8", value=t["force8"]["mae"], role="baseline"),
                 dict(name="始终报零", value=t["zero"]["mae"], role="baseline")],
        per_user=dict(label="每次比较的绝对误差（对照：始终报零）",
                      rows=[dict(subject=f"{r.patient} · {r.protocol}", model=abs(r.prediction - r.reference_raw),
                                 baseline=abs(r.reference_raw)) for r in pairs.itertuples()]),
        sources=["force/change/metrics.json", "force/change/validation.csv",
                 "force/change/test_changes.csv"],
    )


def fatigue():
    m = read_json("fatigue/single_channel_biceps/metrics.json")
    names = {"knn": "kNN", "ridge": "岭分类器"}
    subsets = {"amplitude": "幅值", "frequency": "频谱", "amplitude_and_frequency": "幅值+频谱"}
    candidates = [dict(label=f"{names[r.kind]} · {r.parameter:g} · {subsets.get(r.subset, r.subset)}",
                       family=names[r.kind], score=r.validation_macro_f1, group=names[r.kind])
                  for r in read_csv("fatigue/single_channel_biceps/validation_candidates.csv").itertuples()]
    return dict(
        title="单路肌肉疲劳", metric=dict(label="macro-F1", better="high", digits=3),
        split=dict(subjects=m["subject_groups"],
                   counts={k: f"{v['windows']} 窗" for k, v in m["split_counts"].items()}),
        families=list(names.values()), candidates=candidates, learning=None,
        heldout=[dict(name="岭分类器（获选）", value=m["test"]["macro_f1"], role="model"),
                 dict(name="仅用运动时长", value=m["elapsed_time_baseline"]["macro_f1"], role="baseline"),
                 dict(name="多数类基线", value=m["majority_baseline"]["macro_f1"], role="baseline")],
        per_user=dict(label="每名测试参与者 macro-F1",
                      rows=[dict(subject=s, model=v["macro_f1"]) for s, v in m["per_test_subject"].items()]),
        sources=["fatigue/single_channel_biceps/metrics.json",
                 "fatigue/single_channel_biceps/validation_candidates.csv"],
    )


def fatigue_full():
    m = read_json("fatigue/full_dataset/metrics.json")
    v1 = read_json("fatigue/full_dataset/metrics_v1.json") if (RESULTS / "fatigue/full_dataset/metrics_v1.json").exists() else None
    scopes = {"four": "四路肌电", "agonist": "单路主动力肌", "four_time": "四路肌电 + 时长"}
    models = {"logistic": "逻辑回归", "trees_leaf5": "极端随机树（叶 5）", "trees_leaf20": "极端随机树（叶 20）"}
    candidates = [dict(label=f"{scopes[r.scope]} · {models.get(r.model, r.model)}", family=scopes[r.scope],
                       score=r.validation_subject_macro_f1, group=scopes[r.scope])
                  for r in read_csv("fatigue/full_dataset/validation.csv").itertuples()]
    candidates.sort(key=lambda c: list(scopes.values()).index(c["family"]))
    people = read_csv("fatigue/full_dataset/per_subject.csv")
    cv = m["grouped_cv"]
    heldout = [dict(name="四路肌电（第二版）", value=cv["four"]["macro_f1"], role="model")]
    if v1:
        heldout.append(dict(name="四路肌电（第一版）", value=v1["grouped_cv"]["four"]["macro_f1"], role="baseline"))
    heldout += [dict(name="四路肌电 + 运动时长", value=cv["four_time"]["macro_f1"], role="baseline"),
                dict(name="仅运动时长", value=cv["elapsed"]["macro_f1"], role="baseline"),
                dict(name="多数类", value=cv["majority"]["macro_f1"], role="baseline")]
    return dict(
        title="全试验肌肉疲劳", metric=dict(label="四折 macro-F1", better="high", digits=3),
        split=dict(subjects=m["groups"], counts={}),
        families=list(scopes.values()), candidates=candidates, learning=None, heldout=heldout,
        per_user=dict(label="每名参与者四折 macro-F1（对照：仅运动时长）",
                      rows=[dict(subject=str(r.subject), model=r.four, baseline=r.elapsed) for r in people.itertuples()]),
        sources=["fatigue/full_dataset/metrics.json", "fatigue/full_dataset/validation.csv",
                 "fatigue/full_dataset/per_subject.csv", "fatigue/full_dataset/metrics_v1.json"],
    )


def export():
    data = {name: build() for name, build in
            (("gesture", gesture), ("continuous", continuous), ("levels", levels), ("change", change), ("fatigue", fatigue), ("fatigue_full", fatigue_full))}
    source = {}
    for item in data.values():
        for path in item.pop("sources"):
            if (RESULTS / path).exists():
                source[path] = file_sha256(RESULTS / path)
    write_snapshot("training-data.js", "trainingData", data, source)
    return {"training": source}


if __name__ == "__main__":
    export()
    print("已导出训练过程快照。")
