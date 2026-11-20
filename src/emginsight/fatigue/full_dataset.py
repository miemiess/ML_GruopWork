"""全试验疲劳分类。运行 python -m emginsight.fatigue.full_dataset。

第二版（v2）特征：每路对数幅值与频谱特征先在过去 15 窗（约 30 秒）内因果平滑，再与本试验开头 5 窗（约 10 秒）的
个人基线相减，并附加 15 窗的变化趋势。疲劳评分在每场试验内单调不减，因此预测类别按时间取累计最大值。
设计只在固定划分的 11 名开发用户上按逐人留一比较后确定，测试用户不参与。
"""
from pathlib import Path
import argparse
import json

import joblib
import sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GroupKFold
from sklearn.base import clone
from threadpoolctl import threadpool_limits

from .reading import discover_fatigue_pairs, load_fatigue_record
from .single_channel import (FS, WINDOW, STEP, FEATURE_NAMES, classification_metrics,
                             filter_single_channel, extract_single_channel_features)
from emginsight.common.splitting import split_subject_ids
from .windowing import window_fatigue_record
from emginsight.common.artifacts import file_sha256

MUSCLES = ("BICEPS BRACHII", "DELTOID ANTERIOR", "DELTOID MEDIUS", "DELTOID POSTERIOR")
ROOT = Path("data/raw/EMG_data_for_ZenodoTired")
RUN = Path("results/runs/fatigue/full_dataset")
REPORT = Path("results/fatigue/full_dataset")
SMOOTH = 15      # 因果平滑窗数，约 30 秒
BASELINE = 5     # 个人基线窗数，约 10 秒
TREND = 15       # 趋势跨度窗数
SCOPES = ("four", "agonist", "four_time")


## @brief 疲劳评分在试验内单调不减：同一试验的预测按时间顺序取累计最大值。
# @param pred 逐窗预测类别，与 frame 行一一对应且已按时间排序。
# @param frame 含 subject 与 trial 的窗口表；None 表示整段属于同一试验。
def monotone(pred, frame=None):
    series = pd.Series(np.asarray(pred))
    if frame is None:
        return series.cummax().to_numpy()
    return series.groupby([frame.subject.to_numpy(), frame.trial.to_numpy()], sort=False).cummax().to_numpy()


## @brief 按肌肉名称对齐左右侧，提取因果平滑值、相对开头基线的变化和趋势；只用当前及过去的窗口。
def signal_features(signal, time, names):
    order = [name.split(" ", 1)[1] for name in names]
    if sorted(order) != sorted(MUSCLES):
        raise ValueError(f"肌肉通道不完整 {names}")
    delta = np.diff(time)
    if not np.isclose(1 / np.median(delta), FS, rtol=.01) or np.any(delta <= 0):
        raise ValueError("时间或采样率不符合协议")
    starts = np.arange(0, len(time)-WINDOW+1, STEP)
    gaps = np.r_[0, np.cumsum(delta > .01)]
    starts = starts[gaps[starts+WINDOW-1] == gaps[starts]]
    if not len(starts):
        raise ValueError("没有完整窗口")
    cols = {}
    for muscle in MUSCLES:
        filtered = filter_single_channel(signal[:, order.index(muscle)])
        features = np.vstack([extract_single_channel_features(filtered[chunk[:, None]+np.arange(WINDOW)])
                              for chunk in np.array_split(starts, (len(starts)+63)//64)])
        smooth = pd.DataFrame(features).rolling(SMOOTH, min_periods=1).mean().to_numpy()
        base = pd.DataFrame(features).expanding().mean().to_numpy()
        base[BASELINE:] = base[min(BASELINE, len(base)) - 1]
        lag = np.maximum(np.arange(len(features)) - TREND, 0)
        trend = (features - features[lag]) / TREND
        for j, name in enumerate(FEATURE_NAMES):
            cols[muscle+"_"+name] = smooth[:, j]
            cols[muscle+"_delta_"+name] = smooth[:, j]-base[:, j]
            cols[muscle+"_trend_"+name] = trend[:, j]
    return pd.DataFrame(cols, index=starts)


## @brief 遍历所有试验，按身份配对评分；仅保留真实评分全程一致的窗口。
def prepare(root):
    parts, audit = [], []
    for pair in discover_fatigue_pairs(root):
        try:
            rec = load_fatigue_record(pair.emg_path, pair.rating_path, channel_mode="all")
        except (ValueError, FileNotFoundError) as error:
            audit.append(dict(subject=pair.subject_id,trial=pair.trial_id,windows=0,excluded=str(error)))
            continue
        batch = window_fatigue_record(rec)
        frame = signal_features(rec.signal, rec.time, rec.channel_names).loc[batch.start_indices].copy()
        muscle = rec.main_muscle.split(" ", 1)[1]
        for col in list(frame):
            if col.startswith(muscle+"_"):
                frame["agonist"+col[len(muscle):]] = frame[col]
        frame["label"] = batch.labels
        frame["subject"] = rec.subject_id
        frame["trial"] = rec.trial_id
        frame["seconds"] = rec.time[batch.start_indices] - rec.time[0]
        parts.append(frame)
        audit.append(dict(subject=rec.subject_id, trial=rec.trial_id, windows=len(frame), **batch.dropped, **rec.repairs))
        if rec.trial_id == 12:
            print("Processed subject", rec.subject_id, flush=True)
    return pd.concat(parts, ignore_index=True), pd.DataFrame(audit)


## @brief 三种输入：同侧四路肌电、单路主动力肌、四路肌电加运动时长。
def scope_columns(frame, scope):
    if scope == "agonist":
        return [c for c in frame if c.startswith("agonist")]
    cols = [c for c in frame if c.startswith(MUSCLES)]
    return cols + ["seconds"] if scope == "four_time" else cols


## @brief 在固定验证用户上选型，并对锁定测试用户输出分组误差与对照。
def run():
    REPORT.mkdir(exist_ok=True, parents=True)
    RUN.mkdir(exist_ok=True, parents=True)
    frame, audit = prepare(ROOT)
    audit.to_csv(REPORT/"records.csv", index=False)
    groups = split_subject_ids(frame.subject.tolist(), seed=42)
    train, val, test = [frame[frame.subject.isin(groups[g])].copy() for g in ("train","validation","test")]
    candidates, fitted = [], {}
    for scope in SCOPES:
        cols = scope_columns(frame, scope)
        for name, clf in (("logistic", make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced",max_iter=1500))),
                          ("trees_leaf5", ExtraTreesClassifier(n_estimators=160,min_samples_leaf=5,class_weight="balanced",random_state=42,n_jobs=2)),
                          ("trees_leaf20", ExtraTreesClassifier(n_estimators=160,min_samples_leaf=20,class_weight="balanced",random_state=42,n_jobs=2))):
            clf.fit(train[cols],train.label)
            pred = monotone(clf.predict(val[cols]), val)
            score = np.mean([classification_metrics(val.label[val.subject==s],pred[val.subject==s])["macro_f1"] for s in val.subject.unique()])
            fitted[scope,name] = clf,cols
            candidates.append(dict(scope=scope,model=name,validation_subject_macro_f1=score))
    ranking = pd.DataFrame(candidates).sort_values("validation_subject_macro_f1",ascending=False)
    ranking.to_csv(REPORT/"validation.csv",index=False)
    metrics = dict(groups=groups, records=len(audit), windows=len(frame), test={})
    output = test[["subject","trial","seconds","label"]].copy()
    for scope in SCOPES:
        name = ranking.loc[ranking.scope==scope,"model"].iloc[0]
        clf,cols = fitted[scope,name]
        output[scope] = monotone(clf.predict(test[cols]), test)
        if scope=="four":
            path=RUN/"fatigue_model.joblib"
            joblib.dump(dict(model=clf,columns=cols,muscles=MUSCLES,fs=FS,window=WINDOW,step=STEP),path,compress=3)
            np.testing.assert_array_equal(monotone(joblib.load(path)["model"].predict(test[cols]), test), output[scope])
            metrics["selected"]=name
    output["elapsed"] = KNeighborsClassifier(n_neighbors=61).fit(train[["seconds"]],train.label).predict(test[["seconds"]])
    output["majority"] = train.label.mode()[0]
    for name in (*SCOPES,"elapsed","majority"):
        metrics["test"][name] = classification_metrics(output.label,output[name])
    metrics["per_subject"]={s:classification_metrics(g.label,g.four) for s,g in output.groupby("subject")}
    output.to_csv(REPORT/"test_predictions.csv",index=False)
    # 外层每名用户只预测一次。内层按人预留验证用户，独立选型后重新拟合外层训练集。
    oof = []
    rng = np.random.default_rng(42)
    for fold, (fit_idx, check_idx) in enumerate(GroupKFold(4).split(frame, groups=frame.subject)):
        fit, check = frame.iloc[fit_idx], frame.iloc[check_idx]
        val_ids = rng.permutation(sorted(fit.subject.unique()))[:3]
        inner = fit[~fit.subject.isin(val_ids)]
        hold = fit[fit.subject.isin(val_ids)]
        rows = check[["subject","trial","seconds","label"]].copy()
        for scope in SCOPES:
            options = {}
            for (candidate_scope,name),(est,cols) in fitted.items():
                if scope != candidate_scope:
                    continue
                candidate = clone(est).fit(inner[cols],inner.label)
                predictions = monotone(candidate.predict(hold[cols]), hold)
                options[name] = np.mean([classification_metrics(hold.label[hold.subject==s], predictions[hold.subject==s])["macro_f1"] for s in val_ids])
            name = max(options,key=options.get)
            est,cols = fitted[scope,name]
            rows[scope] = monotone(clone(est).fit(fit[cols],fit.label).predict(check[cols]), check)
            rows[scope+"_model"] = name
        rows["elapsed"] = KNeighborsClassifier(61).fit(fit[["seconds"]],fit.label).predict(check[["seconds"]])
        rows["majority"] = fit.label.mode()[0]
        rows["fold"] = fold
        oof.append(rows)
        print("Completed outer fold",fold,flush=True)
    oof = pd.concat(oof)
    oof.to_csv(REPORT/"grouped_predictions.csv",index=False)
    people = pd.DataFrame([dict(subject=s,windows=len(g),**{
        name:classification_metrics(g.label,g[name])["macro_f1"] for name in (*SCOPES,"elapsed","majority")})
        for s,g in oof.groupby("subject")])
    people.to_csv(REPORT/"per_subject.csv",index=False)
    metrics["grouped_cv"]={k:classification_metrics(oof.label,oof[k]) for k in (*SCOPES,"elapsed","majority")}
    metrics["feature_design"]=dict(version=2, smooth_windows=SMOOTH, baseline_windows=BASELINE, trend_windows=TREND,
                                   monotone=True, chosen_by="固定划分 11 名开发用户逐人留一")
    (REPORT/"metrics.json").write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding="utf8")
    write_report()
    print([(k,v["macro_f1"]) for k,v in metrics["test"].items()],flush=True)


def write_report():
    """从本次评估产物生成报告与模型校验卡。"""
    metrics = json.loads((REPORT/"metrics.json").read_text(encoding="utf8"))
    records = pd.read_csv(REPORT/"records.csv")
    people = pd.read_csv(REPORT/"per_subject.csv")
    from matplotlib.figure import Figure

    figure = Figure(figsize=(8, 4), layout="constrained")
    axes = figure.subplots()
    positions = np.arange(len(people))
    for offset, key, name in [(-0.27, "four", "Four-channel EMG"), (0, "four_time", "EMG + elapsed time"),
                              (0.27, "elapsed", "Elapsed time only")]:
        axes.bar(positions + offset, people[key], width=0.26, label=name)
    axes.set(xticks=positions, xticklabels=people.subject, ylim=(0, 1),
             xlabel="Held-out participant", ylabel="Macro-F1", title="Grouped cross-validation")
    axes.legend()
    figure.savefig(REPORT/"per_subject.png")
    model_path = RUN/"fatigue_model.joblib"
    card = dict(sha256=file_sha256(model_path), path=model_path.as_posix(), sklearn=sklearn.__version__,
                selected=metrics["selected"], groups=metrics["groups"], channels=MUSCLES, fs_hz=FS,
                window=WINDOW, step=STEP, features=joblib.load(model_path)["columns"])
    (REPORT/"model_card.json").write_text(json.dumps(card,indent=2),encoding="utf8")
    previous = REPORT/"metrics_v1.json"
    v1 = json.loads(previous.read_text(encoding="utf8")) if previous.exists() else None
    names = (("four","四路肌电"),("agonist","单路主动力肌"),("four_time","四路肌电 + 运动时长"),
             ("elapsed","仅运动时长（kNN）"),("majority","多数类"))
    report = ["# E · 全试验自感疲劳分类", "",
              f"扫描 {len(records)} 场，{int((records.windows>0).sum())} 场提供有效窗口，共 {metrics['windows']} 窗、{len(people)} 人。排除原因及尾行修正见 [records.csv](records.csv)。",
              "", "固定划分沿用原实验的 8/3/2 人，按验证用户平均 macro-F1 选型。另做四折按人嵌套验证，每折仅从外层训练人群抽取三人选型，再用该折全部训练人员重新拟合。每名参与者只贡献一次外层测试预测。",
              "", "## 第二版特征", "",
              f"156 场试验的评分全部单调不减（0→1→2），而第一版逐窗独立判断、只以首窗为参照，噪声大。第二版对每路特征做过去 {SMOOTH} 窗（约 30 秒）的因果平滑，减去试验开头 {BASELINE} 窗（约 10 秒）的个人基线，加入 {TREND} 窗趋势，并按时间对预测取累计最大值。全部只用当前及过去的窗口。",
              "", "这些设计只在固定划分的 11 名开发用户上按逐人留一比较后确定，两名固定测试用户不参与；四折结果包含这 11 人，可能略偏乐观。",
              "", "| 输入 | 固定测试 macro-F1 | 四折合并 macro-F1 | 四折逐人平均 macro-F1 |" + (" 第一版四折合并 |" if v1 else ""),
              "|---|---:|---:|---:|" + ("---:|" if v1 else "")]
    for k,n in names:
        old = f" {v1['grouped_cv'][k]['macro_f1']:.3f} |" if v1 and k in v1["grouped_cv"] else (" — |" if v1 else "")
        report.append(f"| {n} | {metrics['test'][k]['macro_f1']:.3f} | {metrics['grouped_cv'][k]['macro_f1']:.3f} | {people[k].mean():.3f} |{old}")
    report.extend(["", f"四路模型在 {int((people.four>people.elapsed).sum())}/{len(people)} 名外层测试用户上超过时间基线，加入运动时长后为 {int((people.four_time>people.elapsed).sum())}/{len(people)}。逐人 F1 范围 {people.four.min():.3f}–{people.four.max():.3f}。",
                   "", f"固定测试的两人上，四路肌电为 {metrics['test']['four']['macro_f1']:.3f}，高于时长基线的 {metrics['test']['elapsed']['macro_f1']:.3f}；但只有两人，结论以 13 人的四折结果为准。",
                   "", "**结论**：第二版明显改善了纯肌电模型（四折 macro-F1 第一版 0.388），但在未见用户上仍不如只看运动时长；肌电与时长合用也没有稳定超过时长基线。现有数据不足以证明表面肌电能比运动时长更好地预测新用户的自感疲劳。模型预测的是试验中的主观评分；重叠窗口不独立，小样本和单一采集协议限制泛化。",
                   "", "[逐人指标](per_subject.csv) · [逐窗留出预测](grouped_predictions.csv) · [含三类召回率和混淆矩阵的指标](metrics.json) · [第一版指标](metrics_v1.json) · [模型卡](model_card.json) · [复现与推理](../../../src/emginsight/fatigue/README.md)"])
    report.extend(["", "![四折验证中各参与者的四路肌电与时间基线表现](per_subject.png)"])
    (REPORT/"report.md").write_text("\n".join(report)+"\n",encoding="utf8")


def predict_record(path, artifact):
    """从单侧四路原始试验 CSV 推理，返回各窗的时间、等级与类别概率。"""
    frame = pd.read_csv(path)
    cols = [c for c in frame if ": EMG " in c and c.endswith(" [V]")]
    if len(cols)!=4:
        raise ValueError("需要四路带肌肉名称的原始肌电列")
    positions = [frame.columns.get_loc(c)-1 for c in cols]
    sides = {c.split()[0] for c in cols}
    if sides not in ({"L"}, {"R"}) or any(i < 0 or not frame.columns[i].startswith("X [s]") for i in positions):
        raise ValueError("需要同侧四路，每路肌电前应有秒时间列 X [s]")
    times = frame.iloc[:,positions].to_numpy(dtype=float)
    values = frame[cols].to_numpy(dtype=float)
    if len(frame)<WINDOW or not np.isfinite(times).all() or not np.isfinite(values).all():
        raise ValueError("记录过短或包含非有限值")
    end=len(frame)
    while end>1 and np.any((times[end-1]==0)&(values[end-1]==0)):
        end-=1
    if not np.allclose(times[:end],times[:end,:1],rtol=0,atol=1e-9):
        raise ValueError("四路时间轴不一致")
    features=signal_features(values[:end],times[:end,0],[c.split(":")[0] for c in cols])
    inputs=features[artifact["columns"]]
    labels=monotone(artifact["model"].predict(inputs))
    result=pd.DataFrame(dict(start_seconds=times[features.index,0],end_seconds=times[features.index+WINDOW-1,0],
                             predicted_label=labels))
    for label, probability in zip(artifact["model"].classes_,artifact["model"].predict_proba(inputs).T):
        result[f"prob_{label}"]=probability
    return result


def predict_csv(path, destination, model_path):
    """兼容原单文件预测入口。"""
    result=predict_record(path, joblib.load(model_path))
    destination.parent.mkdir(parents=True,exist_ok=True)
    result.to_csv(destination,index=False)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predict",type=Path)
    parser.add_argument("--model",type=Path,default=RUN/"fatigue_model.joblib")
    parser.add_argument("--output",type=Path,default=RUN/"prediction.csv")
    args=parser.parse_args()
    with threadpool_limits(limits=1):
        if args.predict:
            predict_csv(args.predict,args.output,args.model)
        else:
            run()
