"""从本地 UCI 原始数据和 C 实验生成可离线打开的演示数据。"""
import argparse
import json

import numpy as np
import pandas as pd

from emginsight.features import extract_time_features
from emginsight.uci_gesture import load_uci_gesture_record
from emginsight.uci_gesture.experiment import LABELS, prepare_data
from emginsight.uci_gesture.model import load_gesture_model

from snapshots import GESTURE_REPORT, GESTURE_REPORT_FILES, ROOT, experiment_source, write_snapshot

RAW = ROOT / "data/raw/EMG_data_for_gestures-master"
MODEL = ROOT / "results/runs/uci_gesture/gesture_model.joblib"


def export_simulator(train, model, scaler, columns):
    features, index = train
    templates = {}
    for label in model.classes_:
        positions = np.flatnonzero(index.label.to_numpy() == label)
        scaled = scaler.transform(features[positions][:, columns])
        distance = np.sum((scaled - np.median(scaled, axis=0)) ** 2, axis=1)
        row = index.iloc[positions[distance.argmin()]]
        record = load_uci_gesture_record(RAW / row.subject / (row.record + ".txt"))
        waveform = record.signal[row.start_sample:row.start_sample + 200]
        templates[int(label)] = waveform.tolist()

    trees = []
    for estimator in model.estimators_:
        tree = estimator.tree_
        nodes = []
        for i, left in enumerate(tree.children_left):
            if left < 0:
                weights = tree.value[i, 0]
                nodes.append((weights / weights.sum()).tolist())
            else:
                nodes.append([int(left), int(tree.children_right[i]), int(tree.feature[i]), tree.threshold[i]])
        trees.append(nodes)
    return dict(templates=templates, trees=trees, classes=model.classes_.tolist(), columns=columns.tolist(),
                mean=scaler.mean_.tolist(), scale=scaler.scale_.tolist())


## @brief 为每种实际出现的真值/预测组合选取一段真实记录。
def export_clips(saved, model, scaler, columns):
    clips = []
    for (label, prediction), matches in saved.groupby(["label", "prediction"]):
        preferred = matches[matches.subject == "09"]
        target = (preferred if len(preferred) else matches).iloc[0]
        record = load_uci_gesture_record(RAW / target.subject / (target.record + ".txt"))
        time = record.time - record.time[0]
        start = int(target.start_sample)
        left = max(0, start - 1000)
        right = min(len(time), start + 2000)
        if label == prediction == 2 and target.subject == "09":
            # 保留握拳前的静息和未标注间隔供连续回放。
            left = max(0, start - 5500)
            right = min(len(time), start + 2500)
        rows = saved[(saved.subject == target.subject) & (saved.record == target.record)
                     & (saved.start_sample >= left) & (saved.start_sample + 200 <= right)]
        starts = rows.start_sample.to_numpy()
        features = extract_time_features(record.signal[starts[:, None] + np.arange(200)])
        scaled = scaler.transform(features[:, columns])
        frames = []
        for row, feat, z, prob in zip(rows.itertuples(), features, scaled, model.predict_proba(scaled)):
            frames.append(dict(a=int(time[row.start_sample]), b=int(time[row.start_sample + 199]),
                               start=row.start_sample, pred=row.prediction, label=row.label,
                               f=feat.tolist(), z=z.tolist(), p=prob.tolist()))
        clips.append(dict(label=int(label), pred=int(prediction), subject=target.subject,
                          record=target.record, focus=int(time[start + 199]), frames=frames,
                          t=time[left:right].astype(int).tolist(), truth=record.labels[left:right].tolist(),
                          channels=record.signal[left:right].T.tolist()))
    return clips


## @brief 复用获选手势模型，按实验记录生成回放、模拟和可选教学快照。
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-development", action="store_true", help="只导出手势回放和交互模拟。")
    args = parser.parse_args(argv)

    source = experiment_source(GESTURE_REPORT, GESTURE_REPORT_FILES)
    bundle = load_gesture_model(MODEL, source["model_sha256"])
    model, scaler, columns = bundle.model, bundle.scaler, np.asarray(bundle.columns)
    summary = json.loads((GESTURE_REPORT / "metrics.json").read_text(encoding="utf-8"))
    scores = pd.read_csv(GESTURE_REPORT / "validation.csv")
    best = scores.loc[scores.macro_f1.idxmax()]
    selected = summary["selected"]
    settings = bundle.preprocessing
    if not (
        best.model == selected["model"] == "forest"
        and best.features == selected["features"] == bundle.features == "RMS+MAV+WL"
        and best.parameter == selected["parameter"] == f"trees={getattr(model, 'n_estimators', None)}"
        and np.array_equal(columns, np.arange(24))
        and bundle.classes == LABELS
        and tuple(settings[key] for key in ("window_size", "step_size", "max_gap_ms", "channels")) == (200, 100, 20, 8)
    ):
        raise ValueError("演示需要与 C 实验表格匹配的八通道、24 特征随机森林模型。")

    data, groups, _ = prepare_data(RAW)
    if {group: list(users) for group, users in groups.items()} != summary["subjects"] or (
        groups["train"] != bundle.train_subjects
    ):
        raise ValueError("手势原始数据、模型和实验报告的用户划分不一致。")
    test, index = data["test"]
    saved = pd.read_csv(GESTURE_REPORT / "test_predictions.csv", dtype={"subject": str})
    pd.testing.assert_frame_equal(index, saved[index.columns])
    # 回放和教学使用 C 实验的六类评估口径，与部署时附加的未知拒识阈值独立。
    np.testing.assert_array_equal(model.predict(scaler.transform(test[:, columns])), saved.prediction)

    clips = export_clips(saved, model, scaler, columns)
    payload = dict(clips=clips, selected=summary["selected"], test=summary["test"]["prediction"],
                   matrix=pd.crosstab(saved.label, saved.prediction)
                   .reindex(index=LABELS, columns=LABELS, fill_value=0).values.tolist())
    simulator = export_simulator(data["train"], model, scaler, columns)
    if not args.skip_development:
        from development_export import export_development

        development = export_development(data, scores, model, scaler, columns, summary, RAW)
        write_snapshot("development-data.js", "EMG_DEVELOPMENT", development, source)
    write_snapshot("session.js", "EMG_SESSION", payload, source)
    write_snapshot("simulator-data.js", "EMG_SIMULATOR", simulator, source)
    print(f"已导出手势回放 {len(clips)} 个案例及交互模拟。")


if __name__ == "__main__":
    main()
