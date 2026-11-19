"""从本地 UCI 原始数据和 C 实验生成可离线打开的演示数据。"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from emginsight.features import extract_time_features
from emginsight.uci_gesture import load_uci_gesture_record
from emginsight.uci_gesture.experiment import prepare_data, select_model

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw/EMG_data_for_gestures-master"
RESULTS = ROOT / "results/uci_gesture/classification"


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
def main():
    data, _, _ = prepare_data(RAW)
    scores, (model, scaler, columns) = select_model(data["train"], data["validation"])
    summary = json.loads((RESULTS / "metrics.json").read_text(encoding="utf8"))
    selected = scores.loc[scores.macro_f1.idxmax()]
    if selected.model != "forest" or selected.parameter != summary["selected"]["parameter"] or len(columns) != 24:
        raise ValueError("当前演示需要 C 实验选出的 24 特征随机森林。请核对实验配置。")
    _, index = data["test"]
    saved = pd.read_csv(RESULTS / "test_predictions.csv", dtype={"subject": str})
    pd.testing.assert_frame_equal(index.reset_index(drop=True), saved[index.columns].reset_index(drop=True))
    np.testing.assert_array_equal(model.predict(scaler.transform(data["test"][0][:, columns])), saved.prediction)
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
    payload = dict(clips=clips, selected=summary["selected"], test=summary["test"]["prediction"],
                   matrix=pd.crosstab(saved.label, saved.prediction).reindex(index=range(1,7), columns=range(1,7), fill_value=0).values.tolist())
    dest = Path(__file__).with_name("session.js")
    dest.write_text("window.EMG_SESSION=" + json.dumps(payload, separators=(",", ":"), allow_nan=False) + ";", encoding="utf8")
    simulator = export_simulator(data["train"], model, scaler, columns)
    dest.with_name("simulator-data.js").write_text(
        "window.EMG_SIMULATOR=" + json.dumps(simulator, separators=(",", ":"), allow_nan=False) + ";",
        encoding="utf8",
    )
    print(f"{len(clips)} clips -> {dest}")


if __name__ == "__main__":
    main()
