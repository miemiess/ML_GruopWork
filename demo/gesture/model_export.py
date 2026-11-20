"""将力度三档、两日变化和单路疲劳实验的测试输出打包为静态演示数据。"""
import json

import numpy as np
import pandas as pd
from emginsight.putemg_force.change import FEATURES, load_change_model

from snapshots import ROOT, file_sha256, write_snapshot


def export():
    reports = {
        "levels": ("putemg_force/levels", "example_predictions", "validation"),
        "change": ("putemg_force/change", "test_trials", "validation"),
        "fatigue": ("zenodo_fatigue/single_channel_biceps", "test_predictions", "validation_candidates"),
    }
    data, sources = {}, {}
    for key, (folder, example, candidates) in reports.items():
        report = ROOT / "results" / folder
        metrics = json.loads((report / "metrics.json").read_text(encoding="utf-8"))
        files = ["metrics.json", f"{example}.csv", f"{candidates}.csv"]
        rows = pd.read_csv(report / files[1], dtype={"patient": str, "subject_id": str})
        if key == "levels":
            rows = rows[["time_s", "relative_force", "label", "prediction", "prob_0", "prob_1", "prob_2"]]
        if key == "fatigue":
            model_path = ROOT / "results/runs/single_channel_biceps/model.npz"
            with np.load(model_path) as model:
                features = rows[metrics["feature_names"]].to_numpy()[:, model["columns"]]
                z = (features - model["mean"]) / model["scale"]
                scores = np.column_stack([np.ones(len(z)), z]) @ model["weights"]
            np.testing.assert_array_equal(scores.argmax(axis=1), rows.predicted_label)
            rows[["score_0", "score_1", "score_2"]] = scores
        validation = pd.read_csv(report / files[2])
        item = dict(metrics=metrics, rows=json.loads(rows.to_json(orient="records")),
                    validation=json.loads(validation.to_json(orient="records")))
        if key == "change":
            model_path = ROOT / "results/runs/putemg_force/force_change_model.joblib"
            model = load_change_model(model_path, report / "model_card.json")
            np.testing.assert_allclose(model.predict(rows[list(FEATURES)]), rows.prediction, atol=1e-7)
            item.update(weights=model.weights.tolist(), intercept=model.intercept)
            files.append("test_changes.csv")
            item["pairs"] = json.loads(pd.read_csv(report / files[-1], dtype={"patient": str}).to_json(orient="records"))
        if (report / "model_card.json").exists():
            files.append("model_card.json")
        sources[key] = {f"{folder}/{f}": file_sha256(report / f) for f in files}
        if key in {"change", "fatigue"}:
            sources[key]["model_sha256"] = file_sha256(model_path)
        data[key] = item
    report = ROOT/"results/zenodo_fatigue/full_dataset"
    files = ("metrics.json","model_card.json","validation.csv","per_subject.csv","test_predictions.csv")
    full = {name.split(".")[0]: (json.loads((report/name).read_text(encoding="utf8")) if name.endswith(".json")
                                 else json.loads(pd.read_csv(report/name).to_json(orient="records"))) for name in files}
    source = {name:file_sha256(report/name) for name in files}
    write_snapshot("full-fatigue-data.js", "fullFatigue", full, source)
    sources["full_fatigue"] = source
    write_snapshot("model-data.js", "modelDemos", data, sources)
    return sources


if __name__ == "__main__":
    export()
