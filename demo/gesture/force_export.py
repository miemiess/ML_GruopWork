"""从 D 实验的模型、特征缓存和报告导出可离线播放的指力案例。"""

import json

import numpy as np
import pandas as pd

from emginsight.putemg_force.model import load_force_model, smooth, transform

from snapshots import FORCE_REPORT, FORCE_REPORT_FILES, ROOT, experiment_source, write_snapshot

CACHE = ROOT / "results/runs/putemg_force"


## @brief 核对案例身份、窗口和模型输出，再导出网页使用的实验快照。
def export():
    source = experiment_source(FORCE_REPORT, FORCE_REPORT_FILES)
    metrics = json.loads((FORCE_REPORT / "metrics.json").read_text(encoding="utf-8"))
    example = pd.read_csv(FORCE_REPORT / "example_predictions.csv")
    record = metrics["curve_example"]["record"]
    assert example.record.eq(record).all()
    (cache_file,) = CACHE.glob(f"*/{record}.npz")
    bundle = load_force_model(CACHE / "force_model.joblib", source["model_sha256"])
    with np.load(cache_file) as item:
        X, rest, tare = item["X"], item["rest"], item["tare"]
        np.testing.assert_array_equal(example.start_sample, item["starts"])
        np.testing.assert_allclose(example.time_s, item["time_s"], atol=0.00051, rtol=0)
        calibrated = transform(X, np.broadcast_to(rest, X.shape), bundle.features, bundle.scale)
        z = bundle.scaler.transform(calibrated)
        raw = bundle.model.predict(z)
        truth = item["y"] - tare
    names = metrics["targets"]
    assert tuple(names) == bundle.targets
    observed = example[[f"{n}_true" for n in names]].to_numpy()
    predicted = example[[f"{n}_pred" for n in names]].to_numpy()
    np.testing.assert_allclose(truth, observed, atol=0.00051)
    np.testing.assert_allclose(smooth(raw, example.record, bundle.smoothing), predicted, atol=0.00051)
    snapshot = dict(metrics=metrics, time=example.time_s.tolist(), truth=observed.tolist(),
                    prediction=predicted.tolist(), raw=raw.round(4).tolist(),
                    base=[example[f"{n}_baseline"].iat[0] for n in names],
                    tare=tare.round(3).tolist(), rest=rest[:3].tolist(),
                    features=X[:, :3].round(6).tolist(), z=z[:, :3].round(4).tolist())
    for name in ("validation", "per_target", "per_user"):
        table = pd.read_csv(FORCE_REPORT / f"{name}.csv", dtype={"subject": str})
        snapshot[name] = json.loads(table.to_json(orient="records"))
    write_snapshot("force-data.js", "forceData", snapshot, source)
    print(f"已导出指力回放 {len(example)} 个窗口。")


if __name__ == "__main__":
    export()
