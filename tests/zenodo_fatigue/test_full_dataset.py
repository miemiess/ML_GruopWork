"""全试验模型的同侧通道顺序与时间因果性。"""

import numpy as np
import pytest

from emginsight.zenodo_fatigue.full_dataset import FS, MUSCLES, STEP, WINDOW, signal_features


def test_channel_order_and_prefix_invariance():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(WINDOW + 3 * STEP, 4))
    t = np.arange(len(x)) / FS
    names = ["R " + m for m in MUSCLES]
    full = signal_features(x, t, names)
    swapped = signal_features(x[:, ::-1], t, ["L " + m for m in MUSCLES[::-1]])
    np.testing.assert_allclose(full, swapped)
    np.testing.assert_allclose(full.iloc[:1], signal_features(x[:WINDOW], t[:WINDOW], names))
    assert full.shape == (4, 40)
    with pytest.raises(ValueError, match="肌肉通道"):
        signal_features(x, t, [names[0]] * 4)

## @brief 验证原始 CSV 的逐窗时间、类别概率及左右混接检查。
def test_predict_record_outputs_and_side_check(tmp_path):
    import pandas as pd
    from sklearn.dummy import DummyClassifier
    from emginsight.zenodo_fatigue.full_dataset import predict_record

    signals = np.random.default_rng(5).normal(size=(WINDOW + STEP, 4))
    times = np.arange(len(signals)) / FS
    names = ["R " + muscle for muscle in MUSCLES]
    features = signal_features(signals, times, names)
    model = DummyClassifier(strategy="prior").fit(
        pd.concat([features, features.iloc[:1]]), [0, 1, 2]
    )
    artifact = {"model": model, "columns": features.columns}
    raw = pd.DataFrame()
    for index, name in enumerate(names):
        raw[f"X [s].{index}"] = times
        raw[f"{name}: EMG {index+1} [V]"] = signals[:, index]
    path = tmp_path / "trial.csv"
    raw.to_csv(path, index=False)
    result = predict_record(path, artifact)
    np.testing.assert_allclose(result.start_seconds, [0, STEP/FS])
    np.testing.assert_allclose(result[["prob_0", "prob_1", "prob_2"]].sum(axis=1), 1)
    raw.rename(columns={raw.columns[1]: raw.columns[1].replace("R ", "L ", 1)}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="同侧"):
        predict_record(path, artifact)
