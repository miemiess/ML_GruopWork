"""单路特征、NumPy 分类器、模型恢复和无评分推理测试。"""

import numpy as np
import pytest

from emginsight.fatigue.single_channel import (
    FS, WINDOW, NumpyFatigueClassifier, classification_metrics,
    extract_single_channel_features, filter_single_channel, predict_fatigue_signal,
)


## @brief 验证已知 100 Hz 正弦的 RMS 和频率，以及直流偏置去除。
# @return None: 幅度和频率与理论值一致，特征与偏置无关。
def test_known_sine_features():
    t = np.arange(WINDOW) / FS
    signal = 2e-5 * np.sin(2 * np.pi * 100 * t)
    features = extract_single_channel_features(signal[None, :])[0]
    assert np.isclose(features[0], np.log10(2e-5 / np.sqrt(2)), atol=.001)
    assert abs(features[3] - 100) < .3 and abs(features[4] - 100) < .3
    np.testing.assert_allclose(extract_single_channel_features((signal + 10)[None, :])[0], features, atol=1e-6)


## @brief 验证静默窗口不会产生 NaN 或无穷值。
# @return None: 零功率时 MNF/MDF 定义为零。
def test_zero_power_features():
    features = extract_single_channel_features(np.zeros((2, WINDOW)))
    assert np.isfinite(features).all()
    np.testing.assert_array_equal(features[:, 3:], 0)


## @brief 验证固定因果滤波不受后续样本影响。
# @return None: 修改未来信号不能改变先前滤波输出。
def test_filter_is_causal():
    signal = np.sin(np.arange(WINDOW * 2))
    modified = signal.copy()
    modified[WINDOW:] = 1000
    np.testing.assert_allclose(filter_single_channel(signal)[:WINDOW], filter_single_channel(modified)[:WINDOW])


## @brief 验证 NumPy 模型能区分已知三簇，不使用预测数据重新拟合标准化。
# @param kind 模型类型。
# @return None: 样例分类正确，保存和恢复后预测一致。
@pytest.mark.parametrize("kind", ["knn", "ridge"])
def test_models_and_train_only_scaling(kind, tmp_path):
    # 三角形三簇可由线性判别函数区分；不要求线性岭模型拟合一维的中间区间。
    X = np.array([[-10., 0.], [-9., 0.], [0., 10.], [0., 9.], [9., 0.], [10., 0.]])
    y = np.array([0, 0, 1, 1, 2, 2])
    model = NumpyFatigueClassifier(kind, 1 if kind == "knn" else .01, (0, 1)).fit(X, y)
    mean, scale = model.mean.copy(), model.scale.copy()
    np.testing.assert_array_equal(model.predict([[-9.5, 0.], [0., 9.5], [9.5, 0.]]), [0, 1, 2])
    model.predict([[1000., 1000.]])
    np.testing.assert_array_equal(model.mean, mean)
    np.testing.assert_array_equal(model.scale, scale)
    path = tmp_path / "model.npz"
    model.save(path)
    np.testing.assert_array_equal(model.predict(X), NumpyFatigueClassifier.load(path).predict(X))


## @brief 验证固定三类混淆矩阵方向和宏平均 F1 的手算结果。
# @return None: 两个正确预测、一个误判时准确率为 2/3，macro-F1 为 5/9。
def test_metrics_known_values():
    result = classification_metrics([0, 1, 2], [0, 0, 2])
    assert result["confusion_matrix"] == [[1, 0, 0], [1, 0, 0], [0, 0, 1]]
    assert np.isclose(result["accuracy"], 2 / 3)
    assert np.isclose(result["macro_f1"], 5 / 9)


## @brief 验证无需评分文件也能对连续单路信号输出等级，并拒绝采样率不匹配。
# @return None: 一窗产生一个带中文名称的预测，错误采样率不静默接受。
def test_signal_prediction_without_labels():
    time = np.arange(WINDOW) / FS
    signal = np.sin(2 * np.pi * 100 * time) * 2e-5
    feature = extract_single_channel_features(filter_single_channel(signal)[None, :])
    model = NumpyFatigueClassifier("knn", 1).fit(np.vstack([feature] * 3), [0, 1, 2])
    result = predict_fatigue_signal(signal, time, model)
    assert len(result) == 1 and result.predicted_label.iloc[0] == 0
    assert result.predicted_name.iloc[0] == "无疲劳"
    with pytest.raises(ValueError, match="采样率"):
        predict_fatigue_signal(signal, time * 2, model)
