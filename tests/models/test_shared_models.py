"""验证基于 scikit-learn 的公共模型接口。"""

import numpy as np
import pytest

from emginsight.models import (
    KNNClassifier,
    LogisticRegression,
    MajorityClassifier,
    MeanRegressor,
    RidgeRegressor,
    Standardizer,
)


## @brief 核对训练统计量、常数列以及对留出集的转换。
def test_standardizer_uses_training_statistics():
    train = [[1, 7], [3, 7]]
    scaler = Standardizer().fit(train)
    np.testing.assert_array_equal(scaler.transform(train), [[-1, 0], [1, 0]])
    np.testing.assert_array_equal(scaler.transform([[9, 10]]), [[7, 3]])
    np.testing.assert_array_equal(scaler.mean_, [2, 7])


## @brief 检查最近邻投票、距离并列、票数并列及大偏置下的距离。
def test_knn_votes_and_ties():
    X = [[1e9 - 2], [1e9], [1e9 + 2]]
    y = ["b", "a", "b"]
    one_neighbor = KNNClassifier(k=1).fit(X, y)
    np.testing.assert_array_equal(one_neighbor.predict([[1e9 - 1], [1e9 + 0.25]]), ["b", "a"])
    np.testing.assert_array_equal(KNNClassifier(k=2).fit(X, y).predict([[1e9]]), ["a"])
    np.testing.assert_array_equal(KNNClassifier(k=3).fit(X, y).predict([[1e9]]), ["b"])


## @brief 检查二分类与多分类的预测、概率及原始标签顺序。
@pytest.mark.parametrize("class_count", [2, 3])
def test_logistic_predicts_held_out_clusters(class_count):
    centers = np.array([[-2, -2], [-2, 2], [2, 0]])[:class_count]
    X = (centers[:, None, :] + np.array([[-0.2, 0.1], [0.1, -0.2], [0.3, 0.1]])).reshape(-1, 2)
    labels = np.array([7, 3, 10])[:class_count]
    model = LogisticRegression().fit(X, np.repeat(labels, 3))
    probabilities = model.predict_proba(centers)
    assert model.converged_
    np.testing.assert_array_equal(model.classes_, np.sort(labels))
    assert model.coef_.shape == (class_count, X.shape[1])
    np.testing.assert_allclose(probabilities.sum(axis=1), np.ones(class_count))
    assert np.all(probabilities >= 0)
    np.testing.assert_array_equal(model.predict(centers), labels)
    assert model.predict_proba(np.empty((0, 2))).shape == (0, class_count)
    assert model.predict(np.empty((0, 2))).shape == (0,)


## @brief 检查零正则化仍可拟合，并保留浮点类别编码。
def test_logistic_allows_no_regularization():
    model = LogisticRegression(l2=0).fit([[-2], [-1], [1], [2]], [0.5, 0.5, 2.5, 2.5])
    np.testing.assert_array_equal(model.predict([[-1], [1]]), [0.5, 2.5])


## @brief 验证岭惩罚的解析值和截距，以及一维输出形状。
def test_ridge_penalizes_weight_and_leaves_intercept_free():
    model = RidgeRegressor(alpha=2).fit([[-1], [1]], [3, 7])
    np.testing.assert_allclose(model.coef_, [1])
    np.testing.assert_allclose(model.predict([[0], [2]]), [5, 7])


## @brief 验证零正则下多输出最小二乘及目标列顺序。
def test_ridge_least_squares_preserves_multiple_outputs():
    model = RidgeRegressor(alpha=0).fit([[1], [2], [3]], [[7, -1], [9, -4], [11, -7]])
    np.testing.assert_allclose(model.predict([[0], [4]]), [[5, 2], [13, -10]])


## @brief 验证二维单列目标在普通预测和空批次中均保留该列。
@pytest.mark.parametrize("model", [RidgeRegressor(alpha=0), RidgeRegressor(alpha=1), MeanRegressor()])
def test_regressors_preserve_single_target_column(model):
    model.fit([[0], [1]], [[3], [5]])
    np.testing.assert_allclose(model.predict([[0.5]]), [[4]])
    assert model.predict(np.empty((0, 1))).shape == (0, 1)


## @brief 核对训练多数类、票数并列和单/多输出的训练均值。
def test_baselines_predict_training_statistics():
    train = [[0], [1], [2], [3]]
    test = [[6], [7]]
    np.testing.assert_array_equal(MajorityClassifier().fit(train, [8, 8, 7, 8]).predict(test), [8, 8])
    np.testing.assert_array_equal(MajorityClassifier().fit(train, [8, 8, 7, 7]).predict(test), [7, 7])
    np.testing.assert_allclose(MeanRegressor().fit(train, [1, 2, 3, 4]).predict(test), [2.5, 2.5])
    targets = [[2, 8], [2, 6], [4, 4], [4, 2]]
    np.testing.assert_allclose(MeanRegressor().fit(train, targets).predict(test), [[3, 5], [3, 5]])


## @brief 检查模型预测前须拟合，且预测列数须与训练一致。
@pytest.mark.parametrize("model_type", [KNNClassifier, LogisticRegression, MajorityClassifier, MeanRegressor, RidgeRegressor])
def test_models_validate_prediction_input(model_type):
    model = model_type()
    with pytest.raises(ValueError, match="fit"):
        model.predict([[1]])
    model.fit([[0], [1], [2], [3], [4]], [0, 0, 1, 1, 1])
    with pytest.raises(ValueError, match="列数"):
        model.predict([[0, 1]])


## @brief 检查 X 与 y 的样本数一致，参数和标签形式符合模型要求。
def test_models_validate_training_input():
    with pytest.raises(ValueError, match="k"):
        KNNClassifier(k=2).fit([[0]], [1])
    with pytest.raises(ValueError, match="两个类别"):
        LogisticRegression().fit([[0], [1]], [1, 1])
    with pytest.raises(ValueError, match="等长"):
        MajorityClassifier().fit([[0], [1]], [0])
    with pytest.raises(ValueError, match="回归 y"):
        RidgeRegressor().fit([[0], [1]], [1, np.inf])
