"""无需拟合信号特征的参考基线。"""

import numpy as np

from ._validation import check_features, check_prediction_input, check_targets, encode_classes


class MajorityClassifier:
    """始终预测训练集中出现次数最多的类别；并列时取排序靠前的类别。"""

    ## @brief 统计训练集的多数类。
    # @param X 训练特征，用于检查样本数和特征列数。
    # @param y 一维训练标签。
    # @return self。
    def fit(self, X, y):
        X = check_features(X)
        classes, indices = encode_classes(y, len(X))
        self.majority_ = classes[np.bincount(indices).argmax()]
        self.n_features_in_ = X.shape[1]
        return self

    ## @brief 对每个样本返回训练集多数类。
    # @param X 新数据特征矩阵。
    # @return ndarray 一维分类预测。
    def predict(self, X):
        X = check_prediction_input(self, X)
        return np.repeat(self.majority_, len(X))


class MeanRegressor:
    """始终预测训练集目标的均值，支持同时预测多个力量目标。"""

    ## @brief 保存每个训练目标的均值。
    # @param X 训练特征，用于检查样本数和特征列数。
    # @param y 一维或二维回归目标。
    # @return self。
    def fit(self, X, y):
        X = check_features(X)
        self.target_mean_ = check_targets(y, len(X)).mean(axis=0)
        self.n_features_in_ = X.shape[1]
        return self

    ## @brief 对每行输出训练集均值。
    # @param X 新数据特征矩阵。
    # @return ndarray 保留训练目标为一维或二维的约定。
    def predict(self, X):
        X = check_prediction_input(self, X)
        return np.repeat(np.asarray(self.target_mean_)[None, ...], len(X), axis=0)
