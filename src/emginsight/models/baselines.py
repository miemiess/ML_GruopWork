"""基于 scikit-learn 的多数类和均值基线。"""

import numpy as np
from sklearn.dummy import DummyClassifier, DummyRegressor

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
        model = DummyClassifier(strategy="most_frequent").fit(X, indices)
        self._model = model
        self.classes_ = classes
        self.majority_ = classes[model.class_prior_.argmax()]
        self.n_features_in_ = X.shape[1]
        return self

    ## @brief 对每个样本返回训练集多数类。
    # @param X 新数据特征矩阵。
    # @return ndarray 一维分类预测。
    def predict(self, X):
        X = check_prediction_input(self, X)
        if not len(X):
            return self.classes_[:0]
        return self.classes_[self._model.predict(X)]


class MeanRegressor:
    """始终预测训练集目标的均值，支持同时预测多个力量目标。"""

    ## @brief 保存每个训练目标的均值。
    # @param X 训练特征，用于检查样本数和特征列数。
    # @param y 一维或二维回归目标。
    # @return self。
    def fit(self, X, y):
        X = check_features(X)
        y = check_targets(y, len(X))
        model = DummyRegressor(strategy="mean").fit(X, y)
        self._model = model
        self._target_shape = y.shape[1:]
        self.target_mean_ = model.constant_.reshape(self._target_shape)[()]
        self.n_features_in_ = X.shape[1]
        return self

    ## @brief 对每行输出训练集均值。
    # @param X 新数据特征矩阵。
    # @return ndarray 保留训练目标为一维或二维的约定。
    def predict(self, X):
        X = check_prediction_input(self, X)
        if not len(X):
            return np.empty((0, *self._target_shape))
        return self._model.predict(X).reshape(len(X), *self._target_shape)
