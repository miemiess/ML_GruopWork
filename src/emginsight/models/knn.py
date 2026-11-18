"""NumPy 实现的欧氏距离 kNN 分类器。"""

from numbers import Integral

import numpy as np

from ._validation import check_features, check_prediction_input, encode_classes


class KNNClassifier:
    """按最近邻多数投票；距离相同按训练行顺序，票数相同取排序靠前的类别。"""

    ## @brief 设置投票邻居数量。
    # @param k 正整数，拟合时不得超过训练样本数。
    # @return None。
    def __init__(self, k=5):
        if isinstance(k, bool) or not isinstance(k, Integral) or k < 1:
            raise ValueError("k 必须为正整数。")
        self.k = k

    ## @brief 保存训练特征和标签，用于预测时找邻居。
    # @param X 训练特征，建议先标准化。
    # @param y 一维类别标签。
    # @return self。
    def fit(self, X, y):
        X = check_features(X)
        if self.k > len(X):
            raise ValueError("k 不能超过训练样本数。")
        self.classes_, self.label_indices_ = encode_classes(y, len(X))
        self.training_ = X.copy()
        self.n_features_in_ = X.shape[1]
        return self

    ## @brief 逐行计算最近邻并投票，保持输入顺序。
    # @param X 与训练列数一致的新特征。
    # @return ndarray 原始类别编码的一维预测。
    def predict(self, X):
        X = check_prediction_input(self, X)
        predictions = np.empty(len(X), dtype=np.intp)
        for row, sample in enumerate(X):
            difference = self.training_ - sample
            squared_distance = np.einsum("ij,ij->i", difference, difference)
            nearest = np.argsort(squared_distance, kind="stable")[:self.k]
            votes = np.bincount(self.label_indices_[nearest], minlength=len(self.classes_))
            predictions[row] = votes.argmax()
        return self.classes_[predictions]
