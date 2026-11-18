"""使用 scikit-learn 搜索欧氏距离最近邻。"""

from numbers import Integral

import numpy as np
from sklearn.neighbors import NearestNeighbors

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
        self._neighbors = NearestNeighbors(algorithm="kd_tree", metric="euclidean").fit(self.training_)
        self.n_features_in_ = X.shape[1]
        return self

    ## @brief 查询最近邻并投票，保持输入顺序。
    # @param X 与训练列数一致的新特征。
    # @return ndarray 原始类别编码的一维预测。
    def predict(self, X):
        X = check_prediction_input(self, X)
        if not len(X):
            return self.classes_[:0]

        count = min(self.k + 1, len(self.training_))
        distances, indices = self._neighbors.kneighbors(X, n_neighbors=count)
        nearest = indices[:, :self.k].copy()
        if count > self.k:
            # 多查一位，只有第 k 位遇到同距时才取出所有并列候选。
            for row in np.flatnonzero(distances[:, self.k - 1] == distances[:, self.k]):
                radius = np.nextafter(distances[row, self.k], np.inf)
                candidate_distances, candidates = self._neighbors.radius_neighbors(X[row:row + 1], radius)
                order = np.lexsort((candidates[0], candidate_distances[0]))
                nearest[row] = candidates[0][order[:self.k]]

        predictions = np.empty(len(X), dtype=np.intp)
        for row, neighbors in enumerate(nearest):
            votes = np.bincount(self.label_indices_[neighbors], minlength=len(self.classes_))
            predictions[row] = votes.argmax()
        return self.classes_[predictions]
