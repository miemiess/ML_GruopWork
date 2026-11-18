"""基于 scikit-learn 的二分类及多分类逻辑回归。"""

from numbers import Integral

import numpy as np
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression

from ._validation import check_features, check_prediction_input, encode_classes


class LogisticRegression:
    """用 L-BFGS 拟合逻辑回归，保留按类别排列的系数和预测概率。"""

    ## @brief 设置正则化与迭代条件。
    # @param l2 非负 L2 正则化系数。
    # @param max_iter 最大迭代次数。
    # @param tol 求解器停止阈值。
    # @return None。
    def __init__(self, l2=0.01, max_iter=1000, tol=1e-6):
        if not np.isfinite(l2) or l2 < 0:
            raise ValueError("l2 必须为非负有限数。")
        if isinstance(max_iter, bool) or not isinstance(max_iter, Integral) or max_iter < 1:
            raise ValueError("max_iter 必须为正整数。")
        if not np.isfinite(tol) or tol <= 0:
            raise ValueError("tol 必须为正有限数。")
        self.l2, self.max_iter, self.tol = l2, max_iter, tol

    ## @brief 从训练矩阵和分类标签拟合参数。
    # @param X 训练特征，建议先标准化。
    # @param y 一维标签，至少需要两类。
    # @return self，可通过 converged_ 和 n_iter_ 检查优化状态。
    def fit(self, X, y):
        X = check_features(X)
        classes, indices = encode_classes(y, len(X))
        if len(classes) < 2:
            raise ValueError("逻辑回归至少需要两个类别。")

        # scikit-learn 用 C 对加总损失定标。二分类用单组系数表示两类分数之差。
        binary_scale = 2 if len(classes) == 2 else 1
        C = binary_scale / (len(X) * self.l2) if self.l2 else np.inf
        model = SklearnLogisticRegression(C=C, max_iter=self.max_iter, tol=self.tol, solver="lbfgs")
        model.fit(X, indices)

        self._model = model
        self.classes_ = classes
        self.coef_ = model.coef_
        self.intercept_ = model.intercept_
        if len(classes) == 2:
            self.coef_ = np.concatenate((-model.coef_, model.coef_)) / 2
            self.intercept_ = np.concatenate((-model.intercept_, model.intercept_)) / 2
        self.n_features_in_ = model.n_features_in_
        self.n_iter_ = int(model.n_iter_.max())
        self.converged_ = self.n_iter_ < self.max_iter
        return self

    ## @brief 按训练时的类别顺序输出预测概率。
    # @param X 新数据特征矩阵。
    # @return ndarray 形状为 (样本数, 类别数)，列顺序为 classes_。
    def predict_proba(self, X):
        X = check_prediction_input(self, X)
        if not len(X):
            return np.empty((0, len(self.classes_)))
        return self._model.predict_proba(X)

    ## @brief 选取概率最高的类别。
    # @param X 新数据特征矩阵。
    # @return ndarray 原始类别编码的一维预测。
    def predict(self, X):
        probabilities = self.predict_proba(X)
        return self.classes_[probabilities.argmax(axis=1)]
