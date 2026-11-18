"""基于 scikit-learn 的岭回归，支持多个力量输出。"""

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge

from ._validation import check_features, check_prediction_input, check_targets


class RidgeRegressor:
    """最小化平方误差之和加 alpha × 系数平方和，截距不惩罚。"""

    ## @brief 初始化岭回归。
    # @param alpha 非负正则化系数；零表示普通最小二乘。
    # @return None。
    def __init__(self, alpha=1.0):
        if not np.isfinite(alpha) or alpha < 0:
            raise ValueError("alpha 必须为非负有限数。")
        self.alpha = alpha

    ## @brief 用训练数据求解回归系数和截距。
    # @param X 训练特征矩阵。
    # @param y 一维或二维回归目标。
    # @return self。
    def fit(self, X, y):
        X = check_features(X)
        y = check_targets(y, len(X))
        model = Ridge(alpha=self.alpha) if self.alpha > 0 else LinearRegression()
        model.fit(X, y)

        self._model = model
        self._target_shape = y.shape[1:]
        self.n_features_in_ = model.n_features_in_
        self.coef_ = model.coef_.reshape(*self._target_shape, self.n_features_in_).T
        self.intercept_ = np.asarray(model.intercept_).reshape(self._target_shape)[()]
        return self

    ## @brief 返回连续数值预测。
    # @param X 新数据特征矩阵。
    # @return ndarray 形状为 (样本数,) 或 (样本数, 输出数)，与拟合目标一致。
    def predict(self, X):
        X = check_prediction_input(self, X)
        if not len(X):
            return np.empty((0, *self._target_shape))
        return self._model.predict(X).reshape(len(X), *self._target_shape)
