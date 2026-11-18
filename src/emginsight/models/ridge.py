"""用 NumPy 最小二乘求解岭回归，支持多个力量输出。"""

import numpy as np

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
        x_mean, y_mean = X.mean(axis=0), y.mean(axis=0)
        design, targets = X - x_mean, y - y_mean
        if self.alpha > 0:
            design = np.vstack((design, np.sqrt(self.alpha) * np.eye(X.shape[1])))
            targets = np.concatenate((targets, np.zeros((X.shape[1], *y.shape[1:]))))

        self.coef_ = np.linalg.lstsq(design, targets, rcond=None)[0]
        self.intercept_ = y_mean - x_mean @ self.coef_
        self.n_features_in_ = X.shape[1]
        return self

    ## @brief 返回连续数值预测。
    # @param X 新数据特征矩阵。
    # @return ndarray 形状为 (样本数,) 或 (样本数, 输出数)，与拟合目标一致。
    def predict(self, X):
        X = check_prediction_input(self, X)
        return X @ self.coef_ + self.intercept_
