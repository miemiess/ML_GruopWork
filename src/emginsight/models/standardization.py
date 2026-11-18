"""由调用方只在训练集上拟合的特征标准化。"""

import numpy as np

from ._validation import check_features, check_prediction_input


class Standardizer:
    """按列减训练均值、除训练标准差；常数列的缩放因子设为 1。"""

    ## @brief 从训练特征计算均值和标准差。
    # @param X 训练特征矩阵。
    # @return self。
    def fit(self, X):
        X = check_features(X)
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        self.scale_[self.scale_ == 0] = 1
        self.n_features_in_ = X.shape[1]
        return self

    ## @brief 使用训练时的统计量转换特征。
    # @param X 训练、验证或测试特征矩阵。
    # @return ndarray 标准化矩阵，列顺序不变。
    def transform(self, X):
        X = check_prediction_input(self, X)
        return (X - self.mean_) / self.scale_
