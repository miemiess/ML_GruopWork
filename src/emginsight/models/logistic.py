"""NumPy 实现的二分类及多分类 Softmax 逻辑回归。"""

from numbers import Integral

import numpy as np

from ._validation import check_features, check_prediction_input, encode_classes


## @brief 把各类别分数转换为概率。
# @param scores 各样本的类别分数。
# @return ndarray 按行归一化的概率矩阵。
def _softmax(scores):
    shifted = scores - scores.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


class LogisticRegression:
    """使用全批加速梯度下降最小化交叉熵，L2 正则化仅作用于系数。"""

    ## @brief 设置正则化与迭代条件。
    # @param l2 非负 L2 正则化系数。
    # @param max_iter 最大迭代次数。
    # @param tol 梯度范数收敛阈值。
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

        design = np.column_stack((X, np.ones(len(X))))
        weights = np.zeros((design.shape[1], len(classes)))
        lookahead = weights
        rows = np.arange(len(X))
        # Softmax Hessian 的谱范数至多为 1/2。由此选择固定梯度步长。
        smoothness = np.linalg.norm(design, ord=2) ** 2 / (2 * len(design)) + self.l2
        converged = False
        for iteration in range(1, self.max_iter + 1):
            error = _softmax(design @ lookahead)
            error[rows, indices] -= 1
            gradient = design.T @ error / len(X)
            gradient[:-1] += self.l2 * lookahead[:-1]
            if np.linalg.norm(gradient) <= self.tol:
                weights = lookahead
                converged = True
                break
            updated = lookahead - gradient / smoothness
            # Nesterov 动量加快小正则化系数下的收敛。
            lookahead = updated + (iteration - 1) / (iteration + 2) * (updated - weights)
            weights = updated

        self.classes_ = classes
        self.coef_ = weights[:-1].T
        self.intercept_ = weights[-1]
        self.n_features_in_ = X.shape[1]
        self.converged_ = converged
        self.n_iter_ = iteration
        return self

    ## @brief 按训练时的类别顺序输出预测概率。
    # @param X 新数据特征矩阵。
    # @return ndarray 形状为 (样本数, 类别数)，列顺序为 classes_。
    def predict_proba(self, X):
        X = check_prediction_input(self, X)
        return _softmax(X @ self.coef_.T + self.intercept_)

    ## @brief 选取概率最高的类别。
    # @param X 新数据特征矩阵。
    # @return ndarray 原始类别编码的一维预测。
    def predict(self, X):
        probabilities = self.predict_proba(X)
        return self.classes_[probabilities.argmax(axis=1)]
