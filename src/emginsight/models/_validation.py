"""共享模型的输入校验。"""

import numpy as np


## @brief 校验并转换二维特征矩阵。
# @param X 特征矩阵。
# @param n_features 预测时需要匹配的特征列数；None 表示训练。
# @return ndarray 有限浮点特征矩阵。
def check_features(X, n_features=None):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or not X.shape[1] or not np.isfinite(X).all():
        raise ValueError("X 必须为至少一列的有限二维特征矩阵。")
    if n_features is None and not len(X):
        raise ValueError("训练数据不能为空。")
    if n_features is not None and X.shape[1] != n_features:
        raise ValueError("预测时的特征列数必须与训练时一致。")
    return X


## @brief 校验分类标签并转成连续的内部索引。
# @param y 一维整数、有限浮点数或字符串标签。
# @param n_samples 训练样本数。
# @return tuple 排序后的原始类别与逐样本类别索引。
def encode_classes(y, n_samples):
    y = np.asarray(y)
    if y.shape != (n_samples,) or y.dtype.kind not in "biufU" or (
        y.dtype.kind == "f" and not np.isfinite(y).all()
    ):
        raise ValueError("分类 y 必须为与 X 等长的一维有限数值或字符串标签。")
    return np.unique(y, return_inverse=True)


## @brief 校验单输出或多输出的回归目标。
# @param y 一维或二维数值目标。
# @param n_samples 训练样本数。
# @return ndarray 与训练样本等长的有限浮点目标。
def check_targets(y, n_samples):
    y = np.asarray(y, dtype=float)
    if (y.ndim not in (1, 2) or len(y) != n_samples or not y.size
            or not np.isfinite(y).all()):
        raise ValueError("回归 y 必须为 (样本数,) 或 (样本数, 输出数) 的有限数组。")
    return y


## @brief 获取训练特征数并在预测前验证输入。
# @param estimator 已拟合模型或变换器。
# @param X 新数据特征矩阵。
# @return ndarray 通过检查的预测数据。
def check_prediction_input(estimator, X):
    if not hasattr(estimator, "n_features_in_"):
        raise ValueError("请先调用 fit。")
    return check_features(X, estimator.n_features_in_)
