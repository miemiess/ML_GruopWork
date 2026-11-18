"""使用少量合成窗口演示如何接入共享分类、回归和基线。"""

import numpy as np

from emginsight.features import extract_time_features
from emginsight.models import (
    KNNClassifier,
    LogisticRegression,
    MajorityClassifier,
    MeanRegressor,
    RidgeRegressor,
    Standardizer,
)


## @brief 用已分组的合成窗口演示特征提取、训练、留出预测和基线比较。
# @return None。在终端打印样例结果。
def main():
    train_amplitude = np.array([1.0, 1.2, 1.4, 3.0, 3.2])
    test_amplitude = np.array([1.3, 3.1])
    waveform = np.array([[1, 0], [-1, 1], [1, 0], [-1, -1]])
    X_train = extract_time_features(train_amplitude[:, None, None] * waveform)
    X_test = extract_time_features(test_amplitude[:, None, None] * waveform)
    print(f"合成窗口特征形状，训练 {X_train.shape}，留出 {X_test.shape}")

    scaler = Standardizer().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    labels = np.array([0, 0, 0, 1, 1])
    print("合成分类留出标签", [0, 1])
    for name, model in (
        ("多数类基线", MajorityClassifier()),
        ("kNN", KNNClassifier(k=3)),
        ("逻辑回归", LogisticRegression()),
    ):
        print(name, model.fit(X_train, labels).predict(X_test).tolist())

    targets = np.column_stack((2 * train_amplitude + 1, 6 - train_amplitude))
    held_out_targets = np.column_stack((2 * test_amplitude + 1, 6 - test_amplitude))
    print("合成回归留出目标", held_out_targets.round(2).tolist())
    for name, model in (("均值基线", MeanRegressor()), ("岭回归", RidgeRegressor(alpha=0.1))):
        predictions = model.fit(X_train, targets).predict(X_test)
        print(name, predictions.round(2).tolist())


if __name__ == "__main__":
    main()
