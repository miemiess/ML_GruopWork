"""手势识别的“未知”拒识：最大类别概率低于阈值时输出 0（未知），阈值在验证用户上选择。"""

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

UNKNOWN = 0
THRESHOLDS = tuple(np.round(np.arange(0.30, 0.951, 0.05), 2))


## @brief 切出一条记录的全部无断点窗口，并记录每窗的标注状态，不丢弃未标注或混合窗口。
# @param record 已读取的 GestureRecord。
# @param size 窗口点数。
# @param step 步长。
# @param max_gap 窗口内允许的最大相邻时间差（毫秒）。
# @return tuple 窗口 (W, size, 8)、状态 (W,)：统一标签 0–7，混合标签为 −1。
def status_windows(record, size=200, step=100, max_gap=20.0):
    n = len(record.time)
    starts = np.arange(0, max(n - size + 1, 0), step)
    breaks = np.r_[0, np.cumsum(np.diff(record.time) > max_gap)]
    starts = starts[breaks[starts + size - 1] == breaks[starts]] if len(starts) else starts
    rows = starts[:, None] + np.arange(size)
    labels = record.labels[rows]
    status = np.where((labels == labels[:, :1]).all(axis=1), labels[:, 0], -1)
    return record.signal[rows], status


## @brief 按阈值把低置信度窗口改为未知。
# @param proba 各类概率 (W, 类别数)。
# @param classes 类别编码。
# @param threshold 阈值；None 或 0 表示不拒识。
# @return ndarray 预测，未知为 0。
def apply_threshold(proba, classes, threshold):
    pred = np.asarray(classes)[proba.argmax(axis=1)]
    if threshold:
        pred = np.where(proba.max(axis=1) < threshold, UNKNOWN, pred)
    return pred


## @brief 计算拒识效果：七类（含未知）macro-F1、已知窗口覆盖率与准确率、未知窗口拒识率。
# @param status 窗口状态（1–6 为已知，其余为未知）。
# @param pred 含未知的预测。
# @param classes 已知类别编码。
# @return dict 指标。
def rejection_scores(status, pred, classes):
    known = np.isin(status, classes)
    truth = np.where(known, status, UNKNOWN)
    accepted = known & (pred != UNKNOWN)
    return {"macro_f1_7": float(f1_score(truth, pred, labels=[UNKNOWN, *classes], average="macro", zero_division=0)),
            "known_coverage": float(accepted.sum() / max(known.sum(), 1)),
            "known_accuracy_accepted": float(np.mean(pred[accepted] == status[accepted])) if accepted.any() else float("nan"),
            "unknown_rejection": float(np.mean(pred[~known] == UNKNOWN)) if (~known).any() else float("nan"),
            "known_windows": int(known.sum()), "unknown_windows": int((~known).sum())}


## @brief 在验证用户上比较各阈值，按七类 macro-F1 选择。
# @param proba 验证窗口概率。
# @param status 验证窗口状态。
# @param classes 已知类别编码。
# @return tuple 阈值表与获选阈值。
def choose_threshold(proba, status, classes):
    rows = [dict(threshold=t, **rejection_scores(status, apply_threshold(proba, classes, t), classes))
            for t in (0.0, *THRESHOLDS)]
    table = pd.DataFrame(rows)
    return table, float(table.loc[table.macro_f1_7.idxmax(), "threshold"])
