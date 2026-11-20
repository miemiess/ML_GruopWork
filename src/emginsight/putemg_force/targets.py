"""力量实测目标的定义、三档标注及回归指标。"""

import numpy as np
import pandas as pd
from scipy.ndimage import percentile_filter
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

TARGETS = tuple(f"FORCE_{i}" for i in range(1, 11))
THRESHOLDS = (0.125, 0.375)
BASELINE_SECONDS = 10.0
BASELINE_PERCENTILE = 10


## @brief 由实测力量生成三档标签：去除缓慢零点漂移后，取第二强传感器除以 MVC 分档。
# @param force 单条记录逐窗的已调零力量 (窗口数, 10)，行按时间排序。
# @param mvc 该记录的 FORCE_MVC。
# @param step_seconds 相邻窗口的时间间隔。
# @return tuple (档位 0/1/2, 相对力量 force/MVC)。
# @note 基线是居中的 10 秒滚动 10% 分位数，只用于构造真值，模型输入不使用实测力量。
#       取第二强传感器：每组手指至少两个传感器，单个传感器的漂移不会被当成出力。
def level_labels(force, mvc, step_seconds):
    if mvc <= 0:
        raise ValueError("FORCE_MVC 必须为正数。")
    width = max(1, int(round(BASELINE_SECONDS / step_seconds)))
    corrected = force - percentile_filter(force, BASELINE_PERCENTILE, size=(width, 1), mode="nearest")
    relative = np.sort(corrected, axis=1)[:, -2] / mvc
    return np.digitize(relative, THRESHOLDS), relative


## @brief 汇总每个力量目标的误差、决定系数和相关系数。
# @param y_true 实测力量 (窗口数, 目标数)。
# @param y_pred 预测力量，形状同上。
# @return DataFrame 每个目标一行，误差单位与原始力量相同。
def target_metrics(y_true, y_pred):
    rows = []
    for i, name in enumerate(TARGETS):
        truth, pred = y_true[:, i], y_pred[:, i]
        constant = np.ptp(pred) <= 1e-9 * max(1.0, np.abs(pred).max()) or np.ptp(truth) == 0
        rows.append(dict(target=name, mae=mean_absolute_error(truth, pred),
                         rmse=np.sqrt(mean_squared_error(truth, pred)), r2=r2_score(truth, pred),
                         pearson_r=np.nan if constant else np.corrcoef(truth, pred)[0, 1]))
    return pd.DataFrame(rows)
