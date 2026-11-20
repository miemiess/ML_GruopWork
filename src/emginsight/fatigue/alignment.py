"""按秒时间轴对齐 50 Hz 主观评分，不对离散标签作线性插值。"""

import numpy as np

from .reading import FatigueRecord


## @brief 对每个肌电采样点取最近一次已发生的评分，保留覆盖范围和时效限制。
# @param record 包含两条独立时间轴的疲劳记录。
# @param max_label_age 评分最大保持时间，默认 0.04 秒；长缺口不无限填充。
# @return ndarray (N,) 整数标签；0/1/2 为有效评分，-1 表示未覆盖或评分过期。
def align_fatigue_labels(record: FatigueRecord, *, max_label_age: float = 0.04) -> np.ndarray:
    if not np.isfinite(max_label_age) or max_label_age <= 0:
        raise ValueError("max_label_age 必须为有限正数。")
    if (record.time.ndim != 1 or record.rating_time.ndim != 1 or not len(record.rating_time)
            or record.ratings.shape != record.rating_time.shape):
        raise ValueError("肌电时间和评分时间必须为一维，评分不能为空且形状须一致。")
    if not all(np.isfinite(a).all() for a in (record.time, record.rating_time, record.ratings)):
        raise ValueError("时间或评分包含非有限数值。")
    if np.any(np.diff(record.time) <= 0) or np.any(np.diff(record.rating_time) <= 0):
        raise ValueError("两条时间轴都必须严格递增。")
    if not np.isin(record.ratings, [0, 1, 2]).all():
        raise ValueError("疲劳标签必须为 0/1/2。")
    index = np.searchsorted(record.rating_time, record.time, side="right") - 1
    safe = np.maximum(index, 0)
    valid = ((index >= 0) & (record.time <= record.rating_time[-1])
             & (record.time - record.rating_time[safe] <= max_label_age + 1e-12))
    aligned = np.full(len(record.time), -1, dtype=np.int64)
    aligned[valid] = record.ratings[safe[valid]]
    return aligned
