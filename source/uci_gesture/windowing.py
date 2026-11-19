"""按采样点分窗；不重采样，不把点数直接当作毫秒。"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .reading import GestureRecord


@dataclass
class GestureWindows:
    """窗口数据：signal 为 (窗口数, 窗口点数, 8)，time 为对应毫秒时间戳。

    labels、subject_ids、record_ids、start_indices 均为一维且与窗口逐一对应。
    start_indices 指向原始记录中的起点。dropped 统计丢弃原因，尾部不足窗不计入。
    """

    signal: NDArray[np.float64]
    time: NDArray[np.float64]
    labels: NDArray[np.int64]
    subject_ids: NDArray[np.str_]
    record_ids: NDArray[np.str_]
    start_indices: NDArray[np.int64]
    dropped: dict[str, int]


## @brief 对单条记录分窗，仅保留完整且标签统一的已标注窗口。
# @param record 已读取的单条记录；请先按用户划分，再分别调用本函数。
# @param window_size 每个窗口的采样点数，默认 200，不表示精确 200 毫秒。
# @param step_size 窗口起点的步长，默认 100 点，允许窗口重叠。
# @param max_gap_ms 窗口内部允许的最大相邻时间间隔，默认 20 毫秒。
# @return GestureWindows 包含信号、原始时间戳、标签、用户、记录编号和丢弃统计。
# @exception ValueError 形状、数值、时间顺序、标签或参数不合法。
def window_gesture_record(
    record: GestureRecord,
    *,
    window_size: int = 200,
    step_size: int = 100,
    max_gap_ms: float = 20.0,
) -> GestureWindows:
    for name, value in (("window_size", window_size), ("step_size", step_size)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} 必须为正整数。")
    if not np.isfinite(max_gap_ms) or max_gap_ms <= 0:
        raise ValueError("max_gap_ms 必须为有限正数。")
    signal = np.asarray(record.signal)
    time = np.asarray(record.time)
    labels = np.asarray(record.labels)
    n = len(signal)
    if signal.shape != (n, 8) or time.shape != (n,) or labels.shape != (n,):
        raise ValueError("信号必须为 (采样点数, 8)，时间与标签必须为等长的一维数组。")
    if not (np.isfinite(signal).all() and np.isfinite(time).all() and np.isfinite(labels).all()):
        raise ValueError("记录包含非有限数值。")
    if np.any(np.diff(time) < 0):
        raise ValueError("时间戳必须非递减。")
    if np.any((labels < 0) | (labels > 7) | (labels != np.floor(labels))):
        raise ValueError("标签必须为 0 到 7 的整数。")

    starts = []
    dropped = {"time_gap": 0, "unmarked": 0, "mixed_labels": 0}
    for start in range(0, n - window_size + 1, step_size):
        end = start + window_size
        # 每个候选窗只计一种丢弃原因，优先级为断点、未标注、混合标签。
        if np.any(np.diff(time[start:end]) > max_gap_ms):
            dropped["time_gap"] += 1
        elif np.any(labels[start:end] == 0):
            dropped["unmarked"] += 1
        elif np.any(labels[start:end] != labels[start]):
            dropped["mixed_labels"] += 1
        else:
            starts.append(start)

    indices = np.asarray(starts, dtype=np.int64)
    rows = indices[:, None] + np.arange(window_size)
    return GestureWindows(
        signal=signal[rows].astype(np.float64),
        time=time[rows].astype(np.float64),
        labels=labels[indices].astype(np.int64),
        subject_ids=np.full(len(indices), record.subject_id),
        record_ids=np.full(len(indices), record.record_id),
        start_indices=indices,
        dropped=dropped,
    )
