"""单条疲劳记录分窗，保留纯评分窗口，并按原因统计丢弃数量。"""

from dataclasses import dataclass
import numpy as np

from .alignment import align_fatigue_labels
from .reading import FatigueRecord


@dataclass
class FatigueWindows:
    """signal=(W,L,C)，labels=(W,)，time=(W,L)，来源和肌肉名称一并保留。"""

    signal: np.ndarray
    labels: np.ndarray
    time: np.ndarray
    subject_ids: np.ndarray
    trial_ids: np.ndarray
    record_ids: np.ndarray
    start_indices: np.ndarray
    channel_names: tuple[str, ...]
    dropped: dict[str, int]


## @brief 对齐评分后逐记录分窗，保留 0/1/2 纯标签窗口。
# @param record 先按用户划分的单条疲劳记录。
# @param window_size 默认 5036 点，按名义 1259 Hz 约为 4 秒。
# @param step_size 默认 2518 点，约 2 秒步长；不足完整窗口的尾部不填充。
# @param max_label_age 最大评分保持时间，默认 0.04 秒。
# @param max_gap_seconds 窗口内允许的最大肌电时间间隔，默认 0.01 秒。
# @return FatigueWindows 保留信号、评分、用户/试验/记录信息；丢弃无评分、混合评分和时间断点窗口。
def window_fatigue_record(
    record: FatigueRecord, *, window_size: int = 5036, step_size: int = 2518,
    max_label_age: float = 0.04, max_gap_seconds: float = 0.01,
) -> FatigueWindows:
    for name, value in (("window_size", window_size), ("step_size", step_size)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} 必须为正整数。")
    if not np.isfinite(max_gap_seconds) or max_gap_seconds <= 0:
        raise ValueError("max_gap_seconds 必须为有限正数。")
    n = len(record.time)
    if (record.signal.ndim != 2 or record.signal.shape != (n, len(record.channel_names))
            or not record.channel_names or not np.isfinite(record.signal).all()):
        raise ValueError("信号形状、通道名或数值不合法。")
    labels = align_fatigue_labels(record, max_label_age=max_label_age)
    starts = []
    dropped = {"time_gap": 0, "uncovered": 0, "mixed_labels": 0}
    for start in range(0, n - window_size + 1, step_size):
        end = start + window_size
        if np.any(np.diff(record.time[start:end]) > max_gap_seconds):
            dropped["time_gap"] += 1
        elif np.any(labels[start:end] < 0):
            dropped["uncovered"] += 1
        elif np.any(labels[start:end] != labels[start]):
            dropped["mixed_labels"] += 1
        else:
            starts.append(start)
    indices = np.asarray(starts, dtype=np.int64)
    rows = indices[:, None] + np.arange(window_size)
    return FatigueWindows(record.signal[rows], labels[indices], record.time[rows],
                          np.full(len(indices), record.subject_id), np.full(len(indices), record.trial_id),
                          np.full(len(indices), record.record_id), indices, record.channel_names, dropped)
