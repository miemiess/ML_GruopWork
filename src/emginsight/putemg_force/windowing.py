"""逐记录切取固定采样点窗口，使用窗口内实测力量均值作为回归目标。"""

from dataclasses import dataclass
from collections.abc import Iterator

import numpy as np
from numpy.typing import NDArray

from .reading import ForceRecord


@dataclass
class ForceWindows:
    """signal=(W,L,24)，targets=(W,T)，time=(W,L)，其余来源字段每窗一项。

    targets 是同一窗口内每个 FORCE 目标的均值，不是类别或指令轨迹。
    """

    signal: NDArray[np.float64]
    targets: NDArray[np.float64]
    time: NDArray[np.float64]
    subject_ids: NDArray[np.str_]
    session_ids: NDArray[np.str_]
    record_ids: NDArray[np.str_]
    start_indices: NDArray[np.int64]
    target_names: tuple[str, ...]
    force_unit: str


## @brief 分批返回单条力量记录的完整窗口，降低长记录分窗的内存占用。
# @param record 已读取的实验记录；bias/mvc 校准记录禁止直接作为训练窗。
# @param window_size 每窗点数，默认 1024；名义 5120 Hz 时约 200 毫秒。
# @param step_size 起点步长，默认 512 点；不跨记录、不填充不足窗的尾部。
# @param max_gap_seconds 允许的最大相邻时间差，默认 0.01 秒；超阈值窗口丢弃。
# @param batch_size 每批最多窗口数，默认 128，调用方应逐批提取特征。
# @return Iterator[ForceWindows] 每批含肌电、窗口均值力量、时间与用户/会话/记录编号。
# @exception ValueError 参数或记录形状不合法；异常数据不插值、不归零、不裁剪负力量。
def iter_force_windows(
    record: ForceRecord, *, window_size: int = 1024, step_size: int = 512,
    max_gap_seconds: float = 0.01, batch_size: int = 128,
) -> Iterator[ForceWindows]:
    for name, value in (("window_size", window_size), ("step_size", step_size), ("batch_size", batch_size)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} 必须为正整数。")
    if not np.isfinite(max_gap_seconds) or max_gap_seconds <= 0:
        raise ValueError("max_gap_seconds 必须为有限正数。")
    if record.trial_type in {"bias", "mvc"}:
        raise ValueError("bias/mvc 为校准记录，不直接生成训练窗口。")
    n = len(record.time)
    if record.signal.shape != (n, 24) or record.force.shape != (n, len(record.target_names)) or n != record.time.size:
        raise ValueError("信号、时间、力量和目标列名的形状不一致。")
    if record.time.ndim != 1 or not record.target_names:
        raise ValueError("时间必须为一维，力量目标不能为空。")
    if not all(np.isfinite(array).all() for array in (record.signal, record.time, record.force)):
        raise ValueError("记录包含非有限值。")
    delta = np.diff(record.time)
    if np.any(delta <= 0):
        raise ValueError("时间必须严格递增。")
    starts = np.arange(0, max(n - window_size + 1, 0), step_size, dtype=np.int64)
    # 利用断点累计计数判断窗内有无断点，避免逐窗重复扫描时间。
    breaks = np.r_[0, np.cumsum(delta > max_gap_seconds)]
    if len(starts):
        starts = starts[(breaks[starts + window_size - 1] - breaks[starts]) == 0]
    for offset in range(0, len(starts), batch_size):
        selected = starts[offset:offset + batch_size]
        rows = selected[:, None] + np.arange(window_size)
        yield ForceWindows(
            signal=record.signal[rows], targets=record.force[rows].mean(axis=1), time=record.time[rows],
            subject_ids=np.full(len(selected), record.subject_id),
            session_ids=np.full(len(selected), record.session_id), record_ids=np.full(len(selected), record.record_id),
            start_indices=selected, target_names=record.target_names, force_unit=record.force_unit,
        )
