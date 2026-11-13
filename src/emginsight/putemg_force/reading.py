"""读取 putEMG-Force 已解压的 CSV，不自动进行校准或单位换算。"""

import csv
from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import pandas as pd
from numpy.typing import NDArray


@dataclass
class ForceRecord:
    """单条记录。signal=(N,24)，force=(N,T)，time=(N,) 为原始秒时间轴。

    target_names 定义目标列顺序；force_unit 未确认时不可将指标标为牛顿。
    session_id 为采集日期，record_id 为完整文件主名。trajectory 不作为预测输入。
    """

    signal: NDArray[np.float64]
    time: NDArray[np.float64]
    force: NDArray[np.float64]
    subject_id: str
    session_id: str
    record_id: str
    trial_type: str
    target_names: tuple[str, ...]
    force_unit: str = "原始数值（物理单位待确认）"


## @brief 读取单条 putEMG-Force CSV，按列名提取同步肌电与实测力量。
# @param path 已解压的 emg_force-用户-类型-采集时间.csv 文件路径。
# @param targets 力量目标列名，默认 FORCE_1 至 FORCE_10；不允许 TRAJ 或 FORCE_MVC。
# @return ForceRecord 包含原始信号、力量、秒时间戳、用户和采集信息。
# @exception ValueError 文件名、表头、数值或时间顺序不合法；不自动删除坏行。
def load_putemg_force_record(
    path: str | Path, *, targets: tuple[str, ...] = tuple(f"FORCE_{i}" for i in range(1, 11))
) -> ForceRecord:
    path = Path(path)
    match = re.fullmatch(
        r"emg_force-(\d{2})-([a-z_]+)-(\d{4}-\d{2}-\d{2})-(\d{2}-\d{2}-\d{2}-\d{3})",
        path.stem,
    )
    if not match:
        raise ValueError(f"{path.name}: 文件名不符合 emg_force-用户-类型-日期-时间 格式。")
    targets = tuple(targets)
    allowed = {f"FORCE_{i}" for i in range(1, 11)}
    if not targets or len(set(targets)) != len(targets) or any(name not in allowed for name in targets):
        raise ValueError("targets 必须是不重复的 FORCE_1..FORCE_10 列名，不能用 TRAJ 或 FORCE_MVC。")
    emg_names = [f"EMG_{i}" for i in range(1, 25)]
    with path.open(encoding="utf-8-sig", newline="") as stream:
        header = next(csv.reader(stream), [])
    if not header or len(set(header)) != len(header):
        raise ValueError(f"{path.name}: 表头为空或存在重复列。")
    # 作者 CSV 将秒时间轴导出为未命名索引列；也接受明确命名的时间列。
    time_names = [name for name in ("", "Unnamed: 0", "Timestamp", "TIMESTAMP", "time") if name in header]
    if len(time_names) != 1:
        raise ValueError(f"{path.name}: 必须存在唯一的秒时间列（可为未命名索引列）。")
    missing = [name for name in emg_names + list(targets) if name not in header]
    if missing:
        raise ValueError(f"{path.name}: 缺少列 {missing}。")
    # 只读取时间、肌电和目标，避免无用轨迹列占用内存或意外进入模型输入。
    positions = [i for i, name in enumerate(header) if name in emg_names + list(targets) + time_names]
    try:
        frame = pd.read_csv(path, encoding="utf-8-sig", usecols=positions, dtype=np.float64)
    except (ValueError, pd.errors.ParserError) as error:
        raise ValueError(f"{path.name}: CSV 数值或行格式错误：{error}") from error
    if frame.empty:
        raise ValueError(f"{path.name}: 没有数据行。")
    values = frame.to_numpy()
    if not np.isfinite(values).all():
        row, column = np.argwhere(~np.isfinite(values))[0]
        raise ValueError(f"{path.name}: 数据行 {row + 1} 的 {frame.columns[column]} 含缺失或非有限值。")
    time_column = "Unnamed: 0" if time_names[0] == "" else time_names[0]
    time = frame[time_column].to_numpy(copy=True)
    if np.any(np.diff(time) <= 0):
        raise ValueError(f"{path.name}: 时间必须严格递增，不接受重复或倒退时间。")
    return ForceRecord(
        signal=frame[emg_names].to_numpy(copy=True), time=time,
        force=frame[list(targets)].to_numpy(copy=True), subject_id=match[1],
        session_id=match[3], record_id=path.stem, trial_type=match[2], target_names=targets,
    )
