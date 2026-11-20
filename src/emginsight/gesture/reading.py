"""读取 UCI EMG Data for Gestures 的单条原始记录。"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


@dataclass
class GestureRecord:
    """一份文件的原始数据；数组按原始采样顺序排列。

    signal: (n_samples, 8)，保留原始信号数值，不假设电压单位。
    time: (n_samples,)，单位为毫秒，保留原始时间戳。
    labels: (n_samples,)，整数 0-7
    0 - 未标记数据，
    1 - 手部静止，
    2 - 手握拳，
    3 - 腕关节屈曲，
    4 - 腕关节伸展，
    5 - 桡偏，
    6 - 尺偏，
    7 - 手掌伸展（并非所有受试者都执行了该手势）
    subject_id: 用户目录名，例如 "05"。
    record_id: 文件主名，不含 .txt；与 subject_id 组合标识记录。
    """

    signal: NDArray[np.float64]
    time: NDArray[np.float64]
    labels: NDArray[np.int64]
    subject_id: str
    record_id: str

def load_uci_gesture_record(
    path: str | Path, *, subject_id: str | None = None
) -> GestureRecord:
    """读取一个带 time/channel1..8/class 表头的原始文本文件。

    默认从父目录推断用户编号，要求目录名为数字；移动过的文件可通过
    subject_id 显式指定用户。重复时间戳允许，倒序时间戳报错。
    不过滤标签、不分窗、不标准化，也不根据时间戳猜测采样率。
    文件不存在时抛出 FileNotFoundError；格式错误时抛出 ValueError。
    """
    path = Path(path)
    if subject_id is None:
        subject_id = path.parent.name
        if not subject_id.isdigit():
            raise ValueError("无法从父目录推断用户编号，请显式传入 subject_id。")
    if not isinstance(subject_id, str) or not subject_id.strip():
        raise ValueError("subject_id 必须为非空字符串。")

    expected = ["time", *(f"channel{i}" for i in range(1, 9)), "class"]
    with path.open(encoding="utf-8-sig") as stream:
        header = stream.readline().split()
        if header != expected:
            raise ValueError(f"{path.name}: 表头应为 {expected}，实际为 {header}。")
        # 忽略空白行，但不接受只有表头或含缺失字段的文件。
        numbered_lines = [(number, line) for number, line in enumerate(stream, start=2) if line.strip()]
        lines = [line for _, line in numbered_lines]
    if not lines:
        raise ValueError(f"{path.name}: 文件没有数据行。")
    try:
        values = np.loadtxt(lines, dtype=np.float64, ndmin=2, comments=None)
    except ValueError as exc:
        # 仅在解析失败时逐行定位，行号包含表头和空行，方便在编辑器中跳转。
        for number, line in numbered_lines:
            fields = line.split()
            if len(fields) != 10:
                raise ValueError(
                    f"{path.name}: 第 {number} 行应有 10 列，实际 {len(fields)} 列。"
                ) from exc
            for column, field in enumerate(fields, start=1):
                try:
                    float(field)
                except ValueError:
                    raise ValueError(
                        f"{path.name}: 第 {number} 行第 {column} 列不是数值：{field!r}。"
                    ) from exc
        raise ValueError(f"{path.name}: 数据行必须包含 10 个数值。") from exc
    if values.shape[1] != 10:
        raise ValueError(f"{path.name}: 第 {numbered_lines[0][0]} 行应有 10 列，实际 {values.shape[1]} 列。")
    if not np.isfinite(values).all():
        row, column = np.argwhere(~np.isfinite(values))[0]
        raise ValueError(f"{path.name}: 第 {numbered_lines[row][0]} 行第 {column + 1} 列包含缺失值或无穷值。")

    time = values[:, 0].copy()
    raw_labels = values[:, 9]
    if np.any(np.diff(time) < 0):
        row = int(np.flatnonzero(np.diff(time) < 0)[0]) + 1
        raise ValueError(f"{path.name}: 第 {numbered_lines[row][0]} 行时间戳倒退，必须按非递减顺序排列。")
    invalid_labels = (raw_labels < 0) | (raw_labels > 7) | (raw_labels != np.floor(raw_labels))
    if np.any(invalid_labels):
        row = int(np.flatnonzero(invalid_labels)[0])
        raise ValueError(f"{path.name}: 第 {numbered_lines[row][0]} 行手势标签必须为 0 到 7 的整数。")

    return GestureRecord(
        signal=values[:, 1:9].copy(),
        time=time,
        labels=raw_labels.astype(np.int64),
        subject_id=subject_id,
        record_id=path.stem,
    )
