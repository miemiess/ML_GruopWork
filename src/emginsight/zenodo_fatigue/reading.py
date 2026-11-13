"""读取已解压的 Zenodo 疲劳肌电与独立评分，按身份而非排序位置配对。"""

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import pandas as pd


MAIN_MUSCLES = {
    1: "R DELTOID ANTERIOR", 2: "L DELTOID ANTERIOR",
    3: "R DELTOID POSTERIOR", 4: "L DELTOID POSTERIOR",
    5: "R BICEPS BRACHII", 6: "L BICEPS BRACHII",
    7: "R DELTOID MEDIUS", 8: "L DELTOID MEDIUS",
    9: "R DELTOID ANTERIOR", 10: "L DELTOID ANTERIOR",
    11: "R DELTOID POSTERIOR", 12: "L DELTOID POSTERIOR",
}


@dataclass
class FatiguePair:
    """同一用户、同一试验的两条路径；缺少评分时 rating_path 为 None。"""

    emg_path: Path
    rating_path: Path | None
    subject_id: str
    trial_id: int


@dataclass
class FatigueRecord:
    """signal=(N,C)，time=(N,) 单位秒；独立评分为 rating_time/ratings。

    默认选择协议中的主动力肌，C=1；all 模式为文件内四路肌电而非左右臂八路。
    channel_names、main_muscle 保留肌肉身份，repairs 记录所有末尾修正。
    """

    signal: np.ndarray
    time: np.ndarray
    rating_time: np.ndarray
    ratings: np.ndarray
    subject_id: str
    trial_id: int
    record_id: str
    channel_names: tuple[str, ...]
    main_muscle: str
    repairs: dict[str, int]


## @brief 从 trial 文件名和上级 subject 目录提取身份，忽略大小写。
# @param path 试验 CSV 路径。
# @return tuple 用户编号字符串与试验整数编号。
# @exception ValueError 无法识别用户目录或试验不在 1..12。
def _identity(path: Path) -> tuple[str, int]:
    trial = re.fullmatch(r"trial_(\d+)", path.stem, re.I)
    subject = next((match for parent in path.parents
                    if (match := re.fullmatch(r"subject_(\d+)", parent.name, re.I))), None)
    if not trial or not subject or int(trial[1]) not in MAIN_MUSCLES:
        raise ValueError(f"{path}: 需要 subject_用户/trial_1..12.csv 格式。")
    return str(int(subject[1])), int(trial[1])


## @brief 在已解压数据中发现并按用户和试验配对，不读取 MVC 或 ZIP。
# @param root 包含 sEMG_data 和 self_perceived_fatigue_index 的根目录。
# @return list[FatiguePair] 按用户/试验数字排序；缺少评分保留为 None，交由调用方报告。
# @exception ValueError 同一身份存在多个 CSV 时拒绝含糊配对。
def discover_fatigue_pairs(root: str | Path) -> list[FatiguePair]:
    root = Path(root)
    indexes = []
    for folder in ("sEMG_data", "self_perceived_fatigue_index"):
        index = {}
        for path in (root / folder).rglob("*"):
            if not path.is_file() or path.suffix.lower() != ".csv" or not re.fullmatch(r"trial_\d+", path.stem, re.I):
                continue
            key = _identity(path)
            if key in index:
                raise ValueError(f"身份 {key} 有重复文件：{index[key]} 与 {path}。")
            index[key] = path
        indexes.append(index)
    return [FatiguePair(path, indexes[1].get(key), *key)
            for key, path in sorted(indexes[0].items(), key=lambda item: (int(item[0][0]), item[0][1]))]


## @brief 读取配对肌电和评分，按协议选择肌肉，并对已知末尾导出问题作有限修正。
# @param emg_path 已解压的肌电 trial CSV。
# @param rating_path 同用户、同试验的评分 CSV，不能用排序位置猜测配对。
# @param channel_mode agonist 默认取协议主动力肌；all 保留当前文件全部原始肌电列。
# @param repair_terminal 默认移除全零尾行及同评分的倒退尾行，次数记入 repairs；False 则严格拒绝。
# @return FatigueRecord 保留原始伏特信号与秒时间，未滤波、未归一化，评分尚未对齐。
def load_fatigue_record(
    emg_path: str | Path, rating_path: str | Path | None, *,
    channel_mode: str = "agonist", repair_terminal: bool = True,
) -> FatigueRecord:
    emg_path = Path(emg_path)
    subject, trial = _identity(emg_path)
    if rating_path is None:
        raise FileNotFoundError(f"用户 {subject} 试验 {trial} 缺少对应评分 CSV。")
    rating_path = Path(rating_path)
    if _identity(rating_path) != (subject, trial):
        raise ValueError("肌电与评分的用户/试验身份不一致。")
    if channel_mode not in {"agonist", "all"}:
        raise ValueError("channel_mode 必须为 agonist 或 all。")
    frame = pd.read_csv(emg_path, encoding="utf-8-sig", dtype=float)
    if frame.empty or not np.isfinite(frame.to_numpy()).all():
        raise ValueError(f"{emg_path}: 肌电为空或包含非有限数值。")
    repairs = {"emg_zero_tail_rows": 0, "rating_reversed_tail_rows": 0}
    if repair_terminal:
        end = len(frame)
        while end and np.all(frame.iloc[end - 1].to_numpy() == 0):
            end -= 1
        repairs["emg_zero_tail_rows"] = len(frame) - end
        frame = frame.iloc[:end]
    if frame.empty:
        raise ValueError("肌电没有有效行。")
    columns = list(frame.columns)
    raw_columns = [name for name in columns if re.fullmatch(r".+: EMG \d+ \[V\]", name.strip())]
    selected = raw_columns if channel_mode == "all" else [
        name for name in raw_columns if name.split(":")[0].strip() == MAIN_MUSCLES[trial]]
    if not selected or (channel_mode == "agonist" and len(selected) != 1):
        raise ValueError(f"{emg_path}: 找不到唯一主动力肌 {MAIN_MUSCLES[trial]}，或没有原始肌电列。")
    time_columns = []
    for name in selected:
        pos = columns.index(name)
        if pos == 0 or not columns[pos - 1].startswith("X [s]"):
            raise ValueError(f"{name}: 前一列不是秒时间轴。")
        time_columns.append(columns[pos - 1])
    # 少数试验仅某一肌肉的最后一对“时间/信号”被补零，其他列仍有值。
    # 只剔除末尾且此前时间已大于零的该类行，避免把内部异常或真正起点删掉。
    if repair_terminal:
        end = len(frame)
        while end > 1 and any(
            frame[time_name].iloc[end - 1] == 0 and frame[signal_name].iloc[end - 1] == 0
            and frame[time_name].iloc[end - 2] > 0
            for time_name, signal_name in zip(time_columns, selected)
        ):
            end -= 1
        repairs["emg_zero_tail_rows"] += len(frame) - end
        frame = frame.iloc[:end]
    time = frame[time_columns[0]].to_numpy(copy=True)
    if np.any(np.diff(time) <= 0):
        raise ValueError(f"{emg_path}: 肌电时间重复或倒退。")
    if any(not np.allclose(time, frame[name], rtol=0, atol=1e-9) for name in time_columns[1:]):
        raise ValueError("所选肌电通道时间轴不同步，需单独确认，不自动插值。")
    labels = pd.read_csv(rating_path, encoding="utf-8-sig")
    if not {"time", "label"}.issubset(labels.columns) or labels.empty:
        raise ValueError(f"{rating_path}: 需要非空 time、label 列。")
    label_values = labels[["time", "label"]].to_numpy(dtype=float)
    if not np.isfinite(label_values).all() or not np.isin(label_values[:, 1], [0, 1, 2]).all():
        raise ValueError(f"{rating_path}: 评分必须为有限数值，标签只能为 0/1/2。")
    end = len(label_values)
    if repair_terminal:
        while end > 1 and label_values[end - 1, 0] <= label_values[end - 2, 0] and label_values[end - 1, 1] == label_values[end - 2, 1]:
            end -= 1
    repairs["rating_reversed_tail_rows"] = len(label_values) - end
    label_values = label_values[:end]
    if np.any(np.diff(label_values[:, 0]) <= 0):
        raise ValueError(f"{rating_path}: 评分时间存在重复或倒退，不属于可修正的相同评分尾行。")
    if time[-1] < label_values[0, 0] or time[0] > label_values[-1, 0]:
        raise ValueError("肌电与评分时间范围没有交集。")
    return FatigueRecord(frame[selected].to_numpy(copy=True), time, label_values[:, 0].copy(),
                         label_values[:, 1].astype(np.int64), subject, trial,
                         f"subject_{subject}/trial_{trial}", tuple(name.split(":")[0].strip() for name in selected),
                         MAIN_MUSCLES[trial], repairs)
