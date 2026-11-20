"""putEMG-Force 各实验共用的数据准备：滤波、静息校准、分窗特征和按用户划分后的特征缓存。

回归、三档、跨日变化和学习曲线都通过 prepare_data 读取同一份缓存；推理时只用 record_features。
"""

from dataclasses import replace

import numpy as np
import pandas as pd

from emginsight.common.signal import bandpass
from emginsight.common.splitting import split_subject_ids
from emginsight.features import extract_time_features

from .reading import load_putemg_force_record
from .targets import TARGETS
from .windowing import iter_force_windows

FS = 5120.0
BAND_HZ = (20.0, 500.0)
CALIBRATION_SECONDS = 4.0
SEED = 42
TRIAL = "repeats_long"
CACHE_VERSION = "v2"
FORCE_UNIT = "putEMG-Force 原始力量数值（数据集未给出物理单位，不是牛顿）"


## @brief 对整条记录的 24 路肌电作固定的因果 20～500 Hz 四阶带通滤波。
# @param signal 原始肌电，形状为 (采样点数, 通道数)。
# @param fs 名义采样率，默认 5120 Hz。
# @return ndarray 同形状滤波信号；参数固定，不从任何用户的数据估计。
def filter_emg(signal, fs=FS):
    return bandpass(signal, fs, BAND_HZ)


## @brief 读取记录开头的指令轨迹，确认校准段内没有要求出力。
# @param path 记录 CSV 路径。
# @param samples 校准段采样点数。
# @return bool 校准段内全部 TRAJ 为零时为 True。
def calibration_is_rest(path, samples):
    traj = pd.read_csv(path, usecols=lambda name: name.startswith("TRAJ_"), nrows=samples)
    return traj.shape[1] > 0 and len(traj) == samples and not traj.to_numpy().any()


## @brief 逐批分窗提取特征，并用记录开头的静息段得到力量零点和静息肌电特征。
# @param record 已读取的实验记录。
# @param window_size 每窗点数。
# @param step_size 窗口步长。
# @param calibration 校准段采样点数；该段窗口只用于校准，不参与训练和评估。
# @return dict 特征 X、原始目标 y、零点 tare、静息特征 rest、窗口起点 starts 和时间 time_s。
def record_features(record, window_size, step_size, calibration):
    record = replace(record, signal=filter_emg(record.signal))
    features, targets, starts = [], [], []
    for batch in iter_force_windows(record, window_size=window_size, step_size=step_size):
        features.append(extract_time_features(batch.signal))
        targets.append(batch.targets)
        starts.append(batch.start_indices)
    X, y, starts = np.concatenate(features), np.concatenate(targets), np.concatenate(starts)
    rest = starts + window_size <= calibration
    keep = starts >= calibration
    if not rest.any() or not keep.any():
        raise ValueError(f"{record.record_id}: 记录过短，无法划出校准段和评估段。")
    return dict(X=X[keep], y=y[keep], starts=starts[keep],
                time_s=record.time[starts[keep] + window_size // 2] - record.time[0],
                tare=np.median(record.force[:calibration], axis=0), rest=X[rest].mean(axis=0))


## @brief 先按用户划分，再逐条读取并缓存每份记录的特征，不同时驻留多份原始记录。
# @param root 已解压的 putEMG-Force 根目录。
# @param cache 特征缓存目录；None 表示不缓存。
# @param window_size 每窗点数。
# @param step_size 窗口步长。
# @return tuple 特征、原始目标、零点、静息特征、窗口索引表、用户分组和跳过记录。
def prepare_data(root, cache=None, window_size=1024, step_size=512):
    paths = sorted(root.rglob(f"emg_force-*-{TRIAL}-*.csv"))
    if not paths:
        raise ValueError(f"找不到 {TRIAL} 力量记录，请检查数据目录 {root}")
    groups = split_subject_ids(sorted({path.name.split("-")[1] for path in paths}), seed=SEED)
    owner = {subject: group for group, subjects in groups.items() for subject in subjects}
    calibration = int(CALIBRATION_SECONDS * FS)
    parts, frames, skipped = [], [], []
    for path in paths:
        cached = cache / f"{CACHE_VERSION}-{window_size}-{step_size}" / f"{path.stem}.npz" if cache else None
        if cached is not None and cached.exists():
            item = dict(np.load(cached))
        else:
            if not calibration_is_rest(path, calibration):
                skipped.append({"record": path.stem, "reason": f"开头 {CALIBRATION_SECONDS} 秒内有出力指令或缺少 TRAJ"})
                continue
            record = load_putemg_force_record(path, targets=TARGETS)
            item = record_features(record, window_size, step_size, calibration)
            del record
            if cached is not None:
                cached.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(cached, **item)
        subject, session = path.name.split("-")[1], "-".join(path.stem.split("-")[3:6])
        n = len(item["y"])
        parts.append((item["X"], item["y"], np.tile(item["tare"], (n, 1)), np.tile(item["rest"], (n, 1))))
        frames.append(pd.DataFrame({"subject": subject, "session": session, "record": path.stem,
                                    "group": owner[subject], "start_sample": item["starts"], "time_s": item["time_s"]}))
        print(f"已处理 {path.name}：{n} 窗", flush=True)
    X, y, tare, rest = (np.concatenate(values) for values in zip(*parts))
    index = pd.concat(frames, ignore_index=True)
    for group in groups:
        if not (index.group == group).any():
            raise ValueError(f"{group} 没有可用窗口。")
    return X, y, tare, rest, index, groups, skipped
