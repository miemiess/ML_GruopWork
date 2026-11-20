"""训练和推理共用的 putEMG-Force 滤波、静息校准与分窗。"""

from dataclasses import replace

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfilt

from emginsight.features import extract_time_features

from .windowing import iter_force_windows

FS = 5120.0
BAND_HZ = (20.0, 500.0)
CALIBRATION_SECONDS = 4.0


## @brief 对整条记录的 24 路肌电作固定的因果 20～500 Hz 四阶带通滤波。
# @param signal 原始肌电，形状为 (采样点数, 通道数)。
# @param fs 名义采样率，默认 5120 Hz。
# @return ndarray 同形状滤波信号；参数固定，不从任何用户的数据估计。
def filter_emg(signal, fs=FS):
    sos = butter(4, BAND_HZ, fs=fs, btype="bandpass", output="sos")
    return sosfilt(sos, np.asarray(signal, dtype=float), axis=0)


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
