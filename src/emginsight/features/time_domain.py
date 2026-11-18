"""三种可跨数据集复用的时域肌电特征。"""

import numpy as np


TIME_DOMAIN_FEATURES = ("RMS", "MAV", "WL")


## @brief 对每个窗口和通道计算 RMS、MAV 和波形长度 WL。
# @param windows 形状为 (窗口数, 采样点数, 通道数) 的实数数组。
# @return ndarray 形状为 (窗口数, 通道数 × 3)，按通道排列 RMS、MAV、WL。
# @exception ValueError 输入形状错误或包含 NaN/Inf；至少需要两个采样点和一个通道。
def extract_time_features(windows):
    windows = np.asarray(windows, dtype=float)
    if windows.ndim != 3 or windows.shape[1] < 2 or windows.shape[2] < 1:
        raise ValueError("肌电窗口必须为 (窗口数, 至少两个采样点, 至少一个通道)。")
    if not np.isfinite(windows).all():
        raise ValueError("肌电窗口包含 NaN 或 Inf。")

    rms = np.sqrt(np.mean(windows ** 2, axis=1))
    mav = np.mean(np.abs(windows), axis=1)
    wl = np.sum(np.abs(np.diff(windows, axis=1)), axis=1)
    return np.stack((rms, mav, wl), axis=2).reshape(len(windows), windows.shape[2] * 3)
