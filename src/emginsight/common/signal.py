"""各数据集共用的信号预处理。"""

import numpy as np
from scipy.signal import butter, sosfilt


## @brief 固定参数的因果 Butterworth 带通滤波，只使用当前及过去的采样点。
# @param signal 采样点在第 0 维的信号，一维或二维（采样点 × 通道）。
# @param fs 采样率（Hz）。
# @param band 通带下限和上限（Hz）。
# @param order 滤波器阶数，默认 4。
# @return ndarray 同形状的滤波信号；参数固定，不从任何用户的数据估计。
def bandpass(signal, fs, band, order=4):
    sos = butter(order, band, fs=fs, btype="bandpass", output="sos")
    return sosfilt(sos, np.asarray(signal, dtype=float), axis=0)
