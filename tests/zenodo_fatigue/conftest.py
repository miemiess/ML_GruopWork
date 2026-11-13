"""疲劳数据合成样本，测试无需真实数据。"""

import numpy as np
import pandas as pd
import pytest

from emginsight.zenodo_fatigue import FatigueRecord


## @brief 提供肌电和评分配对 CSV 工厂。
# @param tmp_path 临时根目录。
# @return callable 返回两份 CSV 路径。
@pytest.fixture
def make_pair(tmp_path):
    ## @brief 创建用户试验 1 的两路原始肌电和独立评分。
    # @param subject 用户编号。
    # @return tuple 肌电路径、评分路径。
    def make(subject="1"):
        emg = tmp_path / "sEMG_data" / f"subject_{subject}" / "trial_1.csv"
        label = tmp_path / "self_perceived_fatigue_index" / f"subject_{subject}" / "Trial_1.csv"
        emg.parent.mkdir(parents=True, exist_ok=True)
        label.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"X [s]": np.arange(12) * 0.01, "R BICEPS BRACHII: EMG 1 [V]": np.arange(12) * 2.,
                      "X [s].1": np.arange(12) * 0.01, "R DELTOID ANTERIOR: EMG 2 [V]": np.arange(12) * 3.}).to_csv(emg, index=False)
        pd.DataFrame({"time": [0., .02, .04, .06, .08, .10, .12], "label": [0, 0, 0, 1, 1, 2, 2]}).to_csv(label, index=False)
        return emg, label
    return make


## @brief 提供直接构造疲劳记录的工厂。
# @return callable 用于验证时间对齐及分窗。
@pytest.fixture
def make_record():
    ## @brief 构造逐点评分的单通道记录。
    # @param labels 逐点疲劳等级。
    # @param subject 用户编号。
    # @return FatigueRecord 原始值等于采样点序号，时间间隔 0.01 秒。
    def make(labels, subject="1"):
        n = len(labels)
        return FatigueRecord(np.arange(n, dtype=float)[:, None], np.arange(n) * .01,
                             np.arange(n) * .01, np.asarray(labels), subject, 1,
                             f"subject_{subject}/trial_1", ("R DELTOID ANTERIOR",), "R DELTOID ANTERIOR",
                             {"emg_zero_tail_rows": 0, "rating_reversed_tail_rows": 0})
    return make
