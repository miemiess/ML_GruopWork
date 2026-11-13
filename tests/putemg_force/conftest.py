"""力量数据测试的合成样本，不依赖原始数据。"""

import numpy as np
import pandas as pd
import pytest

from emginsight.putemg_force import ForceRecord


## @brief 提供可自定义样本数、用户和记录类型的记录工厂。
# @return callable 返回不依赖真实数据的 ForceRecord。
@pytest.fixture
def make_force_record():
    ## @brief 构造具有已知通道值和力量值的同步记录。
    # @param n 样本数。
    # @param subject 用户编号。
    # @param trial 记录类型。
    # @return ForceRecord 两个测试目标，原始力量包含负值。
    def make(n=9, subject="03", trial="repeats_long"):
        return ForceRecord(np.repeat(np.arange(n, dtype=float)[:, None], 24, axis=1),
                           np.arange(n) / 5120., np.c_[np.arange(n) - 4., np.arange(n) * 2.],
                           subject, "2018-05-11", f"emg_force-{subject}-{trial}-2018-05-11-11-22-53-467",
                           trial, ("FORCE_1", "FORCE_2"))
    return make


## @brief 提供与实际文件表头一致的 CSV 工厂，额外加入 MVC 和指令列。
# @param tmp_path pytest 临时目录。
# @return callable 返回测试文件路径。
@pytest.fixture
def make_force_csv(tmp_path):
    ## @brief 创建具有全部 24 路肌电和 10 路力量的 CSV。
    # @param subject 用户编号。
    # @param trial 记录类型。
    # @return Path 新建的 CSV 路径。
    def make(subject="03", trial="repeats_long"):
        values = {f"EMG_{i}": np.arange(9) + i for i in range(1, 25)}
        values.update({f"FORCE_{i}": np.arange(9) - i for i in range(1, 11)})
        values.update(FORCE_MVC=np.full(9, 999.), TRAJ_1=np.full(9, 3000.))
        frame = pd.DataFrame(values, index=np.arange(9) / 5120.)
        path = tmp_path / f"emg_force-{subject}-{trial}-2018-05-11-11-22-53-467.csv"
        frame.to_csv(path)
        return path
    return make
