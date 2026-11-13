"""验证读取接口及实测力量目标选择。"""

import numpy as np
import pandas as pd
import pytest

from emginsight.putemg_force import load_putemg_force_record


## @brief 验证未命名时间索引、通道顺序、力量值和元数据原样保留。
# @param make_force_csv 合成 CSV 工厂。
# @return None: 默认目标为十路实测力量，不含 MVC 与 TRAJ。
def test_values_and_metadata(make_force_csv):
    record = load_putemg_force_record(make_force_csv())
    assert record.signal.shape == (9, 24)
    assert record.force.shape == (9, 10)
    np.testing.assert_array_equal(record.signal[:, 0], np.arange(9) + 1)
    np.testing.assert_array_equal(record.signal[:, 23], np.arange(9) + 24)
    np.testing.assert_array_equal(record.force[:, 0], np.arange(9) - 1)
    np.testing.assert_allclose(record.time, np.arange(9) / 5120.)
    assert record.subject_id == "03" and record.session_id == "2018-05-11"
    assert record.trial_type == "repeats_long"
    assert record.target_names == tuple(f"FORCE_{i}" for i in range(1, 11))
    assert "待确认" in record.force_unit


## @brief 验证目标子集顺序和单行维度。
# @param make_force_csv 合成 CSV 工厂。
# @return None: 目标列严格按调用方顺序返回。
def test_target_order_and_one_row(make_force_csv):
    path = make_force_csv()
    frame = pd.read_csv(path)
    frame.head(1).to_csv(path, index=False)
    record = load_putemg_force_record(path, targets=("FORCE_10", "FORCE_1"))
    assert record.signal.shape == (1, 24)
    np.testing.assert_array_equal(record.force, [[-10., -1.]])


## @brief 验证目标不能选指令、MVC、重复列或空列表。
# @param make_force_csv 合成 CSV 工厂。
# @param targets 非法目标配置。
# @return None: 错误配置明确报错。
@pytest.mark.parametrize("targets", [("TRAJ_1",), ("FORCE_MVC",), (), ("FORCE_1", "FORCE_1")])
def test_invalid_targets(make_force_csv, targets):
    with pytest.raises(ValueError, match="targets"):
        load_putemg_force_record(make_force_csv(), targets=targets)


## @brief 验证缺通道、非有限值、空文件与重复时间戳被拒绝。
# @param make_force_csv 合成 CSV 工厂。
# @param defect 当前异常类型。
# @return None: 不静默删除或插补坏数据。
@pytest.mark.parametrize("defect", ["missing", "nan", "empty", "duplicate_time"])
def test_invalid_record(make_force_csv, defect):
    path = make_force_csv()
    frame = pd.read_csv(path)
    if defect == "missing":
        frame = frame.drop(columns="EMG_24")
    elif defect == "nan":
        frame.loc[2, "FORCE_1"] = np.nan
    elif defect == "empty":
        frame = frame.head(0)
    else:
        frame.iloc[1, 0] = frame.iloc[0, 0]
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_putemg_force_record(path)
