"""验证按身份配对、主肌选择及末尾修正规则。"""

import numpy as np
import pandas as pd
import pytest

from emginsight.zenodo_fatigue import discover_fatigue_pairs, load_fatigue_record


## @brief 验证主动力肌列名选择、原始数值和全部通道模式。
# @param make_pair 合成配对文件工厂。
# @return None: 默认取主肌，不按固定列索引猜测。
def test_identity_channel_and_values(make_pair):
    emg, label = make_pair()
    record = load_fatigue_record(emg, label)
    assert record.subject_id == "1" and record.trial_id == 1
    assert record.signal.shape == (12, 1)
    np.testing.assert_array_equal(record.signal[:, 0], np.arange(12) * 3.)
    assert record.channel_names == ("R DELTOID ANTERIOR",)
    assert load_fatigue_record(emg, label, channel_mode="all").signal.shape == (12, 2)


## @brief 验证按用户和试验配对且评分缺失可被明确识别。
# @param make_pair 合成配对文件工厂。
# @return None: 不使用文件列表的排序位置配对。
def test_discovery_and_missing_label(make_pair):
    emg, label = make_pair()
    root = emg.parents[2]
    pairs = discover_fatigue_pairs(root)
    assert len(pairs) == 1 and pairs[0].rating_path == label
    label.unlink()
    pair = discover_fatigue_pairs(root)[0]
    assert pair.rating_path is None
    with pytest.raises(FileNotFoundError, match="缺少"):
        load_fatigue_record(pair.emg_path, pair.rating_path)


## @brief 验证肌电与评分身份不一致时拒绝读取。
# @param make_pair 合成配对文件工厂。
# @return None: 不允许跨用户拼接标签。
def test_mismatched_identity(make_pair):
    emg, _ = make_pair("1")
    _, label = make_pair("2")
    with pytest.raises(ValueError, match="身份"):
        load_fatigue_record(emg, label)


## @brief 验证只修正可识别的末尾零填充与同评分倒退尾行，保留修正次数。
# @param make_pair 合成配对文件工厂。
# @return None: 原始文件未修改，严格模式仍会拒绝异常时间。
def test_terminal_repairs_are_audited(make_pair):
    emg, label = make_pair()
    frame = pd.read_csv(emg)
    frame.loc[len(frame)] = 0.
    frame.to_csv(emg, index=False)
    ratings = pd.read_csv(label)
    ratings.loc[len(ratings)] = [.1, 2]
    ratings.to_csv(label, index=False)
    record = load_fatigue_record(emg, label)
    assert len(record.time) == 12
    assert record.repairs == {"emg_zero_tail_rows": 1, "rating_reversed_tail_rows": 1}
    assert len(pd.read_csv(emg)) == 13 and len(pd.read_csv(label)) == 8
    with pytest.raises(ValueError, match="倒退"):
        load_fatigue_record(emg, label, repair_terminal=False)


## @brief 验证只有主肌的末尾时间/信号补零时也能有限修正。
# @param make_pair 合成配对文件工厂。
# @return None: 仅移除补零尾行，其他原始列存在非零值时也不误判为内部时间异常。
def test_partial_channel_zero_tail(make_pair):
    emg, label = make_pair()
    frame = pd.read_csv(emg)
    frame.loc[len(frame)] = [.12, 24., 0., 0.]
    frame.to_csv(emg, index=False)
    record = load_fatigue_record(emg, label)
    assert len(record.time) == 12
    assert record.repairs["emg_zero_tail_rows"] == 1
    assert len(pd.read_csv(emg)) == 13


## @brief 验证中间倒退、不同评分尾行、非法标签或不同步通道不会被自动修复。
# @param make_pair 合成配对文件工厂。
# @param defect 异常类型。
# @return None: 明确报错，不排序或猜测标签。
@pytest.mark.parametrize("defect", ["middle", "tail_changed", "invalid_label", "channel_time"])
def test_rejects_other_anomalies(make_pair, defect):
    emg, label = make_pair()
    ratings = pd.read_csv(label)
    if defect == "middle":
        ratings.loc[3, "time"] = .01
    elif defect == "tail_changed":
        ratings.loc[len(ratings)] = [.1, 1]
    elif defect == "invalid_label":
        ratings.loc[2, "label"] = 3
    else:
        frame = pd.read_csv(emg)
        frame["X [s].1"] += .001
        frame.to_csv(emg, index=False)
    ratings.to_csv(label, index=False)
    with pytest.raises(ValueError):
        load_fatigue_record(emg, label, channel_mode="all")
