"""验证评分窗口与记录边界。"""

import numpy as np
import pytest

from emginsight.zenodo_fatigue import window_fatigue_record


## @brief 验证保留零级评分、丢弃混合等级，且来源、形状和窗口值一致。
# @param make_record 合成记录工厂。
# @return None: 保留纯 0、纯 2 两窗，混合 1/2 窗丢弃。
def test_zero_class_and_mixed_windows(make_record):
    record = make_record([0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 2, 2])
    batch = window_fatigue_record(record, window_size=4, step_size=4, max_gap_seconds=.02)
    assert batch.signal.shape == (2, 4, 1)
    np.testing.assert_array_equal(batch.labels, [0, 2])
    np.testing.assert_array_equal(batch.start_indices, [0, 8])
    np.testing.assert_array_equal(batch.signal[:, :, 0], [[0, 1, 2, 3], [8, 9, 10, 11]])
    assert batch.dropped["mixed_labels"] == 1
    assert set(batch.subject_ids) == {"1"} and set(batch.trial_ids) == {1}
    assert set(batch.record_ids) == {record.record_id}


## @brief 验证覆盖缺失与肌电断点分别计数，优先统计时间断点。
# @param make_record 合成记录工厂。
# @return None: 坏窗口不进入输出。
def test_uncovered_and_time_gap(make_record):
    record = make_record([0] * 8)
    record.time[6:] += .05
    record.rating_time = np.array([.01, .03])
    record.ratings = np.array([0, 0])
    batch = window_fatigue_record(record, window_size=4, step_size=4, max_gap_seconds=.02)
    assert batch.signal.shape == (0, 4, 1)
    assert batch.dropped == {"time_gap": 1, "uncovered": 1, "mixed_labels": 0}


## @brief 验证短记录不跨文件拼接，尾部不足完整窗口不填充。
# @param make_record 合成记录工厂。
# @return None: 两条短记录分别返回空窗，长记录只有完整窗。
def test_record_boundaries_and_tail(make_record):
    for record in [make_record([0, 0]), make_record([0, 0])]:
        batch = window_fatigue_record(record, window_size=4)
        assert batch.signal.shape == (0, 4, 1)
    batch = window_fatigue_record(make_record([1] * 5), window_size=4, step_size=4, max_gap_seconds=.02)
    assert batch.labels.tolist() == [1]


## @brief 验证窗口参数错误时拒绝执行。
# @param make_record 合成记录工厂。
# @param kwargs 非法配置。
# @return None: 抛出 ValueError。
@pytest.mark.parametrize("kwargs", [{"window_size": 0}, {"step_size": True}, {"max_gap_seconds": 0}])
def test_invalid_window_parameters(make_record, kwargs):
    with pytest.raises(ValueError):
        window_fatigue_record(make_record([0] * 8), **kwargs)
