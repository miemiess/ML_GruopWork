"""验证回归窗口、均值目标、批次与时间边界。"""

import numpy as np
import pytest

from emginsight.putemg_force import iter_force_windows


## @brief 验证逐批分窗、尾部丢弃、目标均值与原始负力量保留。
# @param make_force_record 测试记录工厂。
# @return None: 窗口起点为 0/2/4，分为两批，来源和时间一致。
def test_values_targets_and_batches(make_force_record):
    record = make_force_record()
    batches = list(iter_force_windows(record, window_size=4, step_size=2, batch_size=2))
    assert [len(batch.targets) for batch in batches] == [2, 1]
    np.testing.assert_array_equal(np.concatenate([b.start_indices for b in batches]), [0, 2, 4])
    np.testing.assert_allclose(np.concatenate([b.targets for b in batches]), [[-2.5, 3.], [-0.5, 7.], [1.5, 11.]])
    first = batches[0]
    assert first.signal.shape == (2, 4, 24)
    np.testing.assert_array_equal(first.signal[0, :, 0], [0, 1, 2, 3])
    np.testing.assert_allclose(first.time[0], record.time[:4])
    assert set(first.subject_ids) == {"03"}
    assert set(first.session_ids) == {"2018-05-11"}
    assert set(first.record_ids) == {record.record_id}
    assert first.target_names == record.target_names
    np.testing.assert_array_equal(record.force[:, 0], np.arange(9) - 4.)


## @brief 验证窗口内部时间断点被丢弃，但断点两侧完整窗口可保留。
# @param make_force_record 测试记录工厂。
# @return None: 起点 2 的跨断点窗口被排除。
def test_time_gap(make_force_record):
    record = make_force_record(n=8)
    record.time[4:] += 1.
    batches = list(iter_force_windows(record, window_size=4, step_size=2))
    np.testing.assert_array_equal(batches[0].start_indices, [0, 4])


## @brief 验证两条不足窗的记录不会被拼接。
# @param make_force_record 测试记录工厂。
# @return None: 每条短记录都没有产出窗口。
def test_short_record_boundary(make_force_record):
    for record in [make_force_record(n=2), make_force_record(n=2)]:
        assert list(iter_force_windows(record, window_size=4)) == []


## @brief 验证校准记录不能作为实验窗口输入。
# @param make_force_record 测试记录工厂。
# @param trial 校准类型。
# @return None: bias 和 mvc 都明确报错。
@pytest.mark.parametrize("trial", ["bias", "mvc"])
def test_calibration_excluded(make_force_record, trial):
    with pytest.raises(ValueError, match="校准"):
        list(iter_force_windows(make_force_record(trial=trial)))


## @brief 验证窗口参数和批大小错误时拒绝执行。
# @param make_force_record 测试记录工厂。
# @param kwargs 非法参数。
# @return None: 参数错误抛出 ValueError。
@pytest.mark.parametrize("kwargs", [{"window_size": 0}, {"step_size": True},
                                    {"batch_size": -1}, {"max_gap_seconds": float("nan")}])
def test_invalid_parameters(make_force_record, kwargs):
    with pytest.raises(ValueError):
        list(iter_force_windows(make_force_record(), **kwargs))
