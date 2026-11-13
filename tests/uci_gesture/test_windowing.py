"""通过公开接口检查分窗、标签规则和边界。"""

import numpy as np
import pytest

from emginsight.uci_gesture import (
    GestureRecord, split_records_by_subject, window_gesture_record,
)


## @brief 构造具有已知数值的测试记录。
# @param labels 每个采样点的标签。
# @param subject 用户编号。
# @return GestureRecord 时间为毫秒，每个通道值等于采样点索引。
def make_record(labels, subject="01"):
    n = len(labels)
    return GestureRecord(np.repeat(np.arange(n)[:, None], 8, axis=1),
                         np.arange(n, dtype=float), np.asarray(labels), subject, "record")


## @brief 验证重叠窗口的数值、形状、标签及来源信息，并丢弃不足窗的尾部。
# @return None: 三个窗口起点为 0、2、4，最后一个采样点未进入完整窗口。
def test_window_values_shapes_and_metadata():
    record = make_record([2] * 9)
    batch = window_gesture_record(record, window_size=4, step_size=2)
    assert batch.signal.shape == (3, 4, 8)
    np.testing.assert_array_equal(batch.start_indices, [0, 2, 4])
    np.testing.assert_array_equal(batch.signal[:, :, 0], [[0, 1, 2, 3], [2, 3, 4, 5], [4, 5, 6, 7]])
    np.testing.assert_array_equal(batch.time, batch.signal[:, :, 0])
    np.testing.assert_array_equal(batch.labels, [2, 2, 2])
    assert list(batch.subject_ids) == ["01"] * 3
    assert list(batch.record_ids) == ["record"] * 3
    assert sum(batch.dropped.values()) == 0
    np.testing.assert_array_equal(record.signal[:, 0], np.arange(9))


## @brief 验证未标注与混合标签窗口被丢弃，静息类 1 正常保留。
# @return None: 分别统计两类丢弃原因，保留最后一个纯静息窗口。
def test_unmarked_mixed_and_rest_labels():
    batch = window_gesture_record(make_record([0, 2, 2, 3, 1, 1]), window_size=2, step_size=2)
    np.testing.assert_array_equal(batch.labels, [1])
    np.testing.assert_array_equal(batch.start_indices, [4])
    assert batch.dropped == {"time_gap": 0, "unmarked": 1, "mixed_labels": 1}


## @brief 验证时间断点窗口被排除，重复时间戳与等于阈值的间隔允许。
# @return None: 两个候选窗仅第二个因时间断点被丢弃。
def test_time_discontinuity():
    record = make_record([2] * 6)
    record.time = np.array([0., 0., 20., 21., 100., 101.])
    batch = window_gesture_record(record, window_size=3, step_size=3)
    np.testing.assert_array_equal(batch.start_indices, [0])
    assert batch.dropped["time_gap"] == 1


## @brief 验证记录不足一窗或所有候选窗被丢弃时，返回正确的空数组形状。
# @param labels 测试记录标签。
# @return None: 信号仍保持三维，标签与元数据均为空。
@pytest.mark.parametrize("labels", [[1], [0, 0, 0, 0], []])
def test_empty_output(labels):
    batch = window_gesture_record(make_record(labels), window_size=4)
    assert batch.signal.shape == (0, 4, 8)
    assert batch.time.shape == (0, 4)
    assert batch.labels.shape == batch.subject_ids.shape == batch.record_ids.shape == (0,)


## @brief 验证三组先按用户划分再独立分窗，窗口保持用户互斥且不能跨记录补齐。
# @return None: 每条记录独立产生一个窗口；两条不足窗的记录不会拼接。
def test_split_then_window_preserves_subjects_and_record_boundaries():
    records = [make_record([2] * 5, str(i)) for i in range(10)]
    groups = split_records_by_subject(records)
    seen = []
    for records_in_group in groups.values():
        users = set()
        for record in records_in_group:
            batch = window_gesture_record(record, window_size=4, step_size=4)
            assert len(batch.labels) == 1
            users.update(batch.subject_ids.tolist())
        seen.append(users)
    assert not (seen[0] & seen[1] or seen[0] & seen[2] or seen[1] & seen[2])
    assert set.union(*seen) == {str(i) for i in range(10)}
    for record in [make_record([1, 1]), make_record([1, 1])]:
        assert len(window_gesture_record(record, window_size=4).labels) == 0


## @brief 验证非法窗口参数被拒绝。
# @param kwargs 当前参数化用例的参数。
# @return None: 参数错误时抛出 ValueError。
@pytest.mark.parametrize("kwargs", [
    {"window_size": 0}, {"window_size": True}, {"step_size": -1},
    {"step_size": 1.5}, {"max_gap_ms": 0}, {"max_gap_ms": float("nan")},
])
def test_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        window_gesture_record(make_record([1] * 4), **kwargs)
