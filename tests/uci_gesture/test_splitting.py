"""用户划分测试，不依赖真实数据集。"""

import numpy as np
import pytest

from emginsight.uci_gesture import GestureRecord
from emginsight.uci_gesture import split_records_by_subject, split_subject_ids


## @brief 验证 36 名用户全部分配且三组无交集，重复编号不增加用户数。
# @return None: 默认比例通过最大余数法得到 22、7、7 名用户。
def test_disjoint_complete_and_counts():
    users = [f"{i:02d}" for i in range(1, 37)]
    groups = split_subject_ids(users + users)
    train, validation, test = (set(groups[key]) for key in groups)
    assert [len(group) for group in groups.values()] == [22, 7, 7]
    assert not (train & validation or train & test or validation & test)
    assert train | validation | test == set(users)


## @brief 验证输入顺序不影响固定种子的结果，改变种子可改变划分。
# @return None: 相同种子划分一致，不同种子在本样例中划分不同。
def test_reproducible_and_independent_of_input_order():
    users = [str(i) for i in range(20)]
    assert split_subject_ids(users) == split_subject_ids(reversed(users))
    assert split_subject_ids(users, seed=42) != split_subject_ids(users, seed=43)


## @brief 验证自定义比例和小规模用户的取整结果。
# @return None: 自定义比例按用户计数，三名用户也能各分配一人。
def test_custom_ratios_and_small_population():
    groups = split_subject_ids(map(str, range(10)), ratios=(0.5, 0.3, 0.2))
    assert [len(group) for group in groups.values()] == [5, 3, 2]
    equal_groups = split_subject_ids(["01", "02", "03"], ratios=(1 / 3, 1 / 3, 1 / 3))
    assert all(len(group) == 1 for group in equal_groups.values())


## @brief 验证非法用户、比例、种子和会产生空集合的配置被拒绝。
# @param users 本用例输入的用户编号。
# @param kwargs 本用例输入的划分参数。
# @return None: 非法配置抛出 ValueError。
@pytest.mark.parametrize("users, kwargs", [
    ([], {}),
    (["01", "02"], {}),
    (["01", "02", ""], {}),
    (["01", "02", 3], {}),
    (["01", "02", "03"], {"ratios": (0.8, 0.1, 0.1)}),
    (["01", "02", "03"], {"ratios": (0.6, 0.4, 0)}),
    (["01", "02", "03"], {"ratios": (0.6, 0.2, 0.3)}),
    (["01", "02", "03"], {"ratios": (float("nan"), 0.2, 0.2)}),
    (["01", "02", "03"], {"ratios": (0.5, 0.5)}),
    (["01", "02", "03"], {"seed": -1}),
])
def test_invalid_configuration(users, kwargs):
    with pytest.raises(ValueError):
        split_subject_ids(users, **kwargs)


## @brief 验证同一用户的两条记录始终同组，且原始记录顺序和对象保持不变。
# @return None: 全部记录只分配一次，同一用户不跨组，数组不被修改。
def test_records_stay_with_subject_without_mutation():
    records = [
        GestureRecord(np.full((2, 8), i), np.array([0., 1.]), np.array([0, 1]), str(i), str(j))
        for j in range(2) for i in range(10)
    ]
    groups = split_records_by_subject(iter(records))
    subject_groups = {}
    assigned = []
    for group, items in groups.items():
        positions = []
        for record in items:
            assert subject_groups.setdefault(record.subject_id, group) == group
            positions.append(next(i for i, original in enumerate(records) if original is record))
            np.testing.assert_array_equal(record.signal, np.full((2, 8), int(record.subject_id)))
        assert positions == sorted(positions)
        assigned.extend(positions)
    assert sorted(assigned) == list(range(len(records)))
