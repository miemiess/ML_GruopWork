"""在力量记录上验证公共的按用户划分：多会话同组、用户不足报错、三人自动三等分。"""

import pytest

from emginsight.common.splitting import split_records_by_subject, split_subject_ids


## @brief 验证同一用户多条记录同组。
# @param make_force_record 测试记录工厂。
# @return None: 三组 6/2/2 人，无记录遗漏或用户交集。
def test_disjoint_groups(make_force_record):
    records = [make_force_record(subject=f"{i:02d}") for i in range(1, 11) for _ in range(2)]
    groups = split_records_by_subject(records)
    users = [set(record.subject_id for record in group) for group in groups.values()]
    assert list(map(len, users)) == [6, 2, 2]
    assert not any(users[i] & users[j] for i in range(3) for j in range(i + 1, 3))
    assert sum(map(len, groups.values())) == 20


## @brief 验证一名用户时不能伪造跨用户评估。
# @param make_force_record 测试记录工厂。
# @return None: 用户不足时明确报错。
def test_insufficient_subjects(make_force_record):
    with pytest.raises(ValueError, match="三名"):
        split_records_by_subject([make_force_record()])


## @brief 验证三个用户默认各分一组，同一用户跨日期与实验类型仍同组。
# @param make_force_record 测试记录工厂。
# @return None: 自动采用三等分，完整保留输入记录并支持种子复现。
def test_three_subjects_multiple_sessions(make_force_record):
    records = []
    for user in ("03", "04", "05"):
        for trial in ("repeats_long", "repeats_short", "sequential"):
            for date in ("2018-03-28", "2018-06-18"):
                record = make_force_record(subject=user, trial=trial)
                record.session_id = date
                records.append(record)
    groups = split_records_by_subject(records)
    assert [len(items) for items in groups.values()] == [6, 6, 6]
    user_sets = [set(item.subject_id for item in items) for items in groups.values()]
    assert all(len(users) == 1 for users in user_sets)
    assert set.union(*user_sets) == {"03", "04", "05"}
    assert not any(user_sets[i] & user_sets[j] for i in range(3) for j in range(i + 1, 3))
    assert split_subject_ids(["03", "04", "05"], seed=42) == split_subject_ids(["05", "04", "03"], seed=42)


## @brief 验证显式传入比例时不偷偷切换为三等分。
# @return None: 三用户明确要求 60%/20%/20% 会因空组而报错，三等分则成功。
def test_explicit_ratios_are_respected():
    with pytest.raises(ValueError, match="空集合"):
        split_subject_ids(["03", "04", "05"], ratios=(0.6, 0.2, 0.2))
    groups = split_subject_ids(["03", "04", "05"], ratios=(1 / 3, 1 / 3, 1 / 3))
    assert all(len(users) == 1 for users in groups.values())
