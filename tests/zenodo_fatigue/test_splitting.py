"""验证疲劳试验不跨用户组。"""

from emginsight.zenodo_fatigue import split_fatigue_records_by_subject


## @brief 验证 13 名用户全部试验同组，用户数量取整为 8/3/2。
# @param make_record 合成记录工厂。
# @return None: 所有记录分配一次且用户互斥。
def test_all_trials_stay_with_subject(make_record):
    records = [make_record([0, 1, 2], subject=str(user)) for user in range(1, 14) for _ in range(2)]
    groups = split_fatigue_records_by_subject(records)
    sets = [set(record.subject_id for record in items) for items in groups.values()]
    assert list(map(len, sets)) == [8, 3, 2]
    assert sum(map(len, groups.values())) == 26
    assert not any(sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3))
