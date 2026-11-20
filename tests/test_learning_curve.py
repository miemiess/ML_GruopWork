"""检查学习曲线的用户抽样：人数正确、同一人数内组合不重复、全体只取一次。"""

from emginsight.learning_curve import subsets


def test_subsets_are_distinct_and_sized():
    subjects = [f"{i:02d}" for i in range(1, 11)]
    plans = subsets(subjects, (1, 2, 4, 10), repeats=5)
    by_size = {}
    for size, repeat, users in plans:
        assert len(users) == size == len(set(users))
        assert set(users) <= set(subjects)
        by_size.setdefault(size, []).append(users)
    assert {size: len(v) for size, v in by_size.items()} == {1: 5, 2: 5, 4: 5, 10: 1}
    assert all(len(set(v)) == len(v) for v in by_size.values())
    assert plans == subsets(subjects, (1, 2, 4, 10), repeats=5)
