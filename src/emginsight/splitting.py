"""各实验共用的用户分组规则。"""

from collections.abc import Iterable

import numpy as np


## @brief 将去重后的用户编号随机划分为互不重叠的三组。
# @param subject_ids 用户编号序列，必须为非空字符串；重复编号只计算一次。
# @param ratios 训练、验证、测试的用户比例，默认 0.6、0.2、0.2，均须大于零且总和为一。
# @param seed 非负整数随机种子；相同用户集合和种子产生相同划分。
# @return dict 三组用户编号元组，键为 train、validation、test。
# @exception ValueError 用户少于三人、比例不合法或取整后出现空集合。
def split_subject_ids(
    subject_ids: Iterable[str],
    *,
    ratios: tuple[float, float, float] = (0.6, 0.2, 0.2),
    seed: int = 42,
) -> dict[str, tuple[str, ...]]:
    ids = list(subject_ids)
    if any(not isinstance(value, str) or not value.strip() for value in ids):
        raise ValueError("用户编号必须为非空字符串。")
    users = sorted(set(ids))
    if len(users) < 3:
        raise ValueError("至少需要三名不同用户才能划分三个集合。")
    proportions = np.asarray(ratios, dtype=float)
    if (
        proportions.shape != (3,)
        or not np.isfinite(proportions).all()
        or np.any(proportions <= 0)
        or not np.isclose(proportions.sum(), 1.0, rtol=0, atol=1e-8)
    ):
        raise ValueError("ratios 必须包含三个正数，且总和为一。")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed 必须为非负整数。")

    # 最大余数法：先向下取整，再按小数部分从大到小分配剩余名额。
    # 小数相同时按训练、验证、测试的顺序分配，总人数始终保持不变。
    expected = len(users) * proportions / proportions.sum()
    counts = np.floor(expected).astype(int)
    remainder_order = np.argsort(-(expected - counts), kind="stable")
    for index in remainder_order[:len(users) - int(counts.sum())]:
        counts[index] += 1
    if np.any(counts == 0):
        raise ValueError("当前用户数和比例会产生空集合，请增加用户或调整 ratios。")

    shuffled = np.random.default_rng(seed).permutation(users).tolist()
    train_end = int(counts[0])
    validation_end = train_end + int(counts[1])
    return {
        "train": tuple(shuffled[:train_end]),
        "validation": tuple(shuffled[train_end:validation_end]),
        "test": tuple(shuffled[validation_end:]),
    }
