"""按用户划分力量记录，保持同一用户的全部会话同组。"""

from collections.abc import Iterable

from emginsight.uci_gesture import split_subject_ids as _split_subject_ids

from .reading import ForceRecord


## @brief 按用户编号分组；仅三名用户时默认各分配一人。
# @param subject_ids 用户编号序列，重复编号只计算一次。
# @param ratios 可选显式比例；省略时三人用三等分，其他人数用 0.6/0.2/0.2。
# @param seed 非负整数随机种子。
# @return dict train、validation、test 的用户编号元组，沿用统一的取整和随机规则。
# @exception ValueError 用户不足或显式比例取整产生空组。
def split_subject_ids(
    subject_ids: Iterable[str], *, ratios: tuple[float, float, float] | None = None, seed: int = 42
) -> dict[str, tuple[str, ...]]:
    subject_ids = list(subject_ids)
    if ratios is None:
        # 在验证编号之前，不对无效值执行去重，交由公共函数报告清晰错误。
        valid_ids = all(isinstance(value, str) and value.strip() for value in subject_ids)
        three_users = valid_ids and len(set(subject_ids)) == 3
        ratios = (1 / 3, 1 / 3, 1 / 3) if three_users else (0.6, 0.2, 0.2)
    return _split_subject_ids(subject_ids, ratios=ratios, seed=seed)


## @brief 按用户划分力量实验记录，默认排除 bias 与 mvc 校准记录。
# @param records 已读取的力量记录序列。
# @param ratios 可选用户比例；省略时三人各一组，其他人数默认 0.6/0.2/0.2。
# @param seed 随机种子；复用已测试的用户划分规则，至少需要三名用户。
# @return dict train、validation、test 三组记录列表，保留输入顺序。
# @exception ValueError 用户不足或比例产生空组；不退化为随机按行划分。
def split_force_records_by_subject(
    records: Iterable[ForceRecord], *, ratios: tuple[float, float, float] | None = None, seed: int = 42
) -> dict[str, list[ForceRecord]]:
    records = [record for record in records if record.trial_type not in {"bias", "mvc"}]
    groups = split_subject_ids((record.subject_id for record in records), ratios=ratios, seed=seed)
    owner = {user: name for name, users in groups.items() for user in users}
    result: dict[str, list[ForceRecord]] = {name: [] for name in groups}
    for record in records:
        result[owner[record.subject_id]].append(record)
    return result
