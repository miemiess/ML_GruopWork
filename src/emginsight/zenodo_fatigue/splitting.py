"""复用统一用户分组规则，保持同一用户的所有试验同组。"""

from collections.abc import Iterable

from emginsight.uci_gesture import split_subject_ids

from .reading import FatigueRecord


## @brief 按用户划分疲劳记录，不按试验或窗口随机打散。
# @param records 已读取的记录序列。
# @param ratios 训练/验证/测试的用户比例，默认 0.6/0.2/0.2。
# @param seed 固定随机种子，默认 42。
# @return dict 三组记录，保留组内输入顺序，至少需要三名用户且取整无空组。
def split_fatigue_records_by_subject(
    records: Iterable[FatigueRecord], *, ratios=(0.6, 0.2, 0.2), seed: int = 42,
) -> dict[str, list[FatigueRecord]]:
    records = list(records)
    groups = split_subject_ids((r.subject_id for r in records), ratios=ratios, seed=seed)
    owner = {subject: name for name, subjects in groups.items() for subject in subjects}
    result: dict[str, list[FatigueRecord]] = {name: [] for name in groups}
    for record in records:
        result[owner[record.subject_id]].append(record)
    return result
