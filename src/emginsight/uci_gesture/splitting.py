"""按用户划分数据，避免同一用户跨训练、验证和测试集。"""

from collections.abc import Iterable

from emginsight.splitting import split_subject_ids

from .reading import GestureRecord


## @brief 根据用户编号划分已读取的原始记录，同一用户的所有记录进入同一集合。
# @param records load_uci_gesture_record 返回的记录序列；应在分窗前调用。
# @param ratios 训练、验证、测试的用户比例，不保证采样点数或类别数符合该比例。
# @param seed 非负整数随机种子，默认 42。
# @return dict 三组记录列表；保留每组内的输入顺序，不复制或修改原始数组。
def split_records_by_subject(
    records: Iterable[GestureRecord],
    *,
    ratios: tuple[float, float, float] = (0.6, 0.2, 0.2),
    seed: int = 42,
) -> dict[str, list[GestureRecord]]:
    records = list(records)
    groups = split_subject_ids(
        (record.subject_id for record in records), ratios=ratios, seed=seed
    )
    destination = {
        subject: group for group, subjects in groups.items() for subject in subjects
    }
    result: dict[str, list[GestureRecord]] = {group: [] for group in groups}
    for record in records:
        result[destination[record.subject_id]].append(record)
    return result
