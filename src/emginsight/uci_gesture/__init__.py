"""UCI 手势数据的统一调用入口。"""

from .reading import GestureRecord, load_uci_gesture_record
from .splitting import split_records_by_subject, split_subject_ids
from .windowing import GestureWindows, window_gesture_record

__all__ = [
    "GestureRecord",
    "load_uci_gesture_record",
    "split_subject_ids",
    "split_records_by_subject",
    "GestureWindows",
    "window_gesture_record",
]
