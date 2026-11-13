"""Zenodo 自感疲劳数据处理的统一调用入口。"""

from .reading import FatigueRecord, FatiguePair, discover_fatigue_pairs, load_fatigue_record
from .alignment import align_fatigue_labels
from .splitting import split_subject_ids, split_fatigue_records_by_subject
from .windowing import FatigueWindows, window_fatigue_record

__all__ = ["FatigueRecord", "FatiguePair", "FatigueWindows", "discover_fatigue_pairs", "load_fatigue_record",
           "align_fatigue_labels", "split_subject_ids", "split_fatigue_records_by_subject", "window_fatigue_record"]
