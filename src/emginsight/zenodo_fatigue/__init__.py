"""Zenodo 自感疲劳数据的读取、标签对齐与分窗（数据集专用）。用户划分见 emginsight.common。"""

from .reading import FatiguePair, FatigueRecord, discover_fatigue_pairs, load_fatigue_record
from .alignment import align_fatigue_labels
from .windowing import FatigueWindows, window_fatigue_record

__all__ = ["FatiguePair", "FatigueRecord", "FatigueWindows", "align_fatigue_labels", "discover_fatigue_pairs",
           "load_fatigue_record", "window_fatigue_record"]
