"""验证低频评分对齐，禁止使用未来评分和范围外推。"""

import numpy as np
import pytest

from emginsight.zenodo_fatigue import align_fatigue_labels


## @brief 验证最近一次已发生评分规则，0 是有效类别，覆盖范围外为 -1。
# @param make_record 合成记录工厂。
# @return None: 不把不同采样率数据逐行配对，也不借用未来标签。
def test_causal_alignment_and_bounds(make_record):
    record = make_record([0] * 8)
    record.time = np.array([-.01, 0., .01, .02, .03, .04, .05, .06])
    record.rating_time = np.array([0., .02, .04])
    record.ratings = np.array([0, 1, 2])
    np.testing.assert_array_equal(align_fatigue_labels(record), [-1, 0, 0, 1, 1, 2, -1, -1])


## @brief 验证评分长缺口不能无限沿用旧标签。
# @param make_record 合成记录工厂。
# @return None: 超出保持时间的采样点为 -1。
def test_stale_label_gap(make_record):
    record = make_record([0] * 4)
    record.time = np.array([0., .03, .05, .1])
    record.rating_time = np.array([0., .1])
    record.ratings = np.array([0, 2])
    np.testing.assert_array_equal(align_fatigue_labels(record), [0, 0, -1, 2])


## @brief 验证评分保持参数不合法时明确报错。
# @param make_record 合成记录工厂。
# @param age 非法阈值。
# @return None: 抛出 ValueError。
@pytest.mark.parametrize("age", [0, -1, float("nan")])
def test_invalid_age(make_record, age):
    with pytest.raises(ValueError):
        align_fatigue_labels(make_record([0, 1]), max_label_age=age)
