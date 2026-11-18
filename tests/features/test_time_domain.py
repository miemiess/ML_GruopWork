"""用可手算的窗口检查共享时域特征。"""

import numpy as np
import pytest

from emginsight.features import TIME_DOMAIN_FEATURES, extract_time_features


## @brief 核对每个通道的计算公式和列顺序，并检查相邻窗口的边界独立。
def test_feature_values_and_channel_order():
    windows = [[[3, -2], [-4, -2], [0, -2]], [[1, 0], [1, 2], [1, 0]]]
    expected = [[np.sqrt(25 / 3), 7 / 3, 11, 2, 2, 0], [1, 1, 0, np.sqrt(4 / 3), 2 / 3, 4]]
    assert TIME_DOMAIN_FEATURES == ("RMS", "MAV", "WL")
    np.testing.assert_allclose(extract_time_features(windows), expected)


## @brief 检查没有有效窗口时仍返回正确的特征列数。
def test_empty_batch_preserves_column_count():
    assert extract_time_features(np.empty((0, 2, 4))).shape == (0, 12)


## @brief 拒绝维度错误、过短窗口、零通道及非有限数据。
@pytest.mark.parametrize(
    "windows",
    [np.ones((4, 2)), np.ones((1, 1, 1)), np.ones((1, 2, 0)), [[[0], [np.nan]]]],
)
def test_rejects_invalid_windows(windows):
    with pytest.raises(ValueError):
        extract_time_features(windows)
