"""用独立小样本验证格式、元数据与错误处理，无需下载数据集。该文件用于测试load_uci_gesture_record函数"""

import numpy as np
import pytest

from emginsight.uci_gesture import load_uci_gesture_record


HEADER = "time\t" + "\t".join(f"channel{i}" for i in range(1, 9)) + "\tclass\n"


## @brief 在临时用户目录中创建模拟肌电记录。
# @param tmp_path pytest 提供的独立临时目录。
# @param body 文件的数据行文本。
# @param header 文件表头，默认使用标准的十列表头。
# @return pathlib.Path: 创建的模拟数据文件路径。
def make_record(tmp_path, body, header=HEADER):
    folder = tmp_path / "05"
    folder.mkdir()
    path = folder / "2_raw_data_example.txt"
    path.write_text(header + body, encoding="utf-8")
    return path


## @brief 验证读取后保留信号数值、行顺序、重复时间戳及记录信息。
# @param tmp_path pytest 提供的独立临时目录。
# @return None: 通过断言检查信号、时间、标签类型及用户和记录编号。
def test_preserves_values_order_and_metadata(tmp_path):
    path = make_record(tmp_path, "1\t1\t2\t3\t4\t5\t6\t7\t8\t0\n1\t-1\t-2\t-3\t-4\t-5\t-6\t-7\t-8\t7\n")
    record = load_uci_gesture_record(path)
    np.testing.assert_array_equal(record.signal, [range(1, 9), range(-1, -9, -1)])
    np.testing.assert_array_equal(record.time, [1, 1])
    np.testing.assert_array_equal(record.labels, [0, 7])
    assert record.labels.dtype == np.int64
    assert record.subject_id == "05"
    assert record.record_id == "2_raw_data_example"


## @brief 验证单行数据保留样本维度，并采用显式指定的用户编号。
# @param tmp_path pytest 提供的独立临时目录。
# @return None: 通过断言检查信号形状为 (1, 8)，时间和标签形状为 (1,)。
def test_single_row_keeps_sample_dimension_and_explicit_subject(tmp_path):
    path = make_record(tmp_path, "1 0 0 0 0 0 0 0 0 1\n")
    record = load_uci_gesture_record(path, subject_id="participant-05")
    assert record.signal.shape == (1, 8)
    assert record.time.shape == record.labels.shape == (1,)
    assert record.subject_id == "participant-05"


## @brief 验证读取函数拒绝空数据、缺列、非法标签及异常数值或时间。
# @param tmp_path pytest 提供的独立临时目录。
# @param body 参数化测试提供的某一种异常数据文本。
# @return None: 确认读取函数抛出 ValueError；未抛出时测试失败。
@pytest.mark.parametrize("body", [
    "",  # 只有表头
    "1 0 0 0 0 0 0 0 1\n",  # 缺列
    "1 0 0 0 0 0 0 0 0 8\n",  # 超范围标签
    "1 0 0 0 0 0 0 0 0 1.5\n",  # 非整数标签
    "1 nan 0 0 0 0 0 0 0 1\n",  # 非有限数值
    "2 0 0 0 0 0 0 0 0 1\n1 0 0 0 0 0 0 0 0 1\n",  # 时间倒序
])
def test_rejects_invalid_data(tmp_path, body):
    with pytest.raises(ValueError):
        load_uci_gesture_record(make_record(tmp_path, body))


## @brief 验证错误表头会触发包含“表头”提示的异常。
# @param tmp_path pytest 提供的独立临时目录。
# @return None: 确认读取函数抛出 ValueError，并检查异常提示。
def test_rejects_wrong_header(tmp_path):
    with pytest.raises(ValueError, match="表头"):
        load_uci_gesture_record(make_record(tmp_path, "1 0 0 0 0 0 0 0 0 1\n", header="wrong\n"))


## @brief 验证文件不在数字用户目录中时，必须显式提供用户编号。
# @param tmp_path pytest 提供的独立临时目录。
# @return None: 确认无法推断用户编号时抛出带对应提示的 ValueError。
def test_requires_subject_for_relocated_file(tmp_path):
    path = tmp_path / "record.txt"
    path.write_text(HEADER + "1 0 0 0 0 0 0 0 0 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="用户编号"):
        load_uci_gesture_record(path)


## @brief 验证读取不存在的文件时抛出文件未找到异常。
# @param tmp_path pytest 提供的独立临时目录。
# @return None: 确认读取函数抛出 FileNotFoundError。
def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_uci_gesture_record(tmp_path / "missing.txt", subject_id="05")
