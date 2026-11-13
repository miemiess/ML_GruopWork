"""验证异常记录报告与默认继续处理行为。"""

import json
import sys

import pytest

from emginsight.uci_gesture import load_uci_gesture_record
from emginsight.uci_gesture.example import main


## @brief 创建五名用户的正常记录，以及用户 01 的额外异常记录。
# @param tmp_path pytest 提供的临时目录。
# @return Path 数据集根目录。
@pytest.fixture
def dataset(tmp_path):
    header = "time " + " ".join(f"channel{i}" for i in range(1, 9)) + " class\n"
    for user in range(1, 6):
        folder = tmp_path / f"{user:02d}"
        folder.mkdir()
        rows = "".join(f"{i} 0 0 0 0 0 0 0 0 1\n" for i in range(4))
        (folder / "1_raw_data_good.txt").write_text(header + rows, encoding="utf-8")
    (tmp_path / "01/2_raw_data_bad.txt").write_text(header + "\n1 0 0\n", encoding="utf-8")
    return tmp_path


## @brief 验证默认模式报告坏文件并继续处理，正常用户与记录不被排除。
# @param dataset 临时测试数据集。
# @param monkeypatch pytest 参数替换工具。
# @param capsys pytest 终端输出捕获工具。
# @param output_json 是否使用 JSON 输出模式。
# @return None: 输出合法报告，各组均有窗口，异常包含物理行号。
@pytest.mark.parametrize("output_json", [False, True])
def test_default_reports_and_continues(dataset, monkeypatch, capsys, output_json):
    monkeypatch.setattr(sys, "argv", ["example", "--data-dir", str(dataset),
                                    "--window-size", "2", "--step-size", "2"]
                        + (["--json"] if output_json else []))
    main()
    output = capsys.readouterr()
    assert "01" in output.err and "第 3 行" in output.err
    if not output_json:
        assert "UCI 手势数据处理报告" in output.out
        assert "训练集（用于训练模型）" in output.out
        assert "验证集（用于选择参数）" in output.out
        assert "测试集（用于最终评估）" in output.out
        assert "异常记录：1 份" in output.out
        assert "第 3 行" in output.out
        assert "标签数量：" in output.out
        return
    report = json.loads(output.out)
    assert len(report["rejected_records"]) == 1
    assert report["rejected_records"][0]["subject_id"] == "01"
    assert "第 3 行" in report["rejected_records"][0]["reason"]
    assert sum(group["records"] for group in report["groups"].values()) == 5
    assert sum(group["windows"] for group in report["groups"].values()) == 10
    assert not report["subject_overlap"]


## @brief 验证严格模式仍可在第一条异常记录处停止并报告原因。
# @param dataset 临时测试数据集。
# @param monkeypatch pytest 参数替换工具。
# @param capsys pytest 终端输出捕获工具。
# @return None: 抛出包含物理行号的 ValueError，且已经输出用户信息。
def test_strict_stops(dataset, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["example", "--data-dir", str(dataset),
                                    "--window-size", "2", "--strict"])
    with pytest.raises(ValueError, match="第 3 行"):
        main()
    assert "01" in capsys.readouterr().err


## @brief 验证读取函数准确报告有空行文件中的原始出错行号。
# @param dataset 临时测试数据集。
# @return None: 定位到第 3 行，并说明期望列数和实际列数。
def test_reader_reports_physical_line(dataset):
    with pytest.raises(ValueError, match="第 3 行应有 10 列，实际 3 列"):
        load_uci_gesture_record(dataset / "01/2_raw_data_bad.txt")
