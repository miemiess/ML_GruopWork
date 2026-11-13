"""验证中文报告、JSON 汇总及缺失配对处理。"""

import json
import pytest

from emginsight.zenodo_fatigue.example import main


## @brief 验证报告按用户分组，缺失评分默认跳过，严格模式停止。
# @param make_pair 配对 CSV 工厂。
# @param capsys 终端输出捕获工具。
# @return None: 三组均有窗口，0 级不被当作未标注排除。
def test_report_and_missing_pair(make_pair, capsys):
    for user in range(1, 6):
        emg, label = make_pair(str(user))
    args = ["--data-dir", str(emg.parents[2]), "--window-size", "2", "--step-size", "2", "--max-gap-seconds", ".02"]
    main(args + ["--json"])
    report = json.loads(capsys.readouterr().out)
    assert len(report["records"]) == 5
    assert not report["subject_overlap"]
    assert all(group["label_counts"]["0"] > 0 for group in report["groups"].values())
    main(args)
    assert "0、1、2 均为有效类别" in capsys.readouterr().out
    # 从三人训练组删除一份标签，仍有其他训练记录可供输出。
    training_user = report["groups"]["train"]["subjects"][0]
    _, missing_label = make_pair(training_user)
    missing_label.unlink()
    main(args + ["--json"])
    output = capsys.readouterr()
    assert len(json.loads(output.out)["rejected_records"]) == 1
    assert "缺少对应评分" in output.err
    with pytest.raises(FileNotFoundError):
        main(args + ["--strict"])
