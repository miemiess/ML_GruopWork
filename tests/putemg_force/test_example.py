"""验证单用户示例和异常报告。"""

import json

import pytest

from emginsight.putemg_force.example import main


## @brief 验证仅一名用户时完成读取与分窗，但不虚构训练/验证/测试划分。
# @param make_force_csv 测试 CSV 工厂。
# @param capsys 终端输出捕获工具。
# @return None: 默认中文输出，JSON 中明确划分不可用，校准窗口数为零。
def test_one_subject_reports_limitation(make_force_csv, capsys):
    path = make_force_csv()
    make_force_csv(trial="bias")
    args = ["--data-dir", str(path.parent), "--window-size", "4", "--step-size", "2"]
    main(args)
    assert "尚未划分" in capsys.readouterr().out
    main(args + ["--json"])
    report = json.loads(capsys.readouterr().out)
    assert report["subject_groups"] is None
    assert report["subject_overlap"] is None
    assert report["total_windows"] == 3
    assert next(item for item in report["records"] if item["trial_type"] == "bias")["windows"] == 0


## @brief 验证异常记录默认报告后继续，严格模式则停止。
# @param make_force_csv 测试 CSV 工厂。
# @param capsys 终端输出捕获工具。
# @param strict 是否使用严格模式。
# @return None: 正常记录可继续产出窗口，坏文件列入报告或触发异常。
@pytest.mark.parametrize("strict", [False, True])
def test_invalid_record_handling(make_force_csv, capsys, strict):
    good = make_force_csv()
    bad = make_force_csv(trial="repeats_short")
    bad.write_text("bad header\n", encoding="utf-8")
    args = ["--data-dir", str(good.parent), "--window-size", "4", "--step-size", "2", "--json"]
    if strict:
        with pytest.raises(ValueError):
            main(args + ["--strict"])
        assert "异常记录" in capsys.readouterr().err
    else:
        main(args)
        output = capsys.readouterr()
        report = json.loads(output.out)
        assert report["total_windows"] == 3
        assert len(report["rejected_records"]) == 1
        assert "异常记录" in output.err


## @brief 验证示例在三个用户及多种实验记录下自动分组并汇总结果。
# @param make_force_csv 合成 CSV 工厂。
# @param capsys 终端输出捕获工具。
# @return None: 每组一人，记录、窗口总数与分组汇总一致，校准记录仍排除。
def test_three_subjects_group_report(make_force_csv, capsys):
    for user in ("03", "04", "05"):
        path = make_force_csv(subject=user)
        make_force_csv(subject=user, trial="sequential")
        make_force_csv(subject=user, trial="bias")
    args = ["--data-dir", str(path.parent), "--window-size", "4", "--step-size", "2"]
    main(args + ["--json"])
    report = json.loads(capsys.readouterr().out)
    assert report["split_policy"] == "三名用户各一组"
    assert report["subject_groups"] == {"train": ["05"], "validation": ["04"], "test": ["03"]}
    assert report["summary"] == {"users": 3, "experimental_records": 6, "calibration_records": 3, "total_windows": 18}
    assert all(group["windows"] == 6 and group["records"] == 2 for group in report["group_summary"].values())
    assert report["groups_with_windows"] == ["test", "train", "validation"]
    assert report["subject_overlap"] is False
    assert report["trial_coverage_consistent"] is True
    main(args)
    assert "每组仅一名用户" in capsys.readouterr().out


## @brief 验证不同用户的实验类型覆盖不同时明确报告差异。
# @param make_force_csv 合成 CSV 工厂。
# @param capsys 终端输出捕获工具。
# @return None: 不重新分组，JSON 记录覆盖差异，中文输出警示。
def test_trial_coverage_difference(make_force_csv, capsys):
    for user in ("03", "04", "05"):
        path = make_force_csv(subject=user)
    make_force_csv(subject="04", trial="repeats_short")
    args = ["--data-dir", str(path.parent), "--window-size", "4", "--step-size", "2"]
    main(args + ["--json"])
    report = json.loads(capsys.readouterr().out)
    assert report["trial_coverage_consistent"] is False
    assert report["subject_groups"]["test"] == ["03"]
    main(args)
    assert "实验覆盖不一致" in capsys.readouterr().out
