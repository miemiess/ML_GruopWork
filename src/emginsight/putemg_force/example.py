"""运行方式：python -m emginsight.putemg_force.example。"""

import argparse
import json
from pathlib import Path
import re
import sys

import numpy as np

from emginsight.putemg_force import load_putemg_force_record, split_subject_ids, iter_force_windows


## @brief 读取本地 CSV 并逐批分窗，中文汇报数据形状、力量范围及划分限制。
# @param argv 可选命令行参数列表；默认从终端获取。
# @return None: 打印报告，不写入或修改原始数据；--json 可输出机器可读报告。
def main(argv=None):
    parser = argparse.ArgumentParser(description="putEMG-Force 数据处理检查")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/EMG_data_for_putemgforce"))
    parser.add_argument("--window-size", type=int, default=1024)
    parser.add_argument("--step-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-gap-seconds", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ratios", type=float, nargs=3, metavar=("TRAIN", "VALIDATION", "TEST"),
                        help="显式用户比例；省略时三名用户各分一组，其他人数用 0.6/0.2/0.2")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--strict", action="store_true", help="遇到异常文件时停止，默认报告后继续")
    args = parser.parse_args(argv)
    if min(args.window_size, args.step_size, args.batch_size) <= 0 or args.seed < 0:
        parser.error("窗口、步长和批大小必须大于零，种子不能为负。")
    if not np.isfinite(args.max_gap_seconds) or args.max_gap_seconds <= 0:
        parser.error("时间断点阈值必须为有限正数。")
    if args.ratios is not None and (
        not np.isfinite(args.ratios).all() or min(args.ratios) <= 0
        or not np.isclose(sum(args.ratios), 1., rtol=0, atol=1e-8)
    ):
        parser.error("三组比例必须为有限正数，且总和为一。")
    paths = sorted(args.data_dir.rglob("emg_force-*.csv"))
    if not paths:
        parser.error("没有找到已解压的 emg_force-*.csv 文件。")
    users = sorted({match[1] for path in paths
                    if (match := re.match(r"emg_force-(\d{2})-(?!bias-|mvc-)", path.name))})
    groups = None
    split_reason = ""
    try:
        groups = split_subject_ids(users, ratios=args.ratios, seed=args.seed)
    except ValueError as error:
        split_reason = str(error)
    owner = {user: group for group, members in (groups or {}).items() for user in members}
    report = {"parameters": {**vars(args), "data_dir": str(args.data_dir)},
              "force_unit": "原始数值（物理单位待确认）", "subject_groups": groups,
              "split_unavailable_reason": split_reason, "records": [], "rejected_records": []}
    report["split_ratios"] = args.ratios or ([1 / 3] * 3 if len(users) == 3 else [0.6, 0.2, 0.2])
    report["split_policy"] = "显式比例" if args.ratios is not None else "三名用户各一组" if len(users) == 3 else "默认 60%/20%/20%"
    for path in paths:
        try:
            record = load_putemg_force_record(path)
            delta = np.diff(record.time)
            item = {"subject_id": record.subject_id, "session_id": record.session_id,
                    "record_id": record.record_id, "trial_type": record.trial_type,
                    "samples": len(record.time), "signal_shape": list(record.signal.shape),
                    "force_shape": list(record.force.shape), "target_names": list(record.target_names),
                    "timestamp_rate_hz": float(1 / np.median(delta)) if len(delta) else None,
                    "force_ranges": {name: [float(record.force[:, i].min()), float(record.force[:, i].max())]
                                     for i, name in enumerate(record.target_names)}}
            if record.trial_type in {"bias", "mvc"}:
                item.update(group="calibration", windows=0, skipped_windows=0)
            else:
                count = 0
                for batch in iter_force_windows(record, window_size=args.window_size, step_size=args.step_size,
                                               batch_size=args.batch_size, max_gap_seconds=args.max_gap_seconds):
                    n = len(batch.targets)
                    assert batch.signal.shape == (n, args.window_size, 24)
                    assert batch.targets.shape == (n, len(record.target_names))
                    assert np.all(batch.subject_ids == record.subject_id)
                    assert np.all(batch.record_ids == record.record_id)
                    count += n
                candidates = max(0, (len(record.time) - args.window_size) // args.step_size + 1)
                item.update(group=owner.get(record.subject_id, "inspection"), windows=count,
                            skipped_windows=candidates - count)
            report["records"].append(item)
            del record
        except (ValueError, OSError, UnicodeError) as error:
            print(f"[异常记录] {path}：{error}", file=sys.stderr, flush=True)
            if args.strict:
                raise
            report["rejected_records"].append({"path": str(path), "reason": str(error)})
    experimental = [item for item in report["records"] if item["group"] != "calibration"]
    if not experimental or not any(item["windows"] for item in experimental):
        raise ValueError("没有可用的实验窗口，请检查数据和参数。")
    report["total_windows"] = sum(item["windows"] for item in experimental)
    report["summary"] = {"users": len({item["subject_id"] for item in experimental}),
                         "experimental_records": len(experimental),
                         "calibration_records": len(report["records"]) - len(experimental),
                         "total_windows": report["total_windows"]}
    report["group_summary"] = {}
    if groups:
        sets = [set(members) for members in groups.values()]
        assert not any(sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3))
        report["subject_overlap"] = False
        report["groups_with_windows"] = sorted({item["group"] for item in experimental if item["windows"]})
        for name, members in groups.items():
            items = [item for item in experimental if item["group"] == name]
            report["group_summary"][name] = {
                "subjects": list(members),
                "subjects_with_windows": sorted({item["subject_id"] for item in items if item["windows"]}),
                "records": len(items), "windows": sum(item["windows"] for item in items),
                "trial_counts": {trial: sum(item["trial_type"] == trial for item in items)
                                 for trial in sorted({item["trial_type"] for item in items})},
            }
        actual_sets = [set(item["subjects_with_windows"]) for item in report["group_summary"].values()]
        assert not any(actual_sets[i] & actual_sets[j] for i in range(3) for j in range(i + 1, 3))
        trial_sets = [set(item["trial_counts"]) for item in report["group_summary"].values()]
        report["trial_coverage_consistent"] = all(trials == trial_sets[0] for trials in trial_sets[1:])
    else:
        report["subject_overlap"] = None
        report["groups_with_windows"] = []
        report["trial_coverage_consistent"] = None
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    print("\n========== putEMG-Force 数据处理报告 ==========")
    print(f"窗口：{args.window_size} 点；步长：{args.step_size} 点；每批最多 {args.batch_size} 窗")
    print("目标：实测 FORCE_1..FORCE_10 的窗口均值；不用 TRAJ 指令作为真值。")
    print(f"力量单位：{report['force_unit']}；保留负值，不自动裁剪、校准或 MVC 归一化。")
    if not groups:
        print(f"[尚未划分] 当前实验用户数 {len(users)}：{split_reason}")
        print("当前只检查读取与分窗，不能声称完成未见用户训练/验证/测试。")
    else:
        print(f"用户划分策略：{report['split_policy']}；比例：{report['split_ratios']}")
        for name, members in groups.items():
            summary = report["group_summary"][name]
            print(f"{dict(train='训练集', validation='验证集', test='测试集')[name]}：{len(members)} 人，{members}；"
                  f"实验记录 {summary['records']} 份，窗口 {summary['windows']} 个")
        print("三组用户是否重叠：否（同一用户的不同日期、不同实验均在同组）")
        if len(users) == 3:
            print("[小样本检查] 每组仅一名用户，适合验证流程，不足以稳定评价泛化能力。")
        if not report["trial_coverage_consistent"]:
            print("[实验覆盖不一致] 各组记录类型不同，请在正式实验前统一类型或明确报告差异。")
        if len(report["groups_with_windows"]) != 3:
            print("[需检查] 部分集合没有有效窗口，不可直接用于三组实验。")
    for item in report["records"]:
        print(f"\n记录：{item['record_id']}\n肌电形状：{item['signal_shape']}；力量形状：{item['force_shape']}")
        if item["timestamp_rate_hz"] is not None:
            print(f"时间戳估计行频率：{item['timestamp_rate_hz']:.2f} Hz（不是独立硬件测量）")
        if item["group"] == "calibration":
            print("校准记录：已读取检查，默认不生成训练窗口。")
        else:
            print(f"保留窗口：{item['windows']}；时间断点丢弃：{item['skipped_windows']}")
            print(f"窗口信号：(窗口数, {args.window_size}, 24)；回归目标：(窗口数, 10)")
        print("各目标原始数值范围：" + "；".join(f"{name}: {bounds}" for name, bounds in item["force_ranges"].items()))
    print(f"\n有效实验窗口共 {report['total_windows']} 个；异常记录 {len(report['rejected_records'])} 份。")
    print("本示例不训练模型，不保存大窗口数组；请逐批提取特征，并仅在训练集拟合标准化参数。")


if __name__ == "__main__":
    main()
