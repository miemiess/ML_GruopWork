"""从仓库根目录运行：python -m emginsight.uci_gesture.example。"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from emginsight.uci_gesture import (
    load_uci_gesture_record, split_subject_ids, window_gesture_record,
)


## @brief 将处理报告以中文形式打印，解释窗口、标签和异常统计。
# @param report 数据划分和分窗的统计报告。
# @return None: 在终端输出中文说明。
def print_chinese_report(report):
    params = report["parameters"]
    print("\n========== UCI 手势数据处理报告 ==========")
    print(f"数据目录：{params['data_dir']}")
    print(f"窗口大小：{params['window_size']} 个采样点；步长：{params['step_size']} 点")
    print(f"时间断点阈值：{params['max_gap_ms']} 毫秒；随机种子：{params['seed']}")
    print("说明：按采样点分窗，窗口点数不等于精确的毫秒时长。")
    print("规则：保留纯手势窗口，丢弃含未标注数据、混合手势或时间断点的窗口。")
    label_names = {"1": "静息", "2": "握拳", "3": "腕屈", "4": "腕伸",
                   "5": "桡偏", "6": "尺偏", "7": "手掌伸展"}
    group_names = {"train": "训练集（用于训练模型）",
                   "validation": "验证集（用于选择参数）", "test": "测试集（用于最终评估）"}
    for name, stat in report["groups"].items():
        print(f"\n【{group_names[name]}】")
        print(f"分配用户：{len(stat['subjects'])} 人；有有效窗口的用户：{len(stat['subjects_with_windows'])} 人")
        print("用户名单：" + "、".join(stat["subjects"]))
        print(f"成功读取：{stat['records']} 份记录；保留窗口：{stat['windows']} 个")
        print(f"信号形状：({stat['windows']}, {params['window_size']}, 8)，依次为窗口数、采样点数、通道数")
        print(f"标签数量：{sum(stat['label_counts'].values())}，与窗口数量一致")
        print("各手势窗口数：" + "；".join(
            f"{key} {label}：{stat['label_counts'].get(key, 0)}"
            for key, label in label_names.items()))
        dropped = stat["dropped"]
        print(f"丢弃候选窗：含未标注 0 的 {dropped['unmarked']} 个；"
              f"混合手势 {dropped['mixed_labels']} 个；时间断点 {dropped['time_gap']} 个")
        print("说明：每个丢弃窗只计一种原因；不足一窗的尾部不计入以上统计。")
        missing = [f"{key} {label}" for key, label in label_names.items()
                   if not stat["label_counts"].get(key, 0)]
        if missing:
            print("本组没有这些类别的窗口：" + "、".join(missing))
    print("\n【完整性检查】")
    print("训练、验证、测试的窗口用户是否重叠：" + ("是" if report["subject_overlap"] else "否"))
    rejected = report["rejected_records"]
    print(f"异常记录：{len(rejected)} 份（跳过文件，同一用户其他正常文件继续使用）")
    for item in rejected:
        print(f"  用户 {item['subject_id']}：{item['path']}\n  原因：{item['reason']}")
    print("\n处理完成：以上仅为数据读取与分窗检查，不是模型准确率或训练结果。")


## @brief 演示真实数据的用户划分与逐记录分窗，输出可核对的统计信息。
# @return None: 默认输出中文报告，--json 输出机器可读报告；异常提示输出至标准错误。
def main():
    parser = argparse.ArgumentParser(description="UCI 用户划分与分窗检查")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/EMG_data_for_gestures-master"))
    parser.add_argument("--window-size", type=int, default=200)
    parser.add_argument("--step-size", type=int, default=100)
    parser.add_argument("--max-gap-ms", type=float, default=20.)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json", action="store_true", help="输出 JSON 报告，默认显示中文报告")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--strict", action="store_true", help="遇到异常记录时立即停止")
    mode.add_argument("--skip-invalid", action="store_true", help="兼容旧命令，默认已跳过异常记录")
    args = parser.parse_args()
    paths = sorted(args.data_dir.glob("[0-9]*/*raw_data*.txt"))
    if not paths:
        parser.error("没有找到用户目录中的 raw_data 文本文件。")
    groups = split_subject_ids([path.parent.name for path in paths], seed=args.seed)
    owner = {user: name for name, users in groups.items() for user in users}
    stats = {
        name: {"subjects": list(users), "subjects_with_windows": [], "records": 0,
               "windows": 0, "label_counts": {},
               "dropped": {"time_gap": 0, "unmarked": 0, "mixed_labels": 0}}
        for name, users in groups.items()
    }
    report = {"parameters": vars(args).copy(), "groups": stats, "rejected_records": []}
    report["parameters"]["data_dir"] = str(args.data_dir)
    for path in paths:
        try:
            record = load_uci_gesture_record(path)
        except (ValueError, OSError, UnicodeError) as error:
            print(f"[异常记录] 用户 {path.parent.name} | {path} | {error}", file=sys.stderr, flush=True)
            if args.strict:
                raise
            report["rejected_records"].append({"subject_id": path.parent.name, "record_id": path.stem,
                                                "path": str(path), "reason": str(error)})
            continue
        batch = window_gesture_record(record, window_size=args.window_size,
                                      step_size=args.step_size, max_gap_ms=args.max_gap_ms)
        count = len(batch.labels)
        assert batch.signal.shape == (count, args.window_size, 8)
        assert batch.time.shape == (count, args.window_size)
        assert len(batch.subject_ids) == len(batch.record_ids) == len(batch.start_indices) == count
        assert np.all(batch.subject_ids == record.subject_id)
        assert np.all(batch.record_ids == record.record_id)
        stat = stats[owner[record.subject_id]]
        stat["records"] += 1
        stat["windows"] += count
        if count and record.subject_id not in stat["subjects_with_windows"]:
            stat["subjects_with_windows"].append(record.subject_id)
        values, counts = np.unique(batch.labels, return_counts=True)
        for label, number in zip(values, counts):
            key = str(label)
            stat["label_counts"][key] = stat["label_counts"].get(key, 0) + int(number)
        for reason, number in batch.dropped.items():
            stat["dropped"][reason] += number
        assert sum(stat["label_counts"].values()) == stat["windows"]
    user_sets = [set(stat["subjects_with_windows"]) for stat in stats.values()]
    report["subject_overlap"] = any(user_sets[i] & user_sets[j] for i in range(3) for j in range(i + 1, 3))
    assert not report["subject_overlap"]
    if not all(stat["windows"] for stat in stats.values()):
        raise ValueError("至少一个集合没有可用窗口，请检查数据与分窗参数。")
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print_chinese_report(report)


if __name__ == "__main__":
    main()
