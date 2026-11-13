"""从仓库根目录运行：python -m emginsight.zenodo_fatigue.example。"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from emginsight.zenodo_fatigue import (
    discover_fatigue_pairs, load_fatigue_record, align_fatigue_labels,
    split_subject_ids, window_fatigue_record,
)


## @brief 对全部已解压试验执行配对、用户划分、时间对齐和分窗，输出中文或 JSON 报告。
# @param argv 可选命令行参数列表，默认读取终端参数。
# @return None: 输出汇总；原始文件不修改，异常报告后跳过，可用 --strict 改为停止。
def main(argv=None):
    parser = argparse.ArgumentParser(description="Zenodo 疲劳数据处理检查")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/EMG_data_for_ZenodoTired"))
    parser.add_argument("--window-size", type=int, default=5036)
    parser.add_argument("--step-size", type=int, default=2518)
    parser.add_argument("--max-label-age", type=float, default=0.04)
    parser.add_argument("--max-gap-seconds", type=float, default=0.01)
    parser.add_argument("--channel-mode", choices=["agonist", "all"], default="agonist")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-terminal-repair", action="store_true", help="禁用已知尾行修正")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.window_size <= 0 or args.step_size <= 0 or args.seed < 0:
        parser.error("窗口与步长必须为正整数，种子必须非负。")
    if not all(np.isfinite(v) and v > 0 for v in (args.max_label_age, args.max_gap_seconds)):
        parser.error("评分保持和时间断点阈值必须为有限正数。")
    pairs = discover_fatigue_pairs(args.data_dir)
    if not pairs:
        parser.error("没有找到已解压的 subject_用户/trial_编号.csv。")
    groups = split_subject_ids([pair.subject_id for pair in pairs], seed=args.seed)
    owner = {subject: name for name, subjects in groups.items() for subject in subjects}
    stats = {name: {"subjects": list(subjects), "subjects_with_windows": [], "records": 0, "windows": 0,
                    "label_counts": {"0": 0, "1": 0, "2": 0},
                    "dropped": {"time_gap": 0, "uncovered": 0, "mixed_labels": 0}}
             for name, subjects in groups.items()}
    report = {"parameters": {**vars(args), "data_dir": str(args.data_dir)}, "groups": stats,
              "discovered_pairs": len(pairs), "records": [], "rejected_records": []}
    for pair in pairs:
        try:
            record = load_fatigue_record(pair.emg_path, pair.rating_path, channel_mode=args.channel_mode,
                                         repair_terminal=not args.no_terminal_repair)
            aligned = align_fatigue_labels(record, max_label_age=args.max_label_age)
            batch = window_fatigue_record(record, window_size=args.window_size, step_size=args.step_size,
                                          max_label_age=args.max_label_age, max_gap_seconds=args.max_gap_seconds)
            count = len(batch.labels)
            assert batch.signal.shape == (count, args.window_size, len(record.channel_names))
            assert batch.labels.shape == batch.subject_ids.shape == batch.trial_ids.shape == batch.record_ids.shape
            assert np.all(batch.subject_ids == record.subject_id) and np.all(batch.trial_ids == record.trial_id)
            item = {"record_id": record.record_id, "subject_id": record.subject_id, "trial_id": record.trial_id,
                    "group": owner[record.subject_id], "signal_shape": list(record.signal.shape),
                    "channel_names": list(record.channel_names), "main_muscle": record.main_muscle,
                    "time_range": [float(record.time[0]), float(record.time[-1])],
                    "rating_time_range": [float(record.rating_time[0]), float(record.rating_time[-1])],
                    "repairs": record.repairs, "uncovered_samples": int(np.count_nonzero(aligned < 0)),
                    "windows": count, "label_counts": {str(i): int(np.count_nonzero(batch.labels == i)) for i in range(3)},
                    "dropped": batch.dropped}
            report["records"].append(item)
            stat = stats[item["group"]]
            stat["records"] += 1
            stat["windows"] += count
            if count and record.subject_id not in stat["subjects_with_windows"]:
                stat["subjects_with_windows"].append(record.subject_id)
            for key, value in item["label_counts"].items():
                stat["label_counts"][key] += value
            for key, value in batch.dropped.items():
                stat["dropped"][key] += value
            del batch, record, aligned
        except (ValueError, OSError, UnicodeError) as error:
            print(f"[异常记录] 用户 {pair.subject_id} 试验 {pair.trial_id}：{error}", file=sys.stderr, flush=True)
            if args.strict:
                raise
            report["rejected_records"].append({"subject_id": pair.subject_id, "trial_id": pair.trial_id,
                                                "emg_path": str(pair.emg_path), "reason": str(error)})
    report["total_windows"] = sum(stat["windows"] for stat in stats.values())
    report["repaired_records"] = sum(any(item["repairs"].values()) for item in report["records"])
    report["repair_counts"] = {key: sum(item["repairs"][key] for item in report["records"])
                               for key in ("emg_zero_tail_rows", "rating_reversed_tail_rows")}
    sets = [set(stat["subjects_with_windows"]) for stat in stats.values()]
    report["subject_overlap"] = any(sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3))
    assert not report["subject_overlap"]
    assert all(sum(stat["label_counts"].values()) == stat["windows"] for stat in stats.values())
    if not all(stat["windows"] for stat in stats.values()):
        raise ValueError("至少一组没有有效窗口，请检查数据、用户划分和参数。")
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    print("\n========== Zenodo 自感疲劳数据处理报告 ==========")
    print(f"通道选择：{'协议主动力肌（单通道）' if args.channel_mode == 'agonist' else '文件内全部原始肌肉通道'}；"
          f"窗口 {args.window_size} 点、步长 {args.step_size} 点")
    print("时间对齐：使用最近一次已发生的评分，不向覆盖范围外推；0、1、2 均为有效类别。")
    print(f"发现 {len(pairs)} 份试验；成功 {len(report['records'])}；异常 {len(report['rejected_records'])}。")
    for name, stat in stats.items():
        title = {"train": "训练集", "validation": "验证集", "test": "测试集"}[name]
        print(f"\n【{title}】用户 {stat['subjects']}，实际产窗用户 {stat['subjects_with_windows']}")
        print(f"成功记录 {stat['records']} 份；保留窗口 {stat['windows']} 个；各等级窗口数 {stat['label_counts']}")
        print(f"丢弃候选窗：时间断点 {stat['dropped']['time_gap']}；评分未覆盖/过期 {stat['dropped']['uncovered']}；"
              f"混合疲劳等级 {stat['dropped']['mixed_labels']}")
    print(f"\n修正记录 {report['repaired_records']} 份；肌电补零尾行 {report['repair_counts']['emg_zero_tail_rows']} 行；"
          f"同评分但时间倒退/重复的评分尾行 {report['repair_counts']['rating_reversed_tail_rows']} 行（原始文件未改动）。")
    print(f"有效窗口共 {report['total_windows']}，三组用户重叠：否。")
    print("每窗肌电形状为（窗口点数，所选肌肉数），特征提取后可训练疲劳分类器。")
    print("这是主观疲劳评分研究，不是临床诊断；尚未训练模型，也未滤波或标准化。")


if __name__ == "__main__":
    main()
