"""对同一人的两日动作清单做两步推理，导出逐次、逐日和最终单值。"""

import argparse
import json
from pathlib import Path

from matplotlib.figure import Figure

from .change import FEATURES, RESULTS, RUNS, load_change_model, read_trials, trial_features
from .model import load_force_model


## @brief 加载相互匹配的两个模型，输出两次测量的全部推理结果。
def run(trials_path, stage1_path, model_path, card_path, output):
    trials = read_trials(trials_path, False)
    model = load_change_model(model_path, card_path)
    model.check_visits(trials)
    stage1 = load_force_model(stage1_path, model.stage1_sha256)
    X = trial_features(trials, stage1)
    details = trials[["patient", "protocol", "date", "record", "start_s", "end_s"]].copy()
    details["record"] = details.record.map(lambda path: path.name)
    details[list(FEATURES)] = X
    details["estimate"] = model.predict(X)
    daily = details.groupby("date", sort=True).estimate.agg(["median", "count"])
    delta = float(daily["median"].iloc[1] - daily["median"].iloc[0])
    summary = dict(patient=details.patient.iloc[0], protocol=details.protocol.iloc[0],
                   baseline=daily.index[0], followup=daily.index[1], change=delta, unit=model.unit,
                   stage1_sha256=model.stage1_sha256,
                   model_sha256=json.loads(card_path.read_text(encoding="utf-8"))["sha256"])
    message = f"本次动作力量较基线的估计变化　{delta:+.1f} {model.unit}"
    fig = Figure(figsize=(6, 4), layout="constrained")
    ax = fig.subplots()
    for i, (day, rows) in enumerate(details.groupby("date", sort=True)):
        ax.scatter([i] * len(rows), rows.estimate, color="#36869b", alpha=.65)
    ax.plot([0, 1], daily["median"], "o-", color="#222222", label="Daily median")
    ax.set(xticks=[0, 1], xticklabels=daily.index, ylabel="Estimated force (N)" if model.unit == "N"
           else "Estimated force (putEMG raw units)")
    ax.legend()
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / "force_change.png")
    details.to_csv(output / "trials.csv", index=False)
    daily.to_csv(output / "daily.csv")
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "patient_result.txt").write_text(message + "\n", encoding="utf-8")
    print(message)
    return summary


## @brief 命令行部署入口，只有成功完成两天的计算才输出患者结果。
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=Path, required=True)
    parser.add_argument("--stage1", type=Path, default=RUNS / "force_model.joblib")
    parser.add_argument("--model", type=Path, default=RUNS / "force_change_model.joblib")
    parser.add_argument("--card", type=Path, default=RESULTS / "model_card.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        run(args.trials, args.stage1, args.model, args.card, args.output_dir)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"力量变化预测失败：{error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
