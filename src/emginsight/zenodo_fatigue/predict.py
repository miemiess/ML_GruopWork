"""四路疲劳模型的 CSV 批处理部署入口。"""
import argparse
import json
from pathlib import Path
from time import perf_counter

import joblib
import sklearn
from matplotlib.figure import Figure
from emginsight._batch import expand_inputs, run_batch
from emginsight.models.artifacts import file_sha256
from .full_dataset import REPORT, RUN, predict_record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=RUN/"fatigue_model.joblib")
    parser.add_argument("--card", type=Path, default=REPORT/"model_card.json")
    args = parser.parse_args(argv)
    card = json.loads(args.card.read_text(encoding="utf8"))
    if file_sha256(args.model) != card["sha256"] or card["sklearn"] != sklearn.__version__:
        parser.error("模型校验值或 scikit-learn 版本不匹配。")
    files = expand_inputs(args.input, {".csv"})
    if not files or len({p.stem.casefold() for p in files}) != len(files):
        parser.error("请提供至少一份 CSV，并使用不同的文件名以区分输出。")
    model = joblib.load(args.model)

    def predict(path):
        start = perf_counter()
        frame = predict_record(path, model)
        latency = 1000*(perf_counter()-start)/len(frame)
        frame.to_csv(args.output_dir/f"{path.stem}_predictions.csv", index=False)
        fig = Figure(figsize=(10, 3.5), layout="constrained")
        ax = fig.subplots()
        ax.step(frame.start_seconds, frame.predicted_label, where="post")
        ax.set(xlabel="Window start (s)", ylabel="Estimated self-rated fatigue",
               yticks=[0,1,2], yticklabels=["None","Transition","Fatigue"])
        fig.savefig(args.output_dir/f"{path.stem}_timeline.png")
        return dict(file=str(path),windows=len(frame),ms_per_window=latency,
                    label_counts=frame.predicted_label.value_counts().to_dict()), ""

    return run_batch(files,args.output_dir,"zenodo_fatigue",{"model":card,"unit":"自感疲劳等级 0/1/2"},predict)


if __name__=="__main__":
    raise SystemExit(main())
