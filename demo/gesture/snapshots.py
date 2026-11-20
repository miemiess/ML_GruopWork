"""为网页快照记录所用的模型和实验表格。"""

import json
from pathlib import Path

from emginsight.common.artifacts import file_sha256

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
GESTURE_REPORT = ROOT / "results/uci_gesture/classification"
FORCE_REPORT = ROOT / "results/putemg_force/regression"
GESTURE_REPORT_FILES = ("model_card.json", "metrics.json", "test_predictions.csv", "validation.csv")
FORCE_REPORT_FILES = ("model_card.json", "metrics.json", "example_predictions.csv", "validation.csv",
                      "per_target.csv", "per_user.csv")
SOURCE_PREFIX = "// EMGInsight snapshot "


def experiment_source(report, files):
    card = json.loads((report / "model_card.json").read_text(encoding="utf-8"))
    return {
        "experiment": report.relative_to(ROOT).as_posix(),
        "model_sha256": card["sha256"],
        "report_sha256": {name: file_sha256(report / name) for name in files},
    }


def write_snapshot(filename, variable, data, source):
    header = SOURCE_PREFIX + json.dumps(source, separators=(",", ":"))
    payload = json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    (HERE / filename).write_text(f"{header}\nwindow.{variable}={payload};\n", encoding="utf-8")
