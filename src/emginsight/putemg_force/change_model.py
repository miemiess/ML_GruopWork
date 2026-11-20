"""指力变化第二阶段的模型文件与完整性校验。"""

from dataclasses import dataclass, field
import json
from pathlib import Path

import numpy as np
import sklearn

from emginsight.models.artifacts import load_model_file, save_model_file
from .targets import TARGETS

RUNS = Path("results/runs/putemg_force")
RESULTS = Path("results/putemg_force/change")
FEATURES = tuple(f"{name}_pred" for name in TARGETS)
RAW_UNIT = "putEMG 原始刻度"


@dataclass
class ForceChangeModel:
    """已拟合的十路岭回归，权重已折算至原始特征尺度。"""

    weights: np.ndarray
    intercept: float
    stage1_sha256: str
    unit: str
    protocols: tuple[str, ...]
    patient: str | None = None
    fit_dates: tuple[str, ...] = ()
    validation: dict = field(default_factory=dict)
    sklearn_version: str = sklearn.__version__
    format_version: int = 1

    ## @brief 从每次动作的十路预测中位数计算一次标定后的动作力量。
    def predict(self, features):
        x = np.asarray(features, dtype=float)
        if x.ndim != 2 or x.shape[1] != 10 or not np.isfinite(x).all():
            raise ValueError("每次动作需要十个有限的指力特征。")
        return x @ self.weights + self.intercept

    ## @brief 核对训练协议，并核对牛顿标定的个人身份、日期及留出日误差。
    def check_visits(self, trials):
        if trials.protocol.iloc[0] not in self.protocols:
            raise ValueError(f"动作协议与模型不匹配，可用协议为 {self.protocols}")
        if self.patient is not None:
            if trials.patient.iloc[0] != self.patient or trials.date.min() <= max(self.fit_dates):
                raise ValueError("患者不匹配或比较日期早于个人标定完成日。")
            if not 0 <= self.validation["delta_mae_n"] < self.validation["no_change_mae_n"]:
                raise ValueError("个人标定的跨日误差未优于零变化基线。")


## @brief 保存第二阶段模型和可移植的配套校验卡。
def save_change_model(model, path, card, metadata):
    info = save_model_file(model, path)
    info.update(metadata, unit=model.unit, features=FEATURES, protocols=model.protocols,
                stage1_sha256=model.stage1_sha256, sklearn_version=model.sklearn_version,
                format_version=model.format_version)
    card.parent.mkdir(parents=True, exist_ok=True)
    card.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    return info


## @brief 读取并核对第二阶段文件的类型、版本和 SHA-256。
def load_change_model(path, card):
    info = json.loads(card.read_text(encoding="utf-8"))
    return load_model_file(path, ForceChangeModel, 1, info["sha256"])
