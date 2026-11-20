"""C 任务获选手势分类模型：保存、读取，以及从窗口到六类概率的推理。"""

from dataclasses import asdict, dataclass, field

import numpy as np
import sklearn

from emginsight.features import extract_time_features
from emginsight.models.artifacts import load_model_file, save_model_file

FORMAT_VERSION = 2
UNKNOWN = 0
LABEL_NAMES = {0: "未知", 1: "静息", 2: "握拳", 3: "腕屈", 4: "腕伸", 5: "桡偏", 6: "尺偏"}
LABEL_NAMES_EN = {0: "unknown", 1: "rest", 2: "fist", 3: "wrist flexion", 4: "wrist extension",
                  5: "radial deviation", 6: "ulnar deviation"}


@dataclass
class GestureModel:
    """已训练的标准化器与分类器，以及复现预测所需的特征列和分窗参数。

    输入为 (窗口数, 窗口点数, 8) 的原始 UCI 肌电窗口；输出六类手势及概率，
    类别编码与 UCI 标签一致（1 静息 … 6 尺偏），概率列顺序见 classes。
    """

    model: object
    scaler: object
    columns: tuple[int, ...]
    features: str
    classes: tuple[int, ...]
    preprocessing: dict
    train_subjects: tuple[str, ...]
    sklearn_version: str = sklearn.__version__
    metrics: dict = field(default_factory=dict)
    reject_threshold: float | None = None
    format_version: int = FORMAT_VERSION

    ## @brief 六类预测加“未知”拒识：最大概率低于 reject_threshold 时输出 0。
    # @param windows 原始肌电窗口。
    # @return tuple (含未知的预测, 六类预测, 各类概率)。
    def predict_with_unknown(self, windows):
        proba = self.predict_proba(windows)
        raw = self.predict(windows)
        if not self.reject_threshold:
            return raw, raw, proba
        return np.where(proba.max(axis=1) < self.reject_threshold, UNKNOWN, raw), raw, proba

    ## @brief 提取 RMS/MAV/WL、选列、标准化后输出各类概率。
    # @param windows 原始肌电窗口 (窗口数, 窗口点数, 8)。
    # @return ndarray (窗口数, 类别数) 概率，列顺序同 classes。
    def predict_proba(self, windows):
        X = self._inputs(windows)
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        # 项目的 kNN 只投票不给概率，退化为所选类别的独热向量。
        return (self.model.predict(X)[:, None] == np.asarray(self.classes)[None, :]).astype(float)

    ## @brief 输出每个窗口的手势编码，直接调用分类器自身的 predict，与实验预测一致。
    # @param windows 原始肌电窗口。
    # @return ndarray 一维手势编码。
    def predict(self, windows):
        return self.model.predict(self._inputs(windows))

    ## @brief 提取特征、选列并用训练集统计量标准化。
    # @param windows 原始肌电窗口。
    # @return ndarray 模型输入。
    def _inputs(self, windows):
        return self.scaler.transform(extract_time_features(windows)[:, list(self.columns)])

    ## @brief 汇总不含模型对象的可读配置。
    # @return dict 可写入 JSON 的模型说明。
    def describe(self):
        info = asdict(self)
        for key in ("model", "scaler"):
            info.pop(key)
        info["model_class"] = type(self.model).__name__
        params = self.model.get_params() if hasattr(self.model, "get_params") else {}
        info["model_params"] = {k: v for k, v in params.items() if k != "n_jobs"}
        info["label_names"] = {str(k): v for k, v in LABEL_NAMES.items()}
        return info


## @brief 压缩保存手势模型文件。
# @param model GestureModel。
# @param path 输出路径。
# @return dict 路径、字节数和 SHA-256。
def save_gesture_model(model, path):
    return save_model_file(model, path)


## @brief 读取手势模型文件，检查校验值、格式与 scikit-learn 版本。
# @param path 模型文件路径。
# @param sha256 可选的期望校验值。
# @return GestureModel 已训练模型。
def load_gesture_model(path, sha256=None):
    return load_model_file(path, GestureModel, FORMAT_VERSION, sha256)
