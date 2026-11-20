"""D 任务获选力量回归模型：特征变换、因果平滑及模型文件的保存与读取。"""

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
import sklearn

from emginsight.common.artifacts import load_model_file, save_model_file

FEATURE_SETS = {"RMS+MAV": (0, 1), "RMS+MAV+WL": (0, 1, 2)}
FEATURE_SCALES = ("raw", "log", "log_rest")
FORMAT_VERSION = 1


## @brief 按特征组合和尺度变换得到模型输入。
# @param X 公共 RMS/MAV/WL 特征，(窗口数, 3 × 通道数)。
# @param rest 每个窗口所属记录的静息特征，形状同 X。
# @param features 特征组合名称。
# @param scale raw、log 或 log_rest（减去同记录静息段的对数特征）。
# @return ndarray 选定列的特征矩阵。
def transform(X, rest, features, scale):
    if scale not in FEATURE_SCALES:
        raise ValueError(f"未知特征尺度 {scale}。")
    columns = np.flatnonzero(np.isin(np.arange(X.shape[1]) % 3, FEATURE_SETS[features]))
    X, rest = X[:, columns], rest[:, columns]
    if scale == "raw":
        return X
    logged = np.log(np.maximum(X, 1e-6))
    return logged if scale == "log" else logged - np.log(np.maximum(rest, 1e-6))


## @brief 在每条记录内对预测做因果滑动平均，只用当前及之前的窗口。
# @param pred 预测矩阵，行按记录内时间排序。
# @param records 每行所属记录。
# @param k 平均窗口数；1 表示不平滑。
# @return ndarray 平滑后的预测。
def smooth(pred, records, k):
    if k == 1:
        return pred
    records = np.asarray(records)
    out = np.empty_like(pred)
    for record in pd.unique(records):
        rows = np.flatnonzero(records == record)
        total = np.vstack([np.zeros((1, pred.shape[1])), np.cumsum(pred[rows], axis=0)])
        end = np.arange(1, len(rows) + 1)
        begin = np.maximum(0, end - k)
        out[rows] = (total[end] - total[begin]) / (end - begin)[:, None]
    return out


@dataclass
class ForceModel:
    """已训练的标准化器与回归器，以及复现预测所需的全部预处理配置。

    输入为同一流程提取的原始 RMS/MAV/WL 特征和所属记录的静息特征；
    输出为相对静息零点的力量（原始数值，不是牛顿），列顺序见 targets。
    """

    model: object
    scaler: object
    scale: str
    features: str
    smoothing: int
    targets: tuple[str, ...]
    preprocessing: dict
    train_subjects: tuple[str, ...]
    sklearn_version: str = sklearn.__version__
    metrics: dict = field(default_factory=dict)
    format_version: int = FORMAT_VERSION

    ## @brief 对已提取的特征做变换、标准化、预测和因果平滑。
    # @param X 原始特征 (窗口数, 72)。
    # @param rest 每行所属记录的静息特征，形状同 X。
    # @param records 每行所属记录编号，用于分记录平滑；行须按记录内时间排序。
    # @return ndarray (窗口数, 目标数) 相对静息力量预测。
    def predict(self, X, rest, records):
        X, rest = np.asarray(X, dtype=float), np.asarray(rest, dtype=float)
        if X.shape != rest.shape or X.ndim != 2 or len(records) != len(X):
            raise ValueError("X、rest 与 records 的行数或形状不一致。")
        inputs = self.scaler.transform(transform(X, rest, self.features, self.scale))
        return smooth(self.model.predict(inputs), records, self.smoothing)

    ## @brief 汇总不含模型对象的可读配置。
    # @return dict 可写入 JSON 的模型说明。
    def describe(self):
        info = asdict(self)
        for key in ("model", "scaler"):
            info.pop(key)
        info["model_class"] = type(self.model).__name__
        # scikit-learn 模型提供 get_params；项目的 RidgeRegressor 只保存 alpha。
        params = self.model.get_params() if hasattr(self.model, "get_params") else {"alpha": self.model.alpha}
        info["model_params"] = {k: v for k, v in params.items() if k != "n_jobs"}
        return info


## @brief 压缩保存模型文件，并返回校验信息。
# @param model ForceModel。
# @param path 输出路径，建议放在被 Git 忽略的 results/runs/。
# @return dict 路径、字节数和 SHA-256。
def save_force_model(model, path):
    return save_model_file(model, path)


## @brief 读取模型文件，检查格式与 scikit-learn 版本。
# @param path 模型文件路径。
# @param sha256 可选的期望校验值；不一致时报错。
# @return ForceModel 已训练模型。
# @exception ValueError 文件不是本模块保存的模型、格式或库版本不匹配。
def load_force_model(path, sha256=None):
    return load_model_file(path, ForceModel, FORMAT_VERSION, sha256)


LEVEL_NAMES = {0: "静息", 1: "轻", 2: "重"}
LEVEL_NAMES_EN = {0: "rest", 1: "light", 2: "heavy"}
LEVEL_FORMAT_VERSION = 1


@dataclass
class ForceLevelModel:
    """整只手三档力量分类器：0 静息、1 轻（约 25% MVC）、2 重（约 50% MVC）。

    输入与 ForceModel 相同（原始特征、同记录静息特征、记录编号）；输出因果平滑后的三档概率。
    标签定义见 thresholds 与 preprocessing，仅在训练和评估时使用实测力量与 MVC。
    """

    model: object
    scaler: object
    scale: str
    features: str
    smoothing: int
    classes: tuple[int, ...]
    thresholds: tuple[float, float]
    preprocessing: dict
    train_subjects: tuple[str, ...]
    sklearn_version: str = sklearn.__version__
    metrics: dict = field(default_factory=dict)
    format_version: int = LEVEL_FORMAT_VERSION

    ## @brief 输出因果平滑后的各档概率。
    # @param X 原始特征 (窗口数, 72)。
    # @param rest 每行所属记录的静息特征。
    # @param records 每行所属记录编号；行须按记录内时间排序。
    # @return ndarray (窗口数, 3) 概率，列顺序同 classes。
    def predict_proba(self, X, rest, records):
        X, rest = np.asarray(X, dtype=float), np.asarray(rest, dtype=float)
        if X.shape != rest.shape or X.ndim != 2 or len(records) != len(X):
            raise ValueError("X、rest 与 records 的行数或形状不一致。")
        inputs = self.scaler.transform(transform(X, rest, self.features, self.scale))
        return smooth(self.model.predict_proba(inputs), records, self.smoothing)

    ## @brief 输出平滑概率最大的档位。
    # @return ndarray 一维档位 0/1/2。
    def predict(self, X, rest, records):
        return np.asarray(self.classes)[self.predict_proba(X, rest, records).argmax(axis=1)]

    ## @brief 汇总不含模型对象的可读配置。
    # @return dict 可写入 JSON 的模型说明。
    def describe(self):
        info = asdict(self)
        for key in ("model", "scaler"):
            info.pop(key)
        info["model_class"] = type(self.model).__name__
        params = self.model.get_params() if hasattr(self.model, "get_params") else {"l2": getattr(self.model, "l2", None)}
        info["model_params"] = {k: v for k, v in params.items() if k != "n_jobs"}
        info["level_names"] = LEVEL_NAMES
        return info


## @brief 保存三档力量模型。
def save_level_model(model, path):
    return save_model_file(model, path)


## @brief 读取三档力量模型，检查校验值、格式与 scikit-learn 版本。
def load_level_model(path, sha256=None):
    return load_model_file(path, ForceLevelModel, LEVEL_FORMAT_VERSION, sha256)
