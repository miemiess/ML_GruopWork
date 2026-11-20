"""已训练模型文件的通用保存与读取：压缩保存、SHA-256 校验、类型与库版本检查。"""

import hashlib
from pathlib import Path

import joblib
import sklearn


## @brief 计算文件的 SHA-256。
# @param path 文件路径。
# @return str 十六进制校验值。
def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


## @brief 压缩保存模型对象，并返回校验信息。
# @param model 待保存对象，应带 sklearn_version 与 format_version 属性。
# @param path 输出路径，建议放在被 Git 忽略的 results/runs/。
# @return dict 路径、字节数和 SHA-256。
def save_model_file(model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path, compress=3)
    return {"path": path.as_posix(), "bytes": path.stat().st_size, "sha256": file_sha256(path)}


## @brief 读取模型文件，检查校验值、对象类型、格式版本与 scikit-learn 版本。
# @param path 模型文件路径。
# @param expected_type 期望的模型类。
# @param format_version 期望的格式版本。
# @param sha256 可选的期望校验值。
# @return object 已训练模型。
# @exception ValueError 校验值、类型、格式或库版本不匹配。
def load_model_file(path, expected_type, format_version, sha256=None):
    path = Path(path)
    if sha256 is not None and file_sha256(path) != sha256:
        raise ValueError(f"{path} 的 SHA-256 与期望值不一致，文件可能已损坏或不是同一模型。")
    model = joblib.load(path)
    if not isinstance(model, expected_type) or getattr(model, "format_version", None) != format_version:
        raise ValueError(f"{path} 不是当前版本的 {expected_type.__name__} 文件。")
    if model.sklearn_version != sklearn.__version__:
        raise ValueError(f"模型由 scikit-learn {model.sklearn_version} 训练，当前为 {sklearn.__version__}，"
                         "请安装相同版本后再读取。")
    return model
