"""校验网页快照并生成用于 gh-pages 的静态发布目录（页面、css/、js/、data/、source/）。"""

import json
from pathlib import Path
from shutil import copyfile
from model_export import export as export_models
from training_export import export as export_training

from snapshots import (
    DATA, FORCE_REPORT, FORCE_REPORT_FILES, GESTURE_REPORT, GESTURE_REPORT_FILES, ROOT, WEB,
    SOURCE_PREFIX, experiment_source,
)

OUTPUT = ROOT / "build/pages"
PAGES = ("index.html", "development.html", "force.html", "fatigue.html", "fatigue-single.html")
STATIC = ("css", "js")
GESTURE_SNAPSHOTS = ("session.js", "simulator-data.js", "development-data.js")
FORCE_SNAPSHOTS = ("force-data.js",)
GENERATED = ("model-data.js", "training-data.js", "full-fatigue-data.js")
SOURCE_FILES = (
    "features/time_domain.py", "models/standardization.py",
    "gesture/windowing.py", "gesture/experiment.py",
)


def check_snapshots(names, expected, command):
    for name in names:
        path = DATA / name
        if not path.is_file():
            raise ValueError(f"缺少 {name}，请运行 {command}。")
        with path.open(encoding="utf-8") as stream:
            header = stream.readline()
        if not header.startswith(SOURCE_PREFIX) or json.loads(header.removeprefix(SOURCE_PREFIX)) != expected:
            raise ValueError(f"{name} 与当前模型卡或实验结果不匹配，请运行 {command}。")


def build():
    gesture = experiment_source(GESTURE_REPORT, GESTURE_REPORT_FILES)
    force = experiment_source(FORCE_REPORT, FORCE_REPORT_FILES)
    check_snapshots(GESTURE_SNAPSHOTS, gesture, "python demo/web/export/prepare.py")
    check_snapshots(FORCE_SNAPSHOTS, force, "python demo/web/export/force_export.py")
    model_sources = export_models()
    model_sources.update(export_training())
    assets = [Path(name) for name in PAGES]
    assets += sorted(path.relative_to(WEB) for folder in STATIC for path in (WEB / folder).iterdir() if path.is_file())
    assets += [Path("data") / name for name in GESTURE_SNAPSHOTS + FORCE_SNAPSHOTS + GENERATED]
    files = set(assets)
    files.update(Path("source") / name for name in SOURCE_FILES)
    files.update((Path(".nojekyll"), Path("snapshot-manifest.json")))
    # 重建只覆盖发布脚本管理的文件，目录中有其他文件时要求先检查。
    extra = {path.relative_to(OUTPUT) for path in OUTPUT.rglob("*") if path.is_file()} - files
    if extra:
        names = ", ".join(sorted(path.as_posix() for path in extra))
        raise ValueError(f"{OUTPUT} 包含发布清单之外的文件，请检查后重新构建：{names}")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    for name in assets:
        (OUTPUT / name).parent.mkdir(parents=True, exist_ok=True)
        if name == Path("index.html"):
            page = (WEB / name).read_text(encoding="utf-8")
            (OUTPUT / name).write_text(
                page.replace('data-source-base="../../src/emginsight/"', 'data-source-base="./source/"'),
                encoding="utf-8",
            )
        else:
            copyfile(WEB / name, OUTPUT / name)
    for name in SOURCE_FILES:
        destination = OUTPUT / "source" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        copyfile(ROOT / "src/emginsight" / name, destination)
    (OUTPUT / ".nojekyll").touch()
    manifest = {"gesture": gesture, "force": force, **model_sources}
    (OUTPUT / "snapshot-manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )
    print(f"静态发布目录已就绪：{OUTPUT}")


if __name__ == "__main__":
    try:
        build()
    except (OSError, ValueError) as error:
        raise SystemExit(f"网页构建失败：{error}") from error
