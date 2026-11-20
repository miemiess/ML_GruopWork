"""校验两套网页快照并生成用于 gh-pages 的静态发布目录。"""

import json
from pathlib import Path
from shutil import copyfile

from snapshots import (
    FORCE_REPORT, FORCE_REPORT_FILES, GESTURE_REPORT, GESTURE_REPORT_FILES, HERE, ROOT,
    SOURCE_PREFIX, experiment_source,
)

OUTPUT = ROOT / "build/pages"
PAGES = (
    "index.html", "style.css", "app.js", "hand.js", "simulation.js", "walkthrough.js",
    "development.html", "development.css", "development.js", "development_diagrams.js",
    "force.html", "force.css", "force.js",
)
GESTURE_SNAPSHOTS = ("session.js", "simulator-data.js", "development-data.js")
FORCE_SNAPSHOTS = ("force-data.js",)
SOURCE_FILES = (
    "features/time_domain.py", "models/standardization.py",
    "uci_gesture/windowing.py", "uci_gesture/experiment.py",
)


def check_snapshots(names, expected, command):
    for name in names:
        path = HERE / name
        if not path.is_file():
            raise ValueError(f"缺少 {name}，请运行 {command}。")
        with path.open(encoding="utf-8") as stream:
            header = stream.readline()
        if not header.startswith(SOURCE_PREFIX) or json.loads(header.removeprefix(SOURCE_PREFIX)) != expected:
            raise ValueError(f"{name} 与当前模型卡或实验结果不匹配，请运行 {command}。")


def build():
    gesture = experiment_source(GESTURE_REPORT, GESTURE_REPORT_FILES)
    force = experiment_source(FORCE_REPORT, FORCE_REPORT_FILES)
    check_snapshots(GESTURE_SNAPSHOTS, gesture, "python demo/gesture/prepare.py")
    check_snapshots(FORCE_SNAPSHOTS, force, "python demo/gesture/force_export.py")
    assets = PAGES + GESTURE_SNAPSHOTS + FORCE_SNAPSHOTS
    files = {Path(name) for name in assets}
    files.update(Path("source") / name for name in SOURCE_FILES)
    files.update((Path(".nojekyll"), Path("snapshot-manifest.json")))
    # 重建只覆盖发布脚本管理的文件，目录中有其他文件时要求先检查。
    extra = {path.relative_to(OUTPUT) for path in OUTPUT.rglob("*") if path.is_file()} - files
    if extra:
        names = ", ".join(sorted(path.as_posix() for path in extra))
        raise ValueError(f"{OUTPUT} 包含发布清单之外的文件，请检查后重新构建：{names}")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    for name in assets:
        if name == "index.html":
            page = (HERE / name).read_text(encoding="utf-8")
            (OUTPUT / name).write_text(
                page.replace('data-source-base="../../src/emginsight/"', 'data-source-base="./source/"'),
                encoding="utf-8",
            )
        else:
            copyfile(HERE / name, OUTPUT / name)
    for name in SOURCE_FILES:
        destination = OUTPUT / "source" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        copyfile(ROOT / "src/emginsight" / name, destination)
    (OUTPUT / ".nojekyll").touch()
    manifest = {"uci_gesture": gesture, "putemg_force": force}
    (OUTPUT / "snapshot-manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )
    print(f"静态发布目录已就绪：{OUTPUT}")


if __name__ == "__main__":
    try:
        build()
    except (OSError, ValueError) as error:
        raise SystemExit(f"网页构建失败：{error}") from error
