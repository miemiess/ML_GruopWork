"""手势和指力命令行共用的批处理与运行汇总。"""

import json
import sys


## @brief 展开输入目录第一层的文件，保留直接传入的文件路径。
def expand_inputs(paths, suffixes):
    files = []
    for path in paths:
        if path.is_dir():
            files.extend(child for child in sorted(path.iterdir())
                         if child.is_file() and child.suffix.lower() in suffixes)
        else:
            files.append(path)
    return files


## @brief 逐文件执行推理并写出 summary.json，一份文件失败时继续处理其他文件。
# @param files 待处理的文件路径。
# @param output 预测结果和汇总目录。
# @param task 任务名。
# @param metadata 模型信息和任务专用的汇总字段。
# @param predict 接收文件路径，返回该文件的汇总与额外的日志文本。
# @return int 进程退出码，任一文件失败时为 1。
def run_batch(files, output, task, metadata, predict):
    output.mkdir(parents=True, exist_ok=True)
    results = []
    failed = 0
    for path in files:
        try:
            summary, details = predict(path)
        except (ValueError, OSError) as error:
            failed += 1
            results.append({"file": str(path), "status": "error", "error": str(error)})
            print(f"[失败] {path.name}：{error}", file=sys.stderr, flush=True)
        else:
            results.append({**summary, "status": "ok"})
            print(f"[完成] {path.name}：{summary['windows']} 窗，"
                  f"{summary['ms_per_window']:.2f} ms/窗{details}", flush=True)

    report = {
        "task": task,
        **metadata,
        "counts": {"total": len(results), "succeeded": len(results) - failed, "failed": failed},
        "files": results,
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"结果已写入 {output}")
    return int(failed > 0)
