"""把 models/ 全部源码整合为一个 txt 快照，并附上 outputs/ 的指标汇总。

用法：
    python tools/models_to_txt.py --tag baseline --run Run1

输出：docs/temporary/models_and_metrics_<tag>_<run>.txt（同名文件会覆盖）。
"""
import os
import re
import glob
import argparse

from extract_metrics_to_excel import extract_test_block, parse_block, METRIC_KEYS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "models")
OUTPUTS = os.path.join(ROOT, "outputs")
DOCS_TMP = os.path.join(ROOT, "docs", "temporary")

SKIP_DIRS = {"__pycache__", "build", "csrc", ".git"}
SKIP_EXTS = {".pth", ".pt", ".pyc"}


def collect_code(models_dir=None):
    models_dir = models_dir or MODELS
    files = []
    for root, dirs, names in os.walk(models_dir):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for n in sorted(names):
            ext = os.path.splitext(n)[1]
            if ext in SKIP_EXTS:
                continue
            if ext in (".py", ".yaml", ".yml"):
                files.append(os.path.relpath(os.path.join(root, n), models_dir))
    return files


def collect_metrics():
    rows = []
    for log in sorted(glob.glob(os.path.join(OUTPUTS, "**", "train_log.txt"), recursive=True)):
        rel = os.path.relpath(os.path.dirname(log), OUTPUTS).replace("\\", "/")
        with open(log, encoding="utf-8", errors="replace") as f:
            block = extract_test_block(f.read())
        if block is None:
            continue
        info = parse_block(block)
        if "F1" in info:
            rows.append((rel, info))
    return rows


def collect_incomplete_logs(tag, run):
    """列出选中轮次没有正式 TEST 区块的日志，避免把停训当成零分。"""
    base = os.path.join(OUTPUTS, tag, run)
    rows = []
    for log in sorted(glob.glob(os.path.join(base, "**", "train_log.txt"), recursive=True)):
        with open(log, encoding="utf-8", errors="replace") as f:
            text = f.read()
        if extract_test_block(text) is not None:
            continue
        steps = re.findall(r"^\[ACTUAL-OPT-STEPS\].*cumulative=(\d+/\d+)", text, re.M)
        rel = os.path.relpath(os.path.dirname(log), OUTPUTS).replace("\\", "/")
        rows.append((rel, steps[-1] if steps else "unknown"))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", type=str, default="baseline")
    ap.add_argument("--run", type=str, default="Run1")
    ap.add_argument("--models-dir", type=str, default="models",
                    help="代码目录（相对仓库根）：models | models_v1")
    args = ap.parse_args()

    models_dir = os.path.join(ROOT, args.models_dir)
    assert os.path.isdir(models_dir), f"缺目录 {models_dir}"
    os.makedirs(DOCS_TMP, exist_ok=True)
    out_path = os.path.join(DOCS_TMP, f"models_and_metrics_{args.tag}_{args.run}.txt")

    with open(out_path, "w", encoding="utf-8", newline="\n") as out:
        out.write(f"# CASA-CD models snapshot + metrics\n")
        out.write(f"# tag={args.tag}  run={args.run}  models_dir={args.models_dir}\n")
        out.write("# CODE 为生成时的本地 models 快照；METRICS 来自 outputs/ 的正式 TEST 区块。\n")
        out.write("# 二者是否与远端训练时的代码身份一致，须核对各 run 的 source_code_sha256。\n")

        files = collect_code(models_dir)
        out.write(f"\n{'=' * 80}\n## MODELS CODE ({len(files)} files)\n{'=' * 80}\n")
        for rel in files:
            out.write(f"\n{'=' * 80}\n# FILE: {args.models_dir}/{rel}\n{'=' * 80}\n")
            with open(os.path.join(models_dir, rel), encoding="utf-8", errors="replace") as f:
                out.write(f.read())

        metrics = collect_metrics()
        out.write(f"\n{'=' * 80}\n## METRICS ({len(metrics)} results)\n{'=' * 80}\n")
        out.write("Run | " + " | ".join(METRIC_KEYS) + " | Params(M) | Trainable(M) | FLOPs(G) | Mode\n")
        for rel, info in metrics:
            line = [rel] + [f"{info.get(k):.4f}" if info.get(k) is not None else "-" for k in METRIC_KEYS]
            line += [
                f"{info['Params(M)']:.3f}" if info.get("Params(M)") is not None else "-",
                f"{info['Trainable(M)']:.3f}" if info.get("Trainable(M)") is not None else "-",
                f"{info['FLOPs(G)']:.4f}" if info.get("FLOPs(G)") is not None else "-",
                info.get("Mode") or "-",
            ]
            out.write(" | ".join(line) + "\n")

        incomplete = collect_incomplete_logs(args.tag, args.run)
        out.write(f"\n{'=' * 80}\n## SELECTED RUN LOGS WITHOUT FINAL TEST ({len(incomplete)})\n{'=' * 80}\n")
        for rel, steps in incomplete:
            out.write(f"{rel} | last_logged_steps={steps} | no formal metric\n")

    print(f"已写入 {out_path}")
    print(f"  代码文件 {len(files)} 个，指标 {len(metrics)} 行")


if __name__ == "__main__":
    main()
