"""Collect formal WDMF-STR results using the project metric and snapshot tools."""
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))
from extract_metrics_to_excel import extract_test_block, parse_block


def command(*args):
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)


def main():
    command("tools/download_training_results.py", "--tag", "CASA-WDMF", "--run", "Run1")
    base = ROOT / "outputs/CASA-WDMF/Run1"
    status = json.loads((base / "pipeline_status.json").read_text(encoding="utf-8"))
    state_file = base / "collection_status.json"
    prior = json.loads(state_file.read_text(encoding="utf-8")) if state_file.exists() else {"formal_runs": []}
    rows, formal = [], []
    active = set(status.get("active", {}).values())
    failed = {item["run"] for item in status.get("failed", [])}
    for dataset in ["SYSU-CD-256", "LEVIR-CD-256", "WHU-CD-256", "CDD-CD-256"]:
        for variant in ["M0_CTRL", "M1_HYPER_QKV", "M2_CA_QKV"]:
            name = f"{variant}/{dataset}"
            path = base / name / "train_log.txt"
            text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
            block = extract_test_block(text)
            f1, iou = "—", "—"
            if block:
                metrics = parse_block(block)
                formal.append(name)
                progress = "完成"
                f1, iou = f"{metrics['F1']*100:.2f}", f"{metrics['IoU']*100:.2f}"
            elif name in active:
                steps = re.findall(r"(?:step=|cumulative=)(\d+/\d+)", text)
                progress = f"训练中：{steps[-1]}" if steps else "初始化"
            elif name in failed:
                progress = "失败"
            elif text:
                progress = "已停止"
            else:
                progress = "待启动"
            rows.append(f"| {dataset.split('-')[0]} | `{variant}` | {progress} | {f1} | {iou} |")
    section = f"状态更新：{status.get('updated', '')} CST；队列 `{status['status']}`；正式结果 **{len(formal)}/12**。\n\n"
    section += "| 数据集 | 实验 | 状态 / 步数 | F1 (%) | IoU (%) |\n|---|---|---|---:|---:|\n"
    section += "\n".join(rows)
    readme = ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")
    begin, end = "<!-- WDMF_RESULTS_START -->", "<!-- WDMF_RESULTS_END -->"
    start, stop = text.index(begin) + len(begin), text.index(end)
    readme.write_text(text[:start] + "\n\n" + section + "\n\n" + text[stop:], encoding="utf-8", newline="\n")
    if set(formal) - set(prior["formal_runs"]):
        command("tools/extract_metrics_to_excel.py")
        command("tools/models_to_txt.py", "--tag", "CASA-WDMF", "--run", "Run1")
    state_file.write_text(json.dumps({"formal_runs": formal, "pipeline_status": status["status"]}, indent=2), encoding="utf-8")
    print(f"formal={len(formal)}/12 status={status['status']}")


if __name__ == "__main__":
    main()
