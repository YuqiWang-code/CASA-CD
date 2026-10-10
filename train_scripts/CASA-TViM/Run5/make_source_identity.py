#!/usr/bin/env python
"""生成 Run5 代码/资产身份 `SOURCE_IDENTITY.json`（Run5 设计文档 §11.1 步骤 1）。

记录并核对：
  * 12 个诊断核心源码（Diag1 `DIAG_CODE_FILES`）逐文件 SHA256；
  * Run4 冻结文件（负结果消融，Run5 不得改动）SHA256；
  * Run5 新增文件 SHA256（前测脚本 + 单测 + 本生成器）；
  * 预训练权重 `pretrained_weight/tinyvim_s_1000e.pth`；
  * 四库 `list/train.txt` 与 `list/test.txt`；
  * `git rev-parse HEAD`（本地仓库；服务器工作副本非 git 仓库 → 记 null）。

`--require-all` 用于服务器 S0 门：任一预期文件缺失即非零退出（禁止“假设服务器就是 GitHub HEAD”）。

    python train_scripts/CASA-TViM/Run5/make_source_identity.py --require-all \
        --out train_scripts/CASA-TViM/Run5/SOURCE_IDENTITY.json
"""
import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))

# Diag1 `tvim_diag_common.DIAG_CODE_FILES` 的 12 文件基线（Run5 只允许改 casa_tvim_str_net.py /
# str_tar.py / train.py / eval.py；其余必须逐字节不变，见 Run5 §6.7）
CORE_12 = [
    "models/model/casa_tvim_str_net.py",
    "models/model/tinyvim_s_slim.py",
    "models/model/layers/caacp_ss2d.py",
    "models/model/layers/ss2d.py",
    "models/model/str_tar.py",
    "models/model/str_dcr.py",
    "models/model/str_reparam.py",
    "models/model/metric_tool.py",
    "models/dataset/dataset.py",
    "models/dataset/Transforms.py",
    "models/train.py",
    "models/eval.py",
]

# Run4 负结果消融（冻结；Run5 §6.7 明确“不动”或“保留”）
RUN4_FROZEN = [
    "models/model/str_fine_tap.py",
    "models/model/str_fine_head.py",
    "models/test_run4_fine_tap.py",
    "models/run4_dry_run.py",
    "analyse/run4_fet_report.py",
    "analyse/tvim_run4_tap_preflight.py",
    "analyse/run4_fold_knife_edge.py",
]

# Run5 新增源码
RUN5_NEW = [
    "analyse/tvim_run5_hf_pair_preflight.py",
    "analyse/tests/test_tvim_run5_preflight.py",
    "train_scripts/CASA-TViM/Run5/make_source_identity.py",
]

ASSETS = [
    "pretrained_weight/tinyvim_s_1000e.pth",
]

DATASETS = ["CDD-CD-256", "LEVIR-CD-256", "SYSU-CD-256", "WHU-CD-256"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_head(project):
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=project,
                             capture_output=True, text=True, timeout=20)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:  # noqa: BLE001
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=PROJECT)
    ap.add_argument("--out", default=os.path.join(HERE, "SOURCE_IDENTITY.json"))
    ap.add_argument("--data-root", default="/share_datasets/CD")
    ap.add_argument("--plan-sha256", default=None,
                    help="本地记录的 Run5 设计文档 SHA256（服务器上无该文件时由 CLI 传入）")
    ap.add_argument("--require-all", action="store_true",
                    help="S0 门：任一预期文件缺失即 rc=2")
    args = ap.parse_args()

    groups = {"core12": CORE_12, "run4_frozen": RUN4_FROZEN, "run5_new": RUN5_NEW}
    files, missing = {}, []
    group_of = {}
    for gname, rels in groups.items():
        for rel in rels:
            group_of[rel] = gname
            p = os.path.join(args.project, rel.replace("/", os.sep))
            if not os.path.isfile(p):
                files[rel] = None
                missing.append(rel)
            else:
                files[rel] = sha256(p)

    for rel in ASSETS:
        group_of[rel] = "asset"
        p = os.path.join(args.project, rel.replace("/", os.sep))
        if not os.path.isfile(p):
            files[rel] = None
            missing.append(rel)
        else:
            files[rel] = sha256(p)

    lists = {}
    for ds in DATASETS:
        for split in ("train", "test"):
            rel = f"{args.data_root}/{ds}/list/{split}.txt"
            group_of[rel] = "dataset_list"
            if os.path.isfile(rel):
                files[rel] = sha256(rel)
                lists[f"{ds}/{split}"] = files[rel]
            else:
                files[rel] = None
                lists[f"{ds}/{split}"] = None
                missing.append(rel)

    payload = {
        "run": "Run5",
        "stage": "FHR-TAR preflight (S1) source/asset identity",
        "plan": "docs/temporary/CASA-CD_Run5_原生高频证据复用与结构改进实验设计_2026-10-10.md",
        "plan_sha256": args.plan_sha256,
        "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        "project": args.project,
        "git_head": _git_head(args.project),
        "frozen_git_head_expected": "a6d49bc151674a93a8274aca52e2ca707f8d070a",
        "files_sha256": files,
        "files_sha256_short16": {k: (v[:16] if v else None) for k, v in files.items()},
        "group": group_of,
        "dataset_lists": lists,
        "missing": missing,
        "note": ("Run5 在 S1 阶段只新增 analyse/tvim_run5_hf_pair_preflight.py 与单测；"
                 "models/ 在门 PASS 之前不改。12 文件基线中，Run5 实装阶段只允许改 "
                 "casa_tvim_str_net.py / str_tar.py / train.py / eval.py（§6.7），"
                 "ss2d.py / caacp_ss2d.py / str_dcr.py / str_reparam.py / str_fine_tap.py "
                 "必须保持不变。"),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {args.out}")
    for rel in sorted(files):
        print(f"  {group_of[rel]:13s} {str(files[rel])[:16]:16s} {rel}")
    if missing:
        print(f"[WARN] {len(missing)} missing: {missing}")
        if args.require_all:
            print("[S0][FAIL] --require-all set and files are missing")
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
