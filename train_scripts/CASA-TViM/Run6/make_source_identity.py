#!/usr/bin/env python
"""生成 Run6 代码/资产身份 `SOURCE_IDENTITY.json`（Run6 设计文档 §8 S0 + 交接 §11.5）。

记录并核对：
  * 12 个诊断核心源码（Diag1 `DIAG_CODE_FILES`）逐文件 SHA256 —— **Run6 前测阶段必须零改动**；
  * Run4 负结果冻结文件 SHA256（Run6 不得触碰）；
  * Run5 冻结文件 SHA256（前测脚本 + FP 画像 + 单测；Run6 只在 `edge_field`/`dilate1`/
    `EDGE_Q` 上**复用** `tvim_run5_fp_profile.py`，不得修改它）；
  * Run6 新增文件 SHA256（前测脚本 + 单测 + README + 本生成器）；
  * 预训练权重 `pretrained_weight/tinyvim_s_1000e.pth`；
  * 四库 `list/train.txt` 与 `list/test.txt`；
  * Run6 设计文档 SHA256（若本地存在则现算，否则由 `--plan-sha256` 传入并核对）；
  * `git rev-parse HEAD`（本地仓库；服务器工作副本非 git 仓库 → 记 null）。

`--require-all` 用于服务器 S0 门：任一预期文件缺失即非零退出
（禁止"假设服务器就是 GitHub HEAD"）。`--expect-plan` 额外把设计文档 SHA 与冻结值对拍。

    python train_scripts/CASA-TViM/Run6/make_source_identity.py --require-all \
        --expect-plan \
        --out train_scripts/CASA-TViM/Run6/SOURCE_IDENTITY.json
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

RUN6_PLAN_REL = ("docs/temporary/"
                 "CASA-CD_Run6_变化判别瓶颈可分辨性前测与读出结构设计_2026-10-11.md")
RUN6_PLAN_SHA = "d77cacbd7b52363598487b348b2593ce5f989a6ae66d55dc688a0cd80969eb5b"
RUN6_FROZEN_GIT_HEAD = "b03606f"

# Diag1 `tvim_diag_common.DIAG_CODE_FILES` 的 12 文件基线。
# Run6 前测阶段（S1）**不允许改动其中任何一个**（改动只可能在门 PASS 后的 §5 实装阶段）。
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

# Run4 负结果消融（冻结）
RUN4_FROZEN = [
    "models/model/str_fine_tap.py",
    "models/model/str_fine_head.py",
    "models/test_run4_fine_tap.py",
    "models/run4_dry_run.py",
    "analyse/run4_fet_report.py",
    "analyse/tvim_run4_tap_preflight.py",
    "analyse/run4_fold_knife_edge.py",
]

# Run5 冻结文件（Run6 只读复用 tvim_run5_fp_profile.py 的 edge_field/dilate1，不得修改）
RUN5_FROZEN = [
    "analyse/tvim_run5_hf_pair_preflight.py",
    "analyse/tvim_run5_fp_profile.py",
    "analyse/tests/test_tvim_run5_preflight.py",
    "analyse/tests/test_tvim_run5_fp_profile.py",
    "train_scripts/CASA-TViM/Run5/make_source_identity.py",
    "train_scripts/CASA-TViM/Run5/README.md",
    "train_scripts/CASA-TViM/Run5/SELECTION_LOCK.json",
]

# Run6 新增源码
RUN6_NEW = [
    "analyse/tvim_run6_discriminability.py",
    "analyse/tests/test_tvim_run6_discriminability.py",
    "train_scripts/CASA-TViM/Run6/make_source_identity.py",
    "train_scripts/CASA-TViM/Run6/README.md",
    "train_scripts/CASA-TViM/Run6/SELECTION_LOCK.json",
]

# Run6 前测直接依赖的共享只读模块（身份必须可核验，否则"复现"无意义）
SHARED_DEPS = [
    "analyse/tvim_diag_common.py",
    "analyse/tvim_object_metrics.py",
    "analyse/tvim_linear_probe.py",
    "analyse/tvim_stage_recoverability.py",
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
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=project,
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
                    help="本地记录的 Run6 设计文档 SHA256（服务器上无该文件时传入）")
    ap.add_argument("--expect-plan", action="store_true",
                    help="设计要求：把设计文档 SHA 与冻结值 d77cacbd…eb5b 对拍")
    ap.add_argument("--require-all", action="store_true",
                    help="S0 门：任一预期文件缺失即 rc=2")
    args = ap.parse_args()

    groups = {
        "core12": CORE_12,
        "run4_frozen": RUN4_FROZEN,
        "run5_frozen": RUN5_FROZEN,
        "run6_new": RUN6_NEW,
        "shared_deps": SHARED_DEPS,
    }
    files, missing, group_of = {}, [], {}
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
        files[rel] = sha256(p) if os.path.isfile(p) else None
        if files[rel] is None:
            missing.append(rel)

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

    plan_path = os.path.join(args.project, RUN6_PLAN_REL.replace("/", os.sep))
    plan_sha_local = sha256(plan_path) if os.path.isfile(plan_path) else None
    plan_sha = args.plan_sha256 or plan_sha_local
    plan_ok = (plan_sha == RUN6_PLAN_SHA)
    if args.expect_plan and not plan_ok:
        print(f"[S0][FAIL] plan sha256 mismatch: got {plan_sha} != {RUN6_PLAN_SHA}")

    git_head = _git_head(args.project)
    git_ok = bool(git_head is None or git_head.startswith(RUN6_FROZEN_GIT_HEAD))

    payload = {
        "run": "Run6",
        "stage": "discriminability preflight (S1) source/asset identity",
        "plan": RUN6_PLAN_REL,
        "plan_sha256": plan_sha,
        "plan_sha256_expected": RUN6_PLAN_SHA,
        "plan_sha256_match": plan_ok,
        "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        "project": args.project,
        "git_head": git_head,
        "frozen_git_head_expected": RUN6_FROZEN_GIT_HEAD,
        "git_head_match": git_ok,
        "files_sha256": files,
        "files_sha256_short16": {k: (v[:16] if v else None) for k, v in files.items()},
        "group": group_of,
        "dataset_lists": lists,
        "missing": missing,
        "note": ("Run6 S1 只新增 analyse/tvim_run6_discriminability.py + 单测 + Run6 目录文档；"
                 "models/ 在 gate.json PASS 之前**逐字节不变**。core12 / run4_frozen / "
                 "run5_frozen 三组的 SHA 是\"未改动 models/ 与未改动 Run5\"的证据。"),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"wrote {args.out}")
    for rel in sorted(files):
        print(f"  {group_of[rel]:13s} {str(files[rel])[:16]:16s} {rel}")
    if missing:
        print(f"[WARN] {len(missing)} missing: {missing}")
        if args.require_all:
            print("[S0][FAIL] --require-all set and files are missing")
            return 2
    if args.expect_plan and not plan_ok:
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
