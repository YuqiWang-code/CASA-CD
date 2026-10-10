#!/usr/bin/env python
"""生成 Run4 代码身份 `SOURCE_IDENTITY.json`（方案 §8.2，不改写 Diag1 manifest）。

记录：Run4 涉及的全部模型/训练/评估文件的 SHA256（full + 16 位短哈希）、
Diag1 `RUN_MANIFEST.json:code_identity.files` 的 12 文件基线、以及逐文件差异分类
（SAME / DIFF / NEW）。本地生成、服务器同脚本重算，两者必须逐条相同。

    python train_scripts/CASA-TViM/Run4/make_source_identity.py \
        --diag-manifest docs/temporary/CASA-TViM_Diag1/RUN_MANIFEST.json \
        --out train_scripts/CASA-TViM/Run4/SOURCE_IDENTITY.json
"""
import argparse
import datetime
import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))

# §8.2 明确要求核对的 7 个文件 + §5 改动清单涉及的其余文件
RUN4_FILES = [
    "models/model/casa_tvim_str_net.py",
    "models/model/str_fine_tap.py",
    "models/model/str_tar.py",
    "models/model/str_dcr.py",
    "models/model/str_reparam.py",
    "models/model/utils.py",
    "models/train.py",
    "models/eval.py",
    "models/test_run4_fine_tap.py",
    "models/run4_dry_run.py",
    "models/smoke_test.py",
    "models/model/tinyvim_s_slim.py",
    "models/model/layers/caacp_ss2d.py",
    "models/model/layers/ss2d.py",
    "models/model/metric_tool.py",
    "models/dataset/dataset.py",
    "models/dataset/Transforms.py",
    "analyse/tvim_run4_tap_preflight.py",
    "analyse/run4_fet_report.py",
]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diag-manifest",
                    default=os.path.join(PROJECT, "docs", "temporary",
                                         "CASA-TViM_Diag1", "RUN_MANIFEST.json"))
    ap.add_argument("--out", default=os.path.join(HERE, "SOURCE_IDENTITY.json"))
    ap.add_argument("--project", default=PROJECT)
    args = ap.parse_args()

    diag_files = {}
    if os.path.isfile(args.diag_manifest):
        with open(args.diag_manifest, encoding="utf-8") as f:
            diag_files = (json.load(f).get("code_identity") or {}).get("files", {})

    files, diff = {}, {}
    for rel in RUN4_FILES:
        p = os.path.join(args.project, rel.replace("/", os.sep))
        if not os.path.isfile(p):
            files[rel] = None
            diff[rel] = "MISSING"
            continue
        full = sha256(p)
        files[rel] = full
        old = diag_files.get(rel)
        if old is None:
            diff[rel] = "NEW (not in Diag1 baseline)"
        elif old == full:
            diff[rel] = "SAME"
        else:
            diff[rel] = f"DIFF vs Diag1 (diag={old[:16]} now={full[:16]})"

    payload = {
        "run": "Run4",
        "stage": "R4-FET1 source identity",
        "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        "project": args.project,
        "diag_manifest": os.path.relpath(args.diag_manifest, args.project)
        if args.diag_manifest.startswith(args.project) else args.diag_manifest,
        "files_sha256": files,
        "files_sha256_short16": {k: (v[:16] if v else None) for k, v in files.items()},
        "vs_diag1": diff,
        "note": ("Diag1 的 12 文件身份中，Run4 只改动 casa_tvim_str_net.py / train.py / eval.py，"
                 "新增 str_fine_tap.py，并因 exact-80K 分母修正触及 utils.py；其余 Diag1 文件逐字节相同。"
                 "历史 Diag1/Run1-Run3 产物不被覆盖。"),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {args.out}")
    for rel in RUN4_FILES:
        print(f"  {diff[rel]:45s} {rel}")


if __name__ == "__main__":
    main()
