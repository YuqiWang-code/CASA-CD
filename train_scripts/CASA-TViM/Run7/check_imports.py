#!/usr/bin/env python
"""Run7 接线自检：`model/str_hf_edge.py` **模块级定义的常量** 在各接线文件里使用时是否都已 import。

存在的理由（真实事故）：`models/train.py` 曾漏 import `HF_EDGE_FORM`，而该名字只在
**写 sidecar**（arch.json / run_manifest.json / last.pth）时才被求值 —— 因此
`models/test_run7_hf_edge.py`（直接构造模型）与 `models/run7_dry_run.py`（不走 manifest）
**都测不到**，只有在真实的 `train.py` 上跑一步才会暴露。
波一（CTRL, hf_edge=0）同样不走该分支，所以本检查必须在开波二之前跑。

只用**模块级赋值**定义的 UPPER 名字作为常量集合（避免把 `import torch.nn.functional as F`
这类导入名误判为常量）。

    python train_scripts/CASA-TViM/Run7/check_imports.py
"""
import ast
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
MOD = os.path.join(ROOT, "models", "model", "str_hf_edge.py")
TARGETS = ["models/train.py", "models/eval.py", "models/model/casa_tvim_str_net.py",
           "models/run7_dry_run.py", "models/test_run7_hf_edge.py"]


def module_constants(path):
    """只取模块级 `NAME = ...` / `NAME: T = ...` 且 NAME 全大写的赋值目标。"""
    tree = ast.parse(open(path, encoding="utf-8").read())
    out = set()
    for node in tree.body:
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        for t in targets:
            if isinstance(t, ast.Name) and t.id.isupper():
                out.add(t.id)
    return out


def main():
    exports = module_constants(MOD)
    print(f"str_hf_edge module-level constants: {sorted(exports)}")
    bad = 0
    for rel in TARGETS:
        p = os.path.join(ROOT, rel.replace("/", os.sep))
        if not os.path.isfile(p):
            print(f"  [skip] {rel} absent")
            continue
        tree = ast.parse(open(p, encoding="utf-8").read())
        imported = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.ImportFrom) and n.module and "str_hf_edge" in n.module:
                for a in n.names:
                    imported.add(a.asname or a.name)
        used = {n.id for n in ast.walk(tree)
                if isinstance(n, ast.Name) and n.id in exports}
        missing = sorted(used - imported)
        ok = not missing
        bad += 0 if ok else 1
        print(f"  [{'OK' if ok else 'FAIL'}] {rel}: used={sorted(used)}"
              + ("" if ok else f"  MISSING_IMPORT={missing}"))
    print(f"IMPORT CHECK = {'PASS' if bad == 0 else f'FAIL ({bad} file(s))'}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
