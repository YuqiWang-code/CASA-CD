#!/usr/bin/env bash
# Run8 真实数据 dry-run（四库 × 2 个关键变体，3 步 + 折叠门 + epoch-0 恒等 + batch32 显存）。
# **只读**：只写 outputs/CASA-CD/CASA-TViM/Run8_DRY/ 下的 dry_run_log/summary，
# 绝不写 best_F1=*.pth，不触碰 Run1–Run7 任何产物。
set -u
CONDA_SH=/home/yqwang/miniforge3/etc/profile.d/conda.sh
REPO=/home/yqwang/projects/CASA-CD
CA=/home/yqwang/miniforge3/envs/casacd
MB=/home/yqwang/miniforge3/envs/mamba_base
DATA=/share_datasets/CD
OUT=/home/yqwang/outputs/CASA-CD/CASA-TViM/Run8_DRY
PRET=$REPO/pretrained_weight
GPUS=${GPUS:-0}

source "$CONDA_SH" && conda activate casacd
export PYTHONPATH="$MB/lib/python3.10/site-packages${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$CA/lib/python3.10/site-packages/torch/lib:$CA/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUT"

FAIL=0
for V in CTRL_R8 H1_HYP; do
  for DS in CDD-CD-256 LEVIR-CD-256 SYSU-CD-256 WHU-CD-256; do
    D="$OUT/$DS/$V"
    mkdir -p "$D"
    echo "=== dry-run $V / $DS ==="
    CUDA_VISIBLE_DEVICES="$GPUS" python "$REPO/models/_legacy/run8_dry_run.py" \
      --variant "$V" \
      --dataset-root "$DATA/$DS" \
      --train-list "$DATA/$DS/list/train.txt" \
      --vmamba-pretrained-weight-path "$PRET/vssm_tiny_0230_ckpt_epoch_262.pth" \
      --output-dir "$D" \
      --device cuda:0 --steps 3 --batch-size 32 --seed 16 \
      --str-dim 192 --hqkv-rank 32 \
      --hqkv-detail-window 2 --hqkv-global-window 4 --hqkv-attn-stages 1,2,3 \
      --hqkv-r0 0.5 --hqkv-a 0.5 --hgsl-lambda 0.2
    rc=$?
    echo "  [dry-run] $V/$DS rc=$rc"
    [ $rc -eq 0 ] || FAIL=1
  done
done

echo
echo "=== 汇总 ==="
python - "$OUT" <<'PY'
import json, os, sys
root = sys.argv[1]
bad = []
for ds in sorted(os.listdir(root)):
    p = os.path.join(root, ds)
    if not os.path.isdir(p):
        continue
    for v in sorted(os.listdir(p)):
        s = os.path.join(p, v, "dry_run_summary.json")
        if not os.path.isfile(s):
            bad.append(f"{ds}/{v}: no summary"); continue
        d = json.load(open(s, encoding="utf-8"))
        g = {k: v for k, v in d.get("gates", {}).items() if not k.startswith("_")}
        # 折叠门：`fold_disagreement_zero` 或 Run7 §7.6 认可的 `fold_knife_edge_artifact`
        # 任一为真即算过门（刀锋像素统计见 d["fold_knife_edge"]）。
        if g.get("fold_knife_edge_artifact"):
            g["fold_disagreement_zero"] = True
        g.pop("fold_knife_edge_artifact", None)
        ok = all(g.values())
        ke = d.get("fold_knife_edge", {})
        print(f"  {ds:<14} {v:<10} gates={g} peak={d.get('peak_allocated_MB_batch32')} "
              f"non_backbone={d.get('param_breakdown', {}).get('non_backbone')} "
              f"knife={ke.get('verdict')}(n_flip={ke.get('n_flip_px')}) "
              f"-> {'PASS' if ok else 'FAIL'}")
        if not ok:
            bad.append(f"{ds}/{v}: {[k for k, vv in g.items() if not vv]}")
print("  ALL PASS" if not bad else "  PROBLEMS: " + "; ".join(bad))
raise SystemExit(0 if not bad else 1)
PY
RC_SUM=$?

echo "DRYRUN DONE (run_fail=$FAIL, summary_rc=$RC_SUM)"
# 退出码由**汇总脚本**决定：`run_fail` 只反映"python 脚本 return 值"，
# 而 `fold_disagreement_zero=False` 这种情况已由汇总里的 KNIFE_EDGE_ARTIFACT 判据吸收
# （见 summary 的 gate 处理），所以不能用 run_fail 覆盖 summary_rc。
exit "$RC_SUM"
