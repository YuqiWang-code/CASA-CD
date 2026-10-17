#!/usr/bin/env bash
# Run8b 真实数据 dry-run（四库 × 2 变体，3 步 + 抗自锁证据 + 折叠门 + batch32 显存）。
# **只读**：只写 outputs/CASA-CD/CASA-TViM/Run8b_DRY/；绝不写 best_F1=*.pth，不碰 Run8 产物。
set -u
CONDA_SH=/home/yqwang/miniforge3/etc/profile.d/conda.sh
REPO=/home/yqwang/projects/CASA-CD
CA=/home/yqwang/miniforge3/envs/casacd
MB=/home/yqwang/miniforge3/envs/mamba_base
DATA=/share_datasets/CD
OUT=/home/yqwang/outputs/CASA-CD/CASA-TViM/Run8b_DRY
PRET=$REPO/pretrained_weight
GPUS=${GPUS:-0}

source "$CONDA_SH" && conda activate casacd
export PATH="$CA/bin:${PATH}"
export PYTHONPATH="$MB/lib/python3.10/site-packages${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$CA/lib/python3.10/site-packages/torch/lib:$CA/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUT"

FAIL=0
for V in CTRL_R8B B1_HYP; do
  for DS in CDD-CD-256 LEVIR-CD-256 SYSU-CD-256 WHU-CD-256; do
    D="$OUT/$DS/$V"; mkdir -p "$D"
    echo "=== dry-run $V / $DS ==="
    CUDA_VISIBLE_DEVICES="$GPUS" python "$REPO/models/run8b_dry_run.py" \
      --variant "$V" \
      --dataset-root "$DATA/$DS" \
      --train-list "$DATA/$DS/list/train.txt" \
      --vmamba-pretrained-weight-path "$PRET/vssm_tiny_0230_ckpt_epoch_262.pth" \
      --output-dir "$D" \
      --device cuda:0 --steps 3 --batch-size 32 --seed 16 \
      --str-dim 192 --hqkv-rank 32 \
      --hqkv-detail-window 2 --hqkv-global-window 4 --hqkv-attn-stages 1,2,3 \
      --hqkv-r0 0.5 --hqkv-a 0.5 --hqkv-gamma-init 0.5 --hgsl-lambda 0.2
    rc=$?
    echo "  [dry-run] $V/$DS rc=$rc"
    [ $rc -eq 0 ] || FAIL=1
  done
done

echo
echo "=== 汇总（抗自锁是核心判据）==="
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
        g = {k: vv for k, vv in d.get("gates", {}).items() if not k.startswith("_")}
        ok = all(g.values())
        f = d.get("fold", {})
        if g.get("fold_gate") is False and f.get("verdict") == "KNIFE_EDGE_ARTIFACT":
            g["fold_gate"] = True
            ok = all(g.values())
        cn = d.get("contrib_norm_trace") or []
        rms = d.get("epoch0_rms_estimate")
        print(f"  {ds:<14} {v:<10} gates_ok={ok} "
              f"contrib={cn[-1] if cn else 0:.3e} rms={rms if rms is None else round(rms,4)} "
              f"knife={f.get('verdict')} peak={d.get('peak_allocated_MB_batch32')} "
              f"non_backbone={d.get('param_breakdown', {}).get('non_backbone')}")
        if not ok:
            bad.append(f"{ds}/{v}: {[k for k, vv in g.items() if not vv]}")
print("  ALL PASS" if not bad else "  PROBLEMS: " + "; ".join(bad))
raise SystemExit(0 if not bad else 1)
PY
RC_SUM=$?

echo "DRYRUN DONE (run_fail=$FAIL, summary_rc=$RC_SUM)"
exit "$RC_SUM"
