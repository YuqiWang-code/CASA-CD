#!/usr/bin/env bash
# 全面探查：GPU / Run8 / Exp1 / 吞吐 / 机制读数（只读）
set -u
echo "=== TIME ==="; date '+%F %T %z'
echo "=== GPU ==="
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader
echo "=== PROC ==="
echo "--- Run8 (GPU0) ---"; pgrep -af "[t]rain\.py --dataset" | sed 's/ --dataset_root.*//' | head -3
echo "--- Exp1 (GPU1) ---"; pgrep -af "[t]rain_stgeo\.py --dataset" | sed 's/ --dataset_root.*//' | head -3

echo
echo "=== RUN8 PROGRESS ==="
for v in CTRL_R8 H1_HYP H2_EUC H3_HYP_GS; do
  for d in CDD-CD-256 LEVIR-CD-256 SYSU-CD-256 WHU-CD-256; do
    L="/home/yqwang/outputs/CASA-CD/CASA-TViM/Run8/$v/$d/train_log.txt"
    [ -f "$L" ] || continue
    st=$(grep -a '^\[ACTUAL-OPT-STEPS\] epoch=' "$L" | tail -1 | sed 's/.*cumulative=//')
    ep=$(grep -a '^Epoch ' "$L" | tail -1 | awk '{print $2}')
    f1=$(grep -a '^Epoch ' "$L" | tail -1 | sed 's/.*F1=//;s/ |.*//')
    bl=$(grep -ac '=== END TEST RESULTS ===' "$L")
    printf '  %-9s %-13s steps=%-13s ep=%-6s F1=%-7s blocks=%s\n' "$v" "$d" "${st:-NA}" "${ep:-NA}" "${f1:-NA}" "$bl"
  done
done
echo "--- Run8 supervisor ---"
for f in /tmp/run8/pipeline_single.pid; do
  [ -f "$f" ] && { p=$(cat $f); kill -0 $p 2>/dev/null && echo "  ALIVE pid=$p" || echo "  DEAD pid=$p"; }
done
tail -3 /tmp/run8/pipeline_single.log 2>/dev/null

echo
echo "=== EXP1 PROGRESS ==="
EXP=/home/yqwang/outputs/CASA-CD/CASA-TViM/Exploration/Exp1
for d in LEVIR-CD-256 SYSU-CD-256; do
  L="$EXP/STG_INJ/$d/train_log.txt"
  if [ -f "$L" ]; then
    st=$(grep -a '^\[ACTUAL-OPT-STEPS\] epoch=' "$L" | tail -1 | sed 's/.*cumulative=//')
    bl=$(grep -ac '=== END TEST RESULTS ===' "$L")
    echo "  --- $d steps=$st blocks=$bl ---"
    grep -a '^Epoch ' "$L" | tail -4
    echo "  最近 4 个 epoch 耗时(s): $(grep -a '^Epoch ' "$L" | tail -4 | sed 's/.*t=//;s/s$//' | tr '\n' ' ')"
    echo "  best: $(grep -a '^\[BEST\]' "$L" | tail -1)"
    grep -acE '\[RETRY|\[FAIL|Traceback|CUDA out of memory' "$L" | sed 's/^/  异常行数: /'
  else
    echo "  --- $d (no log yet) ---"
  fi
done
tail -4 "$EXP/run_all.log" 2>/dev/null

echo
echo "=== EXP1 机制读数（从 last.pth 直接读 γ，只读）==="
python - <<'PY' 2>/dev/null || echo "  (读取失败/无 ckpt)"
import glob, torch
for p in sorted(glob.glob("/share_datasets/yqwang/checkpoints/CASA-CD/CASA-TViM/Exploration/Exp1/STG_INJ/*/last.pth")):
    try:
        ck = torch.load(p, map_location="cpu", weights_only=False)
        sd = ck["state_dict"]
        g = float(sd["stgeo_op.gamma"]) if "stgeo_op.gamma" in sd else None
        bn = float(sd["stgeo_op.bn.running_var"].mean()) if "stgeo_op.bn.running_var" in sd else None
        w = float(sd["stgeo_op.out.weight"].abs().mean()) if "stgeo_op.out.weight" in sd else None
        print(f"  {p.split('/')[-2]}: steps={ck.get('actual_steps')} gamma={g:.6e} "
              f"|out.w|mean={w:.4e} bn_var_mean={bn:.4e} best_f1={ck.get('best_f1'):.4f}")
    except Exception as e:
        print(f"  {p.split('/')[-2]}: 读取异常 {type(e).__name__}")
PY
echo "=== DONE ==="
