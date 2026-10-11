#!/usr/bin/env bash
# Run7 收口「一条命令」：等齐 12 个 VALID run → **逐 run 硬门** → 对象级只读复算 →
# 裁决 + 机制归因 → **独立最小复算审计** → 对照表。
#
# 为什么需要它：三波 80K 共约 11–13 h，训练与波形衔接已由 supervise_waves.sh 全自动完成；
# 收口必须在训练结束的那一刻能**无歧义地一条命令跑完**，不依赖"谁在场、记不记得跑哪几个脚本"。
#
#   bash train_scripts/CASA-TViM/Run7/finish_run7.sh
#
# 退出码：0 = 全部步骤完成（裁决已产出：MECHANISM-PASS / MECHANISM-FAIL / GUARDRAIL-FAIL）
#         2 = 尚有 run 未产出完整 TEST 区块（未跑完）—— 此时**不做**昂贵的对象级复算
#         3 = 环境/脚本缺失
#         4 = 逐 run 硬门（check_run.py）有 FAIL —— **不产出裁决**，避免假 PASS。
#             确认无碍后可用 RUN7_ALLOW_CHECKRUN_FAIL=1 显式越过，并在 README 记录原因。
#         5 = 独立审计未全过（n_failed > 0）
set -uo pipefail
source /home/yqwang/miniforge3/etc/profile.d/conda.sh
conda activate casacd || exit 3
export PROJ=/home/yqwang/projects/CASA-CD
export LD_LIBRARY_PATH=/home/yqwang/miniforge3/envs/casacd/lib/python3.10/site-packages/torch/lib:/home/yqwang/miniforge3/envs/casacd/lib:${LD_LIBRARY_PATH:-}
cd "$PROJ" || exit 3

# 路径与变体/库列表可由环境变量覆盖 —— 唯一目的是能在 /tmp 里**排练整条收口链**，
# 而不触碰正式产物（项目纪律：正式跑之前先 DRY 排练）。默认值与正式收口完全一致。
LOG_ROOT="${RUN7_LOG_ROOT:-/home/yqwang/outputs/CASA-CD/CASA-TViM/Run7}"
CKPT_ROOT="${RUN7_CKPT_ROOT:-/share_datasets/yqwang/checkpoints/CASA-CD/CASA-TViM/Run7}"
CKPT_PARENT="$(dirname "$CKPT_ROOT")"   # run7_object_metrics 的 --ckpt-root = <parent>/<run>
OBJ=$LOG_ROOT/_OBJECT_METRICS
RES=$LOG_ROOT/_RESULTS
mkdir -p "$OBJ" "$RES"
VARIANTS="${RUN7_VARIANTS:-M1_R7CTRL E9_HFAE_A E10_HFAE_B}"
DATASETS="${RUN7_DATASETS:-CDD-CD-256 LEVIR-CD-256 SYSU-CD-256 WHU-CD-256}"
V_CSV="$(echo $VARIANTS | tr ' ' ',')"
D_CSV="$(echo $DATASETS | tr ' ' ',')"
# 排练时高声告警：绝不能让排练结果被误当成正式收口结论
if [ "$LOG_ROOT" != "/home/yqwang/outputs/CASA-CD/CASA-TViM/Run7" ] \
   || [ "$CKPT_ROOT" != "/share_datasets/yqwang/checkpoints/CASA-CD/CASA-TViM/Run7" ]; then
  echo "################################################################"
  echo "###  [REHEARSAL] 非正式路径 —— 结果不得当作 Run7 正式结论   ###"
  echo "###  LOG_ROOT=$LOG_ROOT"
  echo "###  CKPT_ROOT=$CKPT_ROOT"
  echo "################################################################"
fi

echo "############ [1/6] 收口门（最后一个完整 TEST 区块 + 80000 步 + 协议串） ############"
# 用与后面各步**同一套** $VARIANTS/$DATASETS 过滤：正式收口时等价于"全部 12 个"，
# 但排练时才能只针对部分 run（否则门永远要求 12/12，排练无法走通）。
python train_scripts/CASA-TViM/Run7/collect_results.py --log-root "$LOG_ROOT" \
  --variants "$V_CSV" --datasets "$D_CSV" \
  --out "$RES/summary.json"
gate_rc=$?
python - "$RES/summary.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1], encoding='utf-8'))
print(f"  VALID {d['n_valid']}/{d['n_total']}")
for r in d['rows']:
    if r['status'] != 'VALID':
        print(f"  NOT READY: {r['variant']}/{r['dataset']} status={r['status']} "
              f"{r.get('invalid_reason', '')}")
PY
if [ "$gate_rc" -ne 0 ]; then
  echo "[FINISH] 尚有 run 未完成 -> 停止（不做对象级复算，避免在未齐时浪费 GPU）"
  exit 2
fi

echo "############ [1b/6] 折叠门：仅对 disagreement>0 的 run 生成「刀锋定位证据」 ############"
# 预注册 §7.6：真实 batch disagreement 必须为 0；若恰因某个像素落在阈值数值误差带内而翻转
# （|p−0.5| < 该次折叠最大绝对误差），则属数值不适定，**但必须**附定位证据才放行。
# 证据缺失或不吻合 => check_run / verdict / audit 三处都会 FAIL（fail-closed）。
KNIFE=$LOG_ROOT/_KNIFE_EDGE
mkdir -p "$KNIFE"
knife_n=0
for v in $VARIANTS; do
  for d in $DATASETS; do
    lg="$LOG_ROOT/$v/$d/train_log.txt"
    [ -f "$lg" ] || continue
    dis=$(grep -oE '\[REPARAM-REAL-ARGMAX-DISAGREE\][[:space:]]+[0-9.eE+-]+' "$lg" | tail -1 | awk '{print $2}')
    mxa=$(grep -oE '\[REPARAM-REAL-MAX-ABS-ERROR\][[:space:]]+[0-9.eE+-]+' "$lg" | tail -1 | awk '{print $2}')
    need=$(python3 -c "d='${dis:-}'
try:
    print(1 if float(d) > 0 else 0)
except Exception:
    print(0)")
    if [ "$need" = "1" ]; then
      echo "  [KNIFE] $v/$d disagreement=$dis maxabs=$mxa -> 生成定位证据"
      python analyse/run7_fold_knife_edge.py --variant "$v" --dataset "$d" \
        --ckpt-root "$CKPT_PARENT" --run Run7 \
        --log-fold-err-real "$mxa" --log-disagree-real "$dis" \
        --out "$KNIFE/KNIFE_EDGE_${d}_${v}.json" > "/tmp/r7_knife_${v}_${d}.log" 2>&1
      krc=$?
      echo "    rc=$krc -> $KNIFE/KNIFE_EDGE_${d}_${v}.json"
      [ "$krc" -eq 0 ] || echo "    [WARN] 证据生成失败；该 run 的折叠门将判 FAIL（fail-closed）"
      knife_n=$((knife_n + 1))
    fi
  done
done
echo "[FINISH] 需要刀锋证据的 run 数 = $knife_n"

echo "############ [2/6] 逐 run 硬门 check_run.py（变体契约 + 机制证据 + sidecar） ############"
cr_fail=0
for v in $VARIANTS; do
  for d in $DATASETS; do
    python train_scripts/CASA-TViM/Run7/check_run.py \
      --log "$LOG_ROOT/$v/$d/train_log.txt" --variant "$v" --dataset "$d" \
      --ckpt-dir "$CKPT_ROOT/$v/$d" --knife-dir "$KNIFE" \
      --json-out "$RES/check_run_${v}_${d}.json" --quiet
    rc=$?
    if [ "$rc" -ne 0 ]; then
      cr_fail=$((cr_fail + 1))
      echo "  [FAIL] $v/$d (rc=$rc) —— 详情："
      python train_scripts/CASA-TViM/Run7/check_run.py \
        --log "$LOG_ROOT/$v/$d/train_log.txt" --variant "$v" --dataset "$d" \
        --ckpt-dir "$CKPT_ROOT/$v/$d" --knife-dir "$KNIFE" 2>&1 | grep -E "^FAIL" | sed 's/^/      /'
    else
      echo "  [PASS] $v/$d"
    fi
  done
done
echo "[FINISH] check_run 未过的 run 数 = $cr_fail"
if [ "$cr_fail" -ne 0 ] && [ "${RUN7_ALLOW_CHECKRUN_FAIL:-0}" != "1" ]; then
  echo "[FINISH] 逐 run 硬门未过 -> **不产出裁决**（fail-closed，避免假 PASS）"
  echo "[FINISH] 若确认无碍，请用 RUN7_ALLOW_CHECKRUN_FAIL=1 显式越过并在 README 记录原因"
  exit 4
fi

echo "############ [3/6] 对象级只读复算（3 变体 × 4 库，用各自 best ckpt） ############"
python analyse/run7_object_metrics.py --run Run7 --all-variants \
  --variants "$V_CSV" --datasets "$D_CSV" --ckpt-root "$CKPT_PARENT" \
  --out "$OBJ" --device "${RUN7_DEVICE:-cuda:0}" \
  --batch-size "${RUN7_OBJ_BATCH:-16}" --num-workers "${RUN7_OBJ_WORKERS:-4}"
obj_rc=$?
echo "[FINISH] object metrics rc=$obj_rc"

echo "############ [4/6] 裁决 + 机制归因 ############"
python train_scripts/CASA-TViM/Run7/run7_verdict.py --log-root "$LOG_ROOT" \
  --object-dir "$OBJ" --knife-dir "$KNIFE" --out "$RES/verdict.json"
verdict_rc=$?

echo "############ [5/6] 独立最小复算审计（不同代码路径；A9 与 verdict 对拍） ############"
python train_scripts/CASA-TViM/Run7/independent_audit.py \
  --log-root "$LOG_ROOT" --ckpt-root "$CKPT_ROOT" --knife-dir "$KNIFE" \
  --verdict "$RES/verdict.json" --out "$RES/INDEPENDENT_AUDIT.json"
audit_rc=$?
echo "[FINISH] independent audit rc=$audit_rc"

echo "############ [6/6] 对象级跨变体对照表（含冻结参考） ############"
python analyse/run7_object_metrics.py --compare --out "$OBJ"

echo "############ [FINISH] 汇总 ############"
echo "  gate_rc=$gate_rc check_run_fail=$cr_fail object_rc=$obj_rc verdict_rc=$verdict_rc audit_rc=$audit_rc"
echo "  summary : $RES/summary.json"
echo "  verdict : $RES/verdict.json (+ .md)"
echo "  audit   : $RES/INDEPENDENT_AUDIT.json"
echo "  per-run : $RES/check_run_*.json"
echo "  objects : $OBJ/object_metrics_*.json"
echo "[FINISH] 完成；请把上面表格与 verdict 结论写入 Run7/README 与根 README"
if [ "$audit_rc" -ne 0 ]; then
  echo "[FINISH][WARN] 独立审计未全过 —— 结论必须按 INVALID 处理，先查因再引用裁决"
  exit 5
fi
exit 0
