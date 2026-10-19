#!/usr/bin/env bash
# 让出 GPU1：停我们的训练 + 记录可续训的精确状态（绝不删除任何产物）
set -u
STAMP=$(date +%Y%m%d_%H%M%S)
EXP=/home/yqwang/outputs/CASA-CD/CASA-TViM/Exploration/Exp1
CK=/share_datasets/yqwang/checkpoints/CASA-CD/CASA-TViM/Exploration/Exp1
STATE=$EXP/_PAUSED_$STAMP
mkdir -p "$STATE"

echo "=== 0) 时间 ==="; date '+%F %T %z'
echo
echo "=== 1) 停我们的进程（只匹配本探索线）==="
pkill -f "Exploration/Exp1/run_all\.sh" && echo "  killed run_all" || echo "  run_all 不在运行"
pkill -f "[t]rain_stgeo\.py" && echo "  killed train_stgeo" || echo "  train_stgeo 不在运行"
sleep 6
pkill -9 -f "[t]rain_stgeo\.py" 2>/dev/null || true
sleep 3

echo
echo "=== 2) 残留与归属检查 ==="
echo "--- 我们自己的训练进程（应为空）---"
pgrep -af "[t]rain_stgeo\.py" | cut -c1-90 || echo "  （无残留）"
echo "--- GPU0 上的 Run8b（另一对话，必须仍在）---"
pgrep -af "[t]rain\.py --dataset" | sed 's/ --dataset_root.*//' | head -3
echo "--- 等卡督导（必须已死，否则会抢卡）---"
pgrep -af "wait_and_launch" | cut -c1-90 || echo "  （无）"
echo "--- pid 文件 ---"; ls -1 /tmp/exp1/*.pid 2>/dev/null || echo "  （无）"
echo "--- GPU ---"; nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader

echo
echo "=== 3) 冻结各臂状态 ==="
{
  echo "# Exp1 暂停状态快照"
  echo
  echo "暂停时间：$(date '+%F %T %z')"
  echo "原因：GPU1 需让给同门作业（用户 2026-10-19 指令）；GPU0 的 Run8b 不受影响。"
  echo "纪律：本脚本**只停进程、只记录**，未删除任何日志/权重。"
  echo
  echo "| 臂 | 数据集 | 严格步数 | 完整 TEST 区块 | best_F1 | 状态 |"
  echo "|---|---|---:|---:|---:|---|"
  for v in STG_INJ STG_V2 STG_SUP STG_REL STG_REL_EUC; do
    for d in CDD-CD-256 LEVIR-CD-256 SYSU-CD-256 WHU-CD-256; do
      L="$EXP/$v/$d/train_log.txt"
      [ -f "$L" ] || continue
      st=$(grep -a '^\[ACTUAL-OPT-STEPS\] epoch=' "$L" | tail -1 | sed 's/.*cumulative=//')
      bl=$(grep -ac '=== END TEST RESULTS ===' "$L")
      bf=$(grep -a '^\[BEST\] F1=' "$L" | tail -1 | sed 's/.*F1=//;s/ .*//')
      if [ "${bl:-0}" -ge 1 ]; then stt="已完成"; else stt="**未完成（可续训）**"; fi
      printf '| %s | %s | %s | %s | %s | %s |\n' "$v" "$d" "${st:-NA}" "${bl:-0}" "${bf:-NA}" "$stt"
    done
  done
  echo
  echo "## 续训所需信息"
  for v in STG_REL; do
    for d in LEVIR-CD-256 SYSU-CD-256; do
      L="$EXP/$v/$d/train_log.txt"
      [ -f "$L" ] || continue
      echo "### $v / $d"
      echo "- 日志：\`$L\`（$(stat -c%s "$L" 2>/dev/null || echo '?') 字节）"
      echo "- 权重目录：\`$CK/$v/$d/\`"
      ls -la "$CK/$v/$d/" 2>/dev/null | sed 's/^/    /'
      echo "- 最新进度行：$(grep -a '^\[ACTUAL-OPT-STEPS\]' "$L" | tail -1)"
      echo "- 最新 epoch 行：$(grep -a '^Epoch ' "$L" | tail -1)"
      echo "- best：$(grep -a '^\[BEST\]' "$L" | tail -1)"
      echo
    done
  done
  echo "## 恢复命令（GPU1 空出后再执行）"
  echo '```bash'
  echo '# 1) 先把未完成曲线改名封存（避免两段曲线混在同一文件里）'
  echo 'L=/home/yqwang/outputs/CASA-CD/CASA-TViM/Exploration/Exp1/STG_REL/LEVIR-CD-256/train_log.txt'
  echo 'mv "$L" "${L}.SEG1_$(date +%Y%m%d_%H%M%S)"'
  echo '# 2) 认证同 run 续训（train_stgeo 会从自己的 last.pth 恢复 model/optimizer/4 项 RNG/actual_steps）'
  echo 'RESUME_SAME_RUN=1 bash train_scripts/Exploration/Exp1/run_one.sh STG_REL LEVIR-CD-256 1'
  echo '# 3) LEVIR 完成后接 SYSU'
  echo 'bash train_scripts/Exploration/Exp1/_exp1_launch_arm.sh STG_REL 1 SYSU-CD-256'
  echo '```'
  echo
  echo "## 说明"
  echo "- \`last.pth\` 含 optimizer + 四项 RNG + actual_steps + protocol_version ⇒ 可**协议级**续训（非逐位，与 Run4/Run7 的既有结论一致）。"
  echo "- 续训前务必把未完成日志改名（本文件已给出命令）；否则正式指标仍只取**最后一个完整 TEST 区块**，但曲线会混段。"
  echo "- 恢复后仍按冻结判据 A-P1..A-P5 裁决，**不得**因为中断而放宽阈值。"
} > "$STATE/STATE.md"

echo
echo "=== 4) 快照写入 ==="
cp "$STATE/STATE.md" "$EXP/_LAST_PAUSED_STATE.md"
echo "  $STATE/STATE.md（并复制到 $EXP/_LAST_PAUSED_STATE.md）"
echo
echo "=== 5) 最终 GPU 状态（GPU1 应交还给同门）==="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
echo
echo "=== PAUSE DONE ==="
sed -n '1,40p' "$STATE/STATE.md"
