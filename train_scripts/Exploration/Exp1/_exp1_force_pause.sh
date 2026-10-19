#!/usr/bin/env bash
# 彻底让出 GPU1：先杀「重试父进程」，再杀 trainer，并连续多轮确认（重试间隔 10 s，必须熬过它）
set -u
STAMP=$(date +%Y%m%d_%H%M%S)
EXP=/home/yqwang/outputs/CASA-CD/CASA-TViM/Exploration/Exp1
CK=/share_datasets/yqwang/checkpoints/CASA-CD/CASA-TViM/Exploration/Exp1
STATE=$EXP/_PAUSED_FORCED_$STAMP
mkdir -p "$STATE"

say(){ echo "[$(date '+%H:%M:%S')] $*"; }
count_ours(){ pgrep -af "[t]rain_stgeo\.py" 2>/dev/null | wc -l; }
gpu1_apps(){
  local u; u=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' '$1==1{print $2}')
  nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | awk -F', ' -v u="$u" '$1==u{c++} END{print c+0}'
}

say "第 1 步：先杀**重试父进程**（run_one.sh / run_all.sh / 任何 _exp1_launch）"
pkill -9 -f "Exploration/Exp1/run_one\.sh" && say "  killed run_one.sh" || say "  run_one.sh 不在"
pkill -9 -f "Exploration/Exp1/run_all\.sh" && say "  killed run_all.sh" || say "  run_all.sh 不在"
pkill -9 -f "_exp1_launch_arm\.sh"         && say "  killed launch_arm" || say "  launch_arm 不在"
pkill -9 -f "_exp1_wait_and_launch\.sh"    && say "  killed wait_and_launch" || say "  wait_and_launch 不在"

say "第 2 步：杀 trainer，并连续 8 轮（每轮 6 s > 重试 sleep 10 s 的一半×2）确认不复活"
round=0; stable=0
while [ $round -lt 8 ]; do
  round=$((round+1))
  n=$(count_ours)
  if [ "$n" -gt 0 ]; then
    pkill -9 -f "[t]rain_stgeo\.py" 2>/dev/null || true
    say "  round $round: 发现并杀掉 $n 个 train_stgeo"
    stable=0
    sleep 6
  else
    say "  round $round: 无 train_stgeo（stable=$stable）"
    stable=$((stable+1))
    sleep 6
    [ $stable -ge 3 ] && break     # 连续 3 轮（≈18 s）为空才收手，足以熬过 10 s 重试
  fi
done

say "第 3 步：最终核实"
echo "  --- 我们的进程 ---"
pgrep -af "[t]rain_stgeo\.py" | cut -c1-80 || echo "    （无）"
pgrep -af "Exploration/Exp1" | cut -c1-80 || echo "    （无 Exp1 脚本）"
echo "  --- GPU1 compute apps（应为 0）---"
echo "    apps_on_gpu1=$(gpu1_apps)"
echo "  --- GPU ---"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
echo "  --- GPU0 的 Run8b（必须仍在）---"
pgrep -af "[t]rain\.py --dataset" | sed 's/ --dataset_root.*//' | head -2

say "第 4 步：重新冻结状态（步数可能已被续训推进）"
{
  echo "# Exp1 强制暂停快照（第一次暂停被训练脚本的 10 s 重试循环复活，故二次强制）"
  echo
  echo "时间：$(date '+%F 5T%H:%M:%S' 2>/dev/null || date '+%F %T %z')"
  echo "教训：**只杀 trainer 不够** —— run_one.sh 的 \`for attempt in 1..3\` 会在 10 s 后从 last.pth 重启。"
  echo "正确顺序：先杀 run_one/run_all 这些**重试父进程**，再杀 trainer，并连续 3 轮（≈18 s）确认为空。"
  echo
  echo "| 臂 | 数据集 | 严格步数 | 完整 TEST | best_F1 | 状态 |"
  echo "|---|---|---:|---:|---:|---|"
  for v in STG_INJ STG_V2 STG_SUP STG_REL STG_REL_EUC; do
    for d in CDD-CD-256 LEVIR-CD-256 SYSU-CD-256 WHU-CD-256; do
      L="$EXP/$v/$d/train_log.txt"; [ -f "$L" ] || continue
      st=$(grep -a '^\[ACTUAL-OPT-STEPS\] epoch=' "$L" | tail -1 | sed 's/.*cumulative=//')
      bl=$(grep -ac '=== END TEST RESULTS ===' "$L")
      bf=$(ls -1 "$CK/$v/$d/"best_F1=*.pth 2>/dev/null | head -1 | sed 's/.*best_F1=//;s/\.pth//')
      if [ "${bl:-0}" -ge 1 ]; then stt="已完成"; else stt="**未完成（可续训）**"; fi
      printf '| %s | %s | %s | %s | %s | %s |\n' "$v" "$d" "${st:-NA}" "${bl:-0}" "${bf:-NA}" "$stt"
    done
  done
  echo
  echo "## 续训（GPU1 空出后）"
  echo '```bash'
  echo 'L=/home/yqwang/outputs/CASA-CD/CASA-TViM/Exploration/Exp1/STG_REL/LEVIR-CD-256/train_log.txt'
  echo 'mv "$L" "${L}.SEG1_$(date +%Y%m%d_%H%M%S)"'
  echo 'RESUME_SAME_RUN=1 bash train_scripts/Exploration/Exp1/run_one.sh STG_REL LEVIR-CD-256 1'
  echo '```'
} > "$STATE/STATE.md"
cp "$STATE/STATE.md" "$EXP/_LAST_PAUSED_STATE.md"
say "快照：$STATE/STATE.md"
cat "$STATE/STATE.md"
say "DONE"
