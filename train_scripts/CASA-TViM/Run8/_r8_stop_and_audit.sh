#!/usr/bin/env bash
# 停掉 Run8 波次并盘点残留（只读盘点 + 精确 kill；不删任何产物）。
set -u
pkill -f 'run_all.sh --wave' 2>/dev/null
sleep 2
for p in $(pgrep -f '^python train\.py' 2>/dev/null); do kill -TERM "$p" 2>/dev/null; done
sleep 3
echo "--- 残留进程 ---"
pgrep -af 'run_all\.sh|python train\.py' || echo "  (无)"
echo "--- CTRL_R8 各库目录内容 ---"
for d in CDD-CD-256 LEVIR-CD-256 SYSU-CD-256 WHU-CD-256; do
  D="/home/yqwang/outputs/CASA-CD/CASA-TViM/Run8/CTRL_R8/$d"
  echo "  $d: $(ls -A "$D" 2>/dev/null | wc -l) files"
  ls -la "$D" 2>/dev/null | tail -n +2 | sed 's/^/      /'
done
echo "--- ckpt 目录（必须为空，否则 run_all.sh 会拒绝启动）---"
for d in CDD-CD-256 LEVIR-CD-256 SYSU-CD-256 WHU-CD-256; do
  C="/share_datasets/yqwang/checkpoints/CASA-CD/CASA-TViM/Run8/CTRL_R8/$d"
  echo "  $d: $(ls -A "$C" 2>/dev/null | wc -l) files"
done
echo "--- GPU ---"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
