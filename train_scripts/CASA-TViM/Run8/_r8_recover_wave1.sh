#!/usr/bin/env bash
# Run8 恢复脚本：把当前**真实**状态收敛到"两卡各跑一个 wave1 余量库"。
#
# ## 触发这个脚本的真实状态（2026-10-15 09:08 实测）
#   - CDD  ：**已完成**（80,000/80,000，有完整 TEST 区块，best_F1=0.9776）
#   - LEVIR：在 GPU0 正常跑（pid 3180554，27.5 GB）—— **不许动**
#   - SYSU ：编排器被守卫拒启（`ckpt 目录非空`，里面只有一个**孤儿** `run_manifest.json`，
#            是"启动了但没训起来"的那次留下的）
#   - WHU  ：未启动
#   - GPU1 ：**完全空闲**（15 MiB）
#   - 并行督导曾经把 SYSU/LEVIR 误判成要跑，两次都 rc=3 退出
#
# ## 本脚本做什么
#   1. 把 SYSU 的孤儿 `run_manifest.json` **改名封存**（`_SUPERSEDED_*`，绝不删除）
#      ⇒ 让 `_r8_run_one_ds.sh` 的"空目录"守卫放行；
#   2. 在 **GPU1** 起 `CTRL_R8/SYSU`（脱离会话）；
#   3. 在 **GPU0** 起 `CTRL_R8/WHU`（脱离会话）——
#      WHU 的 `train.py` 会先尝试占 27.5 GB 而 GPU0 只剩 4.9 GB ⇒
#      **它会自己 OOM 一次**，然后 `_r8_run_one_ds.sh` 的重试机制从自身 `last.pth` 续训。
#      ⚠️ 这会产生一次 OOM 噪声。因此**更好的做法是等 LEVIR 跑完再起 WHU**——
#      本脚本默认**不**起 WHU，只起 SYSU；WHU 交给并行督导在 GPU0 空出来后自动接。
#
# 用法：
#   bash _r8_recover_wave1.sh            # 只起 SYSU（推荐）
#   bash _r8_recover_wave1.sh --with-whu # 同时起 WHU（会先 OOM 一次）
set -u
REPO=/home/yqwang/projects/CASA-CD
RUN_DIR="$REPO/train_scripts/CASA-TViM/Run8"
STATE=/tmp/run8
CKPT_ROOT=/share_datasets/yqwang/checkpoints/CASA-CD/CASA-TViM/Run8
WITH_WHU=0
[ "${1:-}" = "--with-whu" ] && WITH_WHU=1

echo "=== Run8 wave1 恢复 $(date '+%F %T') ==="

echo "--- 1) 封存 SYSU 的孤儿 run_manifest.json ---"
SY="$CKPT_ROOT/CTRL_R8/SYSU-CD-256"
if [ -d "$SY" ]; then
  echo "  封存前内容: $(ls -A "$SY" | tr '\n' ' ')"
  for f in "$SY"/*; do
    [ -e "$f" ] || continue
    b=$(basename "$f")
    case "$b" in
      *_SUPERSEDED_*) echo "  SKIP $b（已封存）" ;;
      *) mv "$f" "${f}.SUPERSEDED_$(date +%Y%m%d_%H%M%S)_orphan_manifest"
         echo "  ARCHIVE $b -> $(basename "${f}.SUPERSEDED_$(date +%Y%m%d_%H%M%S)_orphan_manifest")" ;;
    esac
  done
  echo "  封存后内容: [$(ls -A "$SY" | tr '\n' ' ')]  （应为空）"
else
  echo "  (目录不存在，跳过)"
fi

echo "--- 2) 在 GPU1 起 CTRL_R8/SYSU ---"
if pgrep -f '_r8_run_one_ds.sh CTRL_R8 SYSU-CD-256' >/dev/null; then
  echo "  已在跑，跳过"
else
  setsid nohup bash "$RUN_DIR/_r8_run_one_ds.sh" CTRL_R8 SYSU-CD-256 1 \
    > "$STATE/run_CTRL_R8_SYSU-CD-256.log" 2>&1 < /dev/null &
  echo "  SYSU pid=$!"
fi

if [ "$WITH_WHU" = "1" ]; then
  echo "--- 2b) 在 GPU0 起 CTRL_R8/WHU（预期先 OOM 一次）---"
  if pgrep -f '_r8_run_one_ds.sh CTRL_R8 WHU-CD-256' >/dev/null; then
    echo "  已在跑，跳过"
  else
    setsid nohup bash "$RUN_DIR/_r8_run_one_ds.sh" CTRL_R8 WHU-CD-256 0 \
      > "$STATE/run_CTRL_R8_WHU-CD-256.log" 2>&1 < /dev/null &
    echo "  WHU pid=$!"
  fi
fi

sleep 20
echo "--- 3) 恢复后状态 ---"
bash "$RUN_DIR/_r8_state.sh" 2>&1 | sed -n '1,40p'
