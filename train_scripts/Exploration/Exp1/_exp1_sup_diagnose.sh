#!/usr/bin/env bash
# 看 STG_SUP/LEVIR 为何 blocks=0：日志尾部与异常行
set -u
L=/home/yqwang/outputs/CASA-CD/CASA-TViM/Exploration/Exp1/STG_SUP/LEVIR-CD-256/train_log.txt
echo "=== 文件大小 / 末尾 40 行 ==="
ls -la "$L"; echo
tail -40 "$L"
echo
echo "=== 异常行 ==="
grep -anE '\[RETRY|\[FAIL|Traceback|OutOfMemory|out of memory|Error' "$L" | tail -12
echo
echo "=== run_all 日志尾部 ==="
tail -8 /home/yqwang/outputs/CASA-CD/CASA-TViM/Exploration/Exp1/run_all_STG_SUP.log 2>/dev/null
echo
echo "=== ckpt ==="
ls -la /share_datasets/yqwang/checkpoints/CASA-CD/CASA-TViM/Exploration/Exp1/STG_SUP/LEVIR-CD-256/ 2>/dev/null
echo "=== 是否有 train_stgeo 在跑 ==="; pgrep -af "[t]rain_stgeo\.py" | cut -c1-80 || echo "(无)"
