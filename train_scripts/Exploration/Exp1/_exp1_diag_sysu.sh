#!/usr/bin/env bash
# 诊断 STG_SUP/SYSU 收尾崩溃
set -u
L=/home/yqwang/outputs/CASA-CD/CASA-TViM/Exploration/Exp1/STG_SUP/SYSU-CD-256/train_log.txt
echo "=== 末尾 30 行 ==="; tail -30 "$L"
echo
echo "=== 异常行（含行号）==="
grep -anE 'Traceback|Error|\[RETRY|\[FAIL' "$L" | tail -10
echo
echo "=== 服务端 train_stgeo.py 的 TEST 区块那行（确认补丁是否生效）==="
grep -n "STGEO\] {args.stgeo}" /home/yqwang/projects/CASA-CD/models_v1/train_stgeo.py
echo
echo "=== ckpt ==="; ls -la /share_datasets/yqwang/checkpoints/CASA-CD/CASA-TViM/Exploration/Exp1/STG_SUP/SYSU-CD-256/ 2>/dev/null
