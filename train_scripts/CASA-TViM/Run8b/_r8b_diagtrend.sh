#!/usr/bin/env bash
set -u
ROOT=/home/yqwang/outputs/CASA-CD/CASA-TViM/Run8b
L="$ROOT/B1_HYP/CDD-CD-256/train_log.txt"
echo "LOG=$L  ($(stat -c%s "$L" 2>/dev/null) bytes)"
echo "--- GRAD (step1) ---"
grep -h 'HQKV-GRAD' "$L" | head -2
echo "--- FAILFAST ---"
grep -h 'FAILFAST' "$L" | head -2
echo "--- HQKV-GAMMA / CONTRIB / PART-SHARE (TEST 块内) ---"
grep -h -E '^\[HQKV-(GAMMA|CONTRIB|PART-SHARE|GAMMA-INIT)\]' "$L" | tail -6
echo "--- DIAG 时间线（每 40 条取 1）---"
grep -h 'HQKV-DIAG' "$L" | awk 'NR%40==1' | sed 's/^\[HQKV-DIAG\] //' | cut -c1-120
echo "--- DIAG 最后 2 ---"
grep -h 'HQKV-DIAG' "$L" | tail -2 | cut -c1-160
echo "--- 计数 / DEPLOY ---"
echo "DIAG lines: $(grep -c 'HQKV-DIAG' "$L")"
grep -h -E '^\[DEPLOY' "$L" | tail -2
