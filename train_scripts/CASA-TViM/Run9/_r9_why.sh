#!/usr/bin/env bash
set -u
F=/home/yqwang/outputs/CASA-CD/CASA-TViM/Run9/S1_HSAM/LEVIR-CD-256/train_log.txt
echo "=== $(date '+%F %T')  文件 $F ($(stat -c%s "$F" 2>/dev/null) B) ==="
echo "--- 尾 40 行 ---"
tail -40 "$F" | sed 's/^/  /'
echo "--- 异常关键字 ---"
grep -nE "Traceback|Error|error|Killed|out of memory|OOM|FAIL|RETRY|CUDA|nan|NaN|inf" "$F" | tail -25 | sed 's/^/  /'
echo "--- RESUME/attempt 痕迹 ---"
grep -nE "RESUME|RETRY|attempt|FAILFAST|HSAM-FAILFAST" "$F" | tail -15 | sed 's/^/  /'
echo "--- 末尾的 epoch 行 ---"
grep -E "^Epoch " "$F" | tail -4 | sed 's/^/  /'
echo
echo "--- 检查所有上传脚本是否带 BOM ---"
for f in /home/yqwang/projects/CASA-CD/train_scripts/CASA-TViM/Run9/*.sh; do
  head -c 3 "$f" | od -An -tx1 | grep -q "ef bb bf" && echo "  BOM: $(basename "$f")"
done
echo "  (以上为空则无 BOM)"
echo
echo "--- 磁盘/内存 ---"
df -h /home/yqwang | tail -1 | sed 's/^/  /'
free -g | head -2 | sed 's/^/  /'
