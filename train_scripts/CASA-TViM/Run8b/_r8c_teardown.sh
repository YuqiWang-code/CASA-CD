#!/usr/bin/env bash
# 拆除第二步：把已被归档的 3 个文件从**现行树**移除（服务器侧），并跑回归门。
# 前提：归档已在 train_scripts/CASA-TViM/Run8b/_SUPERSEDED_hqkv_v2_2026-10-20/，
#       且其 MANIFEST 与 SOURCE_SHA256.txt 逐项相等（已在本机核对）。
set -u
REPO=/home/yqwang/projects/CASA-CD
ARC="$REPO/train_scripts/CASA-TViM/Run8b/_SUPERSEDED_hqkv_v2_2026-10-20"
cd "$REPO" || exit 1

echo "=== 拆除：先验归档完整性（不完整就不删） $(date '+%F %T') ==="
echo "--- 逐文件：归档副本 vs MANIFEST 登记的原始路径哈希 ---"
ok=0; bad=0
while IFS= read -r line; do
  line=$(printf '%s' "$line" | tr -d '\r')
  [ -z "$line" ] && continue
  hash=$(printf '%s' "$line" | awk '{print $1}')
  path=$(printf '%s' "$line" | awk '{print $2}' | sed 's#\\#/#g')
  [ -z "$path" ] && continue
  base=$(basename "$path")
  if [ -f "$ARC/$base" ]; then
    got=$(sha256sum "$ARC/$base" | awk '{print $1}')
    if [ "$got" = "$hash" ]; then ok=$((ok+1)); echo "  OK   $base  ${got:0:16}"
    else bad=$((bad+1)); echo "  BAD  $base  归档=${got:0:16} 登记=${hash:0:16}"; fi
  else
    bad=$((bad+1)); echo "  MISS $base"
  fi
done < "$ARC/MANIFEST.sha256"
echo "  归档核对：OK=$ok BAD=$bad"
if [ "$bad" -ne 0 ] || [ "$ok" -eq 0 ]; then echo "  ⇒ 归档不完整，拒绝删除现行文件"; exit 3; fi

echo "--- 删除现行树里的 3 个已归档文件 ---"
for f in models/model/layers/caacp_hqkv_v2.py models/test_run8b_antilock_precheck.py models/run8b_dry_run.py; do
  if [ -f "$f" ]; then rm -f "$f" && echo "  removed $f"; else echo "  (已不在) $f"; fi
done
echo "--- 剩余引用检查（应为空）---"
grep -rn "caacp_hqkv_v2" --include=*.py models/ 2>/dev/null | sed 's/^/  /' || echo "  (无)"
echo "--- hyperbolic 的 live 引用（应只剩 v1 casa_vmamba_net / caacp_hqkv / Run8 自检）---"
grep -rln "layers.hyperbolic" --include=*.py models/ 2>/dev/null | sed 's/^/  /'

echo
echo "=== 回归门 ==="
cd "$REPO/models" || exit 1
source /home/yqwang/miniforge3/etc/profile.d/conda.sh
conda activate casacd
export PYTHONPATH="/home/yqwang/miniforge3/envs/mamba_base/lib/python3.10/site-packages:${PYTHONPATH:-}"
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader | sed 's/^/  /'
LATTICE_TEST_CUDA=1 python test_hqkv_lattice_cpu.py 2>&1 | grep -vE "UserWarning|warnings.warn" | tail -40
