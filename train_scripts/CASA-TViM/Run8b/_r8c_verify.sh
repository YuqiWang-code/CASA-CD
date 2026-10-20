#!/usr/bin/env bash
# 拆除后总验收（只读，不训练、不占 GPU 长时）。
set -u
REPO=/home/yqwang/projects/CASA-CD
ARC="$REPO/train_scripts/CASA-TViM/Run8b/_SUPERSEDED_hqkv_v2_2026-10-20"
cd "$REPO" || exit 1
source /home/yqwang/miniforge3/etc/profile.d/conda.sh
conda activate casacd
export PYTHONPATH="/home/yqwang/miniforge3/envs/mamba_base/lib/python3.10/site-packages:${PYTHONPATH:-}"

echo "=== 拆除总验收 $(date '+%F %T') ==="
echo
echo "1) 归档完整性（6 个文件，必须全 OK）"
ok=0; bad=0
while IFS= read -r line; do
  line=$(printf '%s' "$line" | tr -d '\r'); [ -z "$line" ] && continue
  h=$(printf '%s' "$line" | awk '{print $1}')
  p=$(printf '%s' "$line" | awk '{print $2}')
  b=$(basename "$p")
  got=$(sha256sum "$ARC/$b" 2>/dev/null | awk '{print $1}')
  if [ "$got" = "$h" ]; then ok=$((ok+1)); echo "   OK   $b"
  else bad=$((bad+1)); echo "   BAD  $b"; fi
done < "$ARC/MANIFEST.sha256"
echo "   归档：OK=$ok BAD=$bad"

echo
echo "2) 现行树里几何是否真的没了"
echo -n "   hyperbolic 的 live import 文件："
grep -rl "from model.layers.hyperbolic\|import hyperbolic" --include=*.py models/ 2>/dev/null \
  | grep -v "_SUPERSEDED" | tr '\n' ' '; echo
echo -n "   caacp_hqkv_v2 的引用："
grep -rl "caacp_hqkv_v2" --include=*.py models/ 2>/dev/null | grep -v "_SUPERSEDED" | tr '\n' ' '; echo
echo -n "   lattice 模块里的 geometry 关键字："
grep -c "lorentz\|hyperbolic\|manifold\|curvature" models/model/layers/caacp_hqkv_lattice.py || true

echo
echo "3) train.py CLI 冒烟（--help 必须能出来，且 choices 已更新）"
cd "$REPO/models" || exit 1
python train.py --help > /tmp/r8c_help.txt 2>&1; rc=$?
echo "   --help rc=$rc"
grep -A1 -- "--hqkv_geometry" /tmp/r8c_help.txt | head -3 | sed 's/^/   /'
grep -o "lattice" /tmp/r8c_help.txt | head -1 | sed 's/^/   choices 含 lattice: /'

echo
echo "4) 协议串 / form 标记"
grep -n "PROTOCOL_VERSION = " train.py | sed 's/^/   /'
grep -n "hqkv_v3_lattice" train.py | head -3 | sed 's/^/   /'

echo
echo "5) 回归门（27 项）"
LATTICE_TEST_CUDA=1 python test_hqkv_lattice_cpu.py 2>&1 \
  | grep -vE "UserWarning|warnings.warn" | grep -E "PASS|FAIL|SKIP|结果" | tail -32

echo
echo "6) GPU 现状（确认没留下任何训练进程）"
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader | sed 's/^/   /'
echo -n "   我们的 train.py: "; pgrep -f 'Run8b.*train\.py' | wc -l
