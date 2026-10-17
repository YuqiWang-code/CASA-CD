#!/usr/bin/env bash
# 判断 Run8 的 H2_EUC 是否**同样自锁**（与 H1_HYP 对比）。
# 若同样自锁，则 Run8 的 H2/H3 继续跑没有科学价值（只能得到同一模型 ± 噪声）。
set -u
ROOT=/home/yqwang/outputs/CASA-CD/CASA-TViM/Run8
DS=CDD-CD-256

for V in H1_HYP H2_EUC; do
  F="$ROOT/$V/$DS/train_log.txt"
  echo "################ $V / $DS ################"
  if [ ! -f "$F" ]; then echo "  (无日志)"; continue; fi
  echo "  --- 最新进度 ---"
  grep -E '^\[ACTUAL-OPT-STEPS\] epoch=' "$F" | tail -1 | sed 's/^/    /'
  grep -E '^Epoch ' "$F" | tail -1 | sed 's/^/    /'
  echo "  --- 训练中每 epoch 的 HQKV-DIAG 末 2 次 ---"
  grep -E '^\[HQKV-DIAG-stage' "$F" | tail -2 | sed 's/^/    /'
  echo "  --- 若已完成：TEST 区块里的 gamma ---"
  awk '/^=== TEST RESULTS ===$/{buf="";inb=1;next}/^=== END TEST RESULTS ===$/{last=buf;inb=0;next}inb{buf=buf $0 "\n"}END{printf "%s",last}' "$F" \
    | grep -E '^(Recall=|\[HQKV-GAMMA\])' | sed 's/^/    /'
done

echo
echo "################ 结论辅助：H2 是否出现过非零 contrib_norm ################"
F="$ROOT/H2_EUC/$DS/train_log.txt"
if [ -f "$F" ]; then
  NZ=$(grep -E '^\[HQKV-DIAG-stage' "$F" | grep -c 'contrib_norm=0\.000000e+00')
  TT=$(grep -cE '^\[HQKV-DIAG-stage' "$F")
  echo "  H2: contrib_norm==0 的行数 $NZ / 总 DIAG 行数 $TT"
  grep -E '^\[HQKV-DIAG-stage' "$F" | grep -v 'contrib_norm=0\.000000e+00' | tail -3 | sed 's/^/    非零样本: /'
fi
