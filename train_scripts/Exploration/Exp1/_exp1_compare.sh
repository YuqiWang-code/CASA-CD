#!/usr/bin/env bash
# 取 CTRL_R8（Run8 波一）的正式 TEST 区块，与 Exp1 同源对比（最后完整区块）
set -u
R8=/home/yqwang/outputs/CASA-CD/CASA-TViM/Run8
for d in LEVIR-CD-256 SYSU-CD-256; do
  echo "########## CTRL_R8 / $d ##########"
  awk '/=== TEST RESULTS ===/{f=1} f{print} /=== END TEST RESULTS ===/{f=0}' \
      "$R8/CTRL_R8/$d/train_log.txt" | grep -E "^(Recall=|\[BEST-F1\]|\[ACTUAL-OPT-STEPS\]|\[PROTOCOL-VERSION\]|\[REPARAM-REAL-ARGMAX|\[DEPLOY-PARAMS\])"
  echo
done
echo "########## 同源对照表 ##########"
printf '%-16s %-10s %-10s %-10s %-10s\n' dataset CTRL_R8 STG_INJ delta_pp verdict
python3 - <<'PY'
import re
base = "/home/yqwang/outputs/CASA-CD/CASA-TViM"
def last_block(p):
    try:
        txt = open(p, encoding="utf-8", errors="replace").read()
    except FileNotFoundError:
        return None
    b = re.findall(r"=== TEST RESULTS ===(.*?)=== END TEST RESULTS ===", txt, re.S)
    if not b:
        return None
    m = re.search(r"Recall=([\d.]+) \| Precision=([\d.]+) \| OA=([\d.]+) \| F1=([\d.]+) \| IoU=([\d.]+) \| Kappa=([\d.]+)", b[-1])
    if not m:
        return None
    return dict(zip(["R", "P", "OA", "F1", "IoU", "Kappa"], [float(x) for x in m.groups()]))
for ds in ["LEVIR-CD-256", "SYSU-CD-256"]:
    c = last_block(f"{base}/Run8/CTRL_R8/{ds}/train_log.txt")
    e = last_block(f"{base}/Exploration/Exp1/STG_INJ/{ds}/train_log.txt")
    if c and e:
        d = (e["F1"] - c["F1"]) * 100
        print(f"{ds:<16} {c['F1']:.4f}     {e['F1']:.4f}     {d:+.2f}      "
              f"{'PASS' if (e['F1'] >= c['F1'] and False) else ''}"
              f"{'IMPROVED' if d > 0 else ('TIE' if abs(d) < 1e-9 else 'WORSE')}")
        print(f"   CTRL: R={c['R']:.4f} P={c['P']:.4f} IoU={c['IoU']:.4f}")
        print(f"   EXP1: R={e['R']:.4f} P={e['P']:.4f} IoU={e['IoU']:.4f}")
PY
