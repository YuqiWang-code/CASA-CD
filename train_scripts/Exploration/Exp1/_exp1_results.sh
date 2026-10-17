#!/usr/bin/env bash
# 取 Exp1 的正式 TEST 区块 + γ 机制读数 + 对象级小目标指标所需信息
set -u
EXP=/home/yqwang/outputs/CASA-CD/CASA-TViM/Exploration/Exp1/STG_INJ
CK=/share_datasets/yqwang/checkpoints/CASA-CD/CASA-TViM/Exploration/Exp1/STG_INJ
for d in LEVIR-CD-256 SYSU-CD-256; do
  echo "############ $d 正式 TEST 区块 ############"
  awk '/=== TEST RESULTS ===/{f=1} f{print} /=== END TEST RESULTS ===/{f=0}' "$EXP/$d/train_log.txt" | tail -30
  echo
  echo "--- $d 曲线摘要（第1/中间/最后）---"
  grep -a '^Epoch ' "$EXP/$d/train_log.txt" | sed -n '1p;$p'
  grep -a '^Epoch ' "$EXP/$d/train_log.txt" | awk -F'F1=' '{split($2,a," "); print a[1]}' | sort -g | tail -1 | sed 's/^/  max-F1(曲线内)=/'
  echo
done
echo "############ ckpt / γ ############"
ls -la "$CK"/LEVIR-CD-256/ "$CK"/SYSU-CD-256/ 2>&1 | head -20
python3 - "$CK" <<'PY'
import sys, glob, torch
root = sys.argv[1]
for p in sorted(glob.glob(root + "/*/last.pth")):
    ds = p.split("/")[-2]
    try:
        ck = torch.load(p, map_location="cpu", weights_only=False)
        sd = ck["state_dict"]
        keys = [k for k in sd if k.startswith("stgeo_op.")]
        g = float(sd["stgeo_op.gamma"].reshape(-1)[0]) if "stgeo_op.gamma" in sd else None
        wm = float(sd["stgeo_op.out.weight"].abs().mean()) if "stgeo_op.out.weight" in sd else None
        bnv = float(sd["stgeo_op.bn.running_var"].mean()) if "stgeo_op.bn.running_var" in sd else None
        print(f"[GAMMA] {ds}: steps={ck.get('actual_steps')} best_f1={ck.get('best_f1'):.4f} "
              f"gamma={g:.6e} |out.w|_mean={wm:.4e} bn_var_mean={bnv:.4e} n_stgeo_keys={len(keys)}")
    except Exception as e:
        print(f"[GAMMA] {ds}: FAILED {type(e).__name__}: {e}")
PY
