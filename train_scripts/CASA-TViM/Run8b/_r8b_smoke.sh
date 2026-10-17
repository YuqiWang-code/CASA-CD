#!/usr/bin/env bash
# Run8b 服务器 GPU 预检（**只读**：不训练、不写正式名产物）。
#   0) 环境 / kernel
#   1) CPU 抗自锁预检（26 项，含 γ=0 自锁负对照）
#   2) VMamba 权重逐位加载 + 前向
#   3) **抗自锁证据**：γ=0.5 时三支贡献都 > 0；γ=0 时全部为 0（负对照）
#   4) batch32 峰值显存
set -u
CONDA_SH=/home/yqwang/miniforge3/etc/profile.d/conda.sh
REPO=/home/yqwang/projects/CASA-CD
CA=/home/yqwang/miniforge3/envs/casacd
MB=/home/yqwang/miniforge3/envs/mamba_base

echo "############ Run8b smoke $(date '+%F %T') ############"
source "$CONDA_SH" && conda activate casacd
export PYTHONPATH="$MB/lib/python3.10/site-packages${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$CA/lib/python3.10/site-packages/torch/lib:$CA/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$REPO" || exit 1

echo
echo "=== 0) 环境 ==="
CUDA_VISIBLE_DEVICES=0 python - <<'PY'
import sys, torch
print("  python   ", sys.version.split()[0])
print("  torch    ", torch.__version__, "| cuda", torch.version.cuda)
print("  device   ", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
try:
    import selective_scan_cuda_oflex as k
    print("  oflex    ", k.__file__)
except Exception as e:
    print("  oflex FAIL", type(e).__name__, e)
PY

echo
echo "=== 1) CPU 抗自锁预检（26 项）==="
cd "$REPO/models" && python test_run8b_antilock_precheck.py; RC_CPU=$?
echo "  [smoke] cpu_precheck rc=$RC_CPU"

echo
echo "=== 2) VMamba 权重逐位加载 + 真实前向 ==="
CUDA_VISIBLE_DEVICES=0 python - <<'PY'
import sys, os, torch, time
sys.path.insert(0, os.getcwd())
from model.vmamba_tiny import VMambaBackbone
w = "/home/yqwang/projects/CASA-CD/pretrained_weight/vssm_tiny_0230_ckpt_epoch_262.pth"
m = VMambaBackbone(pretrained_path=w).cuda().eval()
st = m.load_stats()
print(f"  retained={st['retained']}/{st['pretrained_keys']} worst_diff={st['worst_diff']}")
assert st["worst_diff"] == 0.0, f"[P0-FAIL] pretrain not bitwise: {st['worst_diff']}"
print(f"  [PASS] pretrain bitwise; params={m.param_count():,}")
x = torch.randn(2, 3, 256, 256, device="cuda")
with torch.no_grad():
    t0 = time.time(); outs = m(x); torch.cuda.synchronize()
print(f"  fwd OK taps={[tuple(t.shape) for t in outs]} {(time.time()-t0)*1000:.1f} ms")
del m, outs, x; torch.cuda.empty_cache()
PY

echo
echo "=== 3) 抗自锁证据（真实 backbone 特征）==="
CUDA_VISIBLE_DEVICES=0 python - <<'PY'
import sys, os, torch
sys.path.insert(0, os.getcwd())
from model.vmamba_tiny import VMambaBackbone
from model.layers.caacp_hqkv_v2 import CAACPHQKVv2
w = "/home/yqwang/projects/CASA-CD/pretrained_weight/vssm_tiny_0230_ckpt_epoch_262.pth"
m = VMambaBackbone(pretrained_path=w).cuda().eval()
xa = torch.randn(2, 3, 256, 256, device="cuda")
xb = torch.randn(2, 3, 256, 256, device="cuda")
with torch.no_grad():
    oa = m(xa); ob = m(xb)
fa = [t.chunk(2, dim=0)[0] for t in oa]
fb = [t.chunk(2, dim=0)[0] for t in ob]
DIMS = (96, 192, 384, 768)

for tag, gi in (("gamma=0.5 (v2 主实验)", 0.5), ("gamma=0.0 (v1 自锁负对照)", 0.0)):
    op = CAACPHQKVv2(stage_dims=DIMS, rank=32, lite_dim=192, gamma_init=gi).cuda()
    out, diag = op(fa, fb)
    cn = diag.get("contrib_norm", 0.0)
    shares = {k: round(v, 4) for k, v in diag.items() if k.endswith("_share")}
    op.zero_grad()
    o2, _ = op(fa, fb)
    o2.pow(2).mean().backward()
    op.snapshot_gamma_grad()
    gg = op.param_report()["gamma_grad"]
    print(f"  [{tag}] contrib_norm={cn:.6e} shares={shares} gamma_grad={gg}")
    rms = cn / max(1.0, (2 * 192 * 64 * 64) ** 0.5)
    print(f"      rms≈{rms:.4f}")
    del op, out, diag, o2
    torch.cuda.empty_cache()
print("  [GATE] 期望：γ=0.5 三支 share 各 ~1/3 且 gamma_grad 全 > 0；"
      "γ=0 全部为 0（复现 v1 自锁）")
PY

echo
echo "=== 4) batch32 峰值显存 ==="
CUDA_VISIBLE_DEVICES=0 python - <<'PY'
import sys, os, torch
sys.path.insert(0, os.getcwd())
from model.casa_vmamba_net_v2 import CASAVMambaSTRNetV2
from model.hgsl_loss import hgsl_loss
from model.utils import BCEDiceLoss
w = "/home/yqwang/projects/CASA-CD/pretrained_weight/vssm_tiny_0230_ckpt_epoch_262.pth"
m = CASAVMambaSTRNetV2(vmamba_pretrained_path=w, hqkv=True, hgsl=True,
                       gamma_init=0.5).cuda().train()
pre = torch.randn(32, 3, 256, 256, device="cuda")
post = torch.randn(32, 3, 256, 256, device="cuda")
tgt = (torch.rand(32, 1, 256, 256, device="cuda") > 0.9).float()
opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], 2e-4)
try:
    torch.cuda.reset_peak_memory_stats()
    out = m(pre, post, tgt)
    loss = BCEDiceLoss(out, tgt)
    hg = hgsl_loss(m.hgsl_score_maps(), tgt, 0.2)
    if hg is not None: loss = loss + hg
    opt.zero_grad(); loss.backward(); opt.step()
    peak = torch.cuda.max_memory_allocated() / 2**20
    print(f"  [PASS] batch32 fwd+bwd OK peak={peak:.0f} MB loss={float(loss):.4f}")
    d = m.hqkv_diag()
    print(f"  contrib_norm={d.get('contrib_norm'):.6e} gammas={d.get('gammas')}")
except torch.cuda.OutOfMemoryError as e:
    print(f"  [FAIL] batch32 OOM: {e}")
PY

echo
echo "SMOKE DONE"
