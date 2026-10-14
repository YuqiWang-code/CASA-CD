#!/usr/bin/env bash
# Run8 服务器 GPU 预检（**只读**：不训练、不写任何正式名产物）。
# 依次验证：
#   0) conda 环境 / torch / GPU / mamba_base kernel 可用性
#   1) CPU 预检（数学 + 形状 + 参数预算）
#   2) VMamba-Tiny 官方权重 **逐位加载**（worst_diff == 0）+ 真实前向
#   3) selective_scan 后端报告
#   4) HQKV 在真实 backbone 输出上的**非退化性**（detail vs global 排序分歧）
#   5) batch32 峰值显存（决定能否同卡并发）
set -u
CONDA_SH=/home/yqwang/miniforge3/etc/profile.d/conda.sh
REPO=/home/yqwang/projects/CASA-CD
CA=/home/yqwang/miniforge3/envs/casacd
MB=/home/yqwang/miniforge3/envs/mamba_base

echo "############ Run8 smoke $(date '+%F %T') ############"
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
print("  arch_list", torch.cuda.get_arch_list())
try:
    import selective_scan_cuda_oflex as k
    print("  oflex    ", k.__file__)
except Exception as e:
    print("  oflex FAIL", type(e).__name__, e)
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    print("  mamba_ssm selective_scan_fn OK")
except Exception as e:
    print("  mamba_ssm FAIL", type(e).__name__, str(e)[:160])
PY

echo
echo "=== 1) CPU 预检（数学 / 形状 / 参数预算）==="
cd "$REPO/models" && python test_run8_cpu_precheck.py; RC_CPU=$?
echo "  [smoke] cpu_precheck rc=$RC_CPU"

echo
echo "=== 2) VMamba 官方权重逐位加载 + 真实前向 ==="
CUDA_VISIBLE_DEVICES=0 python - <<'PY'
import sys, os, torch, time
sys.path.insert(0, os.getcwd())
from model.vmamba_tiny import VMambaBackbone
from model.layers.vmamba_kernel import backend_report
print("  backend:", backend_report())
w = "/home/yqwang/projects/CASA-CD/pretrained_weight/vssm_tiny_0230_ckpt_epoch_262.pth"
m = VMambaBackbone(pretrained_path=w).cuda().eval()
st = m.load_stats()
print(f"  retained={st['retained']}/{st['pretrained_keys']} worst_diff={st['worst_diff']} "
      f"missing_new={len(st['missing_new'])} dropped={len(st['dropped_intentional'])}")
assert st["worst_diff"] == 0.0, f"[P0-FAIL] pretrain not bitwise: worst_diff={st['worst_diff']}"
x = torch.randn(2, 3, 256, 256, device="cuda")
with torch.no_grad():
    for _ in range(2):
        t0 = time.time(); outs = m(x); torch.cuda.synchronize()
print(f"  fwd OK taps={[tuple(t.shape) for t in outs]} {(time.time()-t0)*1000:.1f} ms/batch2")
print(f"  params={m.param_count():,} ({m.param_count()/1e6:.3f} M)")
del m, outs, x; torch.cuda.empty_cache()
PY

echo
echo "=== 3) HQKV 在真实 backbone 特征上的非退化性（detail vs global）==="
CUDA_VISIBLE_DEVICES=0 python - <<'PY'
import sys, os, torch
sys.path.insert(0, os.getcwd())
from model.vmamba_tiny import VMambaBackbone
from model.layers.caacp_hqkv import CAACPHQKV
from model.layers.hyperbolic import branch_rank_disagreement
w = "/home/yqwang/projects/CASA-CD/pretrained_weight/vssm_tiny_0230_ckpt_epoch_262.pth"
m = VMambaBackbone(pretrained_path=w).cuda().eval()
xa = torch.randn(2, 3, 256, 256, device="cuda")
xb = torch.randn(2, 3, 256, 256, device="cuda")
with torch.no_grad():
    oa = m(xa); ob = m(xb)
fa = [t.chunk(2, dim=0)[0] for t in oa]
fb = [t.chunk(2, dim=0)[0] for t in ob]
op = CAACPHQKV(stage_dims=(96,192,384,768), rank=32, lite_dim=192).cuda()
with torch.no_grad():
    out, diag = op(fa, fb)
print(f"  out {tuple(out.shape)} gamma0_max={float(out.abs().max()):.3e}")
for k, v in sorted(diag["stages"].items()):
    print(f"  {k}: s_d_mean={v['score_detail_mean']:.4f} s_g_mean={v['score_global_mean']:.4f} "
          f"deg_frac={v['deg_d_vs_g_frac_rank_diff']:.4f} tau={v['deg_d_vs_g_kendall_tau']:.4f}")
worst = min(v["deg_d_vs_g_frac_rank_diff"] for v in diag["stages"].values())
print(f"  [GATE] min detail-vs-global rank disagreement = {worst:.4f} (frozen threshold 0.05)"
      f" -> {'PASS' if worst > 0.05 else 'FAIL'}")
PY

echo
echo "=== 4) batch32 峰值显存（单卡，决定并发策略）==="
CUDA_VISIBLE_DEVICES=0 python - <<'PY'
import sys, os, torch
sys.path.insert(0, os.getcwd())
from model.casa_vmamba_net import CASAVMambaSTRNet
from model.hgsl_loss import hgsl_loss
from model.utils import BCEDiceLoss
w = "/home/yqwang/projects/CASA-CD/pretrained_weight/vssm_tiny_0230_ckpt_epoch_262.pth"
m = CASAVMambaSTRNet(vmamba_pretrained_path=w, hqkv=True, hgsl=True).cuda().train()
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
    print(f"  [PASS] batch32 fwd+bwd OK peak={peak:.0f} MB  loss={float(loss):.4f}")
    print(f"  GPU 余量 {32607 - peak:.0f} MB -> 同卡第二个 run "
          f"{'可行' if 32607 - peak > peak else '不可行（保持串行）'}")
except torch.cuda.OutOfMemoryError as e:
    print(f"  [FAIL] batch32 OOM: {e}")
PY

echo
echo "SMOKE DONE"
