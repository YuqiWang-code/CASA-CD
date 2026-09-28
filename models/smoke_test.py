"""Smoke test for CASA-CD (baseline / CASAA / SAA modes, no real data needed).

Builds ChangeViT-Tiny, loads the pretrained DeiT-Tiny weights, and runs the Run1
mechanism test suite (docs/temporary/CASA-CD_CASAA_Run1_修改与实验设计建议.md §8.8):

  T0 full-network smoke:   3-step forward+backward on random 256x256 bi-temporal input,
                           params + FLOPs (per mode).
  T1 baseline equivalence: with --mode baseline, new vit.forward_pair(x1, x2) must be
                           identical (eval, <1e-6) to vit(x1), vit(x2).
  T2 pretrained keys:      CASAA must not add any missing/unexpected key vs baseline
                           (qkv/proj in-place inheritance of DeiT-Tiny weights).
  T3 CASAA mechanism:      output shape, token budget (N=256, K=64, Kc=32, Kb=32),
                           T1/T2 swap symmetry of routing, gradients finite & nonzero.

Usage:
    python smoke_test.py --model_type tiny --pretrained_weight_path <deit_tiny.pth> \
        [--gpu_id 0] [--mode all|baseline|casaa|saa]
"""
import os
import sys
import argparse
from functools import partial

import torch

_MODELS_ROOT = os.path.dirname(os.path.abspath(__file__))
if _MODELS_ROOT not in sys.path:
    sys.path.insert(0, _MODELS_ROOT)

from model.trainer import Trainer
from model.encoder import DinoVisionTransformer
from model.layers import MemEffAttention
from model.layers.block import NestedTensorBlock as Block
from model.utils import BCEDiceLoss

CASAA_LAYERS = [8, 9, 10, 11]
CASAA_KEEP = 0.25
CASAA_SHARE = 0.5


def measure_params(model):
    return sum(p.numel() for p in model.parameters())


def measure_effective_params(model):
    """Params excluding unused ResNet head (layer4 + fc + avgpool never run in forward)."""
    total = measure_params(model)
    dead = 0
    for name, mod in model.named_modules():
        if name in ("encoder.resnet.layer4", "encoder.resnet.fc", "encoder.resnet.avgpool"):
            dead += sum(p.numel() for p in mod.parameters())
    return total - dead


def measure_flops(model, size=256):
    from fvcore.nn import flop_count

    model.eval()
    pre = torch.randn(1, 3, size, size).cuda()
    post = torch.randn(1, 3, size, size).cuda()
    with torch.no_grad():
        counts, unsupported = flop_count(model, (pre, post))
    return sum(counts.values()), len(unsupported)


def build_trainer(pretrained_path, mode, device):
    return Trainer("tiny", pretrained_path=pretrained_path, resnet_pretrained=False,
                   mode=mode, casaa_layers=CASAA_LAYERS, casaa_keep_ratio=CASAA_KEEP,
                   casaa_change_share=CASAA_SHARE,
                   casaa_router=("content" if mode == "saa" else "change")).float().to(device)


def build_vit(casaa_layers):
    return DinoVisionTransformer(
        img_size=256, patch_size=16, embed_dim=192, depth=12, num_heads=6,
        mlp_ratio=4, block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=0, casaa_layers=casaa_layers,
        casaa_keep_ratio=CASAA_KEEP, casaa_change_share=CASAA_SHARE, casaa_router="change")


def load_vit_keys(vit, pretrained_path):
    """Same load procedure as Encoder.__init__; returns (missing, unexpected) sets."""
    sd = torch.load(pretrained_path, map_location="cpu")["model"]
    for k in ["pos_embed", "patch_embed.proj.weight"]:
        sd.pop(k, None)
    msg = vit.load_state_dict(sd, strict=False)
    return set(msg.missing_keys), set(msg.unexpected_keys)


def t1_baseline_equivalence(pretrained_path, device):
    print("[T1] baseline equivalence: forward_pair(x1,x2) == (vit(x1), vit(x2))")
    model = build_trainer(pretrained_path, "baseline", device).eval()
    vit = model.encoder.vit
    assert not vit.casaa_layers
    x1 = torch.randn(2, 3, 256, 256, device=device)
    x2 = torch.randn(2, 3, 256, 256, device=device)
    with torch.no_grad():
        p1, p2 = vit.forward_pair(x1, x2)
        s1, s2 = vit(x1), vit(x2)
    d = max((p1 - s1).abs().max().item(), (p2 - s2).abs().max().item())
    assert d < 1e-6, f"baseline forward_pair mismatch: {d}"
    print(f"  OK, max abs diff = {d:.2e}")
    del model


def t2_pretrained_keys(pretrained_path):
    print("[T2] pretrained keys parity: CASAA must not add attention-related keys")
    base_vit = build_vit(casaa_layers=None)
    m_base, u_base = load_vit_keys(base_vit, pretrained_path)
    casaa_vit = build_vit(casaa_layers=CASAA_LAYERS)
    m_casaa, u_casaa = load_vit_keys(casaa_vit, pretrained_path)
    print(f"  baseline missing ({len(m_base)}): {sorted(m_base)}")
    print(f"  casaa    missing ({len(m_casaa)}): {sorted(m_casaa)}")
    assert m_casaa == m_base, f"missing keys differ:\n  only-casaa: {m_casaa - m_base}\n  only-base: {m_base - m_casaa}"
    assert u_casaa == u_base, f"unexpected keys differ: casaa={u_casaa - u_base} base={u_base - u_casaa}"
    # 关键验收：qkv/proj 不能出现在 missing 里（预训练注意力参数必须原位继承）
    for i in CASAA_LAYERS:
        for key in (f"blocks.{i}.attn.qkv.weight", f"blocks.{i}.attn.qkv.bias",
                    f"blocks.{i}.attn.proj.weight", f"blocks.{i}.attn.proj.bias"):
            assert key not in m_casaa, f"pretrained attn key missing: {key}"
    print("  OK, qkv/proj weights load in place; no new keys")


def t3_mechanism(pretrained_path, mode, device):
    print(f"[T3] CASAA mechanism (mode={mode}): shape / budget / swap symmetry / gradients")
    model = build_trainer(pretrained_path, mode, device).eval()
    vit = model.encoder.vit
    x1 = torch.randn(2, 3, 256, 256, device=device)
    x2 = torch.randn(2, 3, 256, 256, device=device)

    # shape: final change map Bx1x256x256
    with torch.no_grad():
        out = model(x1, x2)
    assert out.shape == (2, 1, 256, 256), f"bad output shape {out.shape}"

    # token budget + routing swap symmetry
    with torch.no_grad():
        _ = vit.forward_pair(x1, x2)
        r_ab = [vit.blocks[i].attn._routing for i in CASAA_LAYERS]
        _ = vit.forward_pair(x2, x1)
        r_ba = [vit.blocks[i].attn._routing for i in CASAA_LAYERS]
    for i, (ra, rb) in zip(CASAA_LAYERS, zip(r_ab, r_ba)):
        assert ra is not None and rb is not None
        assert (ra["N"], ra["K"]) == (256, 64), f"block {i} budget {ra}"
        print(f"  block {i}: N={ra['N']} K={ra['K']} Kc={ra['Kc']} Kb={ra['Kb']}")
        if mode == "casaa":
            assert (ra["Kc"], ra["Kb"]) == (32, 32)
            assert torch.equal(ra["Ic"], rb["Ic"]), f"block {i} change indices not swap-symmetric"
        else:
            assert (ra["Kc"], ra["Kb"]) == (0, 64) and ra["Ic"] is None
        assert torch.equal(ra["assign"], rb["assign"]), f"block {i} bg assignment not swap-symmetric"

    # ViT-level swap symmetry: routing 一致 + 两个时相输出对应交换。
    # （不做整网 swap 等价断言：decoder 的 difference modeling 用 1x1 conv 混合
    #   [x,y,|x-y|] 通道，对 T1/T2 交换并非严格置换对称，baseline 同样如此。）
    with torch.no_grad():
        p1, p2 = vit.forward_pair(x1, x2)
        q1, q2 = vit.forward_pair(x2, x1)
    d_swap = max((q1 - p2).abs().max().item(), (q2 - p1).abs().max().item())
    assert d_swap < 1e-6, f"vit swap symmetry broken: {d_swap}"
    print(f"  vit swap symmetry OK, max abs diff = {d_swap:.2e}")
    del model

    # gradients: qkv/proj/upstream finite & nonzero
    model = build_trainer(pretrained_path, mode, device).train()
    target = torch.randint(0, 2, (2, 1, 256, 256), device=device).float()
    out = model(x1, x2)
    loss = BCEDiceLoss(out, target)
    loss.backward()
    vit = model.encoder.vit
    for i in CASAA_LAYERS:
        attn = vit.blocks[i].attn
        for name, g in (("qkv.weight", attn.qkv.weight.grad), ("proj.weight", attn.proj.weight.grad)):
            assert g is not None, f"block {i} {name} grad is None"
            assert torch.isfinite(g).all(), f"block {i} {name} grad has NaN/Inf"
            assert g.norm() > 0, f"block {i} {name} grad is zero"
    g_pe = vit.patch_embed.proj.weight.grad
    assert g_pe is not None and torch.isfinite(g_pe).all() and g_pe.norm() > 0
    print(f"  gradients OK (qkv/proj per CASAA block + upstream patch_embed)")
    del model


def full_smoke(pretrained_path, mode, device):
    print(f"[T0] full-network smoke (mode={mode})")
    model = build_trainer(pretrained_path, mode, device)
    n_params = measure_params(model)
    n_eff = measure_effective_params(model)
    print(f"  total params = {n_params / 1e6:.3f} M, effective = {n_eff / 1e6:.3f} M")

    pre = torch.randn(2, 3, 256, 256, device=device)
    post = torch.randn(2, 3, 256, 256, device=device)
    target = torch.randint(0, 2, (2, 1, 256, 256), device=device).float()

    optimizer = torch.optim.Adam(model.parameters(), 2e-4)
    model.train()
    for step in range(3):
        out = model(pre, post)
        assert out.shape == (2, 1, 256, 256), f"bad output shape {out.shape}"
        loss = BCEDiceLoss(out, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        assert torch.isfinite(torch.tensor(loss.item())), "loss is NaN/Inf"
        print(f"  step {step + 1}/3 ok, loss = {loss.item():.4f}")

    try:
        flops, n_unsup = measure_flops(model, size=256)
        print(f"  FLOPs = {flops:.4f} G (unsupported_ops={n_unsup})")
    except Exception as e:
        print(f"  [warn] FLOPs measurement failed ({type(e).__name__}: {e})")
    del model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='tiny')
    parser.add_argument('--pretrained_weight_path', type=str, required=True)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--mode', type=str, default='all',
                        choices=['all', 'baseline', 'casaa', 'saa'],
                        help='all | baseline | casaa | saa')
    args = parser.parse_args()

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[SMOKE] mode={args.mode} device={device}")

    if args.mode in ("baseline", "all"):
        t1_baseline_equivalence(args.pretrained_weight_path, device)
        t2_pretrained_keys(args.pretrained_weight_path)
        full_smoke(args.pretrained_weight_path, "baseline", device)
    if args.mode in ("casaa", "all"):
        t3_mechanism(args.pretrained_weight_path, "casaa", device)
        full_smoke(args.pretrained_weight_path, "casaa", device)
    if args.mode in ("saa", "all"):
        t3_mechanism(args.pretrained_weight_path, "saa", device)
        full_smoke(args.pretrained_weight_path, "saa", device)

    print("[SMOKE] ALL OK")


if __name__ == "__main__":
    main()
