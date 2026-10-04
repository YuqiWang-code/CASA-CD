"""Smoke test for CASA-CD (baseline / CASAA / SAA / ORACLE modes, no real data needed).

Builds ChangeViT-Tiny, loads the pretrained DeiT-Tiny weights, and runs the mechanism
test suite (Run1 设计文档 §8.8 + Run2 决策文档 §13):

  T0 qkv sliced equivalence: CASAAAttention 的切片投影（full X 只算 Q、压缩上下文只算
     K/V）必须与整段 fused qkv 后切片数学等价（<1e-6）。
  T1 full-network smoke:   3-step forward+backward on random 256x256 bi-temporal input,
                           params + FLOPs (per mode).
  T2 baseline equivalence: with --mode baseline, new vit.forward_pair(x1, x2) must be
                           identical (eval, <1e-6) to vit(x1), vit(x2).
  T3 pretrained keys:      CASAA must not add any missing/unexpected key vs baseline
                           (qkv/proj in-place inheritance of DeiT-Tiny weights).
  T4 CASAA mechanism:      output shape, token budget (N=256, K=64, Kc=32, Kb=32),
                           T1/T2 swap symmetry of routing, gradients finite & nonzero.
  T5 oracle router:        GT patch occupancy -> 256 scores；TopK 正确；不进入梯度；
                           T1/T2 swap 不改变 routing（DIAGNOSTIC-ONLY）。

Usage:
    python smoke_test.py --model_type tiny --pretrained_weight_path <deit_tiny.pth> \
        [--gpu_id 0] [--mode all|baseline|casaa|saa|oracle]
"""
import os
import sys
import argparse
from functools import partial

import torch
import torch.nn.functional as F

_MODELS_ROOT = os.path.dirname(os.path.abspath(__file__))
if _MODELS_ROOT not in sys.path:
    sys.path.insert(0, _MODELS_ROOT)

from model.trainer import Trainer
from model.encoder import DinoVisionTransformer
from model.layers import MemEffAttention, CASAAAttention, rank_normalize_per_image
from model.layers.block import NestedTensorBlock as Block
from model.utils import BCEDiceLoss
from model.psd_detail import PSDDetail
from model.resnet import resnet18
from model.mobile_detail import MobileDetail
from torchvision.models import mobilenet_v3_small

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
    label = torch.zeros(1, 1, size, size).cuda()
    with torch.no_grad():
        counts, unsupported = flop_count(model, (pre, post, label))
    return sum(counts.values()), len(unsupported)


def build_trainer(pretrained_path, mode, device, router=None, vit_depth=12):
    if mode == "oracle":
        trainer_mode, router = "casaa", "oracle"
    elif mode == "detail":
        trainer_mode, router = "casaa", "detail"
    elif mode == "detail_fused":
        trainer_mode, router = "casaa", "detail_fused"
    else:
        trainer_mode = mode
        if router is None:
            router = "content" if mode == "saa" else "change"
    return Trainer("tiny", pretrained_path=pretrained_path, resnet_pretrained=False,
                   mode=trainer_mode, casaa_layers=CASAA_LAYERS, casaa_keep_ratio=CASAA_KEEP,
                   casaa_change_share=CASAA_SHARE, casaa_router=router,
                   vit_depth=vit_depth).float().to(device)


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


def t0_qkv_sliced_equivalence(device):
    """Smoke-0（Run2 §13）：切片 qkv 投影必须与整段 qkv 后切片数学等价。"""
    print("[T0] qkv sliced projection equivalence (full X -> Q only, context -> K/V only)")
    torch.manual_seed(0)
    B, N, C, H = 2, 256, 192, 6
    x1 = torch.randn(B, N, C, device=device)
    x2 = torch.randn(B, N, C, device=device)

    def ref_forward_pair(attn, xa, xb):
        """参考实现：整段 fused qkv 后切片（Run1 旧逻辑）。"""
        with torch.no_grad():
            Ca, Cb = attn.compute_contexts(xa, xb)
            Kref = Ca.shape[1]
            qkv1 = attn.qkv(xa).reshape(B, N, 3, H, C // H).permute(2, 0, 3, 1, 4)
            qkv2 = attn.qkv(xb).reshape(B, N, 3, H, C // H).permute(2, 0, 3, 1, 4)
            qkv1c = attn.qkv(Ca).reshape(B, Kref, 3, H, C // H).permute(2, 0, 3, 1, 4)
            qkv2c = attn.qkv(Cb).reshape(B, Kref, 3, H, C // H).permute(2, 0, 3, 1, 4)
            q1, k1, v1 = qkv1[0] * attn.scale, qkv1[1], qkv1[2]
            q2, k2, v2 = qkv2[0] * attn.scale, qkv2[1], qkv2[2]
            _, k1c, v1c = qkv1c
            _, k2c, v2c = qkv2c
            y1 = ((q1 @ k1c.transpose(-2, -1)).softmax(-1) @ v1c).transpose(1, 2).reshape(B, N, C)
            y2 = ((q2 @ k2c.transpose(-2, -1)).softmax(-1) @ v2c).transpose(1, 2).reshape(B, N, C)
        return attn.proj_drop(attn.proj(y1)), attn.proj_drop(attn.proj(y2))

    for router in ("change", "content"):
        attn = CASAAAttention(dim=C, num_heads=H, qkv_bias=True, router=router).to(device).eval()
        with torch.no_grad():
            y1, y2 = attn.forward_pair(x1, x2)
            yr1, yr2 = ref_forward_pair(attn, x1, x2)
        d = max((y1 - yr1).abs().max().item(), (y2 - yr2).abs().max().item())
        assert d < 1e-6, f"router={router} sliced vs full-qkv mismatch: {d}"
        print(f"  router={router}: sliced == full-qkv, max_abs_err = {d:.2e}")
    del attn


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


def t5_oracle_router(pretrained_path, device):
    """Smoke-2（Run2 §13）：oracle router 的 GT patch occupancy 机制测试。"""
    print("[T5] oracle router: GT occupancy -> TopK, no grad, swap-invariant")
    model = build_trainer(pretrained_path, "oracle", device).eval()
    vit = model.encoder.vit
    x1 = torch.randn(2, 3, 256, 256, device=device)
    x2 = torch.randn(2, 3, 256, 256, device=device)
    # 假 label：只让特定 patch 区域有变化（patch 16 -> 16x16 grid）
    label = torch.zeros(2, 1, 256, 256, device=device)
    label[:, :, 0:32, 0:32] = 1.0          # 4 个 patch 有变化
    label[:, :, 48:80, 160:192] = 1.0      # 4 个 patch 有变化
    g_ref = torch.nn.functional.avg_pool2d(label, 16).view(2, -1)   # (B, 256)

    with torch.no_grad():
        out = model(x1, x2, label)
    assert out.shape == (2, 1, 256, 256)

    r = vit.blocks[8].attn._routing
    assert (r["N"], r["K"], r["Kc"], r["Kb"]) == (256, 64, 32, 32)
    assert torch.equal(r["s"], g_ref), "oracle score must equal GT patch occupancy"
    assert torch.equal(r["Ic"], torch.topk(g_ref, 32, dim=-1).indices), "TopK indices wrong"

    # T1/T2 交换：routing 只依赖 GT hint，不变
    with torch.no_grad():
        _ = vit.forward_pair(x2, x1, score_hint=g_ref)
        r_sw = vit.blocks[8].attn._routing
    assert torch.equal(r["Ic"], r_sw["Ic"]) and torch.equal(r["assign"], r_sw["assign"])

    # 不进入梯度：score 路径 no-grad（label 无 requires_grad，输出对 label 无梯度）
    model.train()
    target = torch.randint(0, 2, (2, 1, 256, 256), device=device).float()
    out = model(x1, x2, label)
    loss = BCEDiceLoss(out, target)
    loss.backward()
    attn = vit.blocks[8].attn
    assert attn.qkv.weight.grad is not None and torch.isfinite(attn.qkv.weight.grad).all()
    assert attn.qkv.weight.grad.norm() > 0
    print("  occupancy -> score OK; TopK OK; swap-invariant; grads finite & nonzero")
    del model


def t6_detail_router(pretrained_path, device):
    """A4 smoke（决策文档 §12 T6-T10）：detail 1/8 形状、rank 归一化、
    fused score、梯度图、参数一致。"""
    print("[T6] detail-fused router: shapes / fused score / rank norm / swap / grads / params")
    model = build_trainer(pretrained_path, "detail_fused", device).eval()
    enc = model.encoder
    x1 = torch.randn(2, 3, 256, 256, device=device)
    x2 = torch.randn(2, 3, 256, 256, device=device)

    # T6: detail shape alignment（layer3 = 1/8 = 32×32×256）
    with torch.no_grad():
        c1 = enc.detail_capture(x1)
        c2 = enc.detail_capture(x2)
    assert c1[0].shape == (2, 64, 128, 128) and c1[1].shape == (2, 128, 64, 64) \
        and c1[2].shape == (2, 256, 32, 32), f"detail shapes {[tuple(t.shape) for t in c1]}"
    with torch.no_grad():
        sd = enc.detail_score_1_8(c1[2], c2[2])
    assert sd.shape == (2, 256), sd.shape
    assert sd.min().item() >= 0.0 and sd.max().item() <= 1.0

    # T7: rank normalization
    s_rand = torch.rand(4, 256, device=device)
    r_rand = rank_normalize_per_image(s_rand)
    assert r_rand.shape == (4, 256)
    assert r_rand.min().item() == 0.0 and r_rand.max().item() == 1.0
    assert torch.equal(r_rand, rank_normalize_per_image(s_rand))

    # 完整前向：fused score == 0.5*R(sv)+0.5*rd，routing 分量被记录
    with torch.no_grad():
        out = model(x1, x2)
    assert out.shape == (2, 1, 256, 256)
    vit = enc.vit
    r = vit.blocks[8].attn._routing
    assert (r["N"], r["K"], r["Kc"], r["Kb"]) == (256, 64, 32, 32)
    assert "s_v_raw" in r and "s_v_rank" in r and "s_d_rank" in r
    assert torch.allclose(r["s"], 0.5 * r["s_v_rank"] + 0.5 * r["s_d_rank"], atol=1e-6)
    assert r["s"].requires_grad is False and r["s"].grad_fn is None   # T9: routing no-grad

    # T8: swap symmetry（detail cosine 与 fused score 对 T1/T2 交换对称）
    with torch.no_grad():
        _ = vit.forward_pair(x2, x1, score_hint=enc.detail_score_1_8(c2[2], c1[2]))
        r_sw = vit.blocks[8].attn._routing
        _ = vit.forward_pair(x1, x2, score_hint=sd)
        r_ab = vit.blocks[8].attn._routing
    assert torch.allclose(r_ab["s"], r_sw["s"], atol=1e-6)
    assert torch.equal(r_ab["Ic"], r_sw["Ic"])

    # T9: 梯度图——detail branch / decoder 正常反传
    model = build_trainer(pretrained_path, "detail_fused", device).train()
    target = torch.randint(0, 2, (2, 1, 256, 256), device=device).float()
    out = model(x1, x2)
    loss = BCEDiceLoss(out, target)
    loss.backward()
    g_layer3 = model.encoder.resnet.layer3[1].conv2.weight.grad
    g_dec = model.decoder.classfier[0].weight.grad
    assert g_layer3 is not None and torch.isfinite(g_layer3).all() and g_layer3.norm() > 0
    assert g_dec is not None and torch.isfinite(g_dec).all() and g_dec.norm() > 0
    print("  T6 shapes OK; T7 rank OK; T8 swap OK; T9 router no-grad + branch grads OK")
    del model

    # A4-D（detail-only router）：score 必须等于 rank 归一化的 detail hint
    model = build_trainer(pretrained_path, "detail", device).eval()
    enc = model.encoder
    with torch.no_grad():
        out = model(x1, x2)
    r = enc.vit.blocks[8].attn._routing
    assert (r["K"], r["Kc"], r["Kb"]) == (64, 32, 32)
    with torch.no_grad():
        c1 = enc.detail_capture(x1)
        c2 = enc.detail_capture(x2)
        sd_ref = enc.detail_score_1_8(c1[2], c2[2])
    assert torch.equal(r["s"], sd_ref), "detail router must use the rank-normalized detail score"
    assert r["s"].requires_grad is False
    print("  A4-D detail-only router OK (s == rank-normalized detail hint)")
    del model

    # T10: 参数一致性（A1 vs A4 零新增参数，state_dict key 完全一致）
    m_a1 = build_trainer(pretrained_path, "saa", device)
    m_a4 = build_trainer(pretrained_path, "detail_fused", device)
    n1 = sum(p.numel() for p in m_a1.parameters())
    n4 = sum(p.numel() for p in m_a4.parameters())
    assert n1 == n4, f"param mismatch {n1} vs {n4}"
    assert set(m_a1.state_dict().keys()) == set(m_a4.state_dict().keys())
    print(f"  T10 parameter parity OK ({n1 / 1e6:.3f} M, identical keys)")
    del m_a1, m_a4


def t_run4(pretrained_path, device, vit_depth=4):
    """Run4 smoke（决策文档 §33 T0/T1/T3）：corrected DeiT load + depth 截断。"""
    print(f"[RUN4] TinyViT-{vit_depth} smoke: exact pretrain load / depth / params / 3-step")
    sd = torch.load(pretrained_path, map_location="cpu")["model"]
    model = build_trainer(pretrained_path, "baseline", device, vit_depth=vit_depth).eval()
    vit = model.encoder.vit

    # T1: depth 截断——blocks 深度之外完全不存在
    assert len(vit.blocks) == vit_depth, f"blocks={len(vit.blocks)} != {vit_depth}"
    resid = [k for k in model.state_dict()
             if any(k.startswith(f"encoder.vit.blocks.{j}.") for j in range(vit_depth, 12))]
    assert not resid, f"leftover deeper blocks: {resid[:3]}"
    # T0: 原位继承 max_abs_diff == 0
    model_sd = vit.state_dict()
    worst = 0.0
    for i in range(vit_depth):
        for k in ("norm1.weight", "attn.qkv.weight", "mlp.fc1.weight"):
            worst = max(worst, (sd[f"blocks.{i}.{k}"].float() - model_sd[f"blocks.{i}.{k}"].float().cpu()).abs().max().item())
    assert worst == 0.0, f"pretrained blocks not loaded in place (diff {worst})"
    assert tuple(model_sd["pos_embed"].shape) == (1, 256, 192)
    print(f"  exact-load OK (worst={worst:.2e}); blocks={vit_depth}; pos_embed (1,256,192)")

    # T3: 参数预算
    total = measure_params(model)
    vit_params = sum(p.numel() for p in vit.parameters())
    print(f"  ViT params = {vit_params / 1e6:.4f} M; total = {total / 1e6:.4f} M; "
          f"effective = {measure_effective_params(model) / 1e6:.4f} M")

    # 3-step 冒烟
    pre = torch.randn(2, 3, 256, 256, device=device)
    post = torch.randn(2, 3, 256, 256, device=device)
    target = torch.randint(0, 2, (2, 1, 256, 256), device=device).float()
    optimizer = torch.optim.Adam([p for n, p in model.named_parameters() if not n.startswith("encoder.vit")], 2e-4)
    model.train()
    for step in range(3):
        out = model(pre, post)
        assert out.shape == (2, 1, 256, 256)
        loss = BCEDiceLoss(out, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        assert torch.isfinite(torch.tensor(loss.item()))
        print(f"  step {step + 1}/3 ok, loss = {loss.item():.4f}")
    try:
        flops, n_unsup = measure_flops(model, size=256)
        print(f"  FLOPs = {flops:.4f} G (unsupported_ops={n_unsup})")
    except Exception as e:
        print(f"  [warn] FLOPs measurement failed ({type(e).__name__}: {e})")
    del model


def t_run4_light(pretrained_path, device, detail_mode='light'):
    """R4-2 smoke：LightDetail + adapters（形状/参数/梯度/3-step）。"""
    print(f"[RUN4-LIGHT] VIT4 + LightDetail({detail_mode})(+adapters) + old FI/decoder smoke")
    model = Trainer("tiny", pretrained_path=pretrained_path, resnet_pretrained=False,
                    mode="baseline", vit_depth=4, detail_mode=detail_mode).float().to(device).eval()
    enc = model.encoder
    x = torch.randn(2, 3, 256, 256, device=device)
    with torch.no_grad():
        c = enc.detail_capture(x)
    assert [tuple(t.shape) for t in c] == [(2, 64, 128, 128), (2, 128, 64, 64), (2, 256, 32, 32)], \
        [tuple(t.shape) for t in c]
    n_detail = sum(p.numel() for p in enc.detail.parameters())
    n_adapt = sum(p.numel() for p in enc.detail_adapters.parameters())
    if detail_mode == 'light':
        assert n_detail == 37024, n_detail
        assert n_adapt == 43008, n_adapt
    elif detail_mode == 'light_bnrelu':
        # R4-2c：Conv1×1→BN→ReLU adapters（conv 43,008 + BN 896）
        assert n_detail == 37024, n_detail
        assert n_adapt == 43904, n_adapt
    else:
        assert n_detail == 64528, n_detail
        assert n_adapt == 56320, n_adapt
    assert "encoder.resnet.conv1.weight" not in model.state_dict(), "resnet must not exist in light mode"
    assert len(enc.vit.blocks) == 4
    print(f"  shapes OK; LightDetail={n_detail} adapters={n_adapt}; no resnet; blocks=4")

    pre = torch.randn(2, 3, 256, 256, device=device)
    post = torch.randn(2, 3, 256, 256, device=device)
    target = torch.randint(0, 2, (2, 1, 256, 256), device=device).float()
    with torch.no_grad():
        out = model(pre, post)
    assert out.shape == (2, 1, 256, 256)
    # 梯度：LightDetail/adapters/decoder 非零有限，ViT 无梯度（冻结由训练脚本控制，
    # 这里只验证非冻结组件的梯度路径正常）
    model.train()
    out = model(pre, post)
    loss = BCEDiceLoss(out, target)
    loss.backward()
    g = model.encoder.detail.stage8b.pw.weight.grad
    assert g is not None and torch.isfinite(g).all() and g.norm() > 0
    g_dec = model.decoder.classfier[0].weight.grad
    assert g_dec is not None and torch.isfinite(g_dec).all() and g_dec.norm() > 0
    print("  gradients OK (LightDetail / decoder)")

    total = measure_params(model)
    print(f"  total params = {total / 1e6:.4f} M (R4-2 仍含原 FI/decoder，>3M 属预期)")
    try:
        flops, n_unsup = measure_flops(model, size=256)
        print(f"  FLOPs = {flops:.4f} G (unsupported_ops={n_unsup})")
    except Exception as e:
        print(f"  [warn] FLOPs measurement failed ({type(e).__name__}: {e})")
    del model


def t_run4_psd(pretrained_path, device, vit_depth=4):
    """R4-2d smoke（决策文档 §29 T-PSD0..T-PSD5）：PSD-Detail 全链路。"""
    print("[RUN4-PSD] VIT4 + PSD-Detail + old FI/decoder smoke")

    # T-PSD0：pretrained stem —— conv1/bn1 与 ImageNet ResNet18 逐位一致
    psd_ref = PSDDetail(pretrained=True).cpu()
    ref = resnet18(pretrained=True).cpu()
    d_conv = (psd_ref.stem_conv.weight.data - ref.conv1.weight.data).abs().max().item()
    assert d_conv == 0.0, f"stem conv not loaded in place (diff {d_conv})"
    for k in ref.bn1.state_dict():
        assert torch.equal(psd_ref.stem_bn.state_dict()[k], ref.bn1.state_dict()[k]), f"stem bn {k} mismatch"
    print(f"  T-PSD0 pretrained stem OK (conv max_abs_diff == 0; bn affine + running stats == source)")
    del ref

    # T-PSD1：输出形状（B×64×128×128 / B×128×64×64 / B×256×32×32）
    x = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        d2, d4, d8 = psd_ref(x)
    assert [tuple(t.shape) for t in (d2, d4, d8)] == [(2, 64, 128, 128), (2, 128, 64, 64), (2, 256, 32, 32)], \
        [tuple(t.shape) for t in (d2, d4, d8)]
    # T-PSD2：参数 == 78,464
    n_psd = sum(p.numel() for p in psd_ref.parameters())
    assert n_psd == 78464, n_psd
    print(f"  T-PSD1 shapes OK; T-PSD2 params = {n_psd}")

    # T-PSD3：完整 Trainer（psd 模式）state dict 无 layer1-4/fc（无隐藏 ResNet）
    model = Trainer("tiny", pretrained_path=pretrained_path, resnet_pretrained=True,
                    mode="baseline", vit_depth=vit_depth, detail_mode="psd").float().to(device).eval()
    bad = [k for k in model.state_dict() if "encoder.resnet" in k
           or any(s in k for s in ("layer1", "layer2", "layer3", "layer4", ".fc."))]
    assert not bad, f"hidden resnet keys: {bad[:3]}"
    assert len(model.encoder.vit.blocks) == vit_depth
    with torch.no_grad():
        c = model.encoder.detail_capture(torch.randn(2, 3, 256, 256, device=device))
    assert [tuple(t.shape) for t in c] == [(2, 64, 128, 128), (2, 128, 64, 64), (2, 256, 32, 32)], \
        [tuple(t.shape) for t in c]
    print(f"  T-PSD3 no hidden ResNet; blocks={vit_depth}; detail_capture shapes OK")

    # T-PSD4 + T-PSD5：3-step 训练，梯度路径 + 冻结 ViT checksum
    for n, p in model.named_parameters():
        if n.startswith("encoder.vit"):
            p.requires_grad_(False)
    vit_before = {k: v.detach().clone() for k, v in model.encoder.vit.state_dict().items()}
    pre = torch.randn(2, 3, 256, 256, device=device)
    post = torch.randn(2, 3, 256, 256, device=device)
    target = torch.randint(0, 2, (2, 1, 256, 256), device=device).float()
    optimizer = torch.optim.Adam([p for n, p in model.named_parameters() if not n.startswith("encoder.vit")], 2e-4)
    model.train()
    for step in range(3):
        out = model(pre, post)
        assert out.shape == (2, 1, 256, 256)
        loss = BCEDiceLoss(out, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        assert torch.isfinite(torch.tensor(loss.item()))
    g_stem = model.encoder.detail.stem_conv.weight.grad
    g_rds = model.encoder.detail.d2_refine.dw.weight.grad
    g_mix = model.encoder.detail.down8.pw.weight.grad
    g_dec = model.decoder.classfier[0].weight.grad
    for name, g in (("stem", g_stem), ("RDS", g_rds), ("MixDown", g_mix), ("decoder", g_dec)):
        assert g is not None and torch.isfinite(g).all() and g.norm() > 0, f"{name} grad broken"
    g_vit = model.encoder.vit.patch_embed.proj.weight.grad
    assert g_vit is None, "frozen ViT must have no grad"
    worst = 0.0
    for k, v in model.encoder.vit.state_dict().items():
        worst = max(worst, (v.detach() - vit_before[k]).abs().max().item())
    assert worst == 0.0, f"frozen ViT changed after training steps (diff {worst})"
    print(f"  T-PSD4 grads OK (stem/RDS/MixDown/decoder; ViT grad None); T-PSD5 frozen ViT checksum max_abs_change == 0")

    total = measure_params(model)
    print(f"  total params = {total / 1e6:.4f} M (R4-2d 仍含原 FI/decoder，>3M 属预期)")
    try:
        flops, n_unsup = measure_flops(model, size=256)
        print(f"  FLOPs = {flops:.4f} G (unsupported_ops={n_unsup})")
    except Exception as e:
        print(f"  [warn] FLOPs measurement failed ({type(e).__name__}: {e})")
    del psd_ref, model


def _ref_mobilenet_p3(mobile_path):
    """构造 ImageNet 官方 MobileNetV3-Small 并把 features0-3 权重切出来（T-R5-1 基准）。"""
    ref = mobilenet_v3_small(weights=None)
    sd = torch.load(mobile_path, map_location="cpu")
    ref.load_state_dict(sd)
    out = {}
    for i in range(4):
        for k, v in ref.features[i].state_dict().items():
            out[f"f{i}.{k}"] = v
    return out


def t_run5_mobile(pretrained_path, mobile_path, device, vit_depth=4):
    """Run5 R5-1 smoke（方案 §43 T-R5-1/2/3/5/6/7/9）：MobileDetail + legacy head。"""
    print("[RUN5-MOBILE] VIT4 + MobileDetail-P3 + legacy adapters + old FI/decoder smoke")

    # T-R5-1：Mobile features0-3 预训练原位继承（max_abs_diff == 0）
    ref_sd = _ref_mobilenet_p3(mobile_path)
    mob = MobileDetail(pretrained_weight_path=mobile_path).cpu()
    worst = 0.0
    for k, v in ref_sd.items():
        worst = max(worst, (mob.state_dict()[k].float() - v.float()).abs().max().item())
    assert worst == 0.0, f"mobile pretrained not loaded in place (diff {worst})"
    n_mobile = sum(p.numel() for p in mob.parameters())
    assert n_mobile == 10488, n_mobile
    print(f"  T-R5-1 mobile exact-load OK (worst={worst:.2e}); params = {n_mobile}")
    del ref_sd

    # T-R5-2：state dict 无 features.4+/classifier/avgpool（无隐藏 MobileNet）
    bad = [k for k in mob.state_dict() if not any(k.startswith(f"f{i}.") for i in range(4))]
    assert not bad, f"unexpected mobile keys: {bad[:5]}"
    print("  T-R5-2 no hidden MobileNet keys")

    # T-R5-3：形状（D2 16×128×128 / D4 16×64×64 / D8 24×32×32）
    x = torch.randn(2, 3, 256, 256, device=device)
    with torch.no_grad():
        d2, d4, d8 = mob.to(device)(x)
    assert [tuple(t.shape) for t in (d2, d4, d8)] == [(2, 16, 128, 128), (2, 16, 64, 64), (2, 24, 32, 32)], \
        [tuple(t.shape) for t in (d2, d4, d8)]
    print("  T-R5-3 shapes OK (16×128×128 / 16×64×64 / 24×32×32)")
    del mob

    # legacy 模式全链路：adapters（T-R5-5）+ 3-step 梯度 + 冻结 ViT checksum（T-R5-6/7）
    model = Trainer("tiny", pretrained_path=pretrained_path, resnet_pretrained=False,
                    mode="baseline", vit_depth=vit_depth, detail_mode="mobile_p3",
                    head_mode="legacy",
                    mobile_pretrained_weight_path=mobile_path).float().to(device).eval()
    n_ad = sum(p.numel() for p in model.encoder.detail_adapters.parameters())
    assert n_ad == 10112, n_ad
    assert any(k.startswith("encoder.detail_adapters.") for k in model.state_dict())
    with torch.no_grad():
        c = model.encoder.detail_capture(x)
    assert [tuple(t.shape) for t in c] == [(2, 64, 128, 128), (2, 128, 64, 64), (2, 256, 32, 32)], \
        [tuple(t.shape) for t in c]
    print(f"  T-R5-5 legacy adapters OK (params {n_ad}); adapted shapes 64/128/256")

    for n, p in model.named_parameters():
        if n.startswith("encoder.vit"):
            p.requires_grad_(False)
    vit_before = {k: v.detach().clone() for k, v in model.encoder.vit.state_dict().items()}
    pre = torch.randn(2, 3, 256, 256, device=device)
    post = torch.randn(2, 3, 256, 256, device=device)
    target = torch.randint(0, 2, (2, 1, 256, 256), device=device).float()
    optimizer = torch.optim.Adam([p for n, p in model.named_parameters() if not n.startswith("encoder.vit")], 2e-4)
    model.train()
    for step in range(3):
        out = model(pre, post)
        assert out.shape == (2, 1, 256, 256)
        loss = BCEDiceLoss(out, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        assert torch.isfinite(torch.tensor(loss.item()))
    for name, g in (("f0", model.encoder.detail.f0[0].weight.grad),
                    ("f3", model.encoder.detail.f3.block[2][0].weight.grad),
                    ("decoder", model.decoder.classfier[0].weight.grad)):
        assert g is not None and torch.isfinite(g).all() and g.norm() > 0, f"{name} grad broken"
    assert model.encoder.vit.patch_embed.proj.weight.grad is None
    worst = max((v.detach() - vit_before[k]).abs().max().item()
                for k, v in model.encoder.vit.state_dict().items())
    assert worst == 0.0, f"frozen ViT changed (diff {worst})"
    print("  T-R5-7 grads OK (Mobile f0/f3 + decoder; ViT grad None); T-R5-6 frozen ViT checksum == 0")

    total = measure_params(model)
    print(f"  total params = {total / 1e6:.4f} M (R5-1 含旧 FI/decoder，>3M 属预期)")
    try:
        flops, n_unsup = measure_flops(model, size=256)
        print(f"  T-R5-9 FLOPs = {flops:.4f} G (unsupported_ops={n_unsup})")
    except Exception as e:
        print(f"  [warn] FLOPs measurement failed ({type(e).__name__}: {e})")
    del model


def t_run5_sgdp(pretrained_path, mobile_path, device, vit_depth=4):
    """Run5 R5-2 smoke（方案 §43）：MobileDetail + SGDP 最终 2.10M 模型。"""
    print("[RUN5-SGDP] VIT4 + MobileDetail-P3 + SGDP head smoke")
    model = Trainer("tiny", pretrained_path=pretrained_path, resnet_pretrained=False,
                    mode="baseline", vit_depth=vit_depth, detail_mode="mobile_p3",
                    head_mode="sgdp",
                    mobile_pretrained_weight_path=mobile_path).float().to(device).eval()

    # T-R5-5：sgdp 模式 adapters 完全不注册
    assert not any("detail_adapters" in k for k in model.state_dict()), "adapters must not exist in sgdp mode"
    n_mobile = sum(p.numel() for p in model.encoder.detail.parameters())
    n_sgdp = sum(p.numel() for p in model.decoder.parameters())
    assert n_mobile == 10488, n_mobile
    assert n_sgdp == 115267, n_sgdp
    total = measure_params(model)
    assert total == 2102587, total
    assert total <= 2.11e6
    print(f"  T-R5-4 params OK: ViT 1,976,832 + Mobile {n_mobile} + SGDP {n_sgdp} = {total:,} (≤2.11M)")

    # T-R5-3 shapes + T-R5-8 time swap
    pre = torch.randn(2, 3, 256, 256, device=device)
    post = torch.randn(2, 3, 256, 256, device=device)
    with torch.no_grad():
        c = model.encoder.detail_capture(pre)
        assert [tuple(t.shape) for t in c] == [(2, 16, 128, 128), (2, 16, 64, 64), (2, 24, 32, 32)], \
            [tuple(t.shape) for t in c]
        out_ab = model(pre, post)
        out_ba = model(post, pre)
    assert out_ab.shape == (2, 1, 256, 256)
    d_swap = (out_ab - out_ba).abs().max().item()
    assert d_swap < 1e-5, f"time swap symmetry broken: {d_swap}"
    print(f"  T-R5-3 shapes OK; T-R5-8 time swap max_abs_diff = {d_swap:.2e}")

    # T-R5-7 梯度路径（冻结 ViT 由训练脚本控制，此处验证非冻结组件）
    target = torch.randint(0, 2, (2, 1, 256, 256), device=device).float()
    model.train()
    out = model(pre, post)
    loss = BCEDiceLoss(out, target)
    loss.backward()
    for name, g in (("f1", model.encoder.detail.f1.block[0][0].weight.grad),
                    ("sem16", model.decoder.sem16.conv.weight.grad),
                    ("fuse8", model.decoder.fuse8.dw.weight.grad),
                    ("classifier", model.decoder.classifier.weight.grad)):
        assert g is not None and torch.isfinite(g).all() and g.norm() > 0, f"{name} grad broken"
    print("  T-R5-7 grads OK (Mobile f1 / SGDP sem16/fuse8/classifier)")

    try:
        flops, n_unsup = measure_flops(model, size=256)
        print(f"  T-R5-9 FLOPs = {flops:.4f} G (unsupported_ops={n_unsup}; hard gate ≤2.0G)")
    except Exception as e:
        print(f"  [warn] FLOPs measurement failed ({type(e).__name__}: {e})")
    del model


def t_run7_csdp(pretrained_path, device, vit_depth=2):
    """Run7 R7-1 smoke（T-R7）：ViT2 + CSDP head 最终 ~1.18M 模型。"""
    print("[RUN7-CSDP] VIT2 + DepthPyramidHead smoke")
    model = Trainer("tiny", pretrained_path=pretrained_path, resnet_pretrained=False,
                    mode="baseline", vit_depth=vit_depth, detail_mode="depth_pyramid",
                    head_mode="csdp").float().to(device).eval()

    # T-R7-1：无独立 detail / legacy head
    bad = [k for k in model.state_dict() if ("encoder.resnet" in k or "detail_adapters" in k
           or k.startswith("decoder.structure_enhance") or k.startswith("decoder.up_c"))]
    assert not bad, f"forbidden keys: {bad[:5]}"
    # T-R7-0：DeiT depth-2 exact 继承（blocks0-1 原位、blocks2-11 不存在）
    sd = torch.load(pretrained_path, map_location="cpu")["model"]
    vit = model.encoder.vit
    assert len(vit.blocks) == 2
    resid = [k for k in model.state_dict() if any(k.startswith(f"encoder.vit.blocks.{j}.") for j in range(2, 12))]
    assert not resid
    worst = 0.0
    for i in range(2):
        for k in ("norm1.weight", "attn.qkv.weight", "mlp.fc1.weight"):
            worst = max(worst, (sd[f"blocks.{i}.{k}"].float() - vit.state_dict()[f"blocks.{i}.{k}"].cpu().float()).abs().max().item())
    assert worst == 0.0, f"pretrained blocks0-1 not exact (diff {worst})"
    print("  T-R7-0/1 OK: depth-2 exact inherit; no resnet/detail/legacy head keys")

    # T-R7-2：形状（B1/B2 = B×256×192；pred = B×1×256×256）+ 参数
    pre = torch.randn(2, 3, 256, 256, device=device)
    post = torch.randn(2, 3, 256, 256, device=device)
    with torch.no_grad():
        c = model.encoder.depth_pyramid_capture(pre)
        assert [tuple(t.shape) for t in c] == [(2, 256, 192), (2, 256, 192)], [tuple(t.shape) for t in c]
        out_ab = model(pre, post)
        out_ba = model(post, pre)
    assert out_ab.shape == (2, 1, 256, 256)
    n_head = sum(p.numel() for p in model.decoder.parameters())
    n_vit = sum(p.numel() for p in vit.parameters())
    total = measure_params(model)
    assert n_head == 90832, n_head
    assert n_vit == 1087104, n_vit
    assert total == 1177936, total
    assert total <= 1.5e6
    print(f"  T-R7-2 params OK: ViT2 {n_vit:,} + head {n_head:,} = {total:,} (≤1.5M)")

    # T-R7-3：时间交换对称
    d_swap = (out_ab - out_ba).abs().max().item()
    assert d_swap < 1e-6, f"time swap broken: {d_swap}"
    print(f"  T-R7-3 time swap max_abs_diff = {d_swap:.2e}")

    # T-R7-4/5：3-step 梯度 + 冻结 ViT checksum
    for n, p in model.named_parameters():
        if n.startswith("encoder.vit"):
            p.requires_grad_(False)
    vit_before = {k: v.detach().clone() for k, v in vit.state_dict().items()}
    target = torch.randint(0, 2, (2, 1, 256, 256), device=device).float()
    optimizer = torch.optim.Adam([p for n, p in model.named_parameters() if not n.startswith("encoder.vit")], 2e-4)
    model.train()
    for step in range(3):
        out = model(pre, post)
        loss = BCEDiceLoss(out, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        assert torch.isfinite(torch.tensor(loss.item()))
    for name, g in (("p1", model.decoder.p1.weight.grad), ("exp1", model.decoder.exp1[0].weight.grad),
                    ("out", model.decoder.out.weight.grad)):
        assert g is not None and torch.isfinite(g).all() and g.norm() > 0, f"{name} grad broken"
    assert vit.patch_embed.proj.weight.grad is None
    worst = max((v.detach() - vit_before[k]).abs().max().item() for k, v in vit.state_dict().items())
    assert worst == 0.0
    print("  T-R7-4/5 OK: head grads nonzero; frozen ViT checksum == 0")

    try:
        flops, n_unsup = measure_flops(model, size=256)
        print(f"  T-R7-6 FLOPs = {flops:.4f} G (unsupported_ops={n_unsup}; hard gate ≤2.0G)")
    except Exception as e:
        print(f"  [warn] FLOPs measurement failed ({type(e).__name__}: {e})")
    del model


def t_run8_b4_spe(pretrained_path, device, vit_depth=4):
    """Run8 R8-1 smoke（T-R8）：ViT4 + B4-SPE head 最终 ~2.031M 模型。"""
    print("[RUN8-B4SPE] VIT4 + B4SPEHead smoke")

    # T-R8-9：detail_mode=none_b4 时 vit_depth!=4 必须被拒绝
    try:
        Trainer("tiny", pretrained_path=pretrained_path, resnet_pretrained=False,
                mode="baseline", vit_depth=12, detail_mode="none_b4", head_mode="b4_spe")
        raise AssertionError("vit_depth=12 with none_b4 should have raised")
    except AssertionError as e:
        assert "vit_depth=4" in str(e), e
    print("  T-R8-9 OK: vit_depth!=4 rejected by Trainer assert")

    model = Trainer("tiny", pretrained_path=pretrained_path, resnet_pretrained=False,
                    mode="baseline", vit_depth=vit_depth, detail_mode="none_b4",
                    head_mode="b4_spe").float().to(device).eval()

    # T-R8-6：无独立 detail / legacy head
    assert model.encoder.resnet is None
    bad = [k for k in model.state_dict() if ("encoder.resnet" in k or "detail_adapters" in k
           or k.startswith("decoder.structure_enhance") or k.startswith("decoder.up_c"))]
    assert not bad, f"forbidden keys: {bad[:5]}"
    # T-R8-7：DeiT depth-4 exact 继承（blocks0-3 原位、blocks4-11 不存在、pos 插值口径）
    sd = torch.load(pretrained_path, map_location="cpu")["model"]
    vit = model.encoder.vit
    assert len(vit.blocks) == 4
    resid = [k for k in model.state_dict() if any(k.startswith(f"encoder.vit.blocks.{j}.") for j in range(4, 12))]
    assert not resid
    worst = 0.0
    for i in range(4):
        for k in ("norm1.weight", "attn.qkv.weight", "mlp.fc1.weight"):
            worst = max(worst, (sd[f"blocks.{i}.{k}"].float() - vit.state_dict()[f"blocks.{i}.{k}"].cpu().float()).abs().max().item())
    assert worst == 0.0, f"pretrained blocks0-3 not exact (diff {worst})"
    print("  T-R8-6/7 OK: no detail/legacy keys; depth-4 exact inherit")

    # T-R8-1/2/3/4：形状 + 参数 + 对称
    pre = torch.randn(2, 3, 256, 256, device=device)
    post = torch.randn(2, 3, 256, 256, device=device)
    with torch.no_grad():
        out_ab = model(pre, post)
        out_ba = model(post, pre)
    assert out_ab.shape == (2, 1, 256, 256)
    d_swap = (out_ab - out_ba).abs().max().item()
    assert d_swap < 1e-6, f"time swap broken: {d_swap}"
    n_head = sum(p.numel() for p in model.decoder.parameters())
    total = measure_params(model)
    assert n_head == 53872, n_head
    assert total == 2030704, total
    assert total <= 2.10e6
    print(f"  T-R8-1/2/3 OK: pred (2,1,256,256); swap {d_swap:.2e}; head {n_head:,}; total {total:,} (≤2.10M)")

    # T-R8-4/8：3-step 训练 + 冻结 ViT checksum + trainable 计数
    for n, p in model.named_parameters():
        if n.startswith("encoder.vit"):
            p.requires_grad_(False)
    vit_before = {k: v.detach().clone() for k, v in vit.state_dict().items()}
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_train == 53872 and n_train <= 0.06e6, n_train
    target = torch.randint(0, 2, (2, 1, 256, 256), device=device).float()
    optimizer = torch.optim.Adam([p for n, p in model.named_parameters() if not n.startswith("encoder.vit")], 2e-4)
    model.train()
    for step in range(3):
        out = model(pre, post)
        loss = BCEDiceLoss(out, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        assert torch.isfinite(torch.tensor(loss.item()))
    for name, g in (("pair", model.decoder.pair.weight.grad), ("exp1", model.decoder.exp1[0].weight.grad),
                    ("out", model.decoder.out.weight.grad)):
        assert g is not None and torch.isfinite(g).all() and g.norm() > 0, f"{name} grad broken"
    assert vit.patch_embed.proj.weight.grad is None
    worst = max((v.detach() - vit_before[k]).abs().max().item() for k, v in vit.state_dict().items())
    assert worst == 0.0
    print(f"  T-R8-4/8 OK: trainable={n_train:,} (≤0.06M); head grads nonzero; frozen ViT checksum == 0")

    try:
        flops, n_unsup = measure_flops(model, size=256)
        print(f"  T-R8-5 FLOPs = {flops:.4f} G (unsupported_ops={n_unsup}; hard gate ≤2.0G)")
    except Exception as e:
        print(f"  [warn] FLOPs measurement failed ({type(e).__name__}: {e})")
    del model


def t_run9_opre(pretrained_path, device, vit_depth=4):
    """Run9 R9-1 smoke（T-R9）：ViT4 + B4-OPRE head 最终 ~2.037M 模型。"""
    print("[RUN9-OPRE] VIT4 + OPREHead smoke")

    # T-R9-9：detail_mode=opre 时 vit_depth!=4 / head 不配套必须被拒绝
    for bad_kw in (dict(vit_depth=12, detail_mode="opre", head_mode="opre_spe"),
                   dict(vit_depth=4, detail_mode="none_b4", head_mode="opre_spe")):
        try:
            Trainer("tiny", pretrained_path=pretrained_path, resnet_pretrained=False,
                    mode="baseline", **bad_kw)
            raise AssertionError(f"{bad_kw} should have raised")
        except AssertionError as e:
            assert "requires" in str(e), e
    print("  T-R9-9 OK: vit_depth!=4 / head-detail 不配套均被拒绝")

    model = Trainer("tiny", pretrained_path=pretrained_path, resnet_pretrained=False,
                    mode="baseline", vit_depth=vit_depth, detail_mode="opre",
                    head_mode="opre_spe", opre_gate=1).float().to(device).eval()
    vit = model.encoder.vit

    # T-R9-0：functional conv 与 patch_embed.proj 等价（stride=16 无 pad）
    x = torch.randn(2, 3, 256, 256, device=device)
    with torch.no_grad():
        ref = vit.patch_embed.proj(x)
        fun = torch.nn.functional.conv2d(x, vit.patch_embed.proj.weight,
                                         vit.patch_embed.proj.bias, stride=16)
    assert (ref - fun).abs().max().item() < 1e-6
    # T-R9-1：O-PRE shape（共享权重、stride=8 reflect pad4）
    with torch.no_grad():
        ore = model.encoder.overlap_patch_capture(x)
    assert tuple(ore.shape) == (2, 192, 32, 32), tuple(ore.shape)
    # T-R9-2：无 duplicate patch kernel params（encoder 参数只有 ViT4 一份）
    n_enc = sum(p.numel() for p in model.encoder.parameters())
    n_vit = sum(p.numel() for p in vit.parameters())
    assert n_enc == n_vit, (n_enc, n_vit)
    assert not any("encoder.ore" in k or "encoder.opre" in k for k in model.state_dict())
    # T-R9-3：DeiT depth-4 逐位继承
    sd = torch.load(pretrained_path, map_location="cpu")["model"]
    worst = 0.0
    for i in range(4):
        for k in ("norm1.weight", "attn.qkv.weight", "mlp.fc1.weight"):
            worst = max(worst, (sd[f"blocks.{i}.{k}"].float() - vit.state_dict()[f"blocks.{i}.{k}"].cpu().float()).abs().max().item())
    assert worst == 0.0
    print("  T-R9-0/1/2/3 OK: conv equiv; O-PRE (2,192,32,32); no dup params; DeiT exact")

    # T-R9-4/5：输出 + 时间交换对称
    pre = torch.randn(2, 3, 256, 256, device=device)
    post = torch.randn(2, 3, 256, 256, device=device)
    with torch.no_grad():
        out_ab = model(pre, post)
        out_ba = model(post, pre)
    assert out_ab.shape == (2, 1, 256, 256)
    assert torch.isfinite(out_ab).all() and out_ab.min().item() >= 0.0 and out_ab.max().item() <= 1.0
    d_swap = (out_ab - out_ba).abs().max().item()
    assert d_swap < 1e-6, f"swap broken: {d_swap}"
    print(f"  T-R9-4/5 OK: pred (2,1,256,256) in [0,1]; swap {d_swap:.2e}")

    # T-R9-6/7：梯度 + 冻结 checksum + 参数
    for n, p in model.named_parameters():
        if n.startswith("encoder.vit"):
            p.requires_grad_(False)
    vit_before = {k: v.detach().clone() for k, v in vit.state_dict().items()}
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = measure_params(model)
    assert n_train == 60113 and n_train <= 0.065e6, n_train
    assert total == 2036945 and total <= 2.10e6, total
    target = torch.randint(0, 2, (2, 1, 256, 256), device=device).float()
    optimizer = torch.optim.Adam([p for n, p in model.named_parameters() if not n.startswith("encoder.vit")], 2e-4)
    model.train()
    for step in range(3):
        out = model(pre, post)
        loss = BCEDiceLoss(out, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        assert torch.isfinite(torch.tensor(loss.item()))
    for name, g in (("pair", model.decoder.pair.weight.grad),
                    ("local_proj", model.decoder.local_proj[0].weight.grad),
                    ("gate", model.decoder.gate32.weight.grad),
                    ("classifier", model.decoder.classifier.weight.grad)):
        assert g is not None and torch.isfinite(g).all() and g.norm() > 0, f"{name} grad broken"
    assert vit.patch_embed.proj.weight.grad is None
    worst = max((v.detach() - vit_before[k]).abs().max().item() for k, v in vit.state_dict().items())
    assert worst == 0.0
    print(f"  T-R9-6/7 OK: trainable={n_train:,}; total={total:,}; head grads nonzero; frozen ViT checksum == 0")

    try:
        flops, n_unsup = measure_flops(model, size=256)
        print(f"  T-R9-8 FLOPs = {flops:.4f} G (unsupported_ops={n_unsup}; hard gate ≤2.0G)")
    except Exception as e:
        print(f"  [warn] FLOPs measurement failed ({type(e).__name__}: {e})")
    del model


def t_run1_strfusion(pretrained_path, device):
    """Run1 STRFusion smoke（设计文档 §6.2）：ViT4 depth-as-scale + TAR/DCR rep。

    T-SF-1 C0/M1 epoch-0 输出逐位一致（max_diff == 0.0）
    T-SF-2 aux 分支 epoch-0 输出严格为 0
    T-SF-3 aux BN gamma 梯度非零；gamma nudge 后 aux conv 梯度出现
    T-SF-4 3-step 训练 loss 有限；冻结 ViT grad == None；非 ViT grad 非零
    T-SF-5 冻结 ViT4 checksum == corrected DeiT 初始化（depth-4 口径）
    T-SF-6 switch_to_deploy 后无 BN/aux 键；C0/M1 deploy Params 逐位相等且 <=3M
    T-SF-7 swap 不对称量级记录（report-only，不设 gate）
    T-SF-8 FLOPs 测量（fvcore）
    """
    from model.str_fusion import STRFusionNet
    print("[RUN1-STRFUSION] ViT4 depth-as-scale + TAR/DCR rep smoke")

    # ---- T-SF-1 / T-SF-2: epoch-0 identity + aux zero output ----
    torch.manual_seed(16)
    m0 = STRFusionNet(pretrained_path, dim=160, rep_mode="plain").to(device)
    torch.manual_seed(16)
    m1 = STRFusionNet(pretrained_path, dim=160, rep_mode="full").to(device)
    pre = torch.randn(2, 3, 256, 256, device=device)
    post = torch.randn(2, 3, 256, 256, device=device)
    m0.eval()
    m1.eval()
    with torch.no_grad():
        y0 = m0(pre, post)
        y1 = m1(pre, post)
    d01 = (y0 - y1).abs().max().item()
    print(f"  T-SF-1 C0/M1 epoch-0 outputs bitwise equal: max_diff={d01:.3e}")
    assert d01 == 0.0, "C0/M1 epoch-0 outputs must be bitwise identical"

    with torch.no_grad():
        P = torch.randn(2, 192, 64, 64, device=device)
        Q = torch.randn(2, 192, 64, 64, device=device)
        st = m1.tar.stage1.temporal
        z_aux = st.bn_s(st.proj_s(P + Q)) + st.bn_d(st.proj_d(Q - P))
        zm = z_aux.abs().max().item()
        P160 = torch.randn(2, 160, 64, 64, device=device)
        Q160 = torch.randn(2, 160, 64, 64, device=device)
        d1 = m1.decoder.fuse1
        zf = d1.bn_s(d1.fuse_s(P160 + Q160)) + d1.bn_d(d1.fuse_d(Q160 - P160))
        zfm = zf.abs().max().item()
    print(f"  T-SF-2 aux epoch-0 output exactly 0: temporal={zm:.3e}, fuse={zfm:.3e}")
    assert zm == 0.0 and zfm == 0.0

    # ---- T-SF-3: gradient families (two zero-init families, both must train) ----
    # family A (zero-conv): temporal/DW/fuse aux — conv weights zero-init, BN default
    #   (gamma=1). Conv grads are input-dependent -> nonzero @init; BN gamma grad is
    #   exactly 0 @init (x_hat==0) and appears after the conv departs 0.
    # family B (zero-gamma): PW serial aux — BN gamma=beta=0 (STRFusion P0
    #   adaptation, MPCR pattern), Kaiming convs keep x_hat nonzero -> gamma grad
    #   nonzero @init; conv grads are blocked by gamma=0 and appear after a nudge.
    target = torch.randint(0, 2, (2, 1, 256, 256), device=device).float()
    m1.train()
    opt = torch.optim.Adam([p for p in m1.parameters() if p.requires_grad], 2e-4)
    opt.zero_grad()
    loss = BCEDiceLoss(m1(pre, post, target), target)
    loss.backward()
    st = m1.tar.stage1.temporal
    g_conv = st.proj_s.weight.grad
    g_bn = st.bn_s.weight.grad
    assert g_conv is not None and g_conv.abs().max().item() > 0, "family A conv grad must be nonzero @init"
    assert g_bn is not None and g_bn.abs().max().item() == 0.0, "family A BN gamma grad must be 0 @init"
    pw = m1.tar.stage1.block.pw
    g_gammaB = pw.bn_lr.weight.grad.clone()
    g_convB = pw.pw2.weight.grad
    assert g_gammaB is not None and g_gammaB.abs().max().item() > 0, "family B BN gamma grad must be nonzero @init"
    assert g_convB is None or g_convB.abs().max().item() == 0.0, "family B conv grad must be 0 @init"
    print(f"  T-SF-3a @init: family A conv grad={g_conv.abs().max().item():.3e} gamma grad=0; "
          f"family B gamma grad={g_gammaB.abs().max().item():.3e} conv grad=0")
    with torch.no_grad():
        pw.bn_lr.weight += 0.1
    opt.zero_grad()
    loss = BCEDiceLoss(m1(pre, post, target), target)
    loss.backward()
    g_convB2 = pw.pw2.weight.grad
    assert g_convB2 is not None and g_convB2.abs().max().item() > 0, \
        "family B conv grad must appear after gamma nudge"
    print(f"  T-SF-3b family B conv grad nonzero after gamma nudge: |grad|_max={g_convB2.abs().max().item():.3e}")
    with torch.no_grad():
        pw.bn_lr.weight -= 0.1  # restore zero init
    opt.step()  # one step moves the family-A zero-conv away from 0
    opt.zero_grad()
    loss = BCEDiceLoss(m1(pre, post, target), target)
    loss.backward()
    g_bn2 = st.bn_s.weight.grad
    assert g_bn2 is not None and g_bn2.abs().max().item() > 0, "family A gamma grad must appear after 1 step"
    print(f"  T-SF-3c family A gamma grad nonzero after 1 optimizer step: |grad|_max={g_bn2.abs().max().item():.3e}")

    # ---- T-SF-4: 3-step training + frozen ViT ----
    m1.train()
    opt2 = torch.optim.Adam([p for p in m1.parameters() if p.requires_grad], 2e-4)
    for step in range(3):
        out = m1(pre, post, target)
        loss = BCEDiceLoss(out, target)
        opt2.zero_grad()
        loss.backward()
        opt2.step()
        assert torch.isfinite(torch.tensor(loss.item())), "loss is NaN/Inf"
    vit_grads = [p.grad for p in m1.encoder.parameters() if p.grad is not None]
    n_train = sum(p.numel() for p in m1.parameters() if p.requires_grad)
    print(f"  T-SF-4 3-step ok (loss={loss.item():.4f}); frozen ViT grad count={len(vit_grads)}; "
          f"trainable={n_train:,}")
    assert len(vit_grads) == 0 and n_train > 0

    # ---- T-SF-5: frozen ViT4 checksum == corrected DeiT init ----
    sd = torch.load(pretrained_path, map_location="cpu")["model"]
    vit = m1.encoder.vit
    keys = ["patch_embed.proj.weight", "patch_embed.proj.bias", "norm.weight", "norm.bias"]
    for i in range(4):
        keys += [f"blocks.{i}.norm1.weight", f"blocks.{i}.attn.qkv.weight", f"blocks.{i}.attn.qkv.bias"]
    worst = 0.0
    for k in keys:
        if not torch.equal(vit.state_dict()[k].cpu(), sd[k]):
            raise AssertionError(f"checksum mismatch at {k}")
    print("  T-SF-5 frozen ViT4 == corrected DeiT init (patch_embed/blocks0-3/norm)")

    # ---- T-SF-6: deploy branch-free + budget + C0/M1 deploy equality ----
    import copy
    m1.eval()
    m1d = copy.deepcopy(m1)
    m1d.switch_to_deploy()
    keys1 = list(m1d.state_dict().keys())
    bn_keys = [k for k in keys1 if "bn" in k]
    p1 = measure_params(m1d)
    m0.eval()
    m0d = copy.deepcopy(m0)
    m0d.switch_to_deploy()
    p0 = measure_params(m0d)
    print(f"  T-SF-6 deploy branch-free (bn_keys={len(bn_keys)}); deploy params C0={p0:,} M1={p1:,} "
          f"({p1 / 1e6:.3f}M, budget <=3M)")
    assert len(bn_keys) == 0
    assert p0 == p1, "C0/M1 deploy params must be equal"
    assert p1 <= 3.0e6, "deploy params exceed 3M budget"
    with torch.no_grad():
        y1 = m1(pre, post)   # fresh train-graph output AFTER the 3-step updates
        y1d = m1d(pre, post)
    err = (y1 - y1d).abs().max().item()
    flip = ((y1 > 0.5) != (y1d > 0.5)).float()
    disagree = flip.mean().item()
    n_flip = int(flip.sum().item())
    print(f"  T-SF-6b deploy fold after 3 steps (RECORD-ONLY): max_abs_error={err:.3e}, "
          f"disagree={disagree:.3e} (flips={n_flip})")
    # RECORD-ONLY: after 3 steps at batch 2 the BN running stats are degenerate
    # (tiny running_var -> FP32 rounding amplified by 1/sigma), and outputs still
    # crowd near 0.5 — both effects are state artifacts, not fold defects. The
    # architecture-level evidence with HEALTHY running stats + live branches is
    # T2b in test_str_reparam_equivalence.py (measured ~1e-6, disagree=0), and the
    # operational gate is the TRAINED model's [REPARAM-ARGMAX-DISAGREE] == 0 in the
    # formal TEST RESULTS block (dry run reports it at 2 healthy epochs).

    # ---- T-SF-7: swap asymmetry (report only) ----
    with torch.no_grad():
        y_ba = m1(post, pre)
        y_ab = m1(pre, post)
    asym = (y_ab - y_ba).abs().mean().item()
    print(f"  T-SF-7 swap asymmetry |pred(A,B)-pred(B,A)| mean={asym:.4e} (report-only)")

    # ---- T-SF-8: FLOPs ----
    try:
        flops, n_unsup = measure_flops(m1, size=256)
        print(f"  T-SF-8 FLOPs = {flops:.4f} G (unsupported_ops={n_unsup})")
    except Exception as e:
        print(f"  T-SF-8 [warn] FLOPs measurement failed ({type(e).__name__}: {e})")

    del m0, m1, m0d, m1d


def t_run11_tass(pretrained_path, device):
    """Run11 TASS smoke（方案 §8，T-S11-1..6）：
    T-S11-1 shapes（S1/S2/S3 投影后 192 通道 64/32/16；pred 256）
    T-S11-2 C0/M1 shared state dict 0/N 差异 + epoch-0 输出逐位一致（alpha=0）
    T-S11-3 alpha 梯度链：alpha grad 非零；stem conv grad @init 为 0（alpha=0 阻断）；
              alpha nudge 后 stem0/stage1/stage2/stage3 + projector 梯度非零
    T-S11-4 冻结 ViT4 checksum == corrected DeiT init；3-step 后不变
    T-S11-5 switch_to_deploy 只折 TAR/DCR：无 BN/aux 残留；C0/M1 各自 deploy <=5M；
              fold 误差记录（T2/T2b 由等价性套件回归）
    T-S11-6 预算打印：TOTAL/EFFECTIVE/TRAINABLE/DEPLOY/TASS-PARAMS/FLOPs
    """
    import copy
    from model.str_tass_fusion import STRTASSNet
    print("[RUN11-TASS] frozen ViT4 + TAR/DCR + TASS spatial residual smoke")

    # ---- T-S11-2 / T-S11-1: shared init + epoch-0 identity + shapes ----
    torch.manual_seed(16)
    m0 = STRTASSNet(pretrained_path, dim=160, spatial_mode="token").to(device)
    torch.manual_seed(16)
    m1 = STRTASSNet(pretrained_path, dim=160, spatial_mode="tass").to(device)
    s0 = m0.shared_state_dict()
    s1 = m1.shared_state_dict()
    diff_keys = [k for k in s0 if not torch.equal(s0[k], s1[k])]
    print(f"  T-S11-2 shared state dict differ: {len(diff_keys)}/{len(s0)}")
    assert len(diff_keys) == 0

    pre = torch.randn(2, 3, 256, 256, device=device)
    post = torch.randn(2, 3, 256, 256, device=device)
    m0.eval()
    m1.eval()
    with torch.no_grad():
        y0 = m0(pre, post)
        y1 = m1(pre, post)
        sp = m1.tass(pre)
    d01 = (y0 - y1).abs().max().item()
    print(f"  T-S11-2 C0/M1 epoch-0 outputs bitwise equal: max_diff={d01:.3e}")
    assert d01 == 0.0
    shapes = [tuple(t.shape) for t in sp]
    print(f"  T-S11-1 tass outputs: {shapes}; pred={tuple(y1.shape)}")
    assert shapes == [(2, 192, 64, 64), (2, 192, 32, 32), (2, 192, 16, 16)]
    assert y1.shape == (2, 1, 256, 256)

    # ---- T-S11-3: alpha gradient chain ----
    target = torch.randint(0, 2, (2, 1, 256, 256), device=device).float()
    m1.train()
    opt = torch.optim.Adam([p for p in m1.parameters() if p.requires_grad], 2e-4)
    opt.zero_grad()
    loss = BCEDiceLoss(m1(pre, post, target), target)
    loss.backward()
    ag = m1.alpha.grad
    assert ag is not None, "alpha grad must exist"
    n_nonzero = int((ag.abs() > 0).sum().item())
    stem_grads = {n: p.grad for n, p in m1.tass.named_parameters() if p.grad is not None}
    stem_nonzero = sum(1 for g in stem_grads.values() if g.abs().max().item() > 0)
    vit_grads = [p.grad for p in m1.encoder.parameters() if p.grad is not None]
    print(f"  T-S11-3a alpha grad nonzero: {n_nonzero}/3; stem conv grad nonzero @init: {stem_nonzero} "
          f"(expect 0, alpha=0 blocks); ViT grads: {len(vit_grads)}")
    assert n_nonzero >= 2, "at least 2/3 alpha grads must be nonzero"
    assert stem_nonzero == 0, "stem grads must be 0 at init (alpha=0)"
    assert len(vit_grads) == 0

    with torch.no_grad():
        m1.alpha += 0.1
    opt.zero_grad()
    loss = BCEDiceLoss(m1(pre, post, target), target)
    loss.backward()
    stem_grads2 = {n: p.grad for n, p in m1.tass.named_parameters() if p.grad is not None}
    nonzero2 = [n for n, g in stem_grads2.items() if g.abs().max().item() > 0]
    import itertools
    stage_hits = {"stem0": False, "s1": False, "s2": False, "s3": False, "p": False}
    for n in nonzero2:
        if "stem0" in n:
            stage_hits["stem0"] = True
        elif "s1." in n:
            stage_hits["s1"] = True
        elif "s2." in n:
            stage_hits["s2"] = True
        elif "s3." in n:
            stage_hits["s3"] = True
        if "p1" in n or "p2" in n or "p3" in n:
            stage_hits["p"] = True
    print(f"  T-S11-3b after alpha nudge: nonzero stem grads={len(nonzero2)}; stage hits={stage_hits}")
    assert all(stage_hits[k] for k in ("stem0", "s1", "s2", "s3", "p")), \
        "stem0/s1/s2/s3/projector must all have nonzero grads after alpha nudge"
    with torch.no_grad():
        m1.alpha -= 0.1

    # ---- T-S11-4: frozen ViT checksum before/after 3 steps ----
    sd = torch.load(pretrained_path, map_location="cpu")["model"]
    vit = m1.encoder.vit
    keys = ["patch_embed.proj.weight", "patch_embed.proj.bias", "norm.weight", "norm.bias"]
    for i in range(4):
        keys += [f"blocks.{i}.norm1.weight", f"blocks.{i}.attn.qkv.weight", f"blocks.{i}.attn.qkv.bias"]
    for k in keys:
        if not torch.equal(vit.state_dict()[k].cpu(), sd[k]):
            raise AssertionError(f"checksum mismatch at {k}")
    m1.train()
    opt2 = torch.optim.Adam([p for p in m1.parameters() if p.requires_grad], 2e-4)
    for step in range(3):
        out = m1(pre, post, target)
        loss = BCEDiceLoss(out, target)
        opt2.zero_grad()
        loss.backward()
        opt2.step()
        assert torch.isfinite(torch.tensor(loss.item())), "loss is NaN/Inf"
    for k in keys:
        if not torch.equal(vit.state_dict()[k].cpu(), sd[k]):
            raise AssertionError(f"ViT checksum changed after training at {k}")
    print(f"  T-S11-4 frozen ViT checksum unchanged after 3 steps (loss={loss.item():.4f})")

    # ---- T-S11-5: deploy fold ----
    m1.eval()
    m1d = copy.deepcopy(m1)
    m1d.switch_to_deploy()
    keys1 = list(m1d.state_dict().keys())
    # TAR/DCR 的 BN 必须全部折叠；TASS 是单路径静态 stem，其 BN 合法留在部署图（方案 §5.3）
    bn_keys = [k for k in keys1 if "bn" in k and not k.startswith("tass.")]
    p1 = measure_params(m1d)
    with torch.no_grad():
        y1 = m1(pre, post)
        y1d = m1d(pre, post)
    err = (y1 - y1d).abs().max().item()
    flip = ((y1 > 0.5) != (y1d > 0.5)).float()
    print(f"  T-S11-5 deploy fold: non-TASS bn_keys={len(bn_keys)}, max_abs_error={err:.3e} (recorded), "
          f"disagree={flip.mean().item():.3e}, deploy_params={p1:,} ({p1 / 1e6:.3f}M)")
    assert len(bn_keys) == 0

    m0d = copy.deepcopy(m0)
    m0d.switch_to_deploy()
    p0 = measure_params(m0d)
    tass_params = m1.tass.param_count()

    # ---- T-S11-6: budget ----
    total1 = measure_params(m1)
    train1 = sum(p.numel() for p in m1.parameters() if p.requires_grad)
    print(f"  T-S11-6 C0 deploy={p0:,} ({p0 / 1e6:.3f}M); M1 deploy={p1:,} ({p1 / 1e6:.3f}M); "
          f"TASS={tass_params:,}; M1 train-graph total={total1:,} trainable={train1:,}")
    assert p0 <= 5.0e6 and p1 <= 5.0e6, "deploy params exceed 5M budget"
    try:
        flops, n_unsup = measure_flops(m1, size=256)
        print(f"  T-S11-6 M1 FLOPs = {flops:.4f} G (unsupported_ops={n_unsup})")
    except Exception as e:
        print(f"  T-S11-6 [warn] FLOPs measurement failed ({type(e).__name__}: {e})")

    del m0, m1, m0d, m1d


def t_casa_tvim_str(pretrained_path, device):
    """CASA-TViM-STR smoke（CAACP-SS2D 主线，T-CA-1..6）。需服务器 GPU + mamba-ssm。

    T-CA-1 层级 taps 形状：F1(48,64²)/F2(64,32²)/F3(168,16²)/F4(224,8²)
    T-CA-2 TinyViM-S-Slim 预训练逐位继承：stage4 深度裁剪 key 重映射、
             retained keys 100% 原位（worst_diff==0）
    T-CA-3 epoch-0 identity：CAACP β=0 -> caacp0×plain vs caacp1×full 逐位一致；
             plain vs full（rep aux 零初始化）逐位一致
    T-CA-4 梯度链：β grad 非零；SS2D x_proj_weight/A_logs grad 非零；kernel fwd/bwd 无 NaN
    T-CA-5 变化分数对称：score(A,B)==score(B,A)（T1/T2 交换）
    T-CA-6 deploy 折叠 + 预算：TAR/DCR 无 BN 残留；deploy params <= 5M；fold 误差记录
    T-CA-7 CP-CAACP（Run2）：β=0 下 rank/cp 两模式输出逐位一致；零变化图严格退化为
             均匀池化；cell 权重与手算 w=(1+s·r)/Σ(1+s·r) 一致
    T-CA-8 FRH（Run2）：γ=0 修正项精确为 0；输出 128² 内部上采样；deploy 折叠为单
             3×3 Conv(dim->1)、分支属性删除、+768 params；init 态 fold 误差 < 1e-4、
             二值 disagreement == 0；γ 梯度链非零
    """
    import copy
    from model.casa_tvim_str_net import CASATViMSTRNet, ENCODER_DIMS
    from model.layers.caacp_ss2d import change_score_cosine_2d, rank_normalize_2d
    # 折叠等价性测量协议（STR T2）：TF32 off + deterministic，避免 GPU TF32 舍入放大
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    print("[CASA-TViM-STR] TinyViM-S-Slim + CAACP-SS2D + TAR/DCR smoke")

    # ---- T-CA-1 / T-CA-2: taps shapes + pretrain bitwise ----
    torch.manual_seed(16)
    m0 = CASATViMSTRNet(pretrained_path, caacp=False, rep_mode="plain").to(device)
    m0.eval()   # 必须先 eval：train 模式 forward 会污染 BN running stats，破坏 epoch-0 恒等比较
    stats = m0.encoder.load_stats()
    trunk = m0.encoder.param_count()
    print(f"  T-CA-2 slim trunk={trunk:,}; retained={stats['retained']}/{stats['pretrained_keys']} "
          f"worst_diff={stats['worst_diff']:.3e}")
    assert stats["retained"] + len(stats["dropped_intentional"]) == stats["pretrained_keys"], stats
    assert stats["worst_diff"] == 0.0, "pretrained retained keys must load bitwise"
    print(f"  T-CA-2 pretrained bitwise OK (retained={stats['retained']}, worst={stats['worst_diff']:.2e})")

    pre = torch.randn(2, 3, 256, 256, device=device)
    post = torch.randn(2, 3, 256, 256, device=device)
    with torch.no_grad():
        x2b = torch.cat([pre, post], dim=0)
        f1, f2, x = m0.encoder.forward_stem_s2_prefix(x2b)
        x = m0.encoder.forward_caacp_block(x)
        f3 = m0.encoder.norm4(x)
        f4 = m0.encoder.forward_stage4(x)
    shapes = [tuple(t.shape) for t in (f1, f2, f3, f4)]
    want = [(4, 48, 64, 64), (4, 64, 32, 32), (4, 168, 16, 16), (4, 224, 8, 8)]
    assert shapes == want, f"taps {shapes} != {want}"
    assert ENCODER_DIMS == (48, 64, 168, 224)
    print(f"  T-CA-1 taps OK: {[(c, h, w) for _, c, h, w in shapes]}")

    # ---- T-CA-3: epoch-0 identity (caacp=0 plain vs M1 caacp=1 full vs full) ----
    torch.manual_seed(16)
    m1 = CASATViMSTRNet(pretrained_path, caacp=True, rep_mode="full").to(device)
    torch.manual_seed(16)
    m2 = CASATViMSTRNet(pretrained_path, caacp=False, rep_mode="full").to(device)
    for m in (m0, m1, m2):
        m.eval()
    with torch.no_grad():
        y0 = m0(pre, post)
        y1 = m1(pre, post)
        y2 = m2(pre, post)
    d_ca = (y0 - y1).abs().max().item()
    d_rep = (y0 - y2).abs().max().item()
    print(f"  T-CA-3 epoch-0 identity: caacp0-vs-M1 max_diff={d_ca:.3e}; plain-vs-full max_diff={d_rep:.3e}")
    assert d_ca == 0.0, "CAACP beta=0 must give bitwise identity at epoch 0"
    assert d_rep == 0.0, "rep aux zero-init must give bitwise identity at epoch 0"
    assert y0.shape == (2, 1, 256, 256)
    assert torch.isfinite(y0).all(), "output has NaN"

    # ---- T-CA-4: gradient chain (beta / SS2D / upstream) ----
    target = torch.randint(0, 2, (2, 1, 256, 256), device=device).float()
    m1.train()
    opt = torch.optim.Adam([p for p in m1.parameters() if p.requires_grad], 2e-4)
    opt.zero_grad()
    loss = BCEDiceLoss(m1(pre, post), target)
    loss.backward()
    op = m1.encoder.caacp_op
    g_beta = op.beta.grad
    g_xproj = op.x_proj_weight.grad
    g_alog = op.A_logs.grad
    g_stem = m1.encoder.patch_embed[0][0].weight.grad
    for name, g in (("beta", g_beta), ("x_proj", g_xproj), ("A_logs", g_alog), ("stem", g_stem)):
        assert g is not None and torch.isfinite(g).all(), f"{name} grad broken"
    assert g_beta.abs().max().item() > 0, "beta grad must be nonzero @init (moves first)"
    assert g_xproj.abs().max().item() > 0, "SS2D x_proj grad must be nonzero"
    assert g_alog.abs().max().item() > 0, "SS2D A_logs grad must be nonzero"
    assert g_stem.abs().max().item() > 0, "upstream stem grad must be nonzero"
    print(f"  T-CA-4 gradient chain OK: |beta|={g_beta.abs().max().item():.3e} "
          f"|x_proj|={g_xproj.abs().max().item():.3e} |A_logs|={g_alog.abs().max().item():.3e} "
          f"|stem|={g_stem.abs().max().item():.3e}")
    opt.step()

    # ---- T-CA-5: change score swap symmetry ----
    m1.eval()
    with torch.no_grad():
        x2b = torch.cat([pre, post], dim=0)
        _, _, xf = m1.encoder.forward_stem_s2_prefix(x2b)
        xa, xb = xf.chunk(2, dim=0)
        s_ab = rank_normalize_2d(change_score_cosine_2d(xa, xb))
        s_ba = rank_normalize_2d(change_score_cosine_2d(xb, xa))
        x2br = torch.cat([post, pre], dim=0)
        _, _, xfr = m1.encoder.forward_stem_s2_prefix(x2br)
        xbr, xar = xfr.chunk(2, dim=0)
        s_swap = rank_normalize_2d(change_score_cosine_2d(xbr, xar))
    d_s = max((s_ab - s_ba).abs().max().item(), (s_ab - s_swap).abs().max().item())
    assert d_s == 0.0, f"change score not swap-symmetric: {d_s}"
    print(f"  T-CA-5 change score swap-symmetric (max_diff={d_s:.2e})")

    # ---- T-CA-6: deploy fold + budget ----
    m1d = copy.deepcopy(m1)
    m1d.eval()
    m1d.switch_to_deploy()
    keys_d = list(m1d.state_dict().keys())
    # 只要求 TAR/DCR 折叠干净；TinyViM 主干（encoder.*）的 BN 是推理期真实模块，合法保留
    bn_keys = [k for k in keys_d if "bn" in k and not k.startswith("encoder.")]
    p_deploy = measure_params(m1d)
    total = measure_params(m1)
    n_train = sum(p.numel() for p in m1.parameters() if p.requires_grad)
    with torch.no_grad():
        y1 = m1(pre, post)
        y1d = m1d(pre, post)
    err = (y1 - y1d).abs().max().item()
    flip = ((y1 > 0.5) != (y1d > 0.5)).float()
    print(f"  T-CA-6 deploy: bn_keys={len(bn_keys)}; params total={total:,} trainable={n_train:,} "
          f"deploy={p_deploy:,} ({p_deploy / 1e6:.3f}M); fold max_abs={err:.3e} "
          f"disagree={flip.mean().item():.3e} (init-stage, RECORD-ONLY)")
    assert len(bn_keys) == 0, f"deploy graph must be BN-free outside backbone: {bn_keys[:3]}"
    assert p_deploy <= 5.0e6, "deploy params exceed 5M budget"

    try:
        flops, n_unsup = measure_flops(m1, size=256)
        print(f"  T-CA-6 FLOPs = {flops:.4f} G (unsupported_ops={n_unsup})")
    except Exception as e:
        print(f"  T-CA-6 [warn] FLOPs measurement failed ({type(e).__name__}: {e})")

    # ---- T-CA-7: CP-CAACP 公式（Run2 首选一） ----
    from model.layers.caacp_ss2d import confidence_preserving_pool2x2
    torch.manual_seed(16)
    m_rank = CASATViMSTRNet(pretrained_path, caacp=True, rep_mode="full",
                            caacp_score_mode="rank").to(device)
    torch.manual_seed(16)
    m_cp = CASATViMSTRNet(pretrained_path, caacp=True, rep_mode="full",
                          caacp_score_mode="cp").to(device)
    m_rank.eval()
    m_cp.eval()
    with torch.no_grad():
        y_rank = m_rank(pre, post)
        y_cp = m_cp(pre, post)
    d_mode = (y_rank - y_cp).abs().max().item()
    assert d_mode == 0.0, f"beta=0 must make rank/cp modes bitwise identical, got {d_mode}"
    # 公式级：全零变化图（s=0）→ q=0 → 严格退化为均匀池化（无论 rank 如何）
    xp = torch.randn(2, 4, 8, 8, device=device)
    s_zero = torch.zeros(2, 8, 8, device=device)
    s_any = torch.rand(2, 8, 8, device=device)
    c_zero = confidence_preserving_pool2x2(xp, s_zero, s_any)
    c_avg = F.avg_pool2d(xp, 2)
    # 语义：零变化 -> 均匀池化。GPU cudnn avg_pool 与 torch sum 求和顺序不同，
    # 故按 float32 舍入容差断言（CPU float64 手算对照仍要求逐位一致）。
    assert (c_zero - c_avg).abs().max().item() < 1e-6, \
        f"zero-change image must reduce to uniform pool, got {(c_zero - c_avg).abs().max().item():.2e}"
    # 公式级：手算对照 w = (1+q)/Σ(1+q)，q = s*r（2×2 cell，CPU float64）
    x1 = torch.randn(1, 2, 4, 4, dtype=torch.float64)
    a1 = torch.tensor([[[[0.0, 1.0, 0.0, 0.0],
                         [0.5, 0.25, 0.0, 0.0],
                         [0.0, 0.0, 2.0, 0.0],
                         [0.0, 0.0, 0.0, 1.0]]]], dtype=torch.float64)
    r1 = torch.tensor([[[[0.0, 1.0, 0.0, 0.0],
                         [0.2, 0.8, 0.0, 0.0],
                         [0.0, 0.0, 1.0, 0.0],
                         [0.0, 0.0, 0.0, 0.5]]]], dtype=torch.float64)
    c1 = confidence_preserving_pool2x2(x1, a1, r1)
    q = (a1 * r1).view(1, 1, 2, 2, 2, 2)                     # per-cell 视图（与实现同口径）
    wv = (1.0 + q) / (1.0 + q).sum(dim=(3, 5), keepdim=True)
    c_manual = (x1.view(1, 2, 2, 2, 2, 2) * wv).sum(dim=(3, 5))
    # float64 下多 dim sum 的结合顺序差 ~1 ulp，按 1e-12 紧容差断言公式一致性
    assert (c1 - c_manual).abs().max().item() < 1e-12, \
        f"CP weights must match manual w=(1+s*r)/sum, got {(c1 - c_manual).abs().max().item():.2e}"
    # 单调性：同一 cell 内 q 最大的像素获得最大权重
    w = wv.view(1, 1, 4, 4)
    q4 = (a1 * r1)
    w00 = w[0, 0, 0:2, 0:2]
    assert torch.argmax(w00).item() == torch.argmax(q4[0, 0, 0:2, 0:2]).item()
    # 诊断口径：零变化 → 均匀权重 → cell 熵 = ln4（与 CAACP 内部熵记录同口径）
    from model.layers.caacp_ss2d import CAACPSS2D
    with torch.no_grad():
        s0 = torch.zeros(2, 16, 16, device=device)
        s_r = torch.rand(2, 16, 16, device=device)
        w_diag = CAACPSS2D._cp_weights(s0, s_r)
    ent0 = -(w_diag * (w_diag + 1e-12).log()).sum(dim=(3, 5)).mean().item()
    assert abs(ent0 - 1.3862943611198906) < 1e-5, f"uniform cell entropy must be ln4, got {ent0}"
    print(f"  T-CA-7 CP-CAACP OK: rank/cp epoch-0 bitwise (d={d_mode:.1e}); zero-change->uniform; manual w match")

    # ---- T-CA-8: FRH（Run2 首选二） ----
    torch.manual_seed(16)
    mf = CASATViMSTRNet(pretrained_path, caacp=True, rep_mode="full", frh=True).to(device)
    mf.eval()
    with torch.no_grad():
        z = torch.randn(2, 96, 64, 64, device=device)
        y_h = mf.head(z)
        zu = F.interpolate(z, scale_factor=2, mode="bilinear", align_corners=False)
        y_base = mf.head.base(zu)
        assert (y_h - y_base).abs().max().item() == 0.0, "gamma=0 must zero the correction term exactly"
        assert y_h.shape == (2, 1, 128, 128), "FRH internal resolution must be 128^2"
        y_f = mf(pre, post)
    assert y_f.shape == (2, 1, 256, 256)
    mfd = copy.deepcopy(mf)
    mfd.eval()
    mfd.switch_to_deploy()
    sd = mfd.state_dict()
    gone = [k for k in sd if k.startswith(("head.base", "head.dw", "head.pw", "head.gamma"))]
    fused = [k for k in sd if k.startswith("head.fused")]
    assert not gone and fused, f"FRH deploy must delete branches (left={gone[:3]}, fused={fused[:3]})"
    p_fused = sum(v.numel() for k, v in sd.items() if k.startswith("head.fused"))
    assert p_fused == 96 * 9 + 1, f"fused head params {p_fused} != 865"
    p_frh_deploy = measure_params(mfd)
    assert p_frh_deploy - p_deploy == 768, f"FRH deploy delta {p_frh_deploy - p_deploy} != +768"
    assert p_frh_deploy <= 5.0e6
    with torch.no_grad():
        y_fd = mfd(pre, post)
    err_frh = (y_f - y_fd).abs().max().item()
    flip_frh = ((y_f > 0.5) != (y_fd > 0.5)).float().mean().item()
    print(f"  T-CA-8 FRH deploy: head->Conv2d(96,1,k3) {p_fused} params; deploy={p_frh_deploy:,} "
          f"({p_frh_deploy / 1e6:.3f}M, +768 vs plain head); init fold max_abs={err_frh:.3e} "
          f"disagree={flip_frh:.3e}")
    assert err_frh < 1e-4, f"FRH init-stage fold error must be < 1e-4, got {err_frh}"
    assert flip_frh == 0.0, "FRH fold must be binarization-identical at init"
    # γ 梯度链（train 态最后测，避免污染前面 eval 态 BN 比较）
    mf2 = copy.deepcopy(mf)
    mf2.train()
    loss_f = BCEDiceLoss(mf2(pre, post), target)
    loss_f.backward()
    g_gamma = mf2.head.gamma.grad
    assert g_gamma is not None and torch.isfinite(g_gamma).all() and g_gamma.abs().max().item() > 0
    print(f"  T-CA-8 FRH gamma grad chain OK (|g_gamma|={g_gamma.abs().max().item():.3e})")

    del m0, m1, m2, m1d, m_rank, m_cp, mf, mfd, mf2

def full_smoke(pretrained_path, mode, device):
    print(f"[T1] full-network smoke (mode={mode})")
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
        out = model(pre, post, target) if mode == "oracle" else model(pre, post)
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
    parser.add_argument('--mobile_pretrained_weight_path', type=str, default=None,
                        help='MobileNetV3-Small ImageNet weights (run5 modes)')
    parser.add_argument('--tinyvim_pretrained_weight_path', type=str, default=None,
                        help='TinyViM-S 1000e checkpoint (casa_tvim_str mode)')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--mode', type=str, default='all',
                        choices=['all', 'baseline', 'casaa', 'saa', 'oracle', 'detail', 'detail_fused',
                                 'run4', 'run4_light', 'run4_light48', 'run4_light_bnrelu', 'run4_psd',
                                 'run5_mobile', 'run5_sgdp', 'run7_csdp', 'run8_b4_spe', 'run9_opre',
                                 'strfusion', 'tass', 'casa_tvim_str'],
                        help='all | baseline | casaa | saa | oracle | detail | detail_fused | run4 | run4_light | run4_light48 | run4_light_bnrelu | run4_psd | run5_mobile | run5_sgdp | run7_csdp | run8_b4_spe | run9_opre | strfusion | tass | casa_str')
    args = parser.parse_args()

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[SMOKE] mode={args.mode} device={device}")

    t0_qkv_sliced_equivalence(device)
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
    if args.mode in ("oracle", "all"):
        t5_oracle_router(args.pretrained_weight_path, device)
        full_smoke(args.pretrained_weight_path, "oracle", device)
    if args.mode in ("detail", "all"):
        t6_detail_router(args.pretrained_weight_path, device)
        full_smoke(args.pretrained_weight_path, "detail", device)
    if args.mode in ("detail_fused", "all"):
        t6_detail_router(args.pretrained_weight_path, device)
        full_smoke(args.pretrained_weight_path, "detail_fused", device)
    if args.mode in ("run4", "all"):
        t_run4(args.pretrained_weight_path, device, vit_depth=4)
    if args.mode in ("run4_light", "all"):
        t_run4_light(args.pretrained_weight_path, device)
    if args.mode in ("run4_light48", "all"):
        t_run4_light(args.pretrained_weight_path, device, detail_mode='light48')
    if args.mode in ("run4_light_bnrelu", "all"):
        t_run4_light(args.pretrained_weight_path, device, detail_mode='light_bnrelu')
    if args.mode in ("run4_psd", "all"):
        t_run4_psd(args.pretrained_weight_path, device, vit_depth=4)
    if args.mode in ("run5_mobile", "all"):
        t_run5_mobile(args.pretrained_weight_path, args.mobile_pretrained_weight_path, device, vit_depth=4)
    if args.mode in ("run5_sgdp", "all"):
        t_run5_sgdp(args.pretrained_weight_path, args.mobile_pretrained_weight_path, device, vit_depth=4)
    if args.mode in ("run7_csdp", "all"):
        t_run7_csdp(args.pretrained_weight_path, device, vit_depth=2)
    if args.mode in ("run8_b4_spe", "all"):
        t_run8_b4_spe(args.pretrained_weight_path, device, vit_depth=4)
    if args.mode in ("run9_opre", "all"):
        t_run9_opre(args.pretrained_weight_path, device, vit_depth=4)
    if args.mode in ("strfusion", "all"):
        t_run1_strfusion(args.pretrained_weight_path, device)
    if args.mode in ("tass", "all"):
        t_run11_tass(args.pretrained_weight_path, device)
    if args.mode == "casa_tvim_str":   # 需 tinyvim_s_1000e.pth + GPU mamba-ssm，不并入 all（all 使用 DeiT 路径）
        t_casa_tvim_str(args.tinyvim_pretrained_weight_path or args.pretrained_weight_path, device)

    print("[SMOKE] ALL OK")


if __name__ == "__main__":
    main()
