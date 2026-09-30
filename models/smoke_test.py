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
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--mode', type=str, default='all',
                        choices=['all', 'baseline', 'casaa', 'saa', 'oracle', 'detail', 'detail_fused',
                                 'run4', 'run4_light', 'run4_light48', 'run4_light_bnrelu', 'run4_psd'],
                        help='all | baseline | casaa | saa | oracle | detail | detail_fused | run4 | run4_light | run4_light48 | run4_light_bnrelu | run4_psd')
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

    print("[SMOKE] ALL OK")


if __name__ == "__main__":
    main()
