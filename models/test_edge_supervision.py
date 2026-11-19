"""Checks for the Sobel edge target and the two edge heads."""
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, ".")

from model.casa_net import CASANet
from model.edge_supervision import EdgeHead, edge_loss, sobel_edge_map


def test_sobel_ring_and_flat():
    mask = torch.zeros(1, 1, 32, 32)
    mask[:, :, 8:24, 8:24] = 1.0
    edge = sobel_edge_map(mask, dilate=0)
    assert float(edge.max()) <= 1.0 + 1e-6
    assert abs(float(edge[0, 0, 16, 16])) < 1e-6, "interior must be flat"
    assert abs(float(edge[0, 0, 2, 2])) < 1e-6, "background must be flat"
    assert float(edge[0, 0, 8, 16]) > 0.9, "the boundary ring must be high"
    # a step of 1 pixel produces unit Sobel magnitude
    thin = torch.zeros(1, 1, 32, 32)
    thin[:, :, 16, :] = 1.0
    assert abs(float(sobel_edge_map(thin, 0).max()) - 1.0) < 1e-5


def test_dilation_widens_the_ring():
    mask = torch.zeros(1, 1, 32, 32)
    mask[:, :, 8:24, 8:24] = 1.0
    tight = sobel_edge_map(mask, dilate=0)
    wide = sobel_edge_map(mask, dilate=1)
    assert float(wide.sum()) > float(tight.sum())
    # the raw Sobel ring is already two pixels wide (the 3x3 kernel straddles the
    # step); dilation must extend it one pixel further on each side
    assert float(tight[0, 0, 7, 16]) > 0
    assert float(tight[0, 0, 6, 16]) == 0
    assert float(wide[0, 0, 6, 16]) > 0
    assert float(wide[0, 0, 5, 16]) == 0


def test_edge_loss_prefers_a_correct_prediction():
    mask = torch.zeros(1, 1, 32, 32)
    mask[:, :, 8:24, 8:24] = 1.0
    edge = sobel_edge_map(mask, 1)
    good = torch.logit(edge.clamp(1e-4, 1 - 1e-4))
    bad = torch.zeros_like(edge) - 3.0
    assert float(edge_loss(good, edge)) < float(edge_loss(bad, edge))


SIDE = 128  # the coarsest scale must stay at least as large as its 4x4 grid
# MobileNetV2 strides: the four TAR scales are side/4, side/8, side/16, side/32
EXPECTED = [(SIDE // 4, SIDE // 4), (SIDE // 8, SIDE // 8),
            (SIDE // 16, SIDE // 16), (SIDE // 32, SIDE // 32)]


def _edge_maps(module, tar_variant, edge_supervision):
    net = CASANet(module=module, tar_variant=tar_variant, edge_supervision=edge_supervision).train()
    with torch.no_grad():
        _, auxiliary = net(torch.randn(1, 3, SIDE, SIDE), torch.randn(1, 3, SIDE, SIDE),
                           return_aux=True)
    return net, auxiliary


def test_heads_are_opt_in():
    _, plain = _edge_maps("none", "cg", False)
    assert plain.get("edge_logits") is None
    _, armed = _edge_maps("none", "cg", True)
    maps = armed["edge_logits"]
    assert [tuple(m.shape[-2:]) for m in maps] == EXPECTED


def test_bridge_and_retrieval_both_expose_edges():
    _, bridge = _edge_maps("manifold", "cg", True)
    assert len(bridge["edge_logits"]) == 8, "four bridge maps plus four retrieval maps"
    _, retrieval_only = _edge_maps("manifold", "spp", True)
    assert len(retrieval_only["edge_logits"]) == 4
    _, difference = _edge_maps("difference", "spp", True)
    assert len(difference["edge_logits"]) == 4


def test_parameter_cost_is_sixty_five_per_head():
    head = EdgeHead(64)
    assert sum(p.numel() for p in head.parameters()) == 65
    base = CASANet(module="manifold", tar_variant="cg").eval()
    armed = CASANet(module="manifold", tar_variant="cg", edge_supervision=True).eval()
    delta = sum(p.numel() for p in armed.parameters()) - sum(p.numel() for p in base.parameters())
    assert delta == 130, delta


def test_gradients_reach_both_heads():
    net = CASANet(module="manifold", tar_variant="cg", edge_supervision=True).train()
    a, b = torch.randn(2, 3, SIDE, SIDE), torch.randn(2, 3, SIDE, SIDE)
    target = torch.zeros(2, 1, SIDE, SIDE)
    target[:, :, SIDE // 4:3 * SIDE // 4, SIDE // 4:3 * SIDE // 4] = 1.0
    _, auxiliary = net(a, b, return_aux=True)
    edge = sobel_edge_map(target, 1)
    loss = sum(edge_loss(m, F.interpolate(edge, size=m.shape[-2:], mode="area"))
               for m in auxiliary["edge_logits"])
    loss.backward()
    assert net.tar.edge_head.proj.weight.grad is not None
    assert float(net.tar.edge_head.proj.weight.grad.abs().sum()) > 0
    assert net.context.edge_head.proj.weight.grad is not None
    assert float(net.context.edge_head.proj.weight.grad.abs().sum()) > 0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"[PASS] {name}")
