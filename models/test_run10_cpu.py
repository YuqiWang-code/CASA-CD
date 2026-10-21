"""Run10 regressions for the late HSAM failure path (CPU only)."""
from types import SimpleNamespace

import torch

from model.layers.hsam import local_sigma, poincare_pairwise_distance
from train import ChangeViTTrainer


def stage(rank):
    return dict(dim=8, tokens=4, feature_source="pre_norm",
                radius_mode="local_dispersion", logit_norm="std",
                radius_cv=0.1, one_minus_r2_cv=0.2,
                one_minus_r2_min=0.1, at_eps_frac=0.0,
                rank_corr_hd_vs_cos=rank, attn_prob_max=0.5,
                attn_entropy_norm=0.8, attn_entropy_paper=1.0,
                atanh_clip_frac=0.0, agg_over_feat=0.5,
                out_over_in=1.2, finite=True)


class FakeModel:
    hsam_op = object()

    def __init__(self):
        self.rank = 0.7

    def hsam_diag(self):
        return {"stages": [3], "stage3": stage(self.rank)}


def main():
    model = FakeModel()
    logs = []
    trainer = SimpleNamespace(model=model, log=logs.append)
    ChangeViTTrainer._log_and_check_hsam(trainer, 1, 200)
    model.rank = 0.9992
    try:
        ChangeViTTrainer._log_and_check_hsam(trainer, 2, 400)
    except SystemExit as exc:
        assert exc.code == 4
    else:
        raise AssertionError("a later epoch's degeneration was not stopped")

    p = torch.tensor([[[0.2, -0.1], [0.2, -0.1]]], requires_grad=True)
    distance = poincare_pairwise_distance(p, p)
    distance.sum().backward()
    assert distance.abs().max() < 1e-6
    assert torch.isfinite(p.grad).all()

    x = torch.ones(1, 4, 4, 4, requires_grad=True)
    sigma = local_sigma(x)
    sigma.sum().backward()
    assert torch.isfinite(sigma).all() and torch.isfinite(x.grad).all()
    assert sigma.max() - sigma.min() < 1e-8, "padding fabricated radius variation"
    print("Run10 CPU: later-epoch gate, coincident-distance gradient, uniform-radius gradient PASS")


if __name__ == "__main__":
    main()
