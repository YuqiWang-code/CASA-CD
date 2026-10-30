import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvExpert(nn.Module):
    """Lightweight convolutional expert used by IA-MoE and TD-MoE."""

    def __init__(self, indim, outdim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(indim, outdim, 1),
            nn.ReLU(),
            nn.Conv2d(outdim, outdim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(outdim, outdim, 1),
        )

    def forward(self, x):
        return self.net(x)


class InteractionAdaptiveMoELayer(nn.Module):
    """Interaction-Adaptive MoE over difference and summation features."""

    def __init__(self, indim, unified_dim=196, outdim=None, n_experts=4, topk=2, noisy_gating=True):
        super().__init__()
        self.n_experts = n_experts
        self.topk = topk
        self.unified_dim = unified_dim
        self.noisy_gating = noisy_gating

        if outdim is None:
            outdim = indim

        self.shared_reduction = nn.Conv2d(indim, unified_dim, 1)
        self.experts = nn.ModuleList([ConvExpert(unified_dim, unified_dim) for _ in range(n_experts)])
        self.gate_proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(unified_dim * 4, unified_dim // 2),
            nn.ReLU(),
            nn.Linear(unified_dim // 2, n_experts),
        )
        self.channel_expansion = nn.Conv2d(unified_dim, outdim, 1)
        self.identity_proj = nn.Conv2d(indim * 2, outdim, 1)

        nn.init.constant_(self.channel_expansion.weight, 0)
        nn.init.constant_(self.channel_expansion.bias, 0)

    def forward(self, x1, x2, training=True):
        b, _, h, w = x1.shape
        identity_out = self.identity_proj(torch.cat([x1, x2], dim=1))

        f1 = self.shared_reduction(x1)
        f2 = self.shared_reduction(x2)
        feat_diff = f1 - f2
        feat_sum = f1 + f2

        gate_feat = torch.cat([f1, f2], dim=1)
        gate_avg = F.adaptive_avg_pool2d(gate_feat, 1)
        gate_max = F.adaptive_max_pool2d(gate_feat, 1)
        gate_logits = self.gate_proj(torch.cat([gate_avg, gate_max], dim=1))

        if training and self.noisy_gating:
            gate_logits = gate_logits + torch.randn_like(gate_logits) * 0.1

        gate_prob = F.softmax(gate_logits, dim=-1)
        topk_vals, topk_idx = torch.topk(gate_prob, self.topk, dim=-1)
        topk_vals = topk_vals / topk_vals.sum(dim=-1, keepdim=True)

        out_moe = torch.zeros((b, self.unified_dim, h, w), device=x1.device, dtype=x1.dtype)
        split_idx = self.n_experts // 2
        for i in range(b):
            for k in range(self.topk):
                idx = topk_idx[i, k]
                val = topk_vals[i, k]
                expert_input = feat_diff[i : i + 1] if idx < split_idx else feat_sum[i : i + 1]
                out_moe[i] += val * self.experts[idx](expert_input).squeeze(0)

        return identity_out + self.channel_expansion(out_moe)
