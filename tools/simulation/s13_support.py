"""s13_support.py — S13a-0：confidence support 的梯度阻断复现（执行方案 §7.2）

为什么单独一个脚本
------------------
`ManifoldContext.support_term` 在 confidence 模式下算
    support = |2 * sigmoid(support_head(x)).mean_cell - 1|, clamp_min(1e-6)
而 `support_head` 是零初始化。零点处 sigmoid=0.5 → |2*0.5-1| = 0 → 被 clamp 到 1e-6，
同时 abs 在 0 处的次梯度取 0、clamp 在边界上的梯度也是 0 —— 两条路径同时断。
本脚本用最小实例复现，并对比三种替代写法，输出可复现的梯度范数。

不训练主网络、不改冻结 runtime。真实 checkpoint 上的权重/输出分位数由服务器端
`s13_read_ckpt.py` 读取（本地没有 checkpoint），这里只做机制复现。

运行
----
    python tools/simulation/s13_support.py
输出
----
    docs/simulation/hyperbolic_v4/results/log_s13a.txt（UTF-8）
"""
from __future__ import annotations

import io
import json
import math
import os
import sys

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "docs", "simulation", "hyperbolic_v4", "results")
os.makedirs(OUT, exist_ok=True)

LINES: list[str] = []


def say(msg: str = "") -> None:
    print(msg, flush=True)
    LINES.append(msg)


def confidence_support(head: torch.nn.Conv2d, x: torch.Tensor, grid: int = 4) -> torch.Tensor:
    """复刻 manifold_context.support_term 的 confidence 分支（逐字对照源码）。"""
    cell = F.adaptive_avg_pool2d(head(x).sigmoid(), grid)
    share = cell.flatten(2).transpose(1, 2).reshape(cell.shape[0], -1)
    return (2 * share - 1).abs().clamp_min(1e-6)


def share_support(head: torch.nn.Conv2d, x: torch.Tensor, grid: int = 4) -> torch.Tensor:
    """原 mass 分支的 share（无 abs），作为对照。"""
    cell = F.adaptive_avg_pool2d(head(x).sigmoid(), grid)
    share = cell.flatten(2).transpose(1, 2).reshape(cell.shape[0], -1)
    return share.clamp_min(1e-6)


def logsigmoid_support(head: torch.nn.Conv2d, x: torch.Tensor, grid: int = 4) -> torch.Tensor:
    """建议的修复：logsigmoid 直接给 log 先验，零点处非零导数。"""
    cell = F.adaptive_avg_pool2d(head(x), grid)
    share = cell.flatten(2).transpose(1, 2).reshape(cell.shape[0], -1)
    return F.logsigmoid(share)


def grad_norm(head: torch.nn.Conv2d) -> float:
    if head.weight.grad is None:
        return float("nan")
    return float(head.weight.grad.norm().item())


def probe(name: str, fn, zero_init: bool = True, channels: int = 64, seed: int = 0) -> dict:
    torch.manual_seed(seed)
    head = torch.nn.Conv2d(channels, 2, 1)
    if zero_init:
        torch.nn.init.zeros_(head.weight)
        torch.nn.init.zeros_(head.bias)
    else:
        torch.nn.init.kaiming_uniform_(head.weight, a=math.sqrt(5))
        torch.nn.init.zeros_(head.bias)
    head.train()
    x = torch.randn(2, channels, 8, 8)
    prior = fn(head, x)                     # [B, n_cells]
    n = prior.shape[1]
    # 构造一个"确实依赖 prior"的目标：prior 作为 key 的 log 权重参与 softmax，
    # 监督是让某些 key 的质量更大（非恒定目标），否则 softmax 对常数不敏感。
    torch.manual_seed(1)
    logits = torch.randn(2, 5, n)           # 5 个查询
    target = torch.zeros(2, 5, n)
    target[:, :, : n // 2] = 1.0            # 前一半 key 是目标
    logp = torch.log_softmax(logits + prior[:, None, :], dim=-1)
    loss = -(target * logp).sum(-1).mean()
    head.zero_grad(set_to_none=True)
    loss.backward()
    rec = {
        "arm": name,
        "zero_init": zero_init,
        "support_min": float(prior.min().item()),
        "support_max": float(prior.max().item()),
        "support_mean": float(prior.mean().item()),
        "grad_norm": grad_norm(head),
        "grad_nonzero": int((head.weight.grad != 0).sum().item()) if head.weight.grad is not None else -1,
        "loss": float(loss.item()),
    }
    say(f"[{name}] zero_init={zero_init} support∈[{rec['support_min']:.3e},{rec['support_max']:.3e}] "
        f"mean={rec['support_mean']:.3e} | grad_norm={rec['grad_norm']:.3e} "
        f"nonzero={rec['grad_nonzero']} loss={rec['loss']:.4f}")
    return rec


def main() -> int:
    say("===== S13a-0：confidence support 梯度阻断复现（冻结源码逐字对照） =====")
    say("源码：models/model/layers/manifold_context.py::support_term")
    say("  share   = adaptive_avg_pool2d(support_head(x).sigmoid(), grid)")
    say("  support = |2*share - 1|  (confidence) -> cat(...).clamp_min(1e-6)")
    say("  support_head = Conv2d(channels, 2, 1)，weight/bias 均 zeros_ 初始化")
    say("")

    recs = []
    say("--- 1) 现状：confidence 分支 + 零初始化 ---")
    recs.append(probe("confidence + zeros", confidence_support, zero_init=True))
    say("")
    say("--- 2) 对照：confidence 分支 + 非零初始化 ---")
    recs.append(probe("confidence + kaiming", confidence_support, zero_init=False))
    say("")
    say("--- 3) 对照：share（mass 式，无 abs）+ 零初始化 ---")
    recs.append(probe("share + zeros", share_support, zero_init=True))
    say("")
    say("--- 4) 建议修复：logsigmoid + 零初始化 ---")
    recs.append(probe("logsigmoid + zeros", logsigmoid_support, zero_init=True))
    say("")

    say("解释：")
    say("  · 零初始化时 sigmoid=0.5，abs(2*0.5-1)=0，clamp 到 1e-6；abs 在 0 处次梯度为 0，")
    say("    clamp 在边界上梯度也为 0 —— 两条路径同时断，support_head 永远收不到梯度。")
    say("  · 同一权重下 share（无 abs）仍有梯度，说明断点来自 abs+clamp，不是零初始化本身。")
    say("  · logsigmoid 在零点导数 = -0.5·… ≠ 0，可作中性初始化下的修复候选（S13b 才谈训练）。")
    say("  · 这解释了 S7 观察到的 confidence support 恒为 1e-6；但'先验因此没有任务作用'")
    say("    仍需 S13a-1 的冻结先验反事实来判断，不能由梯度为 0 直接推出。")
    say("")

    zero_grad_ok = recs[0]["grad_norm"] == 0.0 and recs[0]["grad_nonzero"] == 0
    nonzero_alt = recs[2]["grad_nonzero"] > 0 and recs[3]["grad_nonzero"] > 0
    say(f"判定：零初始化 confidence 梯度严格为 0 = {zero_grad_ok}；"
        f"share/logsigmoid 有非零梯度 = {nonzero_alt}")

    with io.open(os.path.join(OUT, "log_s13a.txt"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(LINES) + "\n")
    with io.open(os.path.join(OUT, "res_s13a.json"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps({"records": recs, "zero_grad_ok": zero_grad_ok,
                             "nonzero_alt": nonzero_alt}, ensure_ascii=False, indent=2) + "\n")
    say(f"已写 {os.path.relpath(os.path.join(OUT, 'log_s13a.txt'), ROOT)}")
    return 0 if (zero_grad_ok and nonzero_alt) else 1


if __name__ == "__main__":
    sys.exit(main())
