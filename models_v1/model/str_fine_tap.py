"""R4-FET1：FineEvidenceTap1x1 —— 细尺度变化证据直达式 DCR 残差接入（1×1 可折叠版）。

方案来源：docs/temporary/CASA-CD_Run4_小目标瓶颈结构改进与实验设计_2026-10-10.md §2/§3。
插入点：`CASATViMSTRNet.forward()` 中 **DCR refine 输出之后**（`R' = R + T(P, Q)`），
其中 `P = f1a`、`Q = f1b` 来自共享编码器 `encoder.norm0`（1/4, 48C, 64²；Diag1 节点 L01b_norm0）。

训练图（全部 stride=1 / padding=0 / groups=1，无 BN、无激活、无新 loss）：

    D  = |Q - P|                                   # 输入特征构造（非线性留在卷积之前）
    U  = W_pq * [P, Q] + b_pq                      # W_pq: 96x96x1x1, b_pq: 96
    V  = W_diff * D                                # W_diff: 96x48x1x1, bias=False
    T  = gamma * (U + V),  R' = R + T              # gamma: 标量，零初始化

初始化纪律（零初始化 + RNG 隔离）：
  * `W_diff = 0`、`gamma = 0` ⇒ epoch-0 时 `T ≡ 0`、`R' ≡ R`（与关闭 FET 逐位一致）；
  * `W_pq/b_pq` 用**独立局部 RNG**（`torch.random.fork_rng` + 固定常量种子）初始化，
    不消耗全局随机流 ⇒ 开关 fine_tap 不影响既有模块的初始化（T2 逐位对拍的前提）；
  * `gamma=0` 时**不使用 `if gamma == 0: return R` 之类的动态图捷径**，保持计算图，
    保证 gamma 与（开门后的）两路卷积都能拿到梯度。

部署图（`switch_to_deploy()`，FP64 拼核、仅最后一次 cast 到 FP32）：

    W_fused = gamma * [W_pq, W_diff] ∈ R^{96x144x1x1},  b_fused = gamma * b_pq ∈ R^{96}
    T(P,Q) = Conv1x1(cat(P, Q, |Q-P|); W_fused, b_fused)

参数（C=48, D=96）：训练图新增 96*96+96 + 96*48 + 1 = **13,921**；部署图新增 96*144+96 = **13,920**。
"""
import torch
import torch.nn as nn

__all__ = ["FineEvidenceTap1x1", "FET_LOCAL_SEED", "FET_FORM", "FET_SOURCE", "FET_FUSE"]

# sidecar / manifest 使用的固定标识（train.py 与 eval.py 必须逐字段相等核对）
FET_FORM = "pq_abs_1x1_gamma_v1"
FET_SOURCE = "norm0_1_4"
FET_FUSE = "post_dcr_refine"
FET_LOCAL_SEED = 16          # 局部初始化种子（不消耗全局 RNG）


class FineEvidenceTap1x1(nn.Module):
    """细尺度（1/4）双时相证据的线性可折叠旁路，输出与 DCR refine 结果相加。"""

    def __init__(self, in_ch: int = 48, out_ch: int = 96, deploy: bool = False,
                 seed: int = FET_LOCAL_SEED):
        super().__init__()
        self.in_ch = int(in_ch)
        self.out_ch = int(out_ch)
        self.deploy = bool(deploy)
        self.local_seed = int(seed)
        self.fold_count = 0

        if self.deploy:
            self.fused = nn.Conv2d(3 * self.in_ch, self.out_ch, 1, bias=True)
        else:
            # 局部 RNG 隔离：**所有**子模块构造（含 Conv2d 自带 reset_parameters）
            # 与显式初始化都在 fork_rng 内完成，绝不推进全局随机流 ⇒
            # 开关 fine_tap 时既有模块的初始化逐位不变（T2 硬门）。
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(self.local_seed)
                self.pq = nn.Conv2d(2 * self.in_ch, self.out_ch, 1, bias=True)
                self.diff = nn.Conv2d(self.in_ch, self.out_ch, 1, bias=False)
                nn.init.zeros_(self.diff.weight)
                self.gamma = nn.Parameter(torch.zeros(1))
            self.gamma.requires_grad_(True)

    # ------------------------------------------------------------------ forward
    def forward(self, P: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
        D = (Q - P).abs()
        if self.deploy:
            return self.fused(torch.cat([P, Q, D], dim=1))
        U = self.pq(torch.cat([P, Q], dim=1))
        V = self.diff(D)
        return self.gamma * (U + V)

    # ------------------------------------------------------------------ fold
    @torch.no_grad()
    def get_equivalent_kernel_bias(self):
        """返回 (W, b)：W ∈ R^{out×3C×1×1} (FP64), b ∈ R^{out} (FP64)。"""
        if self.deploy:
            return self.fused.weight.detach().double(), self.fused.bias.detach().double()
        W = torch.cat([self.pq.weight.detach().double(),
                       self.diff.weight.detach().double()], dim=1)
        b = self.pq.bias.detach().double()
        g = self.gamma.detach().double().reshape(())
        return g * W, g * b

    @torch.no_grad()
    def switch_to_deploy(self):
        if self.deploy:                      # 幂等：第二次调用直接返回
            return self
        W, b = self.get_equivalent_kernel_bias()
        self.fused = nn.Conv2d(3 * self.in_ch, self.out_ch, 1, bias=True)
        self.fused.weight.data = W.float()
        self.fused.bias.data = b.reshape(-1).float()
        self.deploy = True
        self.fold_count += 1
        for name in ("pq", "diff", "gamma"):
            if hasattr(self, name):
                delattr(self, name)
        return self

    # ------------------------------------------------------------------ diag
    def param_report(self) -> dict:
        """参数与形态报告（T1/T6 使用）。"""
        if self.deploy:
            return {"deploy": True,
                    "fused": int(self.fused.weight.numel() + self.fused.bias.numel()),
                    "fused_shape": [int(self.fused.weight.shape[0]), int(self.fused.weight.shape[1]), 1, 1]}
        return {"deploy": False,
                "train_new": int(sum(p.numel() for p in self.parameters())),
                "pq": int(self.pq.weight.numel() + self.pq.bias.numel()),
                "diff": int(self.diff.weight.numel()),
                "gamma": int(self.gamma.numel()),
                "gamma_value": float(self.gamma.detach().cpu().item())}
