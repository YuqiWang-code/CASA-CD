"""VMamba-Tiny 自包含主干（Run8 换骨干主线）。

## 来源与许可边界（**必须如实标注**）

结构逐项对齐 **官方 VMamba v2**（*VMamba: Visual State Space Model*, NeurIPS 2024；
官方仓库 `MzeroMiko/VMamba`，`classification/models/vmamba.py`）。
本文件是**自包含移植**（去掉 `einops` / `timm` / `fvcore` / `csm_triton` 依赖），
目的是让 `casa_vmamba_net.py` 与 Run1–Run7 的 TinyViM 主线共用同一套工程约束。
**这是复用既有公开架构，不是本项目原创**，论文中不得计为贡献。

## 为什么不用 `mamba_ssm` 的那套

RSML-3 实测：`casacd` 里那个 10-03 编译的 `selective_scan_cuda.so` ABI 已坏
（`undefined symbol: _ZN3c104cuda29c10_cuda_check_implementationE...`）。
而 `selective_scan_cuda_oflex` 是 **Run1–Run7 生产同款**、已验证的内核。
因此本文件默认走 oflex，`mamba_ssm.selective_scan_fn` 只作兜底（见 `vmamba_kernel.py`）。

## 与本项目 TinyViM 主线的三点结构差异（都会影响"换骨干"这一唯一变量，须在论文披露）

| 项 | TinyViM-S-Slim（Run1–Run7） | VMamba-Tiny（Run8） |
|---|---|---|
| 频率解耦 | 有（Laplace mixer：低频进 scan、高频走 `local_conv` RepDW） | **无**（全部通道进 scan，无 dense 高频旁路） |
| drop_path | 0（slim 构造里未启用） | **0.1**（官方默认） |
| 归一化位置 | `Conv2d_BN` 为主 | **pre-norm**（`x + op(norm(x))`） |

⇒ Run8 的 Stage A **不能**用 Run7 的数字当同期对照。

## 官方 v2 权重对齐（`vssm_tiny_0230_ckpt_epoch_262.pth`）

实测 key 形状（2026-10-15）：
`patch_embed.{0,2}` = 48 ch（`Conv2d(3,48,k4,s4)` + `LayerNorm2d(48)`），
`patch_embed.{5,7}` = 96 ch，`layers.{i}.blocks.{j}.{norm,norm2,mlp,op.*}`，
`layers.{i}.downsample.{1,3}`，`classifier.{norm,head}`。
⇒ `patch_size=4` / `downsample_version="v2"` / `norm_layer="ln2d"` / `dims=[96,192,384,768]`
   / `depths=[2,2,9,2]` 与权重逐 key 对齐。
"""
import math
from collections import OrderedDict
from functools import partial
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.layers.vmamba_kernel import (
    load_mamba_selective_scan_fn,
    load_selective_scan_oflex,
)

VMAMBA_TINY = dict(
    patch_size=4,
    in_chans=3,
    depths=(2, 2, 5, 2),          # ⚠️ 实测：vssm_tiny_0230 是 5 不是 9（总 11 个 block）
    dims=(96, 192, 384, 768),
    ssm_d_state=1,                # ⚠️ 实测：A_logs=(768,1)、x_proj dim1=8=6+2*1
    ssm_ratio=2.0,                # d_inner = 2*d_model
    ssm_disable_z=True,           # ⚠️ 实测：in_proj.weight=(192,96) ⇒ 无 z 门控
    ssm_dt_rank="auto",
    ssm_conv=3,
    ssm_conv_bias=False,          # ⚠️ 实测：ckpt 无 op.conv2d.bias
    ssm_bias=False,               # ⚠️ 实测：ckpt 无 in_proj.bias / out_proj.bias
    drop_path_rate=0.1,           # 官方 tiny 常用 0.2；不影响加载
    patch_norm=True,
    downsample_version="v3",      # ⚠️ 实测：downsample.1.weight = (192,96,3,3) ⇒ k3s2
    # ⚠️ 官方 0230 权重用的是 **v2 patch embed**（两个 stride-2 的 3×3 conv，GELU 分隔），
    # 不是 v1 的单个 stride-4 conv。逐 key 由权重反推确认（2026-10-15 服务器实测）：
    #   patch_embed.0.weight (48, 3, 3, 3)   ← Conv2d(3, 48, k3, s2, p1)
    #   patch_embed.2.weight (48,)           ← LayerNorm2d(48)
    #   patch_embed.5.weight (96, 48, 3, 3)  ← Conv2d(48, 96, k3, s2, p1)
    #   patch_embed.7.weight (96,)           ← LayerNorm2d(96)
    # ⇒ 1/4 分辨率、dims[0]=96。
    # 首版曾误用 v1（stride-4 单 conv），smoke 报
    #   "shape mismatch patch_embed.0.weight: ckpt (48,3,3,3) vs model (96,3,4,4)" ⇒ 已修正。
    patchembed_version="v2",
)

# 权重实测（2026-10-15，服务器 `_r8_ckpt_audit.sh`）：
#   vssm_tiny_0230_ckpt_epoch_262.pth  sha256 09739f6d95638e5caf0d33fcbca85b7cff62b8ca16ec2926d781109939c6b201
#   len(ck["model"]) = 222 keys；sum(numel) = 30,705,832（含 1000 类分类头 770,536）
#   ⇒ 纯骨干 = 29,935,296（29.94 M）
#   ⇒ 项目文档里"官方 VMamba-T ≈30M"对这枚权重**成立**，但**必须连同配置一起标注**：
#      depths=[2,2,5,2] / dims=[96,192,384,768] / ssm_ratio=2.0 / d_state=1 / noz /
#      patch_embed v2 / downsample v3 / ln2d / mlp_ratio=4.0 / ImageNet-1k 1000 类。
#   ⚠️ 不得再引用 `others/VMamba/vmamba.py::VSSM.__init__` 的**类默认值**
#      （那里是 [2,2,9,2]/d_state16/v1/v2 ⇒ 42.95M，是错误口径的来源）。

__all__ = ["VMambaBackbone", "build_vmamba_tiny", "VSSM_STAGE_DIMS", "STAGE_RES"]

VSSM_STAGE_DIMS = (96, 192, 384, 768)
STAGE_RES = ("1/4", "1/8", "1/16", "1/32")


# ---------------------------------------------------------------------------
# 基础算子（与官方逐行一致；只去掉外部依赖）
# ---------------------------------------------------------------------------
def _trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    """timm.trunc_normal_ 的无 scipy 实现（与 `layers/ss2d.py` 同口径）。"""
    with torch.no_grad():
        l = (1.0 + math.erf((a - mean) / std / math.sqrt(2.0))) / 2.0
        u = (1.0 + math.erf((b - mean) / std / math.sqrt(2.0))) / 2.0
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0)).add_(mean)
        tensor.clamp_(min=a, max=b)
    return tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        return x * x.new_empty(shape).bernoulli_(keep).div_(keep)


class LayerNorm2d(nn.LayerNorm):
    """官方 `LayerNorm2d`：在 (B,H,W,C) 布局上做 LayerNorm（channel_first 模式下用）。"""

    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1)
        x = super().forward(x)
        return x.permute(0, 3, 1, 2)


class Linear2d(nn.Linear):
    """官方 `Linear2d`：把 nn.Linear 当作 1×1 卷积用（channel-first 模式下必需）。

    ⚠️ 为什么必须用它（2026-10-15 smoke 抓到的真实 bug）：
    channel-first 模式下 `LayerNorm2d` 的输出是 **(B,C,H,W)**，张量**最后一维是 W 不是 C**。
    若这里用普通 `nn.Linear`，`F.linear` 会把 W 当成特征维 ⇒
    `mat1 and mat2 shapes cannot be multiplied (12288x64 and 96x192)`。

    同时**权重形状必须保持 (out, in)**（官方 ckpt 就是 (192,96)），所以不能改写成
    `nn.Conv2d(96,192,1)`（那是 (192,96,1,1)，会与侧车/审计的形状对拍冲突）。
    `Linear2d` 用 `weight[:, :, None, None]` 在 forward 里即时视图成 conv2d 的 4D 权重，
    权重本身仍是 2D ⇒ 两边都满足。`_load_from_state_dict` 里那次 `view` 是为了兼容
    "checkpoint 用 Conv2d 存的 4D 权重" 的历史情况。
    """

    def forward(self, x: torch.Tensor):
        return F.conv2d(x, self.weight[:, :, None, None], self.bias)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)
        state_dict[prefix + "weight"] = state_dict[prefix + "weight"].view(self.weight.shape)


class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x):
        return x.permute(*self.args)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.0, channels_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        # channel-first ⇒ 输入是 (B,C,H,W)，最后一维是 W 不是 C ⇒ 必须用 Linear2d
        Linear = Linear2d if channels_first else nn.Linear
        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop) if drop > 0.0 else nn.Identity()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# ---------------------------------------------------------------------------
# 四向交叉扫描
# ---------------------------------------------------------------------------
class CrossScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        xs = x.new_empty((B, 4, C, H * W))
        xs[:, 0] = x.flatten(2, 3)
        xs[:, 1] = x.transpose(dim0=2, dim1=3).flatten(2, 3)
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
        return xs

    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        B, C, H, W = ctx.shape
        L = H * W
        ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, -1, L)
        y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, -1, L)
        return y.view(B, -1, H, W)


class CrossMerge(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, D, H, W = ys.shape
        ctx.shape = (H, W)
        ys = ys.view(B, K, D, -1)
        ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
        y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, D, -1)
        return y

    @staticmethod
    def backward(ctx, x: torch.Tensor):
        H, W = ctx.shape
        B, C, L = x.shape
        xs = x.new_empty((B, 4, C, L))
        xs[:, 0] = x
        xs[:, 1] = x.view(B, C, H, W).transpose(dim0=2, dim1=3).flatten(2, 3)
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
        return xs.view(B, 4, C, H, W)


def _scan_oflex(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows=1):
    """oflex 内核前向（与 `layers/ss2d.py::SelectiveScan` 完全同调用口径）。"""
    oflex = load_selective_scan_oflex()
    out, x, *rest = oflex.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows, True)
    return out, x


def _scan_mamba_ssm(u, delta, A, B, C, D, delta_bias, delta_softplus):
    """兜底：mamba_ssm 官方 `selective_scan_fn`，按 K 维循环调用。"""
    fn = load_mamba_selective_scan_fn()
    K = B.shape[1]
    Cdim = u.shape[1] // K
    outs = []
    for k in range(K):
        uk = u[:, k * Cdim:(k + 1) * Cdim].permute(0, 2, 1).contiguous()
        dk = delta[:, k * Cdim:(k + 1) * Cdim].permute(0, 2, 1).contiguous()
        ak = A[k * Cdim:(k + 1) * Cdim].contiguous()
        bk = B[:, k].contiguous()
        ck = C[:, k].contiguous()
        dk_vec = D[k * Cdim:(k + 1) * Cdim].contiguous() if D is not None else None
        db = delta_bias[k * Cdim:(k + 1) * Cdim].contiguous() if delta_bias is not None else None
        yk = fn(uk, dk, ak, bk, ck, dk_vec, z=None, delta_bias=db, delta_softplus=delta_softplus)
        outs.append(yk.permute(0, 2, 1).contiguous())
    return torch.cat(outs, dim=1), None


class SelectiveScan(torch.autograd.Function):
    """可切换后端的 selective scan（oflex 优先，mamba_ssm 兜底）。

    入参/返回与官方 `selective_scan_cuda.fwd` 完全一致：
    `u (B, K*C, L)`、`delta (B, K*C, L)`、`A (K*C, N)`、`B/C (B, K, N, L)`、
    `D (K*C,)`、`delta_bias (K*C,)`；输出 `(B, K*C, L)`。
    """

    @staticmethod
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None,
                delta_softplus=False, nrows=1):
        for t in (u, delta, B, C):
            if t.stride(-1) != 1:
                t = t.contiguous()
        if B.dim() == 3:
            B = B.unsqueeze(1)
            ctx.squeeze_B = True
        else:
            ctx.squeeze_B = False
        if C.dim() == 3:
            C = C.unsqueeze(1)
            ctx.squeeze_C = True
        else:
            ctx.squeeze_C = False
        ctx.delta_softplus = delta_softplus
        try:
            out, x = _scan_oflex(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)
            ctx.backend = "oflex"
            ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
            return out
        except Exception:  # noqa: BLE001  —— 兜底到 mamba_ssm
            out, _x = _scan_mamba_ssm(u, delta, A, B, C, D, delta_bias, delta_softplus)
            ctx.backend = "mamba_ssm"
            return out

    @staticmethod
    def backward(ctx, dout, *args):
        if getattr(ctx, "backend", "") != "oflex":
            raise RuntimeError(
                "[SCAN-BACKEND] mamba_ssm 兜底路径的 autograd 由 selective_scan_fn 自己接管，"
                "不应走到这里；请检查是否误用了本类的 apply。")
        oflex = load_selective_scan_oflex()
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = oflex.bwd(
            u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1)
        dB = dB.squeeze(1) if getattr(ctx, "squeeze_B", False) else dB
        dC = dC.squeeze(1) if getattr(ctx, "squeeze_C", False) else dC
        return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None)


def cross_selective_scan(
    x=None, x_proj_weight=None, x_proj_bias=None,
    dt_projs_weight=None, dt_projs_bias=None,
    A_logs=None, Ds=None, delta_softplus=True,
    out_norm=None, out_norm_shape="v0", channel_first=False,
    to_dtype=True, force_fp32=False, nrows=-1, backnrows=-1, ssoflex=True,
    SelectiveScan=SelectiveScan, CrossScan=CrossScan, CrossMerge=CrossMerge,
    no_einsum=False, dt_low_rank=True, **kwargs,
):
    """与官方 `cross_selective_scan` 同语义；`channel_first=True` 时返回 (B,C,H,W)。"""
    B, D, H, W = x.shape
    D, N = A_logs.shape
    K, D, R = dt_projs_weight.shape
    L = H * W
    if nrows == 0:
        nrows = 4 if D % 4 == 0 else (3 if D % 3 == 0 else (2 if D % 2 == 0 else 1))
    if backnrows == 0:
        backnrows = 4 if D % 4 == 0 else (3 if D % 3 == 0 else (2 if D % 2 == 0 else 1))

    def _scan(u, delta, A, B_, C_, Dd=None, delta_bias=None, delta_softplus=True):
        return SelectiveScan.apply(u, delta, A, B_, C_, Dd, delta_bias,
                                   delta_softplus, nrows)

    if no_einsum:
        xs = CrossScan.apply(x)
        x_dbl = F.conv1d(xs.view(B, -1, L), x_proj_weight.view(-1, D, 1),
                         bias=(x_proj_bias.view(-1) if x_proj_bias is not None else None), groups=K)
        dts, Bs, Cs = torch.split(x_dbl.view(B, K, -1, L), [R, N, N], dim=2)
        dts = F.conv1d(dts.contiguous().view(B, -1, L), dt_projs_weight.view(K * D, -1, 1), groups=K)
    else:
        xs = CrossScan.apply(x)
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)
        if x_proj_bias is not None:
            x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)

    xs = xs.view(B, -1, L)
    dts = dts.contiguous().view(B, -1, L)
    As = -torch.exp(A_logs.to(torch.float))
    Bs = Bs.contiguous().view(B, K, N, L)
    Cs = Cs.contiguous().view(B, K, N, L)
    Ds = Ds.to(torch.float)
    delta_bias = dt_projs_bias.view(-1).to(torch.float)

    if force_fp32:
        xs = xs.to(torch.float)
        dts = dts.to(torch.float)
        Bs = Bs.to(torch.float)
        Cs = Cs.to(torch.float)

    ys = _scan(xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus).view(B, K, -1, H, W)
    y = CrossMerge.apply(ys)

    if channel_first:
        y = y.view(B, -1, H, W)
        if out_norm_shape in ("v1",):
            y = out_norm(y)
        else:
            y = out_norm(y.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return (y.to(x.dtype) if to_dtype else y)

    if out_norm_shape in ("v1",):
        y = out_norm(y.view(B, -1, H, W)).permute(0, 2, 3, 1)
    else:
        y = y.transpose(dim0=1, dim1=2).contiguous()
        y = out_norm(y).view(B, H, W, -1)
    return (y.to(x.dtype) if to_dtype else y)


# ---------------------------------------------------------------------------
# SS2D（官方 v2 口径：无频率解耦，全部通道进 scan）
# ---------------------------------------------------------------------------
class SS2D(nn.Module):
    def __init__(self, d_model=96, d_state=16, ssm_ratio=2.0, dt_rank="auto",
                 act_layer=nn.SiLU, d_conv=3, conv_bias=True, dropout=0.0, bias=False,
                 dt_min=0.001, dt_max=0.1, dt_init="random", dt_scale=1.0,
                 dt_init_floor=1e-4, initialize="v0", forward_type="v2",
                 disable_z=False, channel_first=False, **kwargs):
        super().__init__()
        factory_kwargs = {"device": None, "dtype": None}
        d_inner = int(ssm_ratio * d_model)
        dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.d_model = d_model
        self.d_inner = d_inner
        self.d_state = d_state
        self.dt_rank = dt_rank
        self.d_conv = d_conv
        self.channel_first = channel_first
        self.forward_type = forward_type
        self.out_norm_shape = "v1" if channel_first else "v0"
        # `disable_z`：官方 v2 的 `forward_type="v2noz"` 语义 —— in_proj 只出 d_inner
        # （无 z 门控）。**vssm_tiny_0230 权重实测就是这种**（in_proj.weight=(192,96)）。
        self.disable_z = bool(disable_z)

        self.out_norm = LayerNorm2d(d_inner) if channel_first else nn.LayerNorm(d_inner)
        # channel-first ⇒ 必须用 Linear2d（见该类注释：LayerNorm2d 的输出最后一维是 W）
        Linear = Linear2d if channel_first else nn.Linear

        k_group = 4
        d_proj = d_inner if self.disable_z else (d_inner * 2)
        self.in_proj = Linear(d_model, d_proj, bias=bias, **factory_kwargs)
        self.act = act_layer()
        if d_conv > 1:
            self.conv2d = nn.Conv2d(d_inner, d_inner, kernel_size=d_conv,
                                    padding=(d_conv - 1) // 2, groups=d_inner,
                                    bias=conv_bias, **factory_kwargs)

        x_proj = [nn.Linear(d_inner, dt_rank + d_state * 2, bias=False, **factory_kwargs)
                  for _ in range(k_group)]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in x_proj], dim=0))
        del x_proj

        dt_projs = [self.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max,
                                 dt_init_floor, **factory_kwargs) for _ in range(k_group)]
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in dt_projs], dim=0))
        del dt_projs

        self.A_logs = self.A_log_init(d_state, d_inner, copies=k_group, merge=True)
        self.Ds = self.D_init(d_inner, copies=k_group, merge=True)

        self.out_proj = Linear(d_inner, d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001,
                dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError(dt_init)
        dt = torch.exp(torch.rand(d_inner, **factory_kwargs)
                       * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        A = torch.arange(1, d_state + 1, dtype=torch.float32, device=device)
        A = A.unsqueeze(0).repeat(d_inner, 1).contiguous()
        A_log = torch.log(A)
        if copies > 0:
            A_log = A_log.unsqueeze(0).repeat(copies, 1, 1)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = D.unsqueeze(0).repeat(copies, 1)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor, **kwargs):
        return cross_selective_scan(
            x, self.x_proj_weight, None, self.dt_projs_weight, self.dt_projs_bias,
            self.A_logs, self.Ds, delta_softplus=True,
            out_norm=self.out_norm, channel_first=self.channel_first,
            out_norm_shape=self.out_norm_shape, force_fp32=True, **kwargs)

    def forward(self, x: torch.Tensor, **kwargs):
        """x: channel_first 时 (B,C,H,W)，否则 (B,H,W,C)。"""
        x = self.in_proj(x)
        if not self.disable_z:
            x, z = x.chunk(2, dim=(1 if self.channel_first else -1))
            z = self.act(z)
        if not self.channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()
        if self.d_conv > 1:
            x = self.conv2d(x)
        x = self.act(x)
        y = self.forward_core(x)
        if not self.disable_z:
            y = y * z
        return self.dropout(self.out_proj(y))


class VSSBlock(nn.Module):
    def __init__(self, hidden_dim=0, drop_path=0.0, norm_layer=nn.LayerNorm,
                 channel_first=False, ssm_d_state=16, ssm_ratio=2.0, ssm_dt_rank="auto",
                 ssm_act_layer=nn.SiLU, ssm_conv=3, ssm_conv_bias=True, ssm_drop_rate=0.0,
                 ssm_init="v0", forward_type="v2", ssm_disable_z=False, ssm_bias=False,
                 mlp_ratio=4.0, mlp_act_layer=nn.GELU,
                 mlp_drop_rate=0.0, gmlp=False, use_checkpoint=False, **kwargs):
        super().__init__()
        self.ssm_branch = ssm_ratio > 0
        self.mlp_branch = mlp_ratio > 0
        self.use_checkpoint = use_checkpoint
        if self.ssm_branch:
            self.norm = norm_layer(hidden_dim)
            self.op = SS2D(d_model=hidden_dim, d_state=ssm_d_state, ssm_ratio=ssm_ratio,
                           dt_rank=ssm_dt_rank, act_layer=ssm_act_layer, d_conv=ssm_conv,
                           conv_bias=ssm_conv_bias, dropout=ssm_drop_rate, bias=ssm_bias,
                           initialize=ssm_init, forward_type=forward_type,
                           disable_z=ssm_disable_z, channel_first=channel_first)
        self.drop_path = DropPath(drop_path)
        if self.mlp_branch:
            self.norm2 = norm_layer(hidden_dim)
            self.mlp = Mlp(in_features=hidden_dim, hidden_features=int(hidden_dim * mlp_ratio),
                           act_layer=mlp_act_layer, drop=mlp_drop_rate,
                           channels_first=channel_first)

    def _forward(self, inp):
        x = inp
        if self.ssm_branch:
            x = x + self.drop_path(self.op(self.norm(x)))
        if self.mlp_branch:
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def forward(self, inp):
        return self._forward(inp)


# ---------------------------------------------------------------------------
# 主干
# ---------------------------------------------------------------------------
class VMambaBackbone(nn.Module):
    """VMamba-Tiny 特征提取（输出四个 stage 的 (B,C,H,W)，channel-first）。"""

    def __init__(self, pretrained_path=None, cfg=None, out_indices=(0, 1, 2, 3),
                 norm_layer="ln2d"):
        super().__init__()
        c = dict(VMAMBA_TINY)
        if cfg:
            c.update(cfg)
        self.cfg = c
        self.dims = list(c["dims"])
        self.depths = list(c["depths"])
        self.out_indices = tuple(out_indices)
        assert norm_layer.lower() == "ln2d", "Run8 只用 channel-first (ln2d) 口径"
        self.channel_first = True
        nl = LayerNorm2d

        # ---- patch embed：官方 v2 stem（两个 3×3/s2 卷积 + GELU + LN2d）
        # 与权重 key 逐项对齐：.0=Conv(3→48) .2=LN2d(48) .5=Conv(48→96) .7=LN2d(96)
        d0 = self.dims[0]
        self.patch_embed = nn.Sequential(
            nn.Conv2d(c["in_chans"], d0 // 2, kernel_size=3, stride=2, padding=1),
            nl(d0 // 2),
            nn.GELU(),
            nn.Conv2d(d0 // 2, d0, kernel_size=3, stride=2, padding=1),
            nl(d0),
        )

        dpr = [x.item() for x in torch.linspace(0, c["drop_path_rate"], sum(self.depths))]
        ch_first = self.channel_first
        downsample_version = str(c.get("downsample_version", "v3")).lower()
        self.layers = nn.ModuleList()
        for i in range(len(self.depths)):
            if i >= len(self.depths) - 1:
                downsample = nn.Identity()
            elif downsample_version == "v3":
                # 官方 `_make_downsample_v3`：Conv k3 s2 p1（实测权重 k=3）
                downsample = nn.Sequential(
                    nn.Identity() if ch_first else Permute(0, 3, 1, 2),
                    nn.Conv2d(self.dims[i], self.dims[i + 1], kernel_size=3, stride=2, padding=1),
                    nn.Identity() if ch_first else Permute(0, 2, 3, 1),
                    nl(self.dims[i + 1]),
                )
            else:
                # 官方 `_make_downsample`：Conv k2 s2
                downsample = nn.Sequential(
                    nn.Identity() if ch_first else Permute(0, 3, 1, 2),
                    nn.Conv2d(self.dims[i], self.dims[i + 1], kernel_size=2, stride=2),
                    nn.Identity() if ch_first else Permute(0, 2, 3, 1),
                    nl(self.dims[i + 1]),
                )
            blocks = [VSSBlock(
                hidden_dim=self.dims[i], drop_path=dpr[sum(self.depths[:i]) + d],
                norm_layer=nl, channel_first=ch_first,
                ssm_d_state=c["ssm_d_state"], ssm_ratio=c["ssm_ratio"],
                ssm_dt_rank=c["ssm_dt_rank"], ssm_act_layer=nn.SiLU,
                ssm_conv=c["ssm_conv"], ssm_conv_bias=c["ssm_conv_bias"],
                ssm_bias=c["ssm_bias"], ssm_disable_z=c["ssm_disable_z"],
                ssm_drop_rate=0.0, ssm_init="v0", forward_type="v2",
                mlp_ratio=4.0, mlp_act_layer=nn.GELU,
            ) for d in range(self.depths[i])]
            self.layers.append(nn.Sequential(OrderedDict(
                blocks=nn.Sequential(*blocks), downsample=downsample)))

        for i in self.out_indices:
            self.add_module(f"outnorm{i}", nl(self.dims[i]))

        self.apply(self._init_weights)
        self._load_stats = None
        # 官方 v2 对 SS2D 内部参数做一次专门的初始化（必须在 apply 之后）
        for m in self.modules():
            if isinstance(m, SS2D):
                self._init_ss2d(m)
        if pretrained_path is not None:
            self.load_pretrained(pretrained_path)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            _trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            _trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @staticmethod
    def _init_ss2d(m):
        """官方 v2 在 `_init_weights` 之后对 SS2D 的 dt/A/D 重做初始化。

        注意：**本函数只在"无预训练"时有意义**；有预训练时全部 key 会被 load_pretrained
        覆盖，因此这里与权重无关。保留是为了"无预训练 smoke"与官方构建逐位一致。
        """
        nn.init.uniform_(m.dt_projs_weight, -0.1, 0.1)

    # ------------------------------------------------------------------
    def load_pretrained(self, path):
        """加载官方 `vssm_tiny_0230_ckpt_epoch_262.pth`（top-level key = `model`）。

        返回 load stats（含 `worst_diff`），供 smoke 断言"预训练语义继承"。
        """
        ck = torch.load(path, map_location="cpu", weights_only=False)
        src = ck.get("model", ck)
        model_sd = self.state_dict()
        kept, dropped = {}, []
        for k, v in src.items():
            if k.startswith("classifier."):
                dropped.append(k)
                continue
            if k in model_sd:
                assert model_sd[k].shape == v.shape, \
                    f"[VMAMBA-PRETRAIN] shape mismatch {k}: ckpt {tuple(v.shape)} vs model {tuple(model_sd[k].shape)}"
                kept[k] = v
            else:
                dropped.append(k)
        missing = [k for k in model_sd if k not in kept]
        self.load_state_dict(kept, strict=False)
        worst = 0.0
        for k, v in kept.items():
            worst = max(worst, (model_sd[k].float() - v.float()).abs().max().item())
        self._load_stats = {
            "pretrained_keys": len(src),
            "retained": len(kept),
            "worst_diff": worst,
            "missing_new": missing,
            "dropped_intentional": dropped,
            "loaded_keys": list(kept.keys()),
        }
        return self._load_stats

    def load_stats(self):
        return self._load_stats

    # ------------------------------------------------------------------
    def forward_features(self, x):
        """返回 `(f1, f2, f3, f4)`，各自 (B, dim_i, H/4, W/4 ... )，channel-first。"""
        x = self.patch_embed(x)                     # (B, 96, 64, 64)
        outs = []
        for i, layer in enumerate(self.layers):
            o = layer.blocks(x)                     # 该 stage 最后一个 block 的输出
            x = layer.downsample(o)
            if i in self.out_indices:
                outs.append(getattr(self, f"outnorm{i}")(o))
        return outs

    def forward_pre_norm(self, x):
        """返回 `outnorm` **之前**的 stage 激活 `(o0..o3)`，形状与 `forward_features` 相同。

        （2026-10-20 实测结论）`forward_features` 的每个 stage 输出都过了
        `outnorm{i} = LayerNorm2d(dim)`；LayerNorm 在通道维上做零均值单位方差标准化 ⇒
        **每个 token 的范数精确等于 √dim**（实测 `‖v‖/√dim = 1.0000 ± 5e-7`，`1-‖p‖² = 0.4200`
        对所有 token 完全相同）。任何靠**半径**携带信息的流形机制（Poincaré 球双曲距离）
        在这组特征上都会退化成常数半径，从而把双曲距离压成"余弦距离的单调变换"。

        本口给出 `outnorm` 之前的激活，其范数随内容变化（实测 cv：stage2 9.7e-2、stage3 6.8e-1），
        恢复论文（JSTARS 2024 HSAM）依赖的"未归一化 CNN 激活"前提。

        网络主干**仍然**使用 `forward_features`（归一化后）；本口只给旁路模块取数，
        因此不改变任何既有 run 的前向数值。
        """
        x = self.patch_embed(x)
        outs = []
        for i, layer in enumerate(self.layers):
            o = layer.blocks(x)
            x = layer.downsample(o)
            if i in self.out_indices:
                outs.append(o)
        return outs

    def forward(self, x):
        return self.forward_features(x)

    def param_count(self):
        return sum(p.numel() for p in self.parameters())

    def trainable_param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_vmamba_tiny(pretrained_path=None, out_indices=(0, 1, 2, 3), cfg=None):
    return VMambaBackbone(pretrained_path=pretrained_path, cfg=cfg, out_indices=out_indices)


if __name__ == "__main__":
    torch.manual_seed(0)
    m = VMambaBackbone()
    print("vssm_tiny params:", f"{m.param_count():,}", f"({m.param_count()/1e6:.3f} M)")
    outs = m(torch.randn(1, 3, 256, 256))
    print("taps:", [tuple(t.shape) for t in outs])
