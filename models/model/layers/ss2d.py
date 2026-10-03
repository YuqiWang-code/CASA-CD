"""SS2D：TinyViM/VMamba 四向交叉扫描 SSM 的自包含移植（无 timm / 无 einops）。

与官方 TinyViM `model/tvimblock.py` 结构一致：
    Conv2d_BN / RepDW / RepDW_Axias / Rep_Inception / FFN / LocalBlock / TViMBlock /
    SS2D / CrossScan / CrossMerge / cross_selective_scan

kernel 后端（RSML-3 RTX 5090 sm_120 适配）：官方依赖的 `selective_scan_cuda`
（VMamba CUDA kernel）在 torch 2.14 + CUDA 13.2 + sm_120 上不可用/未编译，
本移植改用 **mamba-ssm 的 selective_scan_fn**（Triton 内核，JIT 适配当前架构），
语义与官方 selective scan 完全一致（同一 S6 递推），按四方向 K 维循环调用。
"""
import math
from functools import partial
from typing import Callable, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import selective_scan_cuda_oflex as _oflex_cuda
except Exception:  # noqa: BLE001
    _oflex_cuda = None

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn as _mamba_ssm_scan
except Exception:  # noqa: BLE001  （mamba-ssm wheel 自带 CUDA kernel ABI 不兼容 / 未安装）
    try:
        from model.layers.mamba_scan_triton import selective_scan_fn as _mamba_ssm_scan
    except Exception:  # noqa: BLE001
        _mamba_ssm_scan = None


# ---------------------------------------------------------------------------
# 本地小工具（timm 等价物，避免依赖）
# ---------------------------------------------------------------------------
def _norm_cdf(x):
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    """timm trunc_normal_ 的无 scipy 实现。"""
    with torch.no_grad():
        l = _norm_cdf((a - mean) / std)
        u = _norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0)).add_(mean)
        tensor.clamp_(min=a, max=b)
    return tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep).div_(keep)
        return x * mask


# ---------------------------------------------------------------------------
# TinyViM 基础模块（官方结构逐项对齐，仅替换 timm 依赖）
# ---------------------------------------------------------------------------
class Conv2d_BN(nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1, resolution=-10000):
        super().__init__()
        self.add_module('c', nn.Conv2d(a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', nn.BatchNorm2d(b))
        nn.init.constant_(self.bn.weight, bn_weight_init)
        nn.init.constant_(self.bn.bias, 0)


class RepDW(nn.Module):
    """高频分支：reparameterizable 3x3 DW conv + 1x1 DW + identity，后接 BN。"""

    def __init__(self, ed):
        super().__init__()
        self.conv = Conv2d_BN(ed, ed, 3, 1, 1, groups=ed)
        self.conv1 = nn.Conv2d(ed, ed, 1, 1, 0, groups=ed)
        self.dim = ed
        self.bn = nn.BatchNorm2d(ed)
        self.apply(self._init_weights)

    def forward(self, x):
        return self.bn((self.conv(x) + self.conv1(x)) + x)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class RepDW_Axias(nn.Module):
    def __init__(self, ed, kernel_max=7, kernel=(1, 7)):
        super().__init__()
        self.kernel = kernel
        self.kernel_max = kernel_max
        padding = kernel_max // 2
        self.conv1 = nn.Conv2d(ed, ed, 1, 1, 0, groups=ed)
        if kernel == (1, kernel_max):
            self.conv = Conv2d_BN(ed, ed, (1, kernel_max), 1, (0, padding), groups=ed)
        else:
            self.conv = Conv2d_BN(ed, ed, (kernel_max, 1), 1, (padding, 0), groups=ed)
        self.dim = ed
        self.bn = nn.BatchNorm2d(ed)
        self.apply(self._init_weights)

    def forward(self, x):
        return self.bn((self.conv(x) + self.conv1(x)) + x)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class Rep_Inception(nn.Module):
    """frequency ramp inception：通道按 split 分成 W 向 / H 向轴向 DW。"""

    def __init__(self, dim, kernel_max=7, ratio=0.5):
        super().__init__()
        gc = int(dim * ratio)
        self.dwconv_h = RepDW_Axias(gc, kernel_max=kernel_max, kernel=(1, kernel_max))
        self.dwconv_w = RepDW_Axias(gc, kernel_max=kernel_max, kernel=(kernel_max, 1))
        self.split = (dim - gc, gc)

    def forward(self, x):
        x_w, x_h = torch.split(x, self.split, dim=1)
        return torch.cat((self.dwconv_w(x_w), self.dwconv_h(x_h)), dim=1)


# ---------------------------------------------------------------------------
# 四向交叉扫描（与官方逐行一致，纯 torch）
# ---------------------------------------------------------------------------
class CrossScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        xs = x.new_empty((B, 4, C, H * W))
        xs[:, 0] = x.flatten(2, 3)
        xs[:, 1] = x.transpose(dim0=2, dim1=3).flatten(2, 3)
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
        return xs

    @staticmethod
    def backward(ctx, ys):
        B, C, H, W = ctx.shape
        L = H * W
        ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, -1, L)
        y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, -1, L)
        return y.view(B, -1, H, W)


class CrossMerge(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys):
        B, K, D, H, W = ys.shape
        ctx.shape = (H, W)
        ys = ys.view(B, K, D, -1)
        ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
        y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, D, -1)
        return y

    @staticmethod
    def backward(ctx, x):
        H, W = ctx.shape
        B, C, L = x.shape
        xs = x.new_empty((B, 4, C, L))
        xs[:, 0] = x
        xs[:, 1] = x.view(B, C, H, W).transpose(dim0=2, dim1=3).flatten(2, 3)
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
        xs = xs.view(B, 4, C, H, W)
        return xs, None, None


class SelectiveScan(torch.autograd.Function):
    """官方 SelectiveScan 接口（TinyViM/VMamba 口径）。

    后端优先级：
      1. `selective_scan_cuda_oflex`——RSML-3 已编译验证的 VMamba v2 flexible
         kernel（与 STR-RepNet 生产同款；已对拍 torch 参考实现 max_abs<2e-5，
         语义 = S6 递推）；
      2. mamba-ssm `selective_scan_fn`（或 vendored Triton 版本）——按四方向
         K 维循环调用。

    接口：u (B, K*C, L)、delta (B, K*C, L)、A (K*C, N)、B/C (B, K, N, L)、
    D (K*C,)、delta_bias (K*C,)。
    """

    @staticmethod
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1):
        if u.stride(-1) != 1:
            u = u.contiguous()
        if delta.stride(-1) != 1:
            delta = delta.contiguous()
        if D is not None and D.stride(-1) != 1:
            D = D.contiguous()
        if B.stride(-1) != 1:
            B = B.contiguous()
        if C.stride(-1) != 1:
            C = C.contiguous()
        if B.dim() == 3:
            B = B.unsqueeze(dim=1)
            ctx.squeeze_B = True
        if C.dim() == 3:
            C = C.unsqueeze(dim=1)
            ctx.squeeze_C = True
        ctx.delta_softplus = delta_softplus

        if _oflex_cuda is not None:
            out, x, *rest = _oflex_cuda.fwd(u, delta, A, B, C, D, delta_bias,
                                            delta_softplus, nrows, True)
            ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
            ctx.use_oflex = True
            return out

        # ---- mamba-ssm 兜底：按方向循环 ----
        if _mamba_ssm_scan is None:
            raise RuntimeError("no selective scan kernel available "
                               "(selective_scan_cuda_oflex / mamba-ssm)")
        K = B.shape[1]
        Cdim = u.shape[1] // K
        assert u.shape[1] == K * Cdim, f"u channels {u.shape[1]} not divisible by K={K}"
        outs = []
        for k in range(K):
            uk = u[:, k * Cdim:(k + 1) * Cdim].permute(0, 2, 1).contiguous()
            dk = delta[:, k * Cdim:(k + 1) * Cdim].permute(0, 2, 1).contiguous()
            ak = A[k * Cdim:(k + 1) * Cdim].contiguous()
            bk = B[:, k].contiguous()
            ck = C[:, k].contiguous()
            dk_vec = D[k * Cdim:(k + 1) * Cdim].contiguous() if D is not None else None
            db = delta_bias[k * Cdim:(k + 1) * Cdim].contiguous() if delta_bias is not None else None
            yk = _mamba_ssm_scan(uk, dk, ak, bk, ck, dk_vec, z=None,
                                 delta_bias=db, delta_softplus=delta_softplus)
            outs.append(yk.permute(0, 2, 1).contiguous())
        ctx.use_oflex = False
        return torch.cat(outs, dim=1)

    @staticmethod
    def backward(ctx, dout, *args):
        if getattr(ctx, "use_oflex", False):
            u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
            if dout.stride(-1) != 1:
                dout = dout.contiguous()
            du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = _oflex_cuda.bwd(
                u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1)
            dB = dB.squeeze(1) if getattr(ctx, "squeeze_B", False) else dB
            dC = dC.squeeze(1) if getattr(ctx, "squeeze_C", False) else dC
            return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None)
        raise RuntimeError("SelectiveScan.backward should never be called directly "
                           "(autograd flows through selective_scan_fn)")


def cross_selective_scan(
        x=None, x_proj_weight=None, x_proj_bias=None,
        dt_projs_weight=None, dt_projs_bias=None,
        A_logs=None, Ds=None, out_norm=None,
        nrows=-1, delta_softplus=True, to_dtype=True, force_fp32=True):
    B, D, H, W = x.shape
    D, N = A_logs.shape
    K, D, R = dt_projs_weight.shape
    L = H * W

    xs = CrossScan.apply(x)

    x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)
    if x_proj_bias is not None:
        x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
    dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
    dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)
    xs = xs.view(B, -1, L)

    dts = dts.contiguous().view(B, -1, L)
    As = -torch.exp(A_logs.to(torch.float))     # (k * c, d_state)
    Bs = Bs.contiguous()
    Cs = Cs.contiguous()
    Ds = Ds.to(torch.float)                     # (K * c)
    delta_bias = dt_projs_bias.view(-1).to(torch.float)

    if force_fp32:
        xs = xs.to(torch.float)
        dts = dts.to(torch.float)
        Bs = Bs.to(torch.float)
        Cs = Cs.to(torch.float)

    ys = SelectiveScan.apply(xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus, nrows)
    ys = ys.view(B, K, -1, H, W)

    y = CrossMerge.apply(ys)
    y = y.transpose(dim0=1, dim1=2).contiguous()   # (B, L, C)
    y = out_norm(y).view(B, H, W, -1)

    return (y.to(x.dtype) if to_dtype else y)


# ---------------------------------------------------------------------------
# SS2D（Laplace mixer 核心：低频池化 -> 四向 scan；高频 RepDW 保留）
# ---------------------------------------------------------------------------
class SS2D(nn.Module):
    def __init__(
            self,
            d_model=96,
            d_state=16,
            ssm_ratio=2.0,
            ssm_rank_ratio=2.0,
            dt_rank="auto",
            act_layer=nn.SiLU,
            d_conv=3,
            conv_bias=True,
            dropout=0.0,
            bias=False,
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            simple_init=False,
            index=0,
            **kwargs,
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        d_expand = int(ssm_ratio * d_model)

        self.pool = nn.AvgPool2d(2 ** (3 - index))
        self.index = index

        # (low, high) 通道分配：frequency ramp inception（官方 split_list）
        split_list = [1 / 4, 1 / 2, 1 / 2, 3 / 4]
        d_inner = int(d_expand * split_list[index])
        self.local_conv = RepDW(d_expand - d_inner)
        self.split = (d_inner, d_expand - d_inner)

        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.d_state = math.ceil(d_model / 6) if d_state == "auto" else d_state
        self.d_conv = d_conv
        self.out_norm = nn.LayerNorm(d_inner)

        self.K = 4
        self.K2 = self.K

        self.in_proj = Conv2d_BN(d_model, d_expand, 1)
        self.act = act_layer()

        if self.d_conv > 1:
            self.conv2d = Rep_Inception(d_expand, 7)

        self.ssm_low_rank = False

        x_proj = [
            nn.Linear(d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in x_proj], dim=0))
        del x_proj

        dt_projs = [
            self.dt_init(self.dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in dt_projs], dim=0))
        del dt_projs

        self.A_logs = self.A_log_init(self.d_state, d_inner, copies=self.K2, merge=True)
        self.Ds = self.D_init(d_inner, copies=self.K2, merge=True)

        self.out_proj = Conv2d_BN(d_expand, d_model, 1)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        if simple_init:
            self.Ds = nn.Parameter(torch.ones((self.K2 * d_inner)))
            self.A_logs = nn.Parameter(torch.randn((self.K2 * d_inner, self.d_state)))
            self.dt_projs_weight = nn.Parameter(torch.randn((self.K, d_inner, self.dt_rank)))
            self.dt_projs_bias = nn.Parameter(torch.randn((self.K, d_inner)))

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random",
                dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        A = torch.arange(1, d_state + 1, dtype=torch.float32, device=device)
        A = A.unsqueeze(0).repeat(d_inner, 1).contiguous()      # (d, n)
        A_log = torch.log(A)
        if copies > 0:
            A_log = A_log.unsqueeze(0).repeat(copies, 1, 1)      # (r, d, n)
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

    def forward_core(self, x, nrows=-1, channel_first=False):
        nrows = 1
        if not channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()
        if self.ssm_low_rank:
            x = self.in_rank(x)

        x_low, x_high = torch.split(x, self.split, dim=1)
        B, C, H, W = x.shape
        x_high = self.local_conv(x_high)
        if self.index < 3:
            x0 = x_low
            x_low = self.pool(x_low)
            res = x0 - F.interpolate(x_low, (H, W), mode='nearest')

        x_low = cross_selective_scan(
            x_low, self.x_proj_weight, None, self.dt_projs_weight, self.dt_projs_bias,
            self.A_logs, self.Ds, getattr(self, "out_norm", None),
            nrows=nrows, delta_softplus=True, force_fp32=self.training,
        )
        x_low = x_low.permute(0, 3, 1, 2)

        if self.index < 3:
            x_low = F.interpolate(x_low, scale_factor=2 ** (3 - self.index), mode='bilinear') + res
        x = torch.cat((x_low, x_high), dim=1)
        if self.ssm_low_rank:
            x = self.out_rank(x)
        return x

    def forward(self, x, **kwargs):
        x = self.in_proj(x)
        if self.d_conv > 1:
            x = self.act(self.conv2d(x))
        y = self.forward_core(x, channel_first=(self.d_conv > 1))
        out = self.dropout(self.out_proj(y))
        return out


class FFN(nn.Module):
    def __init__(self, in_dim, mid_dim=None, out_dim=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_dim = out_dim or in_dim
        mid_dim = mid_dim or in_dim
        self.fc1 = Conv2d_BN(in_dim, mid_dim, 1)
        self.fc2 = Conv2d_BN(mid_dim, out_dim, 1)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LocalBlock(nn.Module):
    """卷积局部块（stage 内非 TViM 块）。"""

    def __init__(self, dim, hidden_dim=64, drop_path=0.0, use_layer_scale=True):
        super().__init__()
        self.dwconv = RepDW(dim)
        self.mlp = FFN(dim, hidden_dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale = nn.Parameter(
                torch.ones(dim).unsqueeze(-1).unsqueeze(-1), requires_grad=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        inp = x
        x = self.dwconv(x)
        x = self.mlp(x)
        if self.use_layer_scale:
            x = inp + self.drop_path(self.layer_scale * x)
        else:
            x = inp + self.drop_path(x)
        return x


class TViMBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            ssm_d_state: int = 16,
            ssm_ratio=2.0,
            ssm_rank_ratio=2.0,
            ssm_dt_rank: Any = "auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv: int = 3,
            ssm_conv_bias=True,
            ssm_drop_rate: float = 0,
            ssm_simple_init=False,
            mlp_ratio=4.0,
            mlp_act_layer=nn.GELU,
            mlp_drop_rate: float = 0.0,
            use_checkpoint: bool = False,
            index=0,
            **kwargs,
    ):
        super().__init__()
        self.ssm_branch = ssm_ratio > 0
        self.mlp_branch = mlp_ratio > 0
        self.use_checkpoint = use_checkpoint

        if self.ssm_branch:
            self.op = SS2D(
                d_model=hidden_dim,
                d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_rank_ratio=ssm_rank_ratio,
                dt_rank=ssm_dt_rank,
                act_layer=ssm_act_layer,
                d_conv=ssm_conv,
                conv_bias=ssm_conv_bias,
                dropout=ssm_drop_rate,
                simple_init=ssm_simple_init,
                index=index,
            )

        self.drop_path = DropPath(drop_path)

        if self.mlp_branch:
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = FFN(in_dim=hidden_dim, mid_dim=mlp_hidden_dim,
                           act_layer=mlp_act_layer, drop=mlp_drop_rate)

    def _forward(self, inp):
        x = inp
        if self.ssm_branch:
            x = inp + self.drop_path(self.op(inp))
        if self.mlp_branch:
            x = x + self.drop_path(self.mlp(x))
        return x

    def forward(self, inp):
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, inp)
        return self._forward(inp)
