"""SHViT-S1 truncated hierarchical encoder（主线重构 · CASA-STR 的 backbone）。

Adapted from official SHViT (CVPR 2024) `others/SHViT/model/shvit.py`
（upstream: https://github.com/ysj9909/SHViT, Apache-style LICENSE, retrieved 2026-10-02）。
逐层复刻官方结构（module 名与 state_dict key 完全一致），便于官方 ImageNet-1K
checkpoint 的 retained subset **exact load**：

    patch_embed : 4× Conv2d_BN s2  3->16->32->64->128        (1/2..1/16)
    blocks1     : 2× BasicBlock(128, type="i", 无 SHSA)      (1/16, C128)
    blocks2     : pre(DWconv+FFN) -> PatchMerging 128->224
                  -> post(DWconv+FFN) -> 4× BasicBlock(224, type="s", SHSA pdim=48)
                                                              (1/32, C224)
    （blocks3 与分类头不构造 —— 截断）

多尺度 tap（本地实施注意事项 §2，固定不搜）：
    F1 = patch_embed 第 2 个 Conv 后 ReLU 输出   (B,32,64,64)   1/4
    F2 = patch_embed 第 3 个 Conv 后 ReLU 输出   (B,64,32,32)   1/8
    F3 = blocks1 输出                            (B,128,16,16)  1/16
    F4 = blocks2 输出                            (B,224,8,8)    1/32

Siamese：forward_stem_blocks1 / forward_blocks2 以 2B 拼接 batch 运行（A/B 共享
同一 BN 批次统计；本地实施注意事项 §6）。

S1 配置：embed_dim=[128,224,320], depth=[2,4,5], partial_dim=[32,48,68],
qk_dim=[16,16,16], types=["i","s","s"]（官方 build.py）。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# 官方 S1 配置（build.py）
S1_CONFIG = {
    "embed_dim": [128, 224, 320],
    "depth": [2, 4, 5],
    "partial_dim": [32, 48, 68],
    "qk_dim": [16, 16, 16],
    "types": ["i", "s", "s"],
}


class SqueezeExciteLocal(nn.Module):
    """timm SqueezeExcite 的本地等价实现（避免 timm 依赖）。

    模块命名 fc1/fc2 与官方 SHViT checkpoint 的 key 完全一致（exact load 需要）。
    """

    def __init__(self, in_chs, rd_ratio=0.25):
        super().__init__()
        self.fc1 = nn.Conv2d(in_chs, max(1, int(in_chs * rd_ratio)), 1, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(max(1, int(in_chs * rd_ratio)), in_chs, 1, bias=True)
        self.gate = nn.Sigmoid()

    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        x_se = self.fc1(x_se)
        x_se = self.act1(x_se)
        x_se = self.fc2(x_se)
        return x * self.gate(x_se)


class GroupNorm(nn.GroupNorm):
    """1-group GroupNorm（官方 SHSA pre_norm）。"""

    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)


class Conv2d_BN(nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1):
        super().__init__()
        self.add_module('c', nn.Conv2d(a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', nn.BatchNorm2d(b))
        nn.init.constant_(self.bn.weight, bn_weight_init)
        nn.init.constant_(self.bn.bias, 0)


class PatchMerging(nn.Module):
    def __init__(self, dim, out_dim):
        super().__init__()
        hid_dim = int(dim * 4)
        self.conv1 = Conv2d_BN(dim, hid_dim, 1, 1, 0)
        self.act = nn.ReLU()
        self.conv2 = Conv2d_BN(hid_dim, hid_dim, 3, 2, 1, groups=hid_dim)
        self.se = SqueezeExciteLocal(hid_dim, .25)
        self.conv3 = Conv2d_BN(hid_dim, out_dim, 1, 1, 0)

    def forward(self, x):
        x = self.conv3(self.se(self.act(self.conv2(self.act(self.conv1(x))))))
        return x


class Residual(nn.Module):
    def __init__(self, m, drop=0.):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:
            return x + self.m(x) * torch.rand(x.size(0), 1, 1, 1,
                                              device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        else:
            return x + self.m(x)


class FFN(nn.Module):
    def __init__(self, ed, h):
        super().__init__()
        self.pw1 = Conv2d_BN(ed, h)
        self.act = nn.ReLU()
        self.pw2 = Conv2d_BN(h, ed, bn_weight_init=0)

    def forward(self, x):
        x = self.pw2(self.act(self.pw1(x)))
        return x


class SHSA(nn.Module):
    """Single-Head Self-Attention（官方原样复刻；blocks2 的 type="s" 使用）。"""

    def __init__(self, dim, qk_dim, pdim):
        super().__init__()
        self.scale = qk_dim ** -0.5
        self.qk_dim = qk_dim
        self.dim = dim
        self.pdim = pdim

        self.pre_norm = GroupNorm(pdim)
        self.qkv = Conv2d_BN(pdim, qk_dim * 2 + pdim)
        self.proj = nn.Sequential(nn.ReLU(), Conv2d_BN(dim, dim, bn_weight_init=0))

    def forward(self, x):
        B, C, H, W = x.shape
        x1, x2 = torch.split(x, [self.pdim, self.dim - self.pdim], dim=1)
        x1 = self.pre_norm(x1)
        qkv = self.qkv(x1)
        q, k, v = qkv.split([self.qk_dim, self.qk_dim, self.pdim], dim=1)
        q, k, v = q.flatten(2), k.flatten(2), v.flatten(2)
        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x1 = (v @ attn.transpose(-2, -1)).reshape(B, self.pdim, H, W)
        x = self.proj(torch.cat([x1, x2], dim=1))
        return x


class BasicBlock(nn.Module):
    def __init__(self, dim, qk_dim, pdim, type):
        super().__init__()
        if type == "s":
            self.conv = Residual(Conv2d_BN(dim, dim, 3, 1, 1, groups=dim, bn_weight_init=0))
            self.mixer = Residual(SHSA(dim, qk_dim, pdim))
            self.ffn = Residual(FFN(dim, int(dim * 2)))
        elif type == "i":
            self.conv = Residual(Conv2d_BN(dim, dim, 3, 1, 1, groups=dim, bn_weight_init=0))
            self.mixer = nn.Identity()
            self.ffn = Residual(FFN(dim, int(dim * 2)))

    def forward(self, x):
        return self.ffn(self.mixer(self.conv(x)))


class SHViT(nn.Module):
    """官方 SHViT 全模型（供 pretrain audit / function-equivalence 对照；训练不使用）。"""

    def __init__(self, in_chans=3, num_classes=1000,
                 embed_dim=None, partial_dim=None, qk_dim=None, depth=None,
                 types=None, down_ops=None, distillation=False):
        super().__init__()
        embed_dim = embed_dim or [128, 256, 384]
        partial_dim = partial_dim or [32, 64, 96]
        qk_dim = qk_dim or [16, 16, 16]
        depth = depth or [1, 2, 3]
        types = types or ["s", "s", "s"]
        down_ops = down_ops or [['subsample', 2], ['subsample', 2], ['']]

        self.patch_embed = nn.Sequential(
            Conv2d_BN(in_chans, embed_dim[0] // 8, 3, 2, 1), nn.ReLU(),
            Conv2d_BN(embed_dim[0] // 8, embed_dim[0] // 4, 3, 2, 1), nn.ReLU(),
            Conv2d_BN(embed_dim[0] // 4, embed_dim[0] // 2, 3, 2, 1), nn.ReLU(),
            Conv2d_BN(embed_dim[0] // 2, embed_dim[0], 3, 2, 1))

        self.blocks1, self.blocks2, self.blocks3 = [], [], []
        for i, (ed, kd, pd, dpth, do, t) in enumerate(zip(embed_dim, qk_dim, partial_dim, depth, down_ops, types)):
            for d in range(dpth):
                eval('self.blocks' + str(i + 1)).append(BasicBlock(ed, kd, pd, t))
            if do[0] == 'subsample':
                blk = eval('self.blocks' + str(i + 2))
                blk.append(nn.Sequential(
                    Residual(Conv2d_BN(embed_dim[i], embed_dim[i], 3, 1, 1, groups=embed_dim[i])),
                    Residual(FFN(embed_dim[i], int(embed_dim[i] * 2)))))
                blk.append(PatchMerging(*embed_dim[i:i + 2]))
                blk.append(nn.Sequential(
                    Residual(Conv2d_BN(embed_dim[i + 1], embed_dim[i + 1], 3, 1, 1, groups=embed_dim[i + 1])),
                    Residual(FFN(embed_dim[i + 1], int(embed_dim[i + 1] * 2)))))
        self.blocks1 = nn.Sequential(*self.blocks1)
        self.blocks2 = nn.Sequential(*self.blocks2)
        self.blocks3 = nn.Sequential(*self.blocks3)

        self.head = nn.Identity() if num_classes <= 0 else BN_LinearLocal(embed_dim[-1], num_classes)
        self.distillation = distillation
        if distillation:
            self.head_dist = nn.Identity() if num_classes <= 0 else BN_LinearLocal(embed_dim[-1], num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.blocks1(x)
        x = self.blocks2(x)
        x = self.blocks3(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        if self.distillation:
            x = self.head(x), self.head_dist(x)
            if not self.training:
                x = (x[0] + x[1]) / 2
        else:
            x = self.head(x)
        return x


class BN_LinearLocal(nn.Sequential):
    """官方 BN_Linear 本地等价（避免 timm trunc_normal_ 依赖）。"""

    def __init__(self, a, b, bias=True, std=0.02):
        super().__init__()
        self.add_module('bn', nn.BatchNorm1d(a))
        self.add_module('l', nn.Linear(a, b, bias=bias))
        nn.init.trunc_normal_(self.l.weight, std=std)
        if bias:
            nn.init.constant_(self.l.bias, 0)


def build_shvit_s1(num_classes=1000, distillation=False):
    """官方 SHViT-S1 全模型（pretrain audit / 功能等价对照用）。"""
    cfg = S1_CONFIG
    return SHViT(in_chans=3, num_classes=num_classes,
                 embed_dim=cfg["embed_dim"], partial_dim=cfg["partial_dim"],
                 qk_dim=cfg["qk_dim"], depth=cfg["depth"], types=cfg["types"],
                 down_ops=[['subsample', 2], ['subsample', 2], ['']],
                 distillation=distillation)


class SHViTS1Truncated(nn.Module):
    """SHViT-S1 截断主干：patch_embed + blocks1 + blocks2（state_dict key 与官方一致）。

    pretrained exact load：从官方 checkpoint['model'] 加载 retained keys
    （patch_embed.* / blocks1.* / blocks2.*），blocks3/head 允许 missing，
    retained 不允许 shape mismatch / missing（strict 审计在 audit 脚本完成）。
    """

    def __init__(self, pretrained_path=None, num_classes=0):
        super().__init__()
        cfg = S1_CONFIG
        ed = cfg["embed_dim"]
        full = build_shvit_s1(num_classes=num_classes, distillation=False)
        self.patch_embed = full.patch_embed
        self.blocks1 = full.blocks1
        self.blocks2 = full.blocks2
        del full

        self.embed_dim = ed
        self.pretrained_path = pretrained_path
        if pretrained_path is not None:
            self.load_pretrained(pretrained_path)

    def load_pretrained(self, path):
        # 官方 SHViT checkpoint 含 argparse.Namespace 等非张量对象 -> weights_only=False
        # （官方可信权重，SHA256 在 manifest/审计中记录）
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "model" in ckpt:
            sd = ckpt["model"]
        elif isinstance(ckpt, dict) and "state_dict" in ckpt:
            sd = ckpt["state_dict"]
        else:
            sd = ckpt
        model_sd = self.state_dict()
        retained = [k for k in model_sd]
        missing = [k for k in retained if k not in sd]
        loaded = {k: sd[k] for k in retained if k in sd and sd[k].shape == model_sd[k].shape}
        mismatch = [k for k in retained if k in sd and sd[k].shape != model_sd[k].shape]
        assert len(missing) == 0, f"retained keys missing in checkpoint: {missing[:5]}"
        assert len(mismatch) == 0, f"retained keys shape mismatch: {mismatch[:5]}"
        model_sd.update(loaded)
        self.load_state_dict(model_sd)
        self._load_stats = {
            "retained": len(retained), "exact_loaded": len(loaded),
            "missing": missing, "mismatch": mismatch,
        }
        return self._load_stats

    def forward_stem_blocks1(self, x2b):
        """2B 拼接 batch：stem + blocks1 -> f1_2b, f2_2b, f3_2b。"""
        x = x2b
        f1 = f2 = None
        for i, m in enumerate(self.patch_embed):
            x = m(x)
            if i == 3:      # 第 2 个 Conv 后的 ReLU -> 1/4 C32
                f1 = x
            elif i == 5:    # 第 3 个 Conv 后的 ReLU -> 1/8 C64
                f2 = x
        f3 = self.blocks1(x)
        return f1, f2, f3

    def forward_blocks2(self, x2b):
        """2B 拼接 batch：blocks2 -> f4_2b（1/32 C224）。"""
        return self.blocks2(x2b)

    def param_count(self):
        return sum(p.numel() for p in self.parameters())
