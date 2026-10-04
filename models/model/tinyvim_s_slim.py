"""TinyViM-S-Slim：CAACP-SS2D 主线的预训练骨干（无 timm / 无 einops / 无分类头）。

官方 TinyViM-S（ICCV 2025）：
    widths = [48, 64, 168, 224]    depths = [3, 3, 9, 6]
    Stage3（index=2，1/16）末块为 TViM（SS2D）；Stage4（index=3，1/32）为
    Local×5 + final TViM。

Slim（调研文档 §5.2，预算优先保留 backbone）：
    Stage4 只保留 Local[0..2] + final TViM[5]，删除 Local[3..4]（−0.814M）。
    加载器做显式 old key -> new key 重映射（network.6.5.* -> network.6.3.*）。

CAACP（可选）：Stage3 末块 TViM 的 SS2D 替换为 CAACPSS2D（β=0 精确继承预训练）。

预训练加载：checkpoint 为 timm 训练产物（含 model_ema），取 `model_ema` 权重；
分类头 head/dist_head/norm 与 stage4 被裁剪块为「有意丢弃」；fork_feat 的
norm0/2/4/6 与 β 为「新增模块」（BN 默认初始化，训练适配）。
"""
import torch
import torch.nn as nn

from model.layers.ss2d import (Conv2d_BN, LocalBlock, TViMBlock, trunc_normal_)

WIDTH_S = [48, 64, 168, 224]
DEPTH_S = [3, 3, 9, 6]
STAGE4_KEEP = [0, 1, 2, 5]          # 保留 Local[0..2] + final TViM；删除 Local[3..4]


def _stem(in_chs, out_chs):
    return nn.Sequential(
        Conv2d_BN(in_chs, out_chs // 2, 3, 2, 1),
        nn.GELU(),
        Conv2d_BN(out_chs // 2, out_chs, 3, 2, 1),
        nn.GELU(),
    )


class Embedding(nn.Module):
    """stage 间下采样（3×3 stride2 patch embedding）。"""

    def __init__(self, patch_size=3, stride=2, padding=1, in_chans=48, embed_dim=64):
        super().__init__()
        self.proj = Conv2d_BN(in_chans, embed_dim, patch_size, stride, padding)

    def forward(self, x):
        return self.proj(x)


def _build_stage(dim, index, depth, slim=False):
    """按官方 Stage 规则构建：末 ssm_num=1 块为 TViM；index==2 时中块（depth//2）
    也为 TViM；其余为 LocalBlock。slim=True 只保留 Stage4 的 STAGE4_KEEP 块。"""
    blocks = []
    idxs = STAGE4_KEEP if slim else list(range(depth))
    for bi in idxs:
        if depth - bi <= 1:
            blocks.append(TViMBlock(dim, ssm_d_state=8, ssm_ratio=1.0,
                                    ssm_conv_bias=False, index=index))
        elif index == 2 and bi == depth // 2:
            blocks.append(TViMBlock(dim, ssm_d_state=8, ssm_ratio=1.0,
                                    ssm_conv_bias=False, index=index))
        else:
            blocks.append(LocalBlock(dim=dim, hidden_dim=4 * dim))
    return nn.Sequential(*blocks)


class TinyViMSlim(nn.Module):
    def __init__(self, pretrained_path=None, caacp=False,
                 widths=WIDTH_S, depths=DEPTH_S, caacp_score_mode="rank"):
        super().__init__()
        self.widths = list(widths)
        self.depths = list(depths)
        self.caacp = caacp
        self.caacp_score_mode = caacp_score_mode

        self.patch_embed = _stem(3, self.widths[0])

        network = []
        for i in range(len(self.widths)):
            network.append(_build_stage(self.widths[i], i, self.depths[i], slim=(i == 3)))
            if i >= len(self.widths) - 1:
                break
            network.append(Embedding(3, 2, 1, self.widths[i], self.widths[i + 1]))
        self.network = nn.ModuleList(network)          # [s0, emb, s1, emb, s2, emb, s3]

        self.out_indices = [0, 2, 4, 6]
        for i_emb, i_layer in enumerate(self.out_indices):
            self.add_module(f"norm{i_layer}", nn.BatchNorm2d(self.widths[i_emb]))

        self.apply(self._init_weights)

        self.caacp_block = None
        self.caacp_op = None
        if caacp:
            self._wrap_caacp()

        self._load_stats = None
        if pretrained_path is not None:
            self.load_pretrained(pretrained_path)

    # ------------------------------------------------------------------
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _wrap_caacp(self):
        """Stage3 末块 TViM 的 SS2D -> CAACPSS2D（继承预训练参数，β 新增 zero-init）。"""
        from model.layers.caacp_ss2d import CAACPSS2D
        blk = self.network[4][self.depths[2] - 1]
        assert isinstance(blk, TViMBlock), "CAACP target must be the final TViM of stage3"
        dim = self.widths[2]
        new = CAACPSS2D(d_model=dim, d_state=8, ssm_ratio=1.0, dt_rank="auto",
                        d_conv=3, conv_bias=False, index=2,
                        score_mode=self.caacp_score_mode)
        new.load_state_dict(blk.op.state_dict(), strict=False)   # beta 缺失 → 保持 0
        blk.op = new
        self.caacp_block = blk
        self.caacp_op = new

    # ------------------------------------------------------------------
    def load_pretrained(self, path):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        src = ck.get("model_ema", None) or ck.get("model", ck)
        remap = {}
        for i, keep in enumerate(STAGE4_KEEP):
            remap[f"network.6.{keep}."] = f"network.6.{i}."
        dropped_prefixes = ("head.", "dist_head.", "norm.")
        # slim 裁剪：stage4 的 Local[3]/Local[4]（原始 network.6.3 / network.6.4）有意丢弃
        dropped_stage4 = ("network.6.3.", "network.6.4.")
        sd, dropped = {}, []
        for k, v in src.items():
            if k.startswith(dropped_prefixes) or k.startswith(dropped_stage4):
                dropped.append(k)
                continue
            nk = k
            for old_p, new_p in remap.items():
                if k.startswith(old_p):
                    nk = new_p + k[len(old_p):]
                    break
            sd[nk] = v
        model_sd = self.state_dict()
        loaded = {k: v for k, v in sd.items() if k in model_sd}
        missing_new = [k for k in model_sd if k not in sd]      # fork norm / beta 等新模块
        unexpected = [k for k in sd if k not in model_sd]
        self.load_state_dict(loaded, strict=False)
        worst = 0.0
        for k, v in loaded.items():
            worst = max(worst, (model_sd[k].float() - v.float()).abs().max().item())
        self._load_stats = {
            "pretrained_keys": len(src),
            "retained": len(loaded),
            "worst_diff": worst,
            "missing_new": missing_new,
            "unexpected": unexpected,
            "dropped_intentional": dropped,
            "loaded_keys": list(loaded.keys()),   # exact-loaded 键列表（BACKBONE-ADAPT 分 stage 审计口径）
        }
        return self._load_stats

    def load_stats(self):
        return self._load_stats

    # ------------------------------------------------------------------
    # 分阶段前向（供 paired 网络注入 CAACP 共享 score）
    def forward_stem_s2_prefix(self, x2b):
        """patch_embed + stage0 + emb + stage1 + emb + stage2 前 depth-1 块。
        返回 (f1_tap, f2_tap, x@1/16 stage2-prefix)。"""
        x = self.patch_embed(x2b)
        x = self.network[0](x)                          # stage0 (1/4)
        f1 = self.norm0(x)
        x = self.network[1](x)                          # emb 1/4 -> 1/8
        x = self.network[2](x)                          # stage1 (1/8)
        f2 = self.norm2(x)
        x = self.network[3](x)                          # emb 1/8 -> 1/16
        for blk in self.network[4][: self.depths[2] - 1]:
            x = blk(x)
        return f1, f2, x

    def forward_caacp_block(self, x2b):
        """stage2 末块（CAACP 目标或官方 TViM）。"""
        return self.network[4][self.depths[2] - 1](x2b)

    def forward_stage4(self, x2b):
        """emb 1/16 -> 1/32 + slim stage4。返回 f4_tap。"""
        x = self.network[5](x2b)
        x = self.network[6](x)
        return self.norm6(x)

    def forward(self, x):
        x = self.patch_embed(x)
        outs = []
        for idx, block in enumerate(self.network):
            x = block(x)
            if idx in self.out_indices:
                outs.append(getattr(self, f"norm{idx}")(x))
        return outs

    def param_count(self):
        return sum(p.numel() for p in self.parameters())


def build_tinyvim_s_slim(pretrained_path=None, caacp=False):
    return TinyViMSlim(pretrained_path=pretrained_path, caacp=caacp)


if __name__ == "__main__":
    torch.manual_seed(0)
    m = TinyViMSlim(caacp=False)
    print("slim trunk params:", f"{m.param_count():,}")
    x = torch.randn(2, 3, 256, 256)
    outs = m(x)
    print("taps:", [tuple(t.shape) for t in outs])
