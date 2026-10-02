"""Run11 STRTASSNet: Frozen ViT4 + TAR/DCR + optional TASS spatial residual.

spatial_mode in {"token", "tass"}（Run11 唯一变量）:
    token (C0): t_i = T_i（B1↑4 / B2↑2 / B3 / B4↓2）
    tass  (M1): t_i = T_i + alpha_i * P_i(S_i),  i in {1,2,3};  t4 unchanged
alpha 初始化为 0 -> M1 epoch-0 输出与 C0 逐位一致（zero-init residual injection）。
TAR/DCR 固定 rep_mode=full、dim=160（两个 group 完全相同的 scaffold，不开放旋钮）。

构造顺序（RNG 纪律，方案 §9.2）：encoder -> head -> TAR -> DCR ->（仅 M1）TASS +
alpha；TASS 在全部共享模块之后创建，保证 C0/M1 共享模块逐位同初始化。
switch_to_deploy() 只折叠 TAR/DCR；TASS 是单路径静态 stem，无训练期多分支。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.str_encoder import STRViT4Encoder
from model.str_tar import MultiScaleTAR
from model.str_dcr import DCRDecoder
from model.tass_stem import TASSStem


class STRTASSNet(nn.Module):
    def __init__(self, pretrained_path, dim=160, spatial_mode="token"):
        super().__init__()
        assert spatial_mode in ("token", "tass"), f"spatial_mode must be token|tass, got {spatial_mode}"
        self.dim = dim
        self.spatial_mode = spatial_mode

        self.encoder = STRViT4Encoder(pretrained_path)
        # head 先于 TAR/DCR 的 aux 分支构造（STR RNG 纪律，见 str_fusion.py）
        self.head = nn.Conv2d(dim, 1, 1)
        self.tar = MultiScaleTAR(
            encoder_dims=(192, 192, 192, 192), dim=dim,
            use_temporal_aux=True, use_dcr_aux=True, use_residual=True,
        )
        self.decoder = DCRDecoder(dim=dim, use_aux=True, use_residual=True)

        # TASS 最后构造（M1 only），不消耗任何共享模块之前的全局 RNG
        self.tass = None
        self.alpha = None
        if spatial_mode == "tass":
            self.tass = TASSStem(project_to=192)
            self.alpha = nn.Parameter(torch.zeros(3))

    def forward(self, pre, post, label=None):
        pre_feats = self.encoder(pre)
        post_feats = self.encoder(post)
        if self.tass is not None:
            sp_pre = self.tass(pre)
            sp_post = self.tass(post)
            a = self.alpha
            pre_feats = [pre_feats[0] + a[0] * sp_pre[0],
                         pre_feats[1] + a[1] * sp_pre[1],
                         pre_feats[2] + a[2] * sp_pre[2],
                         pre_feats[3]]
            post_feats = [post_feats[0] + a[0] * sp_post[0],
                          post_feats[1] + a[1] * sp_post[1],
                          post_feats[2] + a[2] * sp_post[2],
                          post_feats[3]]
        feats = self.tar(pre_feats, post_feats)
        x = self.decoder(feats)
        logits = self.head(x)
        logits = F.interpolate(logits, size=pre.shape[-2:], mode="bilinear", align_corners=False)
        return torch.sigmoid(logits)

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()  # frozen ViT stays a feature extractor
        return self

    @torch.no_grad()
    def switch_to_deploy(self):
        self.tar.switch_to_deploy()
        self.decoder.switch_to_deploy()
        return self

    def shared_state_dict(self):
        """C0/M1 共享模块的 state dict（不含 tass.* 与 alpha），用于逐位对拍。"""
        return {k: v for k, v in self.state_dict().items()
                if not k.startswith("tass.") and k != "alpha"}
