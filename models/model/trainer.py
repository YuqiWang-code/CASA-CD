import torch
import torch.nn as nn

from model.encoder import Encoder
from model.decoder import Decoder
from model.sgdp_head import SGDPHead
from model.depth_pyramid_head import DepthPyramidHead

from model.utils import weight_init


class Trainer(nn.Module):
    def __init__(self, model_type='small', pretrained_path=None, resnet_pretrained=True,
                 mode='baseline', casaa_layers=None, casaa_keep_ratio=0.25,
                 casaa_change_share=0.5, casaa_router='change', vit_depth=12,
                 detail_mode='resnet', head_mode='legacy',
                 mobile_pretrained_weight_path=None):
        super().__init__()
        if model_type == 'tiny':
            embed_dim = 192
        elif model_type == 'small':
            embed_dim = 384
        else:
            assert False, r'Trainer: check the vit model type'

        self.mode = mode
        self.head_mode = head_mode
        self.encoder = Encoder(model_type, pretrained_path=pretrained_path,
                               resnet_pretrained=resnet_pretrained,
                               mode=mode, casaa_layers=casaa_layers,
                               casaa_keep_ratio=casaa_keep_ratio,
                               casaa_change_share=casaa_change_share,
                               casaa_router=casaa_router,
                               vit_depth=vit_depth,
                               detail_mode=detail_mode,
                               head_mode=head_mode,
                               mobile_pretrained_weight_path=mobile_pretrained_weight_path)

        if head_mode == 'sgdp':
            # Run5 R5-2：统一 change head（difference-first + semantic gate），
            # 直接消费 MobileDetail raw 16/16/24 + ViT 192。
            self.decoder = SGDPHead()
        elif head_mode == 'csdp':
            # Run7 R7-1：CSDP head（B1+B2 对称差分 + PixelShuffle 金字塔），
            # 消费 encoder 的 depth_pyramid_capture 输出 [B1, B2]。
            self.decoder = DepthPyramidHead()
        else:
            self.decoder = Decoder(in_dim=[64, 128, 256, embed_dim])
            weight_init(self.decoder)
        
    def forward(self, x, y, label=None):
        # label 仅 router='oracle'（DIAGNOSTIC-ONLY）使用，其余模式忽略
        fx, fy = self.encoder(x, y, label=label)
        pred = self.decoder(fx, fy)

        return pred
        