import torch
import torch.nn as nn

from model.encoder import Encoder
from model.decoder import Decoder

from model.utils import weight_init


class Trainer(nn.Module):
    def __init__(self, model_type='small', pretrained_path=None, resnet_pretrained=True,
                 mode='baseline', casaa_layers=None, casaa_keep_ratio=0.25,
                 casaa_change_share=0.5, casaa_router='change'):
        super().__init__()
        if model_type == 'tiny':
            embed_dim = 192
        elif model_type == 'small':
            embed_dim = 384
        else:
            assert False, r'Trainer: check the vit model type'

        self.mode = mode
        self.encoder = Encoder(model_type, pretrained_path=pretrained_path,
                               resnet_pretrained=resnet_pretrained,
                               mode=mode, casaa_layers=casaa_layers,
                               casaa_keep_ratio=casaa_keep_ratio,
                               casaa_change_share=casaa_change_share,
                               casaa_router=casaa_router)

        self.decoder = Decoder(in_dim=[64, 128, 256, embed_dim])
        weight_init(self.decoder)
        
    def forward(self, x, y):
        fx, fy = self.encoder(x, y)
        pred = self.decoder(fx, fy)

        return pred
        