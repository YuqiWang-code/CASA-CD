"""Backbone zoo: MobileNetV2 (default), VMamba-Tiny and a 4-block ViT-L.

All backbones return the same contract so that TAR / DCR stay untouched::

    stem, [c1, c2, c3, c4]

``c1..c4`` live at 64², 32², 16² and 8² for a 256² input; ``stem`` is an extra
fine view (128² for the two transformers, 64² for MobileNetV2) that only the
``cd2`` bridge taps.  Channel counts are exposed as ``out_channels`` and
``stem_channels`` so the model can size its heads from the backbone.
"""
import torch
from torch import nn
from torch.nn import functional as F

from model.mobilenet_v2 import mobilenet_v2

BACKBONES = ("mobilenet_v2", "vmamba_tiny", "vit_base4")
TAR_DIMS = (96, 192, 384, 768)


class MobileNetAdapter(nn.Module):
    name = "mobilenet_v2"
    out_channels = (24, 32, 96, 320)
    stem_channels = 16
    scales = ("64", "32", "16", "8")

    def __init__(self, pretrained_path=None, freeze=False):
        super().__init__()
        self.net = mobilenet_v2(pretrained_path)
        if freeze:
            for parameter in self.net.parameters():
                parameter.requires_grad_(False)

    def forward(self, x):
        features = self.net(x)
        return features[0], features[1:]


class VMambaAdapter(nn.Module):
    name = "vmamba_tiny"
    out_channels = TAR_DIMS
    stem_channels = 96
    scales = ("64", "32", "16", "8")

    def __init__(self, pretrained_path=None, freeze=False):
        super().__init__()
        from model.vmamba_tiny import VMambaBackbone
        if pretrained_path is None:
            raise ValueError("vmamba_tiny needs the vssm_tiny checkpoint (--pretrained)")
        self.net = VMambaBackbone(pretrained_path)
        self.load_stats = self.net.load_stats()
        if freeze:
            for parameter in self.net.parameters():
                parameter.requires_grad_(False)

    def forward(self, x):
        c1, c2, c3, c4 = self.net(x)
        stem = F.interpolate(c1, scale_factor=2, mode="bilinear", align_corners=False)
        return stem, [c1, c2, c3, c4]


class ViT4Adapter(nn.Module):
    """Four blocks of a ViT used as four depths, resampled to the pyramid.

    The official DeiT/ViT positional grid is 14×14 (224²); it is interpolated to
    the 16×16 grid that a 256² input produces.  Depth-as-scale: the block-1..4
    states are resampled to 64²/32²/16²/8² and projected to the TAR widths, so
    the rest of the network keeps its usual shapes.
    """

    name = "vit_base4"
    out_channels = TAR_DIMS
    stem_channels = 64
    scales = ("64", "32", "16", "8")

    def __init__(self, pretrained_path=None, freeze=False, depth=4,
                 model_name="vit_base_patch16_224.augreg_in21k_ft_in1k",
                 image_size=256):
        super().__init__()
        import timm
        vit = timm.create_model(model_name, pretrained=False,
                                num_classes=0, img_size=image_size)
        if pretrained_path is not None:
            state = torch.load(pretrained_path, map_location="cpu", weights_only=False)
            state = state.get("model", state)
            pos = state.get("pos_embed")
            if pos is not None and pos.shape[1] != vit.pos_embed.shape[1]:
                # the checkpoint is a 14x14 grid (224^2); resample it to this grid
                side = int(round((pos.shape[1] - 1) ** 0.5))
                grid = int(round((vit.pos_embed.shape[1] - 1) ** 0.5))
                patch = pos[:, 1:].reshape(1, side, side, -1).permute(0, 3, 1, 2)
                patch = F.interpolate(patch, size=(grid, grid), mode="bicubic",
                                      align_corners=False)
                patch = patch.permute(0, 2, 3, 1).reshape(1, grid * grid, -1)
                state = dict(state)
                state["pos_embed"] = torch.cat((pos[:, :1], patch), 1)
            vit.load_state_dict(state, strict=False)
        self.patch_embed = vit.patch_embed
        self.cls_token = vit.cls_token
        self.pos_embed = vit.pos_embed
        self.blocks = vit.blocks[:depth]
        # drop the unused tail so the encoder reports only what it runs
        vit.blocks = vit.blocks[:depth]
        self.norm = vit.norm
        self.embed_dim = vit.embed_dim
        self.grid = image_size // vit.patch_embed.patch_size[0]
        # the trunk is used at 256^2 in training, but a 128^2 probe is cheap and the
        # grid is resolved from the token count, so no fixed number of tokens is assumed
        self.patch_embed.strict_img_size = False
        self.proj = nn.ModuleList([nn.Conv2d(self.embed_dim, dim, 1) for dim in TAR_DIMS])
        self.stem_proj = nn.Conv2d(self.embed_dim, self.stem_channels, 1)
        self._resize_pos_embed()
        if freeze:
            for parameter in self.parameters():
                parameter.requires_grad_(False)

    def _resize_pos_embed(self):
        tokens = self.pos_embed.shape[1] - 1
        side = int(round(tokens ** 0.5))
        if side == self.grid:
            return
        weight = self.pos_embed[:, 1:].reshape(1, side, side, -1).permute(0, 3, 1, 2)
        weight = F.interpolate(weight, size=(self.grid, self.grid), mode="bicubic",
                               align_corners=False)
        weight = weight.permute(0, 2, 3, 1).reshape(1, self.grid * self.grid, -1)
        self.pos_embed = nn.Parameter(torch.cat((self.pos_embed[:, :1], weight), 1))

    def _positional(self, count):
        """Resample the learned grid to the token count of the current input."""
        if self.pos_embed.shape[1] - 1 == count:
            return self.pos_embed
        side = int(round((self.pos_embed.shape[1] - 1) ** 0.5))
        grid = int(round(count ** 0.5))
        if grid * grid != count:
            raise ValueError(f"token count {count} is not a square grid")
        weight = self.pos_embed[:, 1:].reshape(1, side, side, -1).permute(0, 3, 1, 2)
        weight = F.interpolate(weight, size=(grid, grid), mode="bicubic",
                               align_corners=False)
        weight = weight.permute(0, 2, 3, 1).reshape(1, grid * grid, -1)
        return torch.cat((self.pos_embed[:, :1], weight), 1)

    def forward(self, x):
        tokens = self.patch_embed(x)
        batch, count = tokens.shape[0], tokens.shape[1]
        side = int(round(count ** 0.5))
        if side * side != count:
            raise ValueError(f"token count {count} is not a square grid")
        tokens = torch.cat((self.cls_token.expand(batch, -1, -1), tokens), 1)
        tokens = tokens + self._positional(count)
        states = []
        for block in self.blocks:
            tokens = block(tokens)
            states.append(self.norm(tokens)[:, 1:])
        maps = [state.transpose(1, 2).reshape(batch, self.embed_dim, side, side)
                for state in states]
        targets = (64, 32, 16, 8)
        pyramid = [F.interpolate(self.proj[i](maps[i]), size=(size, size), mode="bilinear",
                                 align_corners=False)
                   for i, size in enumerate(targets)]
        stem = F.interpolate(self.stem_proj(maps[0]), size=(128, 128), mode="bilinear",
                             align_corners=False)
        return stem, pyramid


def build_backbone(name, pretrained_path=None, freeze=False):
    if name == "mobilenet_v2":
        return MobileNetAdapter(pretrained_path, freeze)
    if name == "vmamba_tiny":
        return VMambaAdapter(pretrained_path, freeze)
    if name == "vit_base4":
        return ViT4Adapter(pretrained_path, freeze)
    raise ValueError(f"unknown backbone {name!r}; choose from {BACKBONES}")
