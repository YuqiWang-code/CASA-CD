"""
Utils for LiFT

Code by: Saksham Suri and Matthew Walmer
"""


def num_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def infer_settings(args):
    model_type = args.model_type
    if model_type not in ['dino_vits8', 'dino_vits16', 'dino_vitb8', 'dino_vitb16', 'vit_small_patch8_224',
            'vit_small_patch16_224', 'vit_base_patch8_224', 'vit_base_patch16_224']:
        print('WARNING: model type "%s" may not be supported'%model_type)
    if 'vits8' in model_type or 'vit_small_patch8' in model_type:
        patch, channel, layer = 8, 384, 11
    elif 'vits16' in model_type or 'vit_small_patch16' in model_type:
        patch, channel, layer = 16, 384, 11
    elif 'vitb8' in model_type or 'vit_base_patch8' in model_type:
        patch, channel, layer = 8, 768, 11
    elif 'vitb16' in model_type or 'vit_base_patch16' in model_type:
        patch, channel, layer = 16, 768, 11
    else:
        print('WARNING: model type "%s" not recognized, settings not inferred'%model_type)
        return
    if args.channel is None: args.channel = channel
    if args.patch is None: args.patch = patch
    if args.stride is None: args.stride = args.patch
    if args.layer is None: args.layer = layer


# [B, T, C] --> [B, C, H, W]
def convert_shape(x, H, W):
    x = x.permute(0, 2, 1)
    x = x.reshape(x.shape[0], -1, H, W)
    return x