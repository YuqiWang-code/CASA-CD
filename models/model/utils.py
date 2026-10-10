import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import random


def weight_init(module):
    for n, m in module.named_children():
        print('initialize: '+n)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            for f, g in m.named_children():
                print('initialize: ' + f)
                if isinstance(g, nn.Conv2d):
                    nn.init.kaiming_normal_(g.weight, mode='fan_in', nonlinearity='relu')
                    if g.bias is not None:
                        nn.init.zeros_(g.bias)
                elif isinstance(g, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.ones_(g.weight)
                    if g.bias is not None:
                        nn.init.zeros_(g.bias)
                elif isinstance(g, nn.Linear):
                    nn.init.kaiming_normal_(g.weight, mode='fan_in', nonlinearity='relu')
                    if g.bias is not None:
                        nn.init.zeros_(g.bias)
        elif isinstance(m, nn.AdaptiveAvgPool2d) or isinstance(m, nn.AdaptiveMaxPool2d) or isinstance(m, nn.ModuleList) or isinstance(m, nn.BCELoss):
            a=1
        else:
            pass


def init_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def BCEDiceLoss(inputs, targets):
    # print(inputs.shape, targets.shape)
    bce = F.binary_cross_entropy(inputs, targets)
    inter = (inputs * targets).sum()
    eps = 1e-5
    dice = (2 * inter + eps) / (inputs.sum() + targets.sum() + eps)
    # print(bce.item(), inter.item(), inputs.sum().item(), dice.item())
    return bce + 1 - dice


def BCE(inputs, targets):
    # print(inputs.shape, targets.shape)
    bce = F.binary_cross_entropy(inputs, targets)
    return bce


def adjust_learning_rate(args, optimizer, epoch, iter, max_batches, lr_factor=1):
    if args.lr_mode == 'step':
        lr = args.lr * (0.1 ** (epoch // args.step_loss))
    elif args.lr_mode == 'poly':
        cur_iter = iter
        # R4（方案 §6.2）：exact-80K 协议下 last epoch 被截断，poly 分母必须固定为
        # args.max_steps，否则 lr 曲线与 Run1/Run2/Run3 不可比（max_epochs*max_batches > max_steps）。
        if (getattr(args, "arch", "changevit") == "casa_tvim_str"
                and bool(getattr(args, "exact_max_steps", 0))):
            max_iter = args.max_steps
        else:
            max_iter = max_batches * args.max_epochs
        lr = args.lr * (1 - cur_iter * 1.0 / max_iter) ** 0.9
    else:
        raise ValueError('Unknown lr mode {}'.format(args.lr_mode))
    if epoch == 0 and iter < 200:
        lr = args.lr * 0.9 * (iter + 1) / 200 + 0.1 * args.lr  # warm_up
    lr *= lr_factor
    for param_group in optimizer.param_groups:
        # 主线重构（CASA-STR）：支持 per-group lr_scale（backbone 0.1× / new 1.0×），
        # 否则 backbone_lr_ratio 会被 scheduler 覆盖失效（P0，见本地实施注意事项 §5）。
        param_group['lr'] = lr * param_group.get('lr_scale', 1.0)
    return lr
