import sys
sys.path.insert(0, 'models')
import warnings
warnings.filterwarnings('ignore')
import torch
import copy
from model.casa_str_net import CASASTRNet
from model.shvit_s1_trunc import SHViTS1Truncated

torch.backends.cudnn.benchmark = True

# 1) trunk params（预训练加载路径）
t = SHViTS1Truncated(pretrained_path="/home/yqwang/projects/CASA-CD/pretrained_weight/shvit_s1.pth")
print('trunk params:', t.param_count(), flush=True)

# 2) CASASTRNet + epoch-0 identity（beta=0, none vs change）
torch.manual_seed(16)
n0 = CASASTRNet(pretrained_path="/home/yqwang/projects/CASA-CD/pretrained_weight/shvit_s1.pth",
                attn_mode='none', rep_mode='full').cuda()
torch.manual_seed(16)
n1 = CASASTRNet(pretrained_path="/home/yqwang/projects/CASA-CD/pretrained_weight/shvit_s1.pth",
                attn_mode='change', rep_mode='full').cuda()
pre = torch.randn(2, 3, 256, 256).cuda()
post = torch.randn(2, 3, 256, 256).cuda()
n0.eval()
n1.eval()
with torch.no_grad():
    y0 = n0(pre, post)
    y1 = n1(pre, post)
print('epoch-0 identity (beta=0): max_diff =', (y0 - y1).abs().max().item(), flush=True)
print('out shape:', tuple(y1.shape), flush=True)
print('M1 total params:', sum(p.numel() for p in n1.parameters()) / 1e6, flush=True)

# 3) CASAA 梯度链：beta grad 非零；nudge 后 qkv/proj grad 非零
target = torch.randint(0, 2, (2, 1, 256, 256)).cuda().float()
n1.train()
opt = torch.optim.Adam([p for p in n1.parameters() if p.requires_grad], 2e-4)
from model.utils import BCEDiceLoss
opt.zero_grad()
loss = BCEDiceLoss(n1(pre, post, target), target)
loss.backward()
print('beta grad @step0:', n1.casaa.beta.grad.item(), flush=True)
with torch.no_grad():
    n1.casaa.beta += 0.1
opt.zero_grad()
loss = BCEDiceLoss(n1(pre, post, target), target)
loss.backward()
print('qkv grad after nudge:', n1.casaa.qkv.weight.grad.abs().max().item(), flush=True)
print('proj grad after nudge:', n1.casaa.proj.weight.grad.abs().max().item(), flush=True)
with torch.no_grad():
    n1.casaa.beta -= 0.1

# 4) deploy fold（TAR/DCR 折叠；全模型 train vs deploy）
n1.eval()
n1d = copy.deepcopy(n1)
n1d.switch_to_deploy()
n1d.eval()
with torch.no_grad():
    yd = n1d(pre, post)
    y1e = n1(pre, post)
err = (y1e - yd).abs().max().item()
flip = ((y1e > 0.5) != (yd > 0.5)).float().mean().item()
print('deploy fold max_abs:', err, ' disagree:', flip, flush=True)
print('deploy params:', sum(p.numel() for p in n1d.parameters()) / 1e6, flush=True)

# 5) plain vs full epoch-0（aux 零初始化 -> 逐位一致）
torch.manual_seed(16)
p0 = CASASTRNet(pretrained_path="/home/yqwang/projects/CASA-CD/pretrained_weight/shvit_s1.pth",
                attn_mode='none', rep_mode='plain').cuda()
torch.manual_seed(16)
p1 = CASASTRNet(pretrained_path="/home/yqwang/projects/CASA-CD/pretrained_weight/shvit_s1.pth",
                attn_mode='none', rep_mode='full').cuda()
p0.eval()
p1.eval()
with torch.no_grad():
    z0 = p0(pre, post)
    z1 = p1(pre, post)
print('plain vs full epoch-0 max_diff:', (z0 - z1).abs().max().item(), flush=True)
print('CASA-STR GPU SMOKE ALL OK', flush=True)
