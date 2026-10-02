"""GPU 复现：TF32 on/off 对 A0 CDD 折叠等价性测量的影响。"""
import copy
import torch
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model.casa_str_net import CASASTRNet

CKPT = "/share_datasets/yqwang/checkpoints/CASA-CD/CASA-STR/Run1/A0_BASE_PLAIN/CDD-CD-256/best_F1=0.9467.pth"
PRETRAIN = "/home/yqwang/projects/CASA-CD/pretrained_weight/shvit_s1.pth"

def measure(tf32):
    torch.backends.cudnn.allow_tf32 = tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.deterministic = not tf32
    m = CASASTRNet(PRETRAIN, attn_mode="none", rep_mode="plain").eval().cuda()
    m.load_state_dict(torch.load(CKPT, map_location="cpu", weights_only=False))
    torch.manual_seed(16)
    pre = torch.randn(8, 3, 256, 256).cuda()
    post = torch.randn(8, 3, 256, 256).cuda()
    with torch.no_grad():
        yt = m(pre, post)
        m.switch_to_deploy()
        yd = m(pre, post)
    err = (yt - yd).abs().max().item()
    dis = ((yt > 0.5) != (yd > 0.5)).float().mean().item()
    return err, dis

for tf32 in (True, False):
    err, dis = measure(tf32)
    print(f"cudnn.allow_tf32={tf32} deterministic={not tf32}: max_abs={err:.3e} disagree={dis:.3e}")
