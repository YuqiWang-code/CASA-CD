"""定位 A0 CDD 折叠误差来源：分别只折 TAR / 只折 DCR / 全折。"""
import copy
import torch
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model.casa_str_net import CASASTRNet

CKPT = "/share_datasets/yqwang/checkpoints/CASA-CD/CASA-STR/Run1/A0_BASE_PLAIN/CDD-CD-256/best_F1=0.9467.pth"
PRETRAIN = "/home/yqwang/projects/CASA-CD/pretrained_weight/shvit_s1.pth"

m = CASASTRNet(PRETRAIN, attn_mode="none", rep_mode="plain").eval()
m.load_state_dict(torch.load(CKPT, map_location="cpu", weights_only=False))

torch.manual_seed(16)
pre = torch.randn(8, 3, 256, 256)
post = torch.randn(8, 3, 256, 256)
with torch.no_grad():
    y_full_train = m(pre, post)

# 只折 TAR
m_tar = copy.deepcopy(m).eval()
m_tar.tar.switch_to_deploy()
with torch.no_grad():
    y_tar_folded = m_tar(pre, post)
e_tar = (y_full_train - y_tar_folded).abs().max().item()

# 只折 DCR
m_dcr = copy.deepcopy(m).eval()
m_dcr.decoder.switch_to_deploy()
with torch.no_grad():
    y_dcr_folded = m_dcr(pre, post)
e_dcr = (y_full_train - y_dcr_folded).abs().max().item()

# 全折
m_both = copy.deepcopy(m).eval()
m_both.switch_to_deploy()
with torch.no_grad():
    y_both = m_both(pre, post)
e_both = (y_full_train - y_both).abs().max().item()

def disagree(a, b):
    return ((a > 0.5) != (b > 0.5)).float().mean().item()

print(f"fold-TAR-only  max_abs={e_tar:.3e} disagree={disagree(y_full_train, y_tar_folded):.3e}")
print(f"fold-DCR-only  max_abs={e_dcr:.3e} disagree={disagree(y_full_train, y_dcr_folded):.3e}")
print(f"fold-ALL       max_abs={e_both:.3e} disagree={disagree(y_full_train, y_both):.3e}")

# 若 DCR 是主源：进一步定位 block1/2/3/refine/fuse
if e_dcr > 1e-4:
    from model import str_dcr
    for name, sub in [("block1", m.decoder.block1), ("block2", m.decoder.block2),
                      ("block3", m.decoder.block3), ("refine", m.decoder.refine),
                      ("fuse1", m.decoder.fuse1), ("fuse2", m.decoder.fuse2),
                      ("fuse3", m.decoder.fuse3)]:
        mm = copy.deepcopy(m).eval()
        tgt = getattr(mm.decoder, name)
        try:
            tgt.switch_to_deploy()
        except AttributeError:
            print(f"  {name}: no switch_to_deploy"); continue
        with torch.no_grad():
            yy = mm(pre, post)
        print(f"  DCR[{name}] alone max_abs={(y_full_train - yy).abs().max().item():.3e}")
