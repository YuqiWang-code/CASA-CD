"""诊断：A0 best checkpoint 中 TAR/DCR BN running stats 极值（折叠放大源）。"""
import torch

def diag(path, tag):
    sd = torch.load(path, map_location="cpu", weights_only=False)
    print(f"===== {tag} =====")
    for k in sorted(sd.keys()):
        if k.endswith("running_var"):
            v = sd[k].float()
            gam = sd[k.replace("running_var", "weight")].float()
            factor = (gam / (v + 1e-5).sqrt())
            print(f"{k}: var min={v.min().item():.3e} max={v.max().item():.3e} "
                  f"| gamma/sqrt(var+eps) min={factor.min().item():.3e} max={factor.max().item():.3e}")
        elif k.endswith("running_mean"):
            v = sd[k].float()
            print(f"{k}: mean min={v.min().item():.3e} max={v.max().item():.3e}")

diag("/share_datasets/yqwang/checkpoints/CASA-CD/CASA-STR/Run1/A0_BASE_PLAIN/CDD-CD-256/best_F1=0.9467.pth", "A0 CDD")
diag("/share_datasets/yqwang/checkpoints/CASA-CD/CASA-STR/Run1/A0_BASE_PLAIN/SYSU-CD-256/best_F1=0.8246.pth", "A0 SYSU")
