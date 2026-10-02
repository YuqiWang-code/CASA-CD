"""CASA-STR 正式复测（STR T2 协议）：对已完成 run 的 best checkpoint 用
TF32-off + cudnn deterministic 重新测量折叠等价性与 deploy 图测试集指标。

背景：train.py 初版 TEST 区块在默认 TF32-on 下测折叠误差，GPU TF32 卷积舍入经
BN 因子放大可达 1e-2 级（CDD A0 实测 1.5e-2 vs 协议口径 1.6e-5）。已完成的
run（A0 CDD/SYSU 及后续用旧代码启动的 run）用本脚本复测，输出与 train.py
一致的 TEST RESULTS 区块（作为该 run 的正式数值）。

用法：python remeasure_casa_str.py --ckpt_dir <dir> --dataset <DS> --gpu_id 0
"""
import argparse
import glob
import os
import sys

import torch

_MODELS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
if _MODELS not in sys.path:
    sys.path.insert(0, _MODELS)

from model.casa_str_net import CASASTRNet
from model.metric_tool import ConfuseMatrixMeter
from model.utils import BCEDiceLoss
import dataset.dataset as myDataLoader
import dataset.Transforms as myTransforms


def measure_params(model):
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def run_val(loader, model, on_gpu):
    model.eval()
    meter = ConfuseMatrixMeter(n_class=2)
    losses = []
    for img, target in loader:
        pre, post = img[:, 0:3], img[:, 3:6]
        if on_gpu:
            pre, post, target = pre.cuda(), post.cuda(), target.cuda()
        output = model(pre.float(), post.float(), target.float())
        losses.append(BCEDiceLoss(output, target.float()).item())
        pred = (output > 0.5).long()
        meter.update_cm(pr=pred.cpu().numpy(), gt=target.cpu().numpy())
    return sum(losses) / len(losses), meter.get_scores()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--attn_mode", type=str, default="none")
    parser.add_argument("--rep_mode", type=str, default="plain")
    parser.add_argument("--gpu_id", type=int, default=0)
    args = parser.parse_args()

    torch.cuda.set_device(args.gpu_id)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.deterministic = True

    bests = sorted(glob.glob(os.path.join(args.ckpt_dir, "best_F1=*.pth")))
    if not bests:
        raise FileNotFoundError(f"no best_F1=*.pth in {args.ckpt_dir}")
    best = bests[-1]

    pretrain = "/home/yqwang/projects/CASA-CD/pretrained_weight/shvit_s1.pth"
    model = CASASTRNet(pretrain, attn_mode=args.attn_mode, rep_mode=args.rep_mode).float().cuda()
    model.load_state_dict(torch.load(best, map_location="cpu", weights_only=False))
    model.eval()

    torch.manual_seed(16)
    pre_fix = torch.randn(8, 3, 256, 256).cuda()
    post_fix = torch.randn(8, 3, 256, 256).cuda()
    with torch.no_grad():
        y_train = model(pre_fix, post_fix)
    model.switch_to_deploy()
    model.eval()
    with torch.no_grad():
        y_deploy = model(pre_fix, post_fix)
    fold_err = (y_train - y_deploy).abs().max().item()
    disagree = ((y_train > 0.5) != (y_deploy > 0.5)).float().mean().item()

    ds_root = f"/share_datasets/CD/{args.dataset}"
    transform = myTransforms.Compose([
        myTransforms.Normalize(mean=[0.406, 0.456, 0.485, 0.406, 0.456, 0.485],
                               std=[0.225, 0.224, 0.229, 0.225, 0.224, 0.229]),
        myTransforms.Scale(256, 256),
        myTransforms.ToTensor(),
    ])
    test_data = myDataLoader.Dataset(file_root=ds_root,
                                     list_path=os.path.join(ds_root, "list/test.txt"),
                                     transform=transform)
    test_loader = torch.utils.data.DataLoader(test_data, shuffle=False, batch_size=16,
                                              num_workers=4, pin_memory=True)
    _, scores = run_val(test_loader, model, on_gpu=True)

    print("=== REMEASURED TEST RESULTS (STR T2 protocol) ===")
    print(f"[CKPT] {best}")
    print(f"[NUMERICS] tf32=off deterministic=on")
    print(f"[REPARAM-MAX-ABS-ERROR] {fold_err:.3e}")
    print(f"[REPARAM-ARGMAX-DISAGREE] {disagree:.3e}")
    print(f"[DEPLOY-PARAMS] total={measure_params(model) / 1e6:.3f} M")
    print(f"Recall={scores['recall']:.4f} | Precision={scores['precision']:.4f} | OA={scores['OA']:.4f} | "
          f"F1={scores['F1']:.4f} | IoU={scores['IoU']:.4f} | Kappa={scores['Kappa']:.4f}")
    print("=== END ===")


if __name__ == "__main__":
    main()
