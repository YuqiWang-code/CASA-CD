"""Standalone evaluation of a trained ChangeViT checkpoint.

Loads `best_F1=x.pth` (or an explicit --resume path) from --ckpt_dir, runs the test
set, and prints a `=== TEST RESULTS ===` block in the same format as train.py.
"""
import os
import sys
import argparse

import torch
import torch.backends.cudnn as cudnn

_MODELS_ROOT = os.path.dirname(os.path.abspath(__file__))
if _MODELS_ROOT not in sys.path:
    sys.path.insert(0, _MODELS_ROOT)

from model.trainer import Trainer
from model.str_fusion import STRFusionNet
from model.str_tass_fusion import STRTASSNet
from model.casa_tvim_str_net import CASATViMSTRNet
from model.metric_tool import ConfuseMatrixMeter
from model.utils import BCEDiceLoss

import dataset.dataset as myDataLoader
import dataset.Transforms as myTransforms


@torch.no_grad()
def val(args, val_loader, model):
    model.eval()

    salEvalVal = ConfuseMatrixMeter(n_class=2)
    epoch_loss = []

    for iter, batched_inputs in enumerate(val_loader):
        img, target = batched_inputs
        pre_img = img[:, 0:3]
        post_img = img[:, 3:6]

        if args.onGPU:
            pre_img = pre_img.cuda()
            target = target.cuda()
            post_img = post_img.cuda()

        output = model(pre_img.float(), post_img.float(), target.float())
        loss = BCEDiceLoss(output, target.float())

        pred = torch.where(output > 0.5, torch.ones_like(output), torch.zeros_like(output)).long()

        epoch_loss.append(loss.data.item())
        salEvalVal.update_cm(pr=pred.cpu().numpy(), gt=target.cpu().numpy())

    average_epoch_loss_val = sum(epoch_loss) / len(epoch_loss)
    scores = salEvalVal.get_scores()

    return average_epoch_loss_val, scores


def measure_params(model):
    return sum(p.numel() for p in model.parameters())


def measure_effective_params(model):
    """Params excluding unused ResNet head (layer4 + fc + avgpool never run in forward)."""
    total = measure_params(model)
    dead = 0
    for name, mod in model.named_modules():
        if name in ("encoder.resnet.layer4", "encoder.resnet.fc", "encoder.resnet.avgpool"):
            dead += sum(p.numel() for p in mod.parameters())
    return total - dead


def measure_flops(model, size=256):
    from fvcore.nn import flop_count

    model.eval()
    pre = torch.randn(1, 3, size, size).cuda()
    post = torch.randn(1, 3, size, size).cuda()
    label = torch.zeros(1, 1, size, size).cuda()
    with torch.no_grad():
        counts, unsupported = flop_count(model, (pre, post, label))
    return sum(counts.values()), len(unsupported)


def main():
    parser = argparse.ArgumentParser(description="ChangeViT standalone evaluation (baseline / CASAA / SAA)")
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--dataset_root', type=str, required=True)
    parser.add_argument('--test_list', type=str, required=True)
    parser.add_argument('--pretrained_weight_path', type=str, required=True)
    parser.add_argument('--ckpt_dir', type=str, required=True)
    parser.add_argument('--inWidth', type=int, default=256)
    parser.add_argument('--inHeight', type=int, default=256)
    parser.add_argument('--model_type', type=str, default='tiny')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--resume', default=None, help='explicit checkpoint; default: ckpt_dir/best_F1=*.pth')
    parser.add_argument('--resnet_pretrained', type=int, default=0,
                        help='load ImageNet ResNet18 weights (0: not needed for eval)')
    parser.add_argument('--onGPU', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--gpu_id', default=0, type=int)

    # 必须与训练一致（CASAA 零新增参数，state dict 与 baseline 兼容，
    # 若按 baseline 架构实例化会静默回退成普通 attention —— 这是 P0 正确性问题）
    parser.add_argument('--mode', type=str, default='baseline', choices=['baseline', 'saa', 'casaa'])
    parser.add_argument('--casaa_layers', type=str, default='8,9,10,11')
    parser.add_argument('--casaa_keep_ratio', type=float, default=0.25)
    parser.add_argument('--casaa_change_share', type=float, default=0.50)
    parser.add_argument('--casaa_router', type=str, default='change',
                        choices=['change', 'content', 'oracle', 'detail', 'detail_fused'])
    parser.add_argument('--vit_depth', type=int, default=12,
                        help='ViT depth (12 = original ChangeViT; 4 = Run4 prefix-4)')
    parser.add_argument('--detail_mode', type=str, default='resnet',
                        choices=['resnet', 'light', 'light48', 'light_bnrelu', 'psd', 'mobile_p3',
                                 'depth_pyramid', 'none_b4', 'opre'],
                        help='detail branch: resnet (original) | light | light48 | light_bnrelu (R4-2c) | psd (R4-2d) | mobile_p3 (Run5) | depth_pyramid (Run7) | none_b4 (Run8) | opre (Run9)')
    parser.add_argument('--head_mode', type=str, default='legacy',
                        choices=['legacy', 'sgdp', 'csdp', 'b4_spe', 'opre_spe'],
                        help='downstream head: legacy (original FI+decoder) | sgdp (Run5 R5-2) | csdp (Run7 R7-1) | b4_spe (Run8 R8-1) | opre_spe (Run9 R9-1)')
    parser.add_argument('--opre_gate', type=int, default=1,
                        help='Run9 O-PRE semantic gate: 1 = gated（主）；0 = NOGATE 消融')
    parser.add_argument('--mobile_pretrained_weight_path', type=str, default=None,
                        help='MobileNetV3-Small ImageNet weights (--detail_mode mobile_p3)')

    # STRFusion Run1 / Run11 TASS / CASA-TViM-STR
    parser.add_argument('--arch', type=str, default='changevit',
                        choices=['changevit', 'strfusion', 'str_tass', 'casa_tvim_str'],
                        help='model architecture: changevit | strfusion | str_tass | casa_tvim_str (must match training)')
    parser.add_argument('--str_dim', type=int, default=160,
                        help='decoder width D (must match training)')
    parser.add_argument('--str_rep_mode', type=str, default='full', choices=['plain', 'full'],
                        help='STRFusion rep mode (must match training)')
    parser.add_argument('--spatial_mode', type=str, default='token', choices=['token', 'tass'],
                        help='STRTASS spatial mode (must match training)')

    # CASA-TViM-STR（CAACP-SS2D 主线）
    parser.add_argument('--caacp', type=int, default=0)
    parser.add_argument('--rep_mode', type=str, default='full', choices=['plain', 'full'])
    parser.add_argument('--tinyvim_pretrained_weight_path', type=str, default=None,
                        help='TinyViM-S 1000e checkpoint (tinyvim_s_1000e.pth)')

    parser.add_argument('--mean', type=float, nargs=6,
                        default=[0.406, 0.456, 0.485, 0.406, 0.456, 0.485])
    parser.add_argument('--std', type=float, nargs=6,
                        default=[0.225, 0.224, 0.229, 0.225, 0.224, 0.229])
    args = parser.parse_args()

    args.casaa_layers = [int(s) for s in args.casaa_layers.split(',') if s.strip() != ''] \
        if args.casaa_layers else []
    if args.mode == 'saa':
        args.casaa_router = 'content'

    if args.onGPU:
        torch.cuda.set_device(args.gpu_id)
    torch.backends.cudnn.benchmark = True

    # pick best checkpoint if not explicitly given
    if args.resume is None:
        import glob as _glob
        cands = sorted(_glob.glob(os.path.join(args.ckpt_dir, "best_F1=*.pth")))
        if not cands:
            raise FileNotFoundError(f"no best_F1=*.pth found in {args.ckpt_dir}")
        args.resume = cands[-1]

    # T-R5-10：与训练侧 sidecar 架构参数核对，不一致直接拒绝（防止静默跑错结构）
    arch_path = os.path.join(args.ckpt_dir, "arch.json")
    if os.path.isfile(arch_path):
        import json as _json
        with open(arch_path, encoding="utf-8") as f:
            arch = _json.load(f)
        if args.arch == "casa_tvim_str":
            cli = {"arch": "casa_tvim_str", "backbone": "tinyvim_s_slim",
                   "caacp": args.caacp, "rep_mode": args.rep_mode,
                   "str_dim": args.str_dim}
        elif args.arch == "str_tass":
            cli = {"arch": "str_tass", "vit_depth": 4,
                   "str_dim": args.str_dim, "spatial_mode": args.spatial_mode}
        elif args.arch == "strfusion":
            cli = {"arch": "strfusion", "vit_depth": 4,
                   "str_dim": args.str_dim, "str_rep_mode": args.str_rep_mode}
        else:
            cli = {"vit_depth": args.vit_depth, "detail_mode": args.detail_mode,
                   "head_mode": args.head_mode, "mode": args.mode,
                   "opre_gate": args.opre_gate}
        if arch != cli:
            raise SystemExit(f"[ARCH-MISMATCH] ckpt arch={arch} vs cli={cli}; refusing to eval")
        print(f"[ARCH] eval arch matches ckpt sidecar: {arch}")

    if args.arch == "casa_tvim_str":
        # STR 折叠纪律测量协议（设计文档 §4.3 T2）：TF32 off + cudnn deterministic，
        # 与 train.py TEST 区块口径一致（TF32 on 会把折叠等价性读数放大到 1e-2 级）
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.deterministic = True
        model = CASATViMSTRNet(args.tinyvim_pretrained_weight_path, caacp=bool(args.caacp),
                               rep_mode=args.rep_mode, str_dim=args.str_dim).float()
        if args.onGPU:
            model = model.cuda()
        state_dict = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(state_dict)
        model.switch_to_deploy()
        model.eval()
        print(f"[RESUME] loaded {args.resume}; deploy graph folded")
    elif args.arch == "str_tass":
        model = STRTASSNet(args.pretrained_weight_path, dim=args.str_dim,
                           spatial_mode=args.spatial_mode).float()
        if args.onGPU:
            model = model.cuda()
        state_dict = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(state_dict)
        model.switch_to_deploy()
        model.eval()
        print(f"[RESUME] loaded {args.resume}; deploy graph folded")
    elif args.arch == "strfusion":
        model = STRFusionNet(args.pretrained_weight_path, dim=args.str_dim,
                             rep_mode=args.str_rep_mode).float()
        if args.onGPU:
            model = model.cuda()
        state_dict = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(state_dict)
        model.switch_to_deploy()
        model.eval()
        print(f"[RESUME] loaded {args.resume}; deploy graph folded")
    else:
        model = Trainer(args.model_type, pretrained_path=args.pretrained_weight_path,
                        resnet_pretrained=bool(args.resnet_pretrained),
                        mode=args.mode, casaa_layers=args.casaa_layers,
                        casaa_keep_ratio=args.casaa_keep_ratio,
                        casaa_change_share=args.casaa_change_share,
                        casaa_router=args.casaa_router,
                        vit_depth=args.vit_depth,
                        detail_mode=args.detail_mode,
                        head_mode=args.head_mode,
                        mobile_pretrained_weight_path=args.mobile_pretrained_weight_path,
                        opre_gate=args.opre_gate).float()
        if args.onGPU:
            model = model.cuda()

        state_dict = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(state_dict)
        print(f"[RESUME] loaded {args.resume}")

    val_transform = myTransforms.Compose([
        myTransforms.Normalize(mean=args.mean, std=args.std),
        myTransforms.Scale(args.inWidth, args.inHeight),
        myTransforms.ToTensor(),
    ])
    test_data = myDataLoader.Dataset(file_root=args.dataset_root, list_path=args.test_list, transform=val_transform)
    test_loader = torch.utils.data.DataLoader(
        test_data, shuffle=False, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=True)

    loss_test, score_test = val(args, test_loader, model)

    total_params = measure_params(model)
    try:
        flops, n_unsup = measure_flops(model, size=args.inWidth)
        flops_line = f"{flops:.4f} G   (input 2x3x{args.inWidth}x{args.inHeight}, unsupported_ops={n_unsup})"
    except Exception as e:  # fvcore 对 CASAA 动态 routing 覆盖不全时不阻塞 eval
        flops_line = f"measurement failed ({type(e).__name__}: {e})"

    print("=== TEST RESULTS ===")
    if args.arch == "casa_tvim_str":
        print("[MODEL] CASA-TViM-STR (TinyViM-S-Slim + CAACP-SS2D + TAR/DCR)")
        print("[BACKBONE] tinyvim_s_slim")
        print(f"[CAACP] {args.caacp}")
        print(f"[REP-MODE] {args.rep_mode}")
        print(f"[STR-DIM] {args.str_dim}")
        print("[DEPLOY-NUMERICS] tf32=off deterministic=on (STR T2 protocol)")
        print("[DEPLOY] folded deploy graph")
        print(f"[DEPLOY-PARAMS] total={total_params / 1e6:.3f} M "
              f"effective={measure_effective_params(model) / 1e6:.3f} M")
        print(f"[DEPLOY-FLOPS] {flops_line}")
        print(f"Recall={score_test['recall']:.4f} | Precision={score_test['precision']:.4f} | OA={score_test['OA']:.4f} | "
              f"F1={score_test['F1']:.4f} | IoU={score_test['IoU']:.4f} | Kappa={score_test['Kappa']:.4f}")
        print("=== END TEST RESULTS ===")
        return

    if args.arch in ("strfusion", "str_tass"):
        if args.arch == "str_tass":
            print("[MODEL] STRTASS (frozen ViT4 + TASS + TAR/DCR)")
            print(f"[TASS-MODE] {args.spatial_mode}")
        else:
            print("[MODEL] STRFusion (frozen ViT4 depth-as-scale + TAR/DCR)")
            print(f"[STR-REP-MODE] {args.str_rep_mode}")
        print(f"[ARCH] {args.arch}")
        print(f"[STR-DIM] {args.str_dim}")
        print("[DEPLOY] folded deploy graph")
        print(f"[DEPLOY-PARAMS] total={total_params / 1e6:.3f} M "
              f"effective={measure_effective_params(model) / 1e6:.3f} M")
        print(f"[DEPLOY-FLOPS] {flops_line}")
        print(f"Recall={score_test['recall']:.4f} | Precision={score_test['precision']:.4f} | OA={score_test['OA']:.4f} | "
              f"F1={score_test['F1']:.4f} | IoU={score_test['IoU']:.4f} | Kappa={score_test['Kappa']:.4f}")
        print("=== END TEST RESULTS ===")
        return

    print(f"[MODEL] ChangeViT-{args.model_type.upper()} {args.mode}")
    print(f"[MODE] {args.mode}")
    if args.mode != "baseline":
        print(f"[CASAA-LAYERS] {','.join(str(i) for i in args.casaa_layers)}")
        print(f"[CASAA-KEEP-RATIO] {args.casaa_keep_ratio}")
        print(f"[CASAA-CHANGE-SHARE] {args.casaa_change_share}")
        print(f"[CASAA-ROUTER] {args.casaa_router}")
        if args.casaa_router == "oracle":
            print("[DIAGNOSTIC-ONLY] oracle routing uses GT and is not deployable")
        if args.casaa_router == "detail_fused":
            print("[CASAA-DETAIL-SCALE] 1/8")
            print("[CASAA-DETAIL-FUSION] rank")
            print("[CASAA-VIT-WEIGHT] 0.5")
            print("[CASAA-DETAIL-WEIGHT] 0.5")
    print(f"[TOTAL-PARAMS] {total_params / 1e6:.3f} M")
    print(f"[EFFECTIVE-PARAMS] {measure_effective_params(model) / 1e6:.3f} M")
    print(f"[FLOPS] {flops_line}")
    print(f"Recall={score_test['recall']:.4f} | Precision={score_test['precision']:.4f} | OA={score_test['OA']:.4f} | "
          f"F1={score_test['F1']:.4f} | IoU={score_test['IoU']:.4f} | Kappa={score_test['Kappa']:.4f}")
    print("=== END TEST RESULTS ===")


if __name__ == "__main__":
    main()
