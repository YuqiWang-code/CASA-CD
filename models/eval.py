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
                        choices=['changevit', 'strfusion', 'str_tass', 'casa_tvim_str',
                                 'casa_vmamba_str'],
                        help='model architecture: changevit | strfusion | str_tass | casa_tvim_str | casa_vmamba_str (must match training)')
    parser.add_argument('--str_dim', type=int, default=160,
                        help='decoder width D (must match training)')
    parser.add_argument('--str_rep_mode', type=str, default='full', choices=['plain', 'full'],
                        help='STRFusion rep mode (must match training)')
    parser.add_argument('--spatial_mode', type=str, default='token', choices=['token', 'tass'],
                        help='STRTASS spatial mode (must match training)')

    # CASA-TViM-STR（CAACP-SS2D 主线）
    parser.add_argument('--caacp', type=int, default=0)
    parser.add_argument('--caacp_score_mode', type=str, default='rank', choices=['rank', 'cp'])
    parser.add_argument('--caacp_residual_mode', type=str, default='current', choices=['current', 'avg_anchor'])
    parser.add_argument('--frh', type=int, default=0)
    parser.add_argument('--fs_tar', type=int, default=0)
    parser.add_argument('--fine_tap', type=int, default=0, choices=[0, 1],
                        help='casa_tvim_str R4 FET1 FineEvidenceTap1x1 gate (must match ckpt sidecar exactly)')
    parser.add_argument('--hf_edge', type=int, default=0, choices=[0, 1],
                        help='casa_tvim_str R7 HFAE gate (must match ckpt sidecar exactly; fail-closed)')
    parser.add_argument('--hf_edge_form', type=str, default='none',
                        choices=['none', 'aux_only', 'aux_inject'],
                        help='R7 HFAE deploy form (must match ckpt sidecar exactly)')
    parser.add_argument('--hf_edge_taps', type=str, default='1_4_1_8',
                        help='R7 HFAE tap scales (must match ckpt sidecar exactly)')
    parser.add_argument('--hf_edge_lambda', type=float, default=0.2,
                        help='R7 HFAE aux loss weight (must match ckpt sidecar exactly)')
    parser.add_argument('--rep_mode', type=str, default='full', choices=['plain', 'full'])
    parser.add_argument('--tinyvim_pretrained_weight_path', type=str, default=None,
                        help='TinyViM-S 1000e checkpoint (tinyvim_s_1000e.pth)')

    # Run8：CASA-VMamba-STR（VMamba-Tiny 骨干 + CAACP-HQKV + TAR/DCR + HGSL）。
    # 与 train.py 同名同默认值、一一对应：这些开关共同决定结构，少给一个就重建出与训练
    # 不同的网络（arch.json 逐字段校验会在加载前拦住，这里只是把契约暴露出来）。
    parser.add_argument('--vmamba_pretrained_weight_path', type=str, default=None,
                        help='Run8 VMamba-Tiny official v2 checkpoint (vssm_tiny_0230_ckpt_epoch_262.pth); '
                             'its basename is compared against ckpt arch.json backbone_weight')
    parser.add_argument('--hqkv', type=int, default=1, choices=[0, 1],
                        help='Run8 CAACP-HQKV gate: 1 = hyperbolic dual-branch SAT-QKV aggregation; 0 = CTRL')
    parser.add_argument('--hqkv_geometry', type=str, default='hyperbolic',
                        choices=['hyperbolic', 'euclidean'],
                        help='Run8 geometry: hyperbolic (Lorentz geodesic) | euclidean (curvature counterfactual)')
    parser.add_argument('--hqkv_branch', type=str, default='both',
                        choices=['both', 'detail_only', 'global_only'],
                        help='Run8 branch specialisation: both | detail_only | global_only')
    parser.add_argument('--hqkv_inject_mode', type=str, default='dcr1',
                        choices=['none', 'dcr1', 'dcr1_16'],
                        help='Run8 injection point: dcr1 = DCR 1/4 input only | dcr1_16 = also 1/16 skip | none')
    parser.add_argument('--hqkv_rank', type=int, default=32,
                        help='Run8 Q/K/V channel rank (must match ckpt sidecar exactly)')
    parser.add_argument('--hqkv_lite_dim', type=int, default=192,
                        help='Run8 per-stage output width before summation (must match ckpt sidecar exactly)')
    parser.add_argument('--hqkv_detail_window', type=int, default=2,
                        help='Run8 detail-branch K/V token compression window')
    parser.add_argument('--hqkv_global_window', type=int, default=4,
                        help='Run8 global-branch K/V token compression window')
    parser.add_argument('--hqkv_attn_stages', type=str, default='1,2,3',
                        help='Run8 stages that run attention ("1,2,3"); parsed into a tuple[int]')
    parser.add_argument('--hqkv_r0', type=float, default=0.5,
                        help='Run8 hyperbolic radius floor r0 (PRE-REGISTERED and frozen)')
    parser.add_argument('--hqkv_a', type=float, default=0.5,
                        help='Run8 hyperbolic radius span a, r in [r0, r0+a] (PRE-REGISTERED and frozen)')
    parser.add_argument('--hgsl', type=int, default=1, choices=[0, 1],
                        help='Run8 HGSL (hyperbolic geometric supervision loss) gate')
    parser.add_argument('--hgsl_detach_score', type=int, default=1, choices=[0, 1],
                        help='HGSL gradient path: 1 = score detached (loss only reshapes score consumption)')
    parser.add_argument('--hgsl_lambda', type=float, default=0.2,
                        help='HGSL loss weight (PRE-REGISTERED and fixed at 0.2; searching it is forbidden)')

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
    if args.arch == "casa_tvim_str" and args.fine_tap and not os.path.isfile(arch_path):
        # R4 fail-closed：FET checkpoint 没有 sidecar 时无法核对 fine_tap 契约，直接拒绝
        raise SystemExit(f"[ARCH-MISSING] --fine_tap 1 but no arch.json in {args.ckpt_dir}; refusing to eval")
    if args.arch == "casa_tvim_str" and args.hf_edge and not os.path.isfile(arch_path):
        # R7 fail-closed：HFAE checkpoint 没有 sidecar 时无法核对 form/taps/lambda 契约，直接拒绝
        raise SystemExit(f"[ARCH-MISSING] --hf_edge 1 but no arch.json in {args.ckpt_dir}; refusing to eval")
    if args.arch == "casa_vmamba_str" and not os.path.isfile(arch_path):
        # Run8 fail-closed：casa_vmamba_str 的结构完全由 sidecar 定义（hqkv/hgsl/geometry/inject/
        # rank/attn_stages/...）。没有 arch.json 就没有任何东西能证明"重建结构与训练一致"，
        # 此时按 CLI 默认值硬跑等于用未经验证的结构出一个 F1 —— 直接拒绝，不给静默回退路径。
        raise SystemExit(f"[ARCH-MISSING] casa_vmamba_str requires arch.json in {args.ckpt_dir}; refusing to eval")
    if os.path.isfile(arch_path):
        import json as _json
        with open(arch_path, encoding="utf-8") as f:
            arch = _json.load(f)
        if args.arch == "casa_tvim_str":
            cli = {"arch": "casa_tvim_str", "backbone": "tinyvim_s_slim",
                   "caacp": args.caacp, "caacp_score_mode": args.caacp_score_mode,
                   "caacp_residual_mode": args.caacp_residual_mode,
                   "frh": args.frh, "fs_tar": args.fs_tar, "fine_tap": args.fine_tap,
                   "rep_mode": args.rep_mode, "str_dim": args.str_dim}
            # Run1/Run2 旧版 arch.json 无新字段 → 按当时默认补齐后比较
            arch_cmp = dict(arch)
            for k, v in (("caacp_score_mode", "rank"), ("caacp_residual_mode", "current"),
                         ("frh", 0), ("fs_tar", 0)):
                if k not in arch_cmp:
                    arch_cmp[k] = v
            if "fine_tap" not in arch_cmp:
                # Run1/Run2/Run3 sidecar 早于 R4：只有显式声明 --fine_tap 0 才允许按 0 补齐
                if args.fine_tap:
                    raise SystemExit(f"[ARCH-MISMATCH] ckpt has no fine_tap field but --fine_tap={args.fine_tap}; "
                                     "refusing to eval")
                arch_cmp["fine_tap"] = 0
            # ---- R7 HFAE（fail-closed）：sidecar 必须齐备且与 CLI 完全一致
            if "hf_edge" not in arch_cmp:
                if args.hf_edge:
                    raise SystemExit(f"[ARCH-MISMATCH] ckpt has no hf_edge field but --hf_edge={args.hf_edge}; "
                                     "refusing to eval")
                arch_cmp["hf_edge"] = 0
            else:
                if int(arch_cmp["hf_edge"]) != int(args.hf_edge):
                    raise SystemExit(f"[ARCH-MISMATCH] ckpt hf_edge={arch_cmp['hf_edge']} vs "
                                     f"cli={args.hf_edge}; refusing to eval")
                exp_form = args.hf_edge_form if args.hf_edge else "none"
                exp_taps = args.hf_edge_taps if args.hf_edge else "none"
                exp_lam = float(args.hf_edge_lambda) if args.hf_edge else 0.0
                for k, exp in (("hf_edge_form", exp_form), ("hf_edge_taps", exp_taps),
                               ("hf_edge_lambda", exp_lam)):
                    got = arch_cmp.get(k)
                    if k == "hf_edge_lambda":
                        ok = (got is not None) and abs(float(got) - exp) < 1e-12
                    else:
                        ok = (got == exp)
                    if not ok:
                        raise SystemExit(f"[ARCH-MISMATCH] ckpt {k}={got} vs expected {exp}; "
                                         "refusing to eval")
            for k in ("hf_edge", "hf_edge_form", "hf_edge_taps", "hf_edge_lambda",
                      "hf_edge_tag", "hf_edge_tap_specs"):
                arch_cmp.pop(k, None)
            # FET 派生字段：sidecar 有则必须一致（防手改口径）
            for k in ("fine_tap_form", "fine_tap_source", "fine_tap_fuse"):
                if k in arch:
                    expected = {"fine_tap_form": "pq_abs_1x1_gamma_v1",
                                "fine_tap_source": "norm0_1_4",
                                "fine_tap_fuse": "post_dcr_refine"}[k] if args.fine_tap else "none"
                    if arch[k] != expected:
                        raise SystemExit(f"[ARCH-MISMATCH] ckpt {k}={arch[k]} vs expected {expected}; refusing to eval")
                    arch_cmp.pop(k, None)
            # 身份字段（Run4 起）：不参与结构比较，但 FET checkpoint 必须齐备
            proto = arch_cmp.pop("protocol_version", None)
            src = arch_cmp.pop("source_code_sha256", None)
            if args.fine_tap:
                if proto != "run4_exact80k_v1":
                    raise SystemExit(f"[ARCH-MISMATCH] FET ckpt protocol_version={proto} "
                                     "!= run4_exact80k_v1; refusing to eval")
                if not src:
                    raise SystemExit("[ARCH-MISMATCH] FET ckpt sidecar lacks source_code_sha256; refusing to eval")
            if args.hf_edge:
                if proto != "run7_hf_edge_exact80k_v1":
                    raise SystemExit(f"[ARCH-MISMATCH] HFAE ckpt protocol_version={proto} "
                                     "!= run7_hf_edge_exact80k_v1; refusing to eval")
                if not src:
                    raise SystemExit("[ARCH-MISMATCH] HFAE ckpt sidecar lacks source_code_sha256; "
                                     "refusing to eval")
            if proto or src:
                print(f"[ARCH] ckpt protocol_version={proto} source_code_sha256={src}")
            if arch_cmp != cli:
                raise SystemExit(f"[ARCH-MISMATCH] ckpt arch={arch_cmp} vs cli={cli}; refusing to eval")
            print(f"[ARCH] eval arch matches ckpt sidecar: {arch_cmp}")
            arch_checked = True
        elif args.arch == "casa_vmamba_str":
            # Run8 fail-closed：期望字典的字段名与 train.py `_save_last()` 里 casa_vmamba_str 的
            # arch dict **完全一致**，逐字段与 arch.json 核对；缺字段或值不符一律 SystemExit。
            # 为什么不像上面 casa_tvim_str 那样"缺字段按当时默认补齐"：那是 Run1–Run3 旧 sidecar 的
            # 历史包袱；casa_vmamba_str 从第一版起就是严格契约 —— 少一个开关就无法证明重建结构与
            # 训练一致，静默补齐会把"手工拼出的错误实验臂"评成一个看起来正常的 F1。
            # backbone_lr_ratio 在 train.py / Run8 run_all.sh 里是预注册固定值 0.1，eval 侧不开放这个
            # 旋钮（它不是结构开关，而是优化器口径）：sidecar 里出现别的值即视为外来 checkpoint。
            _vmamba_bb_lr_ratio = 0.1
            # 协议身份硬编码（与上面 casa_tvim_str 的 run4_exact80k_v1 / run7_hf_edge_exact80k_v1 同风格）：
            # 只接受 Run8 exact-80K 协议产出的 sidecar，防止把别的实验线/手写 json 当 Run8 评。
            _vmamba_protocol = "run8_vmamba_hqkv_exact80k_v1"
            cli = {"arch": "casa_vmamba_str",
                   "backbone": "vmamba_tiny_v2",
                   "backbone_weight": os.path.basename(args.vmamba_pretrained_weight_path or ""),
                   "hqkv": int(args.hqkv),
                   "hqkv_form": ("hqkv_v1" if int(args.hqkv) else "none"),
                   "hqkv_geometry": (args.hqkv_geometry if int(args.hqkv) else "none"),
                   "hqkv_branch": (args.hqkv_branch if int(args.hqkv) else "none"),
                   "hqkv_inject_mode": (args.hqkv_inject_mode if int(args.hqkv) else "none"),
                   "hqkv_rank": (int(args.hqkv_rank) if int(args.hqkv) else 0),
                   "hqkv_lite_dim": (int(args.hqkv_lite_dim) if int(args.hqkv) else 0),
                   "hqkv_detail_window": (int(args.hqkv_detail_window) if int(args.hqkv) else 0),
                   "hqkv_global_window": (int(args.hqkv_global_window) if int(args.hqkv) else 0),
                   "hqkv_attn_stages": (args.hqkv_attn_stages if int(args.hqkv) else "none"),
                   "hqkv_r0": (float(args.hqkv_r0) if int(args.hqkv) else 0.0),
                   "hqkv_a": (float(args.hqkv_a) if int(args.hqkv) else 0.0),
                   "hgsl": int(args.hgsl),
                   "hgsl_detach_score": (int(args.hgsl_detach_score) if int(args.hgsl) else None),
                   "hgsl_lambda": (float(args.hgsl_lambda) if int(args.hgsl) else 0.0),
                   "fs_tar": int(args.fs_tar),
                   "rep_mode": args.rep_mode,
                   "str_dim": args.str_dim,
                   "backbone_lr_ratio": _vmamba_bb_lr_ratio}
            arch_cmp = dict(arch)
            # 身份字段（与 casa_tvim_str 块同策略）：不参与结构比较，但必须齐备 ——
            # 缺 protocol_version / source_code_sha256 说明 sidecar 不是本协议写出来的。
            proto = arch_cmp.pop("protocol_version", None)
            src = arch_cmp.pop("source_code_sha256", None)
            if proto != _vmamba_protocol:
                raise SystemExit(f"[ARCH-MISMATCH] ckpt protocol_version={proto} "
                                 f"!= {_vmamba_protocol}; refusing to eval")
            if not src:
                raise SystemExit("[ARCH-MISMATCH] ckpt sidecar lacks source_code_sha256; refusing to eval")
            # 逐字段严格比对：**缺字段也拒绝**（新 arch 不允许"缺字段就按默认补齐"的旧兼容逻辑）。
            # float 字段用 1e-12 容差比较（JSON 往返），其余按值相等比较（含 None == None）。
            _float_keys = ("hqkv_r0", "hqkv_a", "hgsl_lambda", "backbone_lr_ratio")
            for k, exp in cli.items():
                if k not in arch_cmp:
                    raise SystemExit(f"[ARCH-MISMATCH] ckpt sidecar missing field {k} "
                                     f"(expected {exp}); refusing to eval")
                got = arch_cmp[k]
                if k in _float_keys:
                    try:
                        ok = abs(float(got) - float(exp)) < 1e-12
                    except (TypeError, ValueError):
                        ok = False
                else:
                    ok = (got == exp)
                if not ok:
                    hint = ("; pass the same --vmamba_pretrained_weight_path as training"
                            if k == "backbone_weight" else "")
                    raise SystemExit(f"[ARCH-MISMATCH] ckpt {k}={got} vs cli={exp}{hint}; refusing to eval")
            # sidecar 的额外字段不参与比较（与 train.py run_manifest 的 compare_keys 约定一致）
            print(f"[ARCH] ckpt protocol_version={proto} source_code_sha256={src}")
            print(f"[ARCH] eval arch matches ckpt sidecar: {cli}")
            arch_checked = True
        elif args.arch == "str_tass":
            cli = {"arch": "str_tass", "vit_depth": 4,
                   "str_dim": args.str_dim, "spatial_mode": args.spatial_mode}
            arch_checked = False
        elif args.arch == "strfusion":
            cli = {"arch": "strfusion", "vit_depth": 4,
                   "str_dim": args.str_dim, "str_rep_mode": args.str_rep_mode}
            arch_checked = False
        else:
            cli = {"vit_depth": args.vit_depth, "detail_mode": args.detail_mode,
                   "head_mode": args.head_mode, "mode": args.mode,
                   "opre_gate": args.opre_gate}
            arch_checked = False
        if not arch_checked and arch != cli:
            raise SystemExit(f"[ARCH-MISMATCH] ckpt arch={arch} vs cli={cli}; refusing to eval")
        if not arch_checked:
            print(f"[ARCH] eval arch matches ckpt sidecar: {arch}")

    if args.arch == "casa_tvim_str":
        # STR 折叠纪律测量协议（设计文档 §4.3 T2）：TF32 off + cudnn deterministic，
        # 与 train.py TEST 区块口径一致（TF32 on 会把折叠等价性读数放大到 1e-2 级）
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.deterministic = True
        model = CASATViMSTRNet(args.tinyvim_pretrained_weight_path, caacp=bool(args.caacp),
                               rep_mode=args.rep_mode, str_dim=args.str_dim,
                               caacp_score_mode=args.caacp_score_mode, frh=bool(args.frh),
                               caacp_residual_mode=args.caacp_residual_mode,
                               fs_tar=bool(args.fs_tar), fine_tap=bool(args.fine_tap),
                               hf_edge=bool(args.hf_edge), hf_edge_form=args.hf_edge_form,
                               hf_edge_taps=args.hf_edge_taps,
                               hf_edge_lambda=args.hf_edge_lambda).float()
        if args.onGPU:
            model = model.cuda()
        state_dict = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(state_dict)
        model.switch_to_deploy()
        model.eval()
        print(f"[RESUME] loaded {args.resume}; deploy graph folded")
    elif args.arch == 'casa_vmamba_str':
        # Run8 主线：VMamba-Tiny 骨干 + CAACP-HQKV + TAR/DCR（+ HGSL）。
        # 为什么 import 放在分支内部：VMamba 依赖（selective_scan 扩展 / mamba 站点包）在独立
        # eval 环境里未必存在；放顶层会让 changevit / strfusion / str_tass / casa_tvim_str 的
        # eval 一起 import 失败，违反"新增 arch 不得影响既有分支"的纪律。
        # 数值模式与 train.py test_best_strfusion 完全一致（TF32 off + cudnn deterministic）：
        # deploy 折叠口径必须在同一数值模式下取得，否则 TF32 卷积舍入会污染等价性读数。
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.deterministic = True
        from model.casa_vmamba_net import CASAVMambaSTRNet
        # 逐项抄 train.py 的 CASAVMambaSTRNet(...) 调用（顺序/开关一一对应）：
        # hqkv_attn_stages 从逗号字符串转 tuple[int]，fs_tar 走 bool()，其余原样透传。
        model = CASAVMambaSTRNet(
            vmamba_pretrained_path=args.vmamba_pretrained_weight_path,
            hqkv=bool(args.hqkv), str_dim=args.str_dim,
            geometry=args.hqkv_geometry, branch=args.hqkv_branch,
            hgsl=bool(args.hgsl), hgsl_detach_score=bool(args.hgsl_detach_score),
            inject_mode=args.hqkv_inject_mode,
            hqkv_rank=args.hqkv_rank, hqkv_lite_dim=args.hqkv_lite_dim,
            hqkv_detail_window=args.hqkv_detail_window,
            hqkv_global_window=args.hqkv_global_window,
            hqkv_attn_stages=tuple(int(s) for s in args.hqkv_attn_stages.split(",") if s != ""),
            rep_mode=args.rep_mode, fs_tar=bool(args.fs_tar),
            hqkv_r0=args.hqkv_r0, hqkv_a=args.hqkv_a,
        ).float()
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
    if args.arch == "casa_vmamba_str":
        # 口径与 train.py is_vmamba TEST 区块一致：deploy 图 Params/FLOPs + test 集六指标。
        # 这里只做打印，total_params / flops_line / score_test 全部复用 main() 里既有的
        # measure_params / measure_flops / val()，不重写任何评估循环代码。
        import json as _json
        print("[MODEL] CASA-VMamba-STR (VMamba-Tiny + CAACP-HQKV + TAR/DCR)")
        print("[ARCH] casa_vmamba_str")
        print(f"[HQKV] {int(args.hqkv)} geometry={args.hqkv_geometry} branch={args.hqkv_branch} "
              f"inject={args.hqkv_inject_mode} rank={args.hqkv_rank}")
        print(f"[HGSL] {int(args.hgsl)} detach_score={int(args.hgsl_detach_score)} "
              f"lambda={args.hgsl_lambda}")
        print(f"[PARAM-BREAKDOWN] {_json.dumps(model.param_breakdown())}")
        print("[DEPLOY-NUMERICS] tf32=off deterministic=on (STR T2 protocol)")
        print("[DEPLOY] folded deploy graph")
        print(f"[DEPLOY-PARAMS] total={total_params / 1e6:.3f} M "
              f"effective={measure_effective_params(model) / 1e6:.3f} M")
        print(f"[DEPLOY-FLOPS] {flops_line}")
        print(f"Recall={score_test['recall']:.4f} | Precision={score_test['precision']:.4f} | OA={score_test['OA']:.4f} | "
              f"F1={score_test['F1']:.4f} | IoU={score_test['IoU']:.4f} | Kappa={score_test['Kappa']:.4f}")
        print("=== END TEST RESULTS ===")
        return

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
