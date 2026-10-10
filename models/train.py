"""Train ChangeViT (Tiny/Small) baseline for binary remote-sensing change detection.

Adapted from the official ChangeViT main.py
(https://github.com/zhuduowang/ChangeViT). Training protocol is kept faithful to the
official repo:

- Loss: BCE + Dice (BCEDiceLoss)
- Optimizer: Adam(lr=2e-4, betas=(0.9, 0.99), eps=1e-8, weight_decay=1e-4)
- Schedule: poly LR (power 0.9) over `max_steps`, with a 200-iteration linear warmup
- Budget: `max_steps` iterations (official 80000), batch 16, input 256x256
- Protocol: test set is used as the validation set; the best checkpoint is selected by
  test F1 (same as the official repo), and the final test is re-run on that checkpoint.

Adaptations for this repo (CASA-CD):
- A/B/label + list/*.txt dataset format (lab contract, label threshold gray>=128)
- Pretrained DeiT-Tiny / DINOv2 path passed via `--pretrained_weight_path`
- Logging: full config header + params/FLOPs at start; one line per epoch
  (loss + Recall/Precision/OA/F1/IoU/Kappa); final `=== TEST RESULTS ===` block
- Checkpoints: `last.pth` (resume, includes optimizer) + `best_F1=x.pth` (state dict)
"""
import os
import sys
import time
import argparse
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn

# Make the repo importable (train.py lives at models/ root).
_MODELS_ROOT = os.path.dirname(os.path.abspath(__file__))
if _MODELS_ROOT not in sys.path:
    sys.path.insert(0, _MODELS_ROOT)

from model.trainer import Trainer
from model.str_fusion import STRFusionNet
from model.str_tass_fusion import STRTASSNet
from model.metric_tool import ConfuseMatrixMeter
from model.utils import BCEDiceLoss, init_seed, adjust_learning_rate

import dataset.dataset as myDataLoader
import dataset.Transforms as myTransforms


# R4（方案 §5/§6.2）：训练协议版本号。写入 run_manifest.json / arch.json / last.pth，
# resume 时严格校验，防止不同协议（例如非 exact-80K）的 checkpoint 被当作同一条曲线。
PROTOCOL_VERSION = "run4_exact80k_v1"

# R4（方案 §5）：代码身份核验范围——主要模型/训练文件，SHA256 写入两个 sidecar。
SOURCE_IDENTITY_FILES = (
    "models/train.py",
    "models/eval.py",
    "models/model/casa_tvim_str_net.py",
    "models/model/str_fine_tap.py",
    "models/model/str_tar.py",
    "models/model/str_dcr.py",
    "models/model/str_reparam.py",
    "models/model/utils.py",
)


# -----------------------------------------------------------------------------
# Params / FLOPs measurement
# -----------------------------------------------------------------------------
def measure_params(model):
    return sum(p.numel() for p in model.parameters())


def measure_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def measure_effective_params(model):
    """Params excluding unused ResNet head (layer4 + fc + avgpool never run in forward).

    The paper's 11.68M for ChangeViT-T counts effective params only; the released
    code keeps the full ResNet18 module (~8.9M dead params).
    """
    total = measure_params(model)
    dead = 0
    for name, mod in model.named_modules():
        if name in ("encoder.resnet.layer4", "encoder.resnet.fc", "encoder.resnet.avgpool"):
            dead += sum(p.numel() for p in mod.parameters())
    return total - dead


def measure_flops(model, size=256):
    """Return flops (G) for a (1,3,size,size) bi-temporal pair (fvcore).

    附带一张全零 label（仅 router='oracle' 需要，其余模式忽略）。
    """
    from fvcore.nn import flop_count

    model.eval()
    pre = torch.randn(1, 3, size, size).cuda()
    post = torch.randn(1, 3, size, size).cuda()
    label = torch.zeros(1, 1, size, size).cuda()
    with torch.no_grad():
        counts, unsupported = flop_count(model, (pre, post, label))
    total = sum(counts.values())
    return total, len(unsupported)


def fmt_flops(total):
    # fvcore.flop_count already returns counts in giga-flops.
    return f"{total:.4f}"


def fmt_params(n):
    return f"{n / 1e6:.3f}"


# -----------------------------------------------------------------------------
# Train / val steps (faithful to the official ChangeViT main.py)
# -----------------------------------------------------------------------------
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

        pre_img_var = pre_img.float()
        post_img_var = post_img.float()
        target_var = target.float()

        output = model(pre_img_var, post_img_var, target_var)
        loss = BCEDiceLoss(output, target_var)

        pred = torch.where(output > 0.5, torch.ones_like(output), torch.zeros_like(output)).long()

        epoch_loss.append(loss.data.item())
        salEvalVal.update_cm(pr=pred.cpu().numpy(), gt=target_var.cpu().numpy())

    average_epoch_loss_val = sum(epoch_loss) / len(epoch_loss)
    scores = salEvalVal.get_scores()

    return average_epoch_loss_val, scores


def train_epoch(args, train_loader, model, optimizer, epoch, max_batches, cur_iter=0,
                lr_factor=1., adapt_log_fn=None, remaining_steps=None):
    """一个 epoch 的优化循环。

    remaining_steps（R4 exact-80K 修正）：若给出，则在完成该数量的 optimizer update 后
    **提前 break**，返回实际执行步数；poly 的分母固定为 args.max_steps（见 model/utils.py）。
    """
    model.train()

    salEvalVal = ConfuseMatrixMeter(n_class=2)
    epoch_loss = []
    executed_steps = 0
    lr = None

    for iter, batched_inputs in enumerate(train_loader):
        if remaining_steps is not None and executed_steps >= remaining_steps:
            break
        img, target = batched_inputs
        pre_img = img[:, 0:3]
        post_img = img[:, 3:6]

        # adjust the learning rate (poly + warmup, official schedule)
        lr = adjust_learning_rate(args, optimizer, epoch, iter + cur_iter, max_batches, lr_factor=lr_factor)

        # casa_tvim_str：per-group lr_scale 验证（P0）——在关键 iter 打印分组 lr
        if getattr(args, "arch", "changevit") == "casa_tvim_str" and (iter + cur_iter) in (0, 99, 198, 199, 200, 201):
            lrs = [f"{pg.get('name', '?')}={pg['lr']:.3e}" for pg in optimizer.param_groups]
            print(f"[LR-GROUPS] iter={iter + cur_iter} " + " ".join(lrs), flush=True)

        # BACKBONE-ADAPT 审计（调研文档 §3.3，每 1000 iter）
        if adapt_log_fn is not None and (iter + cur_iter) % 1000 == 0 and iter > 0:
            adapt_log_fn(model, iter + cur_iter)

        if args.onGPU:
            pre_img = pre_img.cuda()
            target = target.cuda()
            post_img = post_img.cuda()

        pre_img_var = pre_img.float()
        post_img_var = post_img.float()
        target_var = target.float()

        output = model(pre_img_var, post_img_var, target_var)
        loss = BCEDiceLoss(output, target_var)

        pred = torch.where(output > 0.5, torch.ones_like(output), torch.zeros_like(output)).long()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        executed_steps += 1

        epoch_loss.append(loss.data.item())

        # computing F-measure on GPU, then accumulate the confusion matrix
        salEvalVal.update_cm(pr=pred.detach().cpu().numpy(), gt=target_var.detach().cpu().numpy())

    average_epoch_loss_train = sum(epoch_loss) / max(1, len(epoch_loss))
    scores = salEvalVal.get_scores()

    return average_epoch_loss_train, scores, lr, executed_steps


def source_code_identity():
    """R4 §5：主要模型/训练文件的 SHA256（写入 arch.json 与 run_manifest.json，eval 侧逐字段核对）。"""
    import hashlib as _hashlib
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = {}
    for rel in SOURCE_IDENTITY_FILES:
        p = os.path.join(repo_root, rel.replace("/", os.sep))
        if not os.path.isfile(p):
            out[rel] = "MISSING"
            continue
        h = _hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        out[rel] = h.hexdigest()[:16]
    return out


# -----------------------------------------------------------------------------
# Trainer
# -----------------------------------------------------------------------------
class ChangeViTTrainer(object):
    def __init__(self, args, log):
        self.args = args
        self.log = log

        init_seed(args.seed)
        torch.backends.cudnn.benchmark = True

        # CASA-TViM-STR（CAACP-SS2D 主线，加性分支，不改动 changevit/strfusion/str_tass 路径）。
        # 从头训练纪律 + backbone/new 双学习率组（scheduler 经 utils lr_scale 生效）。
        if getattr(args, "arch", "changevit") == "casa_tvim_str":
            assert args.resume is None, \
                "from-scratch discipline: --resume is forbidden at launch (crash recovery uses own last.pth)"
            assert args.seed == 16, f"from-scratch discipline: seed must be 16, got {args.seed}"
            assert os.path.basename(args.tinyvim_pretrained_weight_path) == "tinyvim_s_1000e.pth", \
                f"pretrained path must be tinyvim_s_1000e.pth, got {args.tinyvim_pretrained_weight_path}"
            from model.casa_tvim_str_net import CASATViMSTRNet
            self.model = CASATViMSTRNet(
                args.tinyvim_pretrained_weight_path, caacp=bool(args.caacp),
                rep_mode=args.rep_mode, str_dim=args.str_dim,
                caacp_score_mode=args.caacp_score_mode, frh=bool(args.frh),
                caacp_residual_mode=args.caacp_residual_mode, fs_tar=bool(args.fs_tar),
                fine_tap=bool(args.fine_tap),
            ).float()
            if args.onGPU:
                self.model = self.model.cuda()
            backbone_params = [p for n, p in self.model.named_parameters() if n.startswith("encoder.")]
            new_params = [p for n, p in self.model.named_parameters() if not n.startswith("encoder.")]
            self.optimizer = torch.optim.Adam(
                [{"params": backbone_params, "lr": args.lr * args.backbone_lr_ratio,
                  "lr_scale": args.backbone_lr_ratio, "name": "backbone"},
                 {"params": new_params, "lr": args.lr,
                  "lr_scale": 1.0, "name": "new"}],
                args.lr, (0.9, 0.99), eps=1e-08, weight_decay=1e-4)
            # BACKBONE-ADAPT 审计参考（调研文档 §3.3 / Run2 §12.1A）：
            # 只快照 exact-loaded 预训练键（新模块/新 norm 不计入 drift 口径）
            loaded_keys = (self.model.encoder.load_stats() or {}).get("loaded_keys")
            self._backbone_ref = {k: v.detach().cpu().clone()
                                  for k, v in self.model.encoder.state_dict().items()
                                  if loaded_keys is None or k in loaded_keys}
            self.start_epoch = 0
            self.best_f1 = -1.0
            self.best_epoch = -1
            self.cur_iter = 0
            resume_path = args.resume
            if resume_path is None:
                auto = os.path.join(args.ckpt_dir, "last.pth")
                if os.path.isfile(auto):
                    resume_path = auto
            os.makedirs(args.ckpt_dir, exist_ok=True)
            self._write_casa_tvim_manifest(resume_path)   # 严格校验（resume 时逐字段比对）
            if resume_path is not None and os.path.isfile(resume_path):
                self._load_resume(resume_path)
            self.log(f"[LR-GROUPS] backbone={args.lr * args.backbone_lr_ratio:.2e} (x{args.backbone_lr_ratio}) "
                     f"new={args.lr:.2e} (x1.0)")
            self.log(f"[CONFIG] caacp={int(self.args.caacp)} score_mode={self.args.caacp_score_mode} "
                     f"residual_mode={self.args.caacp_residual_mode} frh={int(self.args.frh)} "
                     f"fs_tar={int(self.args.fs_tar)} rep={self.args.rep_mode} str_dim={self.args.str_dim}")
            self.log(f"[CONFIG] fine_tap={int(self.args.fine_tap)} arch={self.args.arch} "
                     f"protocol={PROTOCOL_VERSION} exact_max_steps={int(args.exact_max_steps)} "
                     f"max_steps={args.max_steps} batch={args.batch_size} seed={args.seed}")
            if int(self.args.fine_tap):
                _rep = self.model.fine_evidence_tap.param_report()
                self.log(f"[FET] form={self.model.fine_tap_form} source={self.model.fine_tap_source} "
                         f"fuse={self.model.fine_tap_fuse} gamma_init={self.model.fet_gamma():.1e} "
                         f"train_new_params={_rep['train_new']} (pq={_rep['pq']} diff={_rep['diff']} gamma={_rep['gamma']})")
            ls = self.model.encoder.load_stats() or {}
            self.log(f"[PRETRAIN-LOAD] retained={ls.get('retained')}/{ls.get('pretrained_keys')} "
                     f"worst_diff={ls.get('worst_diff'):.3e} new_modules={len(ls.get('missing_new', []))} "
                     f"dropped={len(ls.get('dropped_intentional', []))}")
            return

        # STRFusion / STRTASS 加性分支（不改动 changevit 路径）。
        # 从头训练纪律（Run11）：禁止 --resume、seed 必须 16、只允许官方 DeiT-Tiny pth。
        if getattr(args, "arch", "changevit") in ("strfusion", "str_tass"):
            assert args.resume is None, \
                "from-scratch discipline: --resume is forbidden at launch (crash recovery uses own last.pth)"
            assert args.seed == 16, f"from-scratch discipline: seed must be 16, got {args.seed}"
            assert os.path.basename(args.pretrained_weight_path) == "deit_tiny_patch16_224-a1311bcf.pth", \
                f"pretrained path must be the official DeiT-Tiny pth, got {args.pretrained_weight_path}"
            if args.arch == "strfusion":
                self.model = STRFusionNet(
                    args.pretrained_weight_path, dim=args.str_dim, rep_mode=args.str_rep_mode,
                ).float()
            else:
                self.model = STRTASSNet(
                    args.pretrained_weight_path, dim=args.str_dim, spatial_mode=args.spatial_mode,
                ).float()
            if args.onGPU:
                self.model = self.model.cuda()
            enc_trainable = sum(p.numel() for p in self.model.encoder.parameters() if p.requires_grad)
            assert enc_trainable == 0, "STRFusion/STRTASS ViT must stay frozen (enforced by the model)"
            self.optimizer = torch.optim.Adam(
                [p for p in self.model.parameters() if p.requires_grad],
                args.lr, (0.9, 0.99), eps=1e-08, weight_decay=1e-4)
            self.start_epoch = 0
            self.best_f1 = -1.0
            self.best_epoch = -1
            self.cur_iter = 0
            resume_path = args.resume
            if resume_path is None:
                auto = os.path.join(args.ckpt_dir, "last.pth")
                if os.path.isfile(auto):
                    resume_path = auto
            if resume_path is not None and os.path.isfile(resume_path):
                self._load_resume(resume_path)
            os.makedirs(args.ckpt_dir, exist_ok=True)
            self._vit_ref_hash = self._vit_hash()
            self._write_run_manifest(resume_path)
            return

        self.model = Trainer(
            args.model_type,
            pretrained_path=args.pretrained_weight_path,
            resnet_pretrained=args.resnet_pretrained,
            mode=args.mode,
            casaa_layers=args.casaa_layers,
            casaa_keep_ratio=args.casaa_keep_ratio,
            casaa_change_share=args.casaa_change_share,
            casaa_router=args.casaa_router,
            vit_depth=args.vit_depth,
            detail_mode=args.detail_mode,
            head_mode=args.head_mode,
            mobile_pretrained_weight_path=args.mobile_pretrained_weight_path,
            opre_gate=args.opre_gate,
        ).float()
        if args.onGPU:
            self.model = self.model.cuda()

        # Run2 起：ViT 崩溃保护（见 train_scripts/CASAA/Run2/README.md）。
        # 实测 ChangeViT 官方协议（统一 lr=2e-4）在 LEVIR 上 ~1600 steps 内把 ViT
        # 训练成精确零权重（pos_embed 4.43→0.0007、patch_embed 8.0→0.03），
        # 一旦归零即无梯度（吸收态），checkpoint 的 ViT 全零且不可恢复。
        # --freeze_vit 1：冻结 ViT（Run2 机制实验用）；--vit_lr_ratio 可调小 ViT lr。
        if args.freeze_vit:
            for n, p in self.model.named_parameters():
                if n.startswith("encoder.vit"):
                    p.requires_grad_(False)
        vit_params = [p for n, p in self.model.named_parameters() if n.startswith("encoder.vit")]
        rest_params = [p for n, p in self.model.named_parameters() if not n.startswith("encoder.vit")]
        self.optimizer = torch.optim.Adam(
            [{"params": rest_params, "lr": args.lr},
             {"params": vit_params, "lr": args.lr * args.vit_lr_ratio}],
            args.lr, (0.9, 0.99), eps=1e-08, weight_decay=1e-4)

        self.start_epoch = 0
        self.best_f1 = -1.0
        self.best_epoch = -1
        self.cur_iter = 0

        # auto-resume from ckpt_dir/last.pth (or an explicit --resume tar)
        resume_path = args.resume
        if resume_path is None:
            auto = os.path.join(args.ckpt_dir, "last.pth")
            if os.path.isfile(auto):
                resume_path = auto
        if resume_path is not None and os.path.isfile(resume_path):
            self._load_resume(resume_path)

        os.makedirs(args.ckpt_dir, exist_ok=True)

    def _load_resume(self, path):
        self.log(f"[RESUME] loading {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(checkpoint["state_dict"])
        if "optimizer" in checkpoint and checkpoint["optimizer"] is not None:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.start_epoch = checkpoint.get("epoch", 0)
        self.best_f1 = checkpoint.get("best_f1", -1.0)
        self.best_epoch = checkpoint.get("best_epoch", -1)
        if "actual_steps" in checkpoint:
            # R4 §6.2：exact-80K 下 epoch 数可能被截断，实际步数才是唯一进度真值
            self.cur_iter = int(checkpoint["actual_steps"])
            iters = checkpoint.get("iters_per_epoch", 0)
            self.log(f"[RESUME] actual_steps={self.cur_iter} (epoch={self.start_epoch}, iters/epoch={iters})")
        else:
            self.cur_iter = self.start_epoch * checkpoint.get("iters_per_epoch", 0)
        if checkpoint.get("protocol_version") not in (None, PROTOCOL_VERSION):
            raise SystemExit(
                f"[PROTOCOL-MISMATCH] ckpt protocol_version={checkpoint.get('protocol_version')} "
                f"!= {PROTOCOL_VERSION}; aborting")
        rng = checkpoint.get("rng_state")
        if rng is not None:
            torch.set_rng_state(rng["torch"])
            np.random.set_state(rng["numpy"])
            random.setstate(rng["python"])
            if rng.get("cuda") is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(rng["cuda"])
            self.log("[RESUME] rng_state restored (torch/numpy/python/cuda)")

    def _write_casa_tvim_manifest(self, resume_path):
        """casa_tvim_str run_manifest.json：fresh 写入 / resume 逐字段严格校验。"""
        import json as _json
        import hashlib as _hashlib
        path = os.path.join(self.args.ckpt_dir, "run_manifest.json")
        pretrain_sha = ""
        if os.path.isfile(self.args.tinyvim_pretrained_weight_path):
            h = _hashlib.sha256()
            with open(self.args.tinyvim_pretrained_weight_path, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            pretrain_sha = h.hexdigest()[:16]
        manifest = {
            "arch": "casa_tvim_str",
            "backbone": "tinyvim_s_slim",
            "backbone_weight_sha256": pretrain_sha,
            "feature_taps": ["stage1_1_4_48", "stage2_1_8_64", "stage3_1_16_168", "stage4_1_32_224"],
            "caacp": int(self.args.caacp),
            "caacp_score_mode": self.args.caacp_score_mode,
            "caacp_residual_mode": self.args.caacp_residual_mode,
            "frh": int(self.args.frh),
            "fs_tar": int(self.args.fs_tar),
            "fine_tap": int(self.args.fine_tap),
            "fine_tap_form": ("pq_abs_1x1_gamma_v1" if int(self.args.fine_tap) else "none"),
            "fine_tap_source": ("norm0_1_4" if int(self.args.fine_tap) else "none"),
            "fine_tap_fuse": ("post_dcr_refine" if int(self.args.fine_tap) else "none"),
            "rep_mode": self.args.rep_mode,
            "str_dim": self.args.str_dim,
            "backbone_lr_ratio": self.args.backbone_lr_ratio,
            "data_contract": "legacy_6ch_reverse_v1",
            "seed": self.args.seed,
            "max_steps": self.args.max_steps,
            "batch_size": self.args.batch_size,
            "dataset": self.args.dataset,
            "protocol_version": PROTOCOL_VERSION,
            "exact_max_steps": int(getattr(self.args, "exact_max_steps", 0)),
            "source_code_sha256": source_code_identity(),
            "resume_source": resume_path,
        }
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as f:
                old = _json.load(f)
            for k in ("arch", "backbone", "backbone_weight_sha256", "caacp",
                      "caacp_score_mode", "caacp_residual_mode", "frh", "fs_tar",
                      "fine_tap", "fine_tap_form", "fine_tap_source", "fine_tap_fuse",
                      "rep_mode", "str_dim", "backbone_lr_ratio",
                      "data_contract", "seed", "max_steps", "batch_size", "dataset",
                      "protocol_version", "exact_max_steps"):
                if old.get(k) != manifest[k]:
                    raise SystemExit(f"[MANIFEST-MISMATCH] field {k}: old={old.get(k)} new={manifest[k]}; aborting")
            # 代码身份：resume 时允许训练脚本本身被格式化（不阻断），但必须留痕
            if old.get("source_code_sha256") != manifest["source_code_sha256"]:
                diff = [k for k in manifest["source_code_sha256"]
                        if old.get("source_code_sha256", {}).get(k) != manifest["source_code_sha256"][k]]
                self.log(f"[MANIFEST] WARNING source_code_sha256 differs on resume: {diff}")
            self.log(f"[MANIFEST] validated existing run_manifest.json (resume_source={resume_path})")
        else:
            with open(path, "w", encoding="utf-8") as f:
                _json.dump(manifest, f, indent=2)
            self.log(f"[MANIFEST] run_manifest.json written (resume_source={resume_path})")

    def _vit_hash(self):
        """冻结 ViT 权重的确定性哈希（TEST 区块 [VIT-CHECKSUM] 对照基准）。"""
        import hashlib
        h = hashlib.sha256()
        for k in sorted(self.model.encoder.state_dict().keys()):
            v = self.model.encoder.state_dict()[k]
            h.update(k.encode())
            h.update(v.detach().cpu().numpy().tobytes())
        return h.hexdigest()[:16]

    def _backbone_adapt_log(self, model, it):
        """BACKBONE-ADAPT 审计（调研文档 §3.3 / Run2 §12.1A）：梯度范数 + 相对预训练 L2 drift。

        只统计从 checkpoint 原位继承（exact-loaded）的 key，按 stage 分组输出；
        新建 norm / CAACP β 等新模块不计入（避免把"新模块从零训练"误读成"预训练漂移"）。
        """
        with torch.no_grad():
            g_sq = 0.0
            p_sq = 0.0
            named = dict(model.named_parameters())
            sd = model.encoder.state_dict()
            ref = self._backbone_ref
            # stage 分组：patch_embed->stem；network.0/2/4->stage0/1/2；network.6->stage3(retained)
            groups = {}
            for k in ref:
                if k.startswith("patch_embed."):
                    g = "stem"
                elif k.startswith("network.0."):
                    g = "stage0"
                elif k.startswith("network.2."):
                    g = "stage1"
                elif k.startswith("network.4."):
                    g = "stage2"
                elif k.startswith("network.6."):
                    g = "stage3-retained"
                else:
                    g = "other"
                v = sd[k]
                r = ref[k].to(v.device)
                d2 = ((v - r).float().norm().item()) ** 2
                n2 = (r.float().norm().item()) ** 2
                groups.setdefault(g, [0.0, 0.0]).__setitem__(0, groups[g][0] + d2)
                groups[g][1] += n2
                full_k = "encoder." + k
                if full_k in named:
                    p = named[full_k]
                    p_sq += p.norm().item() ** 2
                    if p.grad is not None:
                        g_sq += p.grad.norm().item() ** 2
            per = " ".join(
                f"{g}={(d / (n + 1e-8)) ** 0.5:.3e}" for g, (d, n) in sorted(groups.items()))
            drift_sq = sum(d for d, _ in groups.values())
            ref_norm_sq = sum(n for _, n in groups.values())
            rel = (drift_sq / (ref_norm_sq + 1e-8)) ** 0.5
            self.log(f"[BACKBONE-ADAPT] iter={it} grad_norm={g_sq ** 0.5:.3e} "
                     f"param_norm={p_sq ** 0.5:.3e} rel_L2_from_pretrain={rel:.4e}")
            self.log(f"[PRETRAIN-DRIFT] {per} (exact-loaded keys only)")

    def _backbone_adapt_final(self, model):
        """best checkpoint 的 backbone 适配终值（TEST 区块记录，exact-loaded keys 口径）。"""
        with torch.no_grad():
            drift_sq = 0.0
            ref_norm_sq = 0.0
            sd = model.encoder.state_dict()
            for k, r in self._backbone_ref.items():
                v = sd[k].to(r.device)
                drift_sq += ((v - r).float().norm().item()) ** 2
                ref_norm_sq += (r.float().norm().item()) ** 2
        return (drift_sq / (ref_norm_sq + 1e-8)) ** 0.5

    def _write_run_manifest(self, resume_path):
        """run_manifest.json（方案 §14）：首次启动写一次，记录从头训练契约。"""
        import json as _json
        import hashlib as _hashlib
        path = os.path.join(self.args.ckpt_dir, "run_manifest.json")
        if os.path.isfile(path):
            return
        pretrain_sha = ""
        if os.path.isfile(self.args.pretrained_weight_path):
            h = _hashlib.sha256()
            with open(self.args.pretrained_weight_path, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            pretrain_sha = h.hexdigest()[:16]
        manifest = {
            "arch": self.args.arch,
            "spatial_mode": getattr(self.args, "spatial_mode", "token"),
            "str_rep_mode": getattr(self.args, "str_rep_mode", "full"),
            "str_dim": getattr(self.args, "str_dim", 160),
            "seed": self.args.seed,
            "pretrain_sha256": pretrain_sha,
            "dataset": self.args.dataset,
            "max_steps": self.args.max_steps,
            "optimizer": "Adam(2e-4, 0.9/0.99, wd=1e-4)",
            "lr": self.args.lr,
            "loss": "BCE+Dice",
            "checkpoint_dir": self.args.ckpt_dir,
            "resume_source": resume_path,
        }
        with open(path, "w", encoding="utf-8") as f:
            _json.dump(manifest, f, indent=2)
        self.log(f"[MANIFEST] run_manifest.json written (resume_source={resume_path})")

    def _save_last(self, epoch, iters_per_epoch):
        arch_name = getattr(self.args, "arch", "changevit")
        if arch_name == "casa_tvim_str":
            arch = {
                "arch": "casa_tvim_str",
                "backbone": "tinyvim_s_slim",
                "caacp": int(self.args.caacp),
                "caacp_score_mode": self.args.caacp_score_mode,
                "caacp_residual_mode": self.args.caacp_residual_mode,
                "frh": int(self.args.frh),
                "fs_tar": int(self.args.fs_tar),
                "fine_tap": int(self.args.fine_tap),
                "fine_tap_form": ("pq_abs_1x1_gamma_v1" if int(self.args.fine_tap) else "none"),
                "fine_tap_source": ("norm0_1_4" if int(self.args.fine_tap) else "none"),
                "fine_tap_fuse": ("post_dcr_refine" if int(self.args.fine_tap) else "none"),
                "rep_mode": self.args.rep_mode,
                "str_dim": self.args.str_dim,
                "protocol_version": PROTOCOL_VERSION,
                "source_code_sha256": source_code_identity(),
            }
        elif arch_name == "str_tass":
            arch = {
                "arch": "str_tass",
                "vit_depth": 4,
                "str_dim": self.args.str_dim,
                "spatial_mode": self.args.spatial_mode,
            }
        elif arch_name == "strfusion":
            arch = {
                "arch": "strfusion",
                "vit_depth": 4,
                "str_dim": self.args.str_dim,
                "str_rep_mode": self.args.str_rep_mode,
            }
        else:
            # T-R5-10：checkpoint 内记录架构参数，eval 侧核对一致
            arch = {
                "vit_depth": self.args.vit_depth,
                "detail_mode": self.args.detail_mode,
                "head_mode": self.args.head_mode,
                "mode": self.args.mode,
                "opre_gate": self.args.opre_gate,
            }
        ckpt = {
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": epoch + 1,
            "best_f1": self.best_f1,
            "best_epoch": self.best_epoch,
            "iters_per_epoch": iters_per_epoch,
            "arch": arch,
        }
        if arch_name == "casa_tvim_str" and bool(getattr(self.args, "exact_max_steps", 0)):
            # R4 §6.2：exact-80K 契约下的进度与 RNG 状态，resume 必须逐位一致
            ckpt["actual_steps"] = int(self.cur_iter)
            ckpt["protocol_version"] = PROTOCOL_VERSION
            ckpt["rng_state"] = {
                "torch": torch.get_rng_state(),
                "numpy": np.random.get_state(),
                "python": random.getstate(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            }
        torch.save(ckpt, os.path.join(self.args.ckpt_dir, "last.pth"))
        # 独立 eval 用 sidecar（best_F1=*.pth 是裸 state_dict，不能混入非权重 key）
        arch_path = os.path.join(self.args.ckpt_dir, "arch.json")
        if not os.path.isfile(arch_path):
            import json as _json
            with open(arch_path, "w", encoding="utf-8") as f:
                _json.dump(ckpt["arch"], f)

    def _make_loader(self, list_path, batch_size, shuffle):
        if shuffle:
            transform = myTransforms.Compose([
                myTransforms.Normalize(mean=self.args.mean, std=self.args.std),
                myTransforms.Scale(self.args.inWidth, self.args.inHeight),
                myTransforms.RandomCropResize(int(7. / 224. * self.args.inWidth)),
                myTransforms.RandomFlip(),
                myTransforms.RandomExchange(),
                myTransforms.ToTensor(),
            ])
        else:
            transform = myTransforms.Compose([
                myTransforms.Normalize(mean=self.args.mean, std=self.args.std),
                myTransforms.Scale(self.args.inWidth, self.args.inHeight),
                myTransforms.ToTensor(),
            ])
        dataset = myDataLoader.Dataset(file_root=self.args.dataset_root, list_path=list_path, transform=transform)
        return torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle,
            num_workers=self.args.num_workers, pin_memory=True, drop_last=False)

    def train(self):
        train_loader = self._make_loader(self.args.train_list, self.args.batch_size, True)
        test_loader = self._make_loader(self.args.test_list, self.args.test_batch_size, False)

        max_batches = len(train_loader)
        self.args.max_epochs = int(np.ceil(self.args.max_steps / max_batches))
        self.log(f"[MAX-STEPS] {self.args.max_steps}  [train_iters/epoch] {max_batches}  [max_epochs] {self.args.max_epochs}")
        self.log("EPOCH | Loss | Recall | Precision | OA | F1 | IoU | Kappa | t(s)")

        cur_iter = self.cur_iter
        exact = bool(getattr(self.args, "exact_max_steps", 0))
        if exact:
            self.log(f"[EXACT-STEPS] protocol={PROTOCOL_VERSION} enabled: "
                     f"stop exactly at optimizer step {self.args.max_steps} (cur_iter={cur_iter})")
        for epoch in range(self.start_epoch, self.args.max_epochs):
            t0 = time.time()
            remaining = (self.args.max_steps - cur_iter) if exact else None
            if remaining is not None and remaining <= 0:
                self.log(f"[EXACT-STEPS] budget already reached ({cur_iter} >= {self.args.max_steps}), stopping")
                break
            loss_tr, score_tr, lr, executed = train_epoch(
                self.args, train_loader, self.model, self.optimizer, epoch, max_batches, cur_iter,
                adapt_log_fn=(self._backbone_adapt_log
                              if getattr(self.args, "arch", "changevit") == "casa_tvim_str" else None),
                remaining_steps=remaining)
            cur_iter += (executed if exact else max_batches)
            self.cur_iter = cur_iter
            if exact:
                self.log(f"[ACTUAL-OPT-STEPS] epoch={epoch} executed={executed} "
                         f"cumulative={cur_iter}/{self.args.max_steps}")

            torch.cuda.empty_cache()

            # official protocol: skip evaluation after the first epoch
            if epoch == 0:
                self._save_last(epoch, max_batches)
                if exact and cur_iter >= self.args.max_steps:
                    break
                continue

            loss_val, score_val = val(self.args, test_loader, self.model)
            torch.cuda.empty_cache()

            line = (f"Epoch {epoch}/{self.args.max_epochs} | Loss={loss_val:.4f} | "
                    f"Recall={score_val['recall']:.4f} | Precision={score_val['precision']:.4f} | "
                    f"OA={score_val['OA']:.4f} | F1={score_val['F1']:.4f} | "
                    f"IoU={score_val['IoU']:.4f} | Kappa={score_val['Kappa']:.4f} | "
                    f"t={time.time() - t0:.0f}s")
            self.log(line)

            # P2（Run2 §18）：best 元数据先更新并保存，再写 last.pth ——
            # 使 last.pth 内 best_f1/best_epoch 与磁盘 best 文件一致（崩溃恢复元数据不落后一拍）。
            if score_val["F1"] > self.best_f1:
                prev_best = os.path.join(self.args.ckpt_dir, f"best_F1={self.best_f1:.4f}.pth")
                if self.best_f1 >= 0 and os.path.isfile(prev_best):
                    os.remove(prev_best)
                self.best_f1 = score_val["F1"]
                self.best_epoch = epoch
                torch.save(self.model.state_dict(),
                           os.path.join(self.args.ckpt_dir, f"best_F1={self.best_f1:.4f}.pth"))

            self._save_last(epoch, max_batches)
            if exact and cur_iter >= self.args.max_steps:
                self.log(f"[EXACT-STEPS] reached {cur_iter}/{self.args.max_steps} optimizer steps, stopping")
                break

        self.log(f"[BEST] F1={self.best_f1:.4f} at epoch {self.best_epoch}")

    def test_best(self):
        best_path = os.path.join(self.args.ckpt_dir, f"best_F1={self.best_f1:.4f}.pth")
        if not os.path.isfile(best_path):
            self.log("[TEST] best checkpoint not found, skipping final test")
            return

        self.model.load_state_dict(torch.load(best_path, map_location="cpu", weights_only=False))
        self.model = self.model.cuda()

        if getattr(self.args, "arch", "changevit") in ("strfusion", "str_tass", "casa_tvim_str"):
            self.test_best_strfusion(best_path)
            return

        total_params = measure_params(self.model)
        trainable = measure_trainable_params(self.model)
        effective_params = measure_effective_params(self.model)
        try:
            flops, n_unsup = measure_flops(self.model, size=self.args.inWidth)
            flops_line = (f"[FLOPS] {fmt_flops(flops)} G   "
                          f"(input 2x3x{self.args.inWidth}x{self.args.inHeight}, unsupported_ops={n_unsup})")
        except Exception as e:  # fvcore 对 CASAA 动态 routing 覆盖不全时不阻塞训练
            flops_line = f"[FLOPS] measurement failed ({type(e).__name__}: {e})"

        test_loader = self._make_loader(self.args.test_list, self.args.test_batch_size, False)
        _, score_test = val(self.args, test_loader, self.model)

        self.log("=== TEST RESULTS ===")
        self.log(f"[MODEL] ChangeViT-{self.args.model_type.upper()} {self.args.mode}")
        self.log(f"[MODE] {self.args.mode}")
        if self.args.mode != "baseline":
            self.log(f"[CASAA-LAYERS] {','.join(str(i) for i in self.args.casaa_layers)}")
            self.log(f"[CASAA-KEEP-RATIO] {self.args.casaa_keep_ratio}")
            self.log(f"[CASAA-CHANGE-SHARE] {self.args.casaa_change_share}")
            self.log(f"[CASAA-ROUTER] {self.args.casaa_router}")
            if self.args.casaa_router == "oracle":
                self.log("[DIAGNOSTIC-ONLY] oracle routing uses GT and is not deployable")
            if self.args.casaa_router == "detail_fused":
                self.log("[CASAA-DETAIL-SCALE] 1/8")
                self.log("[CASAA-DETAIL-FUSION] rank")
                self.log("[CASAA-VIT-WEIGHT] 0.5")
                self.log("[CASAA-DETAIL-WEIGHT] 0.5")
        self.log(f"[FREEZE-VIT] {int(self.args.freeze_vit)}")
        self.log(f"[VIT-LR-RATIO] {self.args.vit_lr_ratio}")
        self.log(f"[VIT-DEPTH] {self.args.vit_depth}")
        self.log(f"[DETAIL-MODE] {self.args.detail_mode}")
        self.log(f"[HEAD-MODE] {self.args.head_mode}")
        if self.args.detail_mode == 'opre':
            self.log("[OPRE-KERNEL] 16")
            self.log("[OPRE-STRIDE] 8")
            self.log("[OPRE-PADDING] reflect4")
            self.log("[OPRE-WEIGHT-SHARED] 1")
            self.log(f"[OPRE-GATE] {int(self.args.opre_gate)}")
        self.log(f"[TOTAL-PARAMS] {fmt_params(total_params)} M")
        self.log(f"[EFFECTIVE-PARAMS] {fmt_params(effective_params)} M")
        self.log(f"[TRAINABLE-PARAMS] {fmt_params(trainable)} M")
        self.log(flops_line)
        self.log(f"Recall={score_test['recall']:.4f} | Precision={score_test['precision']:.4f} | OA={score_test['OA']:.4f} | "
                 f"F1={score_test['F1']:.4f} | IoU={score_test['IoU']:.4f} | Kappa={score_test['Kappa']:.4f}")
        self.log(f"[BEST-F1] {self.best_f1:.4f} (epoch {self.best_epoch})")
        self.log("=== END TEST RESULTS ===")

    def test_best_strfusion(self, best_path):
        """STRFusion 正式测试：折叠为 deploy 图后评估（STR 部署纪律）。

        1) 固定随机 batch 上记录 train-graph vs deploy-graph 的 max_abs_error 与
           0.5 阈值二值化 disagreement（硬要求 = 0，STR argmax=0 的单通道类比）；
        2) 报告 deploy 图 Params / FLOPs（部署预算硬门槛 <3M 由预算审计裁决，此处记录）；
        3) 在 deploy 图上跑完整 test 集，写 TEST RESULTS 区块。
        """
        self.model.eval()
        # STR 折叠等价性测量协议（设计文档 §4.3 T2）：TF32 off + cudnn deterministic。
        # GPU TF32 卷积舍入经 BN 因子放大可达 1e-2 级（CDD 实测 1.5e-2），污染折叠
        # 误差读数（TF32 off 后 1.6e-5）；train/deploy 两图必须在同一数值模式下比较。
        prev_tf32 = torch.backends.cudnn.allow_tf32
        prev_det = torch.backends.cudnn.deterministic
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(16)
        pre_fix = torch.randn(8, 3, self.args.inWidth, self.args.inHeight).cuda()
        post_fix = torch.randn(8, 3, self.args.inWidth, self.args.inHeight).cuda()
        # 主线文档 T6：随机输入与真实数据 batch 都测。真实数据概率极化，二值 mask 应逐位一致。
        real_loader = self._make_loader(self.args.test_list, 16, False)
        img_r, _ = next(iter(real_loader))
        pre_r = img_r[:, 0:3].cuda()
        post_r = img_r[:, 3:6].cuda()
        with torch.no_grad():
            y_train = self.model(pre_fix, post_fix)
            y_train_real = self.model(pre_r.float(), post_r.float())
        # R4：FET 折叠前状态（gamma / 等效卷积核），必须在 switch_to_deploy 之前读取
        fet_gamma_final = None
        fet_l1_final = None
        if getattr(self.args, "arch", "changevit") == "casa_tvim_str" and int(self.args.fine_tap):
            fet_gamma_final = self.model.fet_gamma()
            w_fused, b_fused = self.model.fine_evidence_tap.get_equivalent_kernel_bias()
            fet_l1_final = (float(w_fused.abs().mean().item()), float(b_fused.abs().mean().item()))
        self.model.switch_to_deploy()
        self.model.eval()
        with torch.no_grad():
            y_deploy = self.model(pre_fix, post_fix)
            y_deploy_real = self.model(pre_r.float(), post_r.float())
        fold_err = (y_train - y_deploy).abs().max().item()
        disagree = ((y_train > 0.5) != (y_deploy > 0.5)).float().mean().item()
        fold_err_real = (y_train_real - y_deploy_real).abs().max().item()
        disagree_real = ((y_train_real > 0.5) != (y_deploy_real > 0.5)).float().mean().item()

        deploy_total = measure_params(self.model)
        deploy_trainable = measure_trainable_params(self.model)
        deploy_effective = measure_effective_params(self.model)
        try:
            flops, n_unsup = measure_flops(self.model, size=self.args.inWidth)
            flops_line = (f"[DEPLOY-FLOPS] {fmt_flops(flops)} G   "
                          f"(input 2x3x{self.args.inWidth}x{self.args.inHeight}, unsupported_ops={n_unsup})")
        except Exception as e:
            flops_line = f"[DEPLOY-FLOPS] measurement failed ({type(e).__name__}: {e})"

        test_loader = self._make_loader(self.args.test_list, self.args.test_batch_size, False)
        _, score_test = val(self.args, test_loader, self.model)

        is_tass = getattr(self.args, "arch", "changevit") == "str_tass"
        is_casa = getattr(self.args, "arch", "changevit") == "casa_tvim_str"
        self.log("=== TEST RESULTS ===")
        if is_casa:
            self.log("[MODEL] CASA-TViM-STR (TinyViM-S-Slim + CAACP-SS2D + TAR/DCR)")
            self.log("[ARCH] casa_tvim_str")
            self.log(f"[CAACP] {int(self.args.caacp)}")
            self.log(f"[CAACP-SCORE-MODE] {self.args.caacp_score_mode}")
            self.log(f"[CAACP-RESIDUAL-MODE] {self.args.caacp_residual_mode}")
            self.log(f"[FRH] {int(self.args.frh)}")
            self.log(f"[FS-TAR] {int(self.args.fs_tar)}")
            self.log(f"[FINE-TAP] {int(self.args.fine_tap)}")
            self.log(f"[REP-MODE] {self.args.rep_mode}")
            self.log(f"[BACKBONE-LR-RATIO] {self.args.backbone_lr_ratio}")
            self.log(f"[DATA-CONTRACT] legacy_6ch_reverse_v1")
            self.log("[DEPLOY-NUMERICS] tf32=off deterministic=on (STR T2 protocol)")
            import hashlib as _hashlib
            if os.path.isfile(self.args.tinyvim_pretrained_weight_path):
                h = _hashlib.sha256()
                with open(self.args.tinyvim_pretrained_weight_path, "rb") as f:
                    for chunk in iter(lambda: f.read(1 << 20), b""):
                        h.update(chunk)
                self.log(f"[PRETRAIN-SHA256] {h.hexdigest()[:16]}")
            load_stats = getattr(self.model.encoder, "_load_stats", None)
            if load_stats:
                self.log(f"[PRETRAIN-LOAD] retained={load_stats['retained']}/{load_stats['pretrained_keys']} "
                         f"worst_diff={load_stats['worst_diff']:.3e}")
            if self.model.encoder.caacp_op is not None:
                self.log(f"[CAACP-BETA] {self.model.caacp_beta():.4e}")
                self.log(f"[CAACP-SCORE-DELTA] {self.model.caacp_score_delta():.4e}")
                ent = self.model.caacp_weight_entropy()
                am = self.model.caacp_abs_mean()
                if ent is not None:
                    self.log(f"[CAACP-WEIGHT-ENTROPY] {ent:.4e} (uniform cell = ln4 ~ 1.386)")
                if am is not None:
                    self.log(f"[CAACP-ABS-MEAN] {am:.4e}")
                if self.args.caacp_score_mode == "cp":
                    self.log("[CAACP-BUDGET] dense=16x16 context=8x8 (2x2 cell CP pooling w~1+s*r, no TopK)")
                else:
                    self.log("[CAACP-BUDGET] dense=16x16 context=8x8 (2x2 cell change-aware pooling, no TopK)")
            if self.args.frh:
                self.log("[FRH-DEPLOY] head folded to single Conv2d(96->1,k3) (+768 params); base/dw/pw/gamma branches deleted")
            if int(self.args.fine_tap):
                self.log(f"[FET-FORM] {self.model.fine_tap_form} (source={self.model.fine_tap_source} fuse={self.model.fine_tap_fuse})")
                if fet_gamma_final is not None:
                    self.log(f"[FET-GAMMA] {fet_gamma_final:.6e} (zero-init residual gate; train-graph value before folding)")
                    self.log(f"[FET-FUSED-WEIGHT-ABS-MEAN] {fet_l1_final[0]:.6e} "
                             f"[FET-FUSED-BIAS-ABS-MEAN] {fet_l1_final[1]:.6e}")
                self.log("[FET-DEPLOY] tap folded to single Conv2d(144->96,k1,no bias) added after DCR refine; "
                         "pq/diff/gamma branches deleted")
                if self.model.fet_deploy_conv_shape() is not None:
                    self.log(f"[FET-FOLD] conv1x1 {self.model.fet_deploy_conv_shape()} "
                             f"params={self.model.fine_evidence_tap.param_report()['fused']} "
                             f"fold_count={self.model.fine_evidence_tap.fold_count}")
            self.log(f"[BACKBONE-ADAPT-FINAL] rel_L2_from_pretrain={self._backbone_adapt_final(self.model):.4e}")
        elif is_tass:
            self.log(f"[MODEL] {('STRTASS (frozen ViT4 + TASS + TAR/DCR)' if is_tass else 'STRFusion (frozen ViT4 depth-as-scale + TAR/DCR)')}")
            self.log(f"[ARCH] {getattr(self.args, 'arch', 'changevit')}")
            self.log(f"[TASS-MODE] {self.args.spatial_mode}")
            if self.model.alpha is not None:
                a = [f"{v.item():.4f}" for v in self.model.alpha]
                self.log(f"[ALPHA] {','.join(a)}")
            tass_params = self.model.tass.param_count() if self.model.tass is not None else 0
            self.log(f"[TASS-PARAMS] {tass_params:,}")
            cur_hash = self._vit_hash()
            self.log(f"[VIT-CHECKSUM] ref={self._vit_ref_hash} now={cur_hash} "
                     f"unchanged={cur_hash == self._vit_ref_hash}")
        else:
            self.log(f"[MODEL] STRFusion (frozen ViT4 depth-as-scale + TAR/DCR)")
            self.log(f"[ARCH] {getattr(self.args, 'arch', 'changevit')}")
            self.log(f"[STR-REP-MODE] {self.args.str_rep_mode}")
        self.log(f"[STR-DIM] {self.args.str_dim}")
        if is_casa:
            self.log("[BACKBONE] TinyViM-S-Slim (Stage4 = Localx3 + final TViM; 1000e EMA weights)")
            self.log("[FEATURE-TAPS] stage1_1_4(48) stage2_1_8(64) stage3_1_16(168) stage4_1_32(224)")
        else:
            self.log(f"[STR-FREEZE-VIT] 1")
            self.log(f"[STR-VIT-DEPTH] 4")
            self.log(f"[STR-SCALES] B1->64x64 B2->32x32 B3->16x16 B4->8x8")
        self.log(f"[REPARAM-MAX-ABS-ERROR] {fold_err:.3e} (train-graph vs deploy-graph, fixed batch)")
        self.log(f"[REPARAM-ARGMAX-DISAGREE] {disagree:.3e} (0.5-binarization; hard gate: must be 0)")
        self.log(f"[REPARAM-REAL-MAX-ABS-ERROR] {fold_err_real:.3e} (train vs deploy, fixed real-data batch 16)")
        self.log(f"[REPARAM-REAL-ARGMAX-DISAGREE] {disagree_real:.3e} (0.5-binarization; operational gate: must be 0)")
        self.log(f"[DEPLOY-PARAMS] total={fmt_params(deploy_total)} M "
                 f"effective={fmt_params(deploy_effective)} M trainable={fmt_params(deploy_trainable)} M")
        self.log(flops_line)
        self.log(f"Recall={score_test['recall']:.4f} | Precision={score_test['precision']:.4f} | OA={score_test['OA']:.4f} | "
                 f"F1={score_test['F1']:.4f} | IoU={score_test['IoU']:.4f} | Kappa={score_test['Kappa']:.4f}")
        self.log(f"[BEST-F1] {self.best_f1:.4f} (epoch {self.best_epoch})")
        if getattr(self.args, "arch", "changevit") == "casa_tvim_str":
            # R4（设计文档 §6.2 / T10）：exact-80K 的步数证据必须落在**最后一个完整 TEST 区块内**，
            # 训练期的 per-epoch [ACTUAL-OPT-STEPS] 行在区块之外、不作为正式结果来源。
            self.log(f"[ACTUAL-OPT-STEPS] {int(self.cur_iter)}")
            self.log(f"[PROTOCOL-VERSION] {PROTOCOL_VERSION}")
        self.log("=== END TEST RESULTS ===")
        torch.backends.cudnn.allow_tf32 = prev_tf32
        torch.backends.cudnn.deterministic = prev_det


# -----------------------------------------------------------------------------
# Config header
# -----------------------------------------------------------------------------
def write_header(args, log):
    # 从 ckpt_dir 推导实验路径（如 .../CASA-CD/CASAA/Run2/... -> train_scripts/CASAA/Run2）
    script_tag = None
    if "CASA-CD/" in args.ckpt_dir:
        tail = args.ckpt_dir.split("CASA-CD/")[-1].split("/")
        if len(tail) >= 2:
            script_tag = f"train_scripts/{tail[0]}/{tail[1]}"
    if script_tag is None:
        script_tag = "train_scripts/baseline/Run1" if args.mode == "baseline" else "train_scripts/CASAA/Run1"
    log("=" * 72)
    log(f"ChangeViT-{args.model_type.upper()}  |  mode={args.mode}  |  {script_tag}")
    log("=" * 72)
    log("[CONFIG]")
    for k, v in vars(args).items():
        log(f"  {k}: {v}")
    if args.mode != "baseline":
        log("[MODE] " + args.mode)
        log("[CASAA-LAYERS] " + ",".join(str(i) for i in args.casaa_layers))
        log(f"[CASAA-KEEP-RATIO] {args.casaa_keep_ratio}")
        log(f"[CASAA-CHANGE-SHARE] {args.casaa_change_share}")
        log(f"[CASAA-ROUTER] {args.casaa_router}")
        if args.casaa_router == "oracle":
            log("[DIAGNOSTIC-ONLY] oracle routing uses GT and is not deployable")
        if args.casaa_router == "detail_fused":
            log("[CASAA-DETAIL-SCALE] 1/8")
            log("[CASAA-DETAIL-FUSION] rank")
            log("[CASAA-VIT-WEIGHT] 0.5")
            log("[CASAA-DETAIL-WEIGHT] 0.5")
    log(f"[FREEZE-VIT] {int(args.freeze_vit)}")
    log(f"[VIT-LR-RATIO] {args.vit_lr_ratio}")
    log("=" * 72)


def main():
    parser = argparse.ArgumentParser(description="ChangeViT training (baseline / CASAA / SAA)")
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--dataset_root', type=str, required=True)
    parser.add_argument('--train_list', type=str, required=True)
    parser.add_argument('--test_list', type=str, required=True)
    parser.add_argument('--pretrained_weight_path', type=str, required=True)
    parser.add_argument('--ckpt_dir', type=str, required=True)
    parser.add_argument('--inWidth', type=int, default=256, help='Width of RGB image')
    parser.add_argument('--inHeight', type=int, default=256, help='Height of RGB image')
    parser.add_argument('--max_steps', type=int, default=80000, help='Max. number of iterations')
    parser.add_argument('--num_workers', type=int, default=4, help='No. of parallel threads')
    parser.add_argument('--model_type', type=str, default='tiny', help='select vit model type | tiny | small')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--test_batch_size', type=int, default=16, help='Test batch size')
    parser.add_argument('--step_loss', type=int, default=100, help='Decrease learning rate after how many epochs (step mode)')
    parser.add_argument('--lr', type=float, default=2e-4, help='Initial learning rate')
    parser.add_argument('--lr_mode', default='poly', help='Learning rate policy, step or poly')
    parser.add_argument('--seed', type=int, default=16, help='initialization seed number')
    parser.add_argument('--resume', default=None, help='Use this checkpoint to continue training (auto: ckpt_dir/last.pth)')
    parser.add_argument('--resnet_pretrained', type=int, default=1, help='load ImageNet ResNet18 weights for detail branch')
    parser.add_argument('--onGPU', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='Run on CPU or GPU. If TRUE, then GPU.')
    parser.add_argument('--gpu_id', default=0, type=int, help='GPU id number')

    # CASAA (创新主线一)：Full Query + 变化感知压缩 K/V
    parser.add_argument('--mode', type=str, default='baseline', choices=['baseline', 'saa', 'casaa'],
                        help='baseline | saa (A1 SAA-style content control) | casaa (A2 change-aware)')
    parser.add_argument('--casaa_layers', type=str, default='8,9,10,11',
                        help='0-based ViT block indices replaced by CASAA (comma separated)')
    parser.add_argument('--casaa_keep_ratio', type=float, default=0.25,
                        help='K/N compressed K/V token budget')
    parser.add_argument('--casaa_change_share', type=float, default=0.50,
                        help='share of K kept as change tokens (router=change)')
    parser.add_argument('--casaa_router', type=str, default='change',
                        choices=['change', 'content', 'oracle', 'detail', 'detail_fused'],
                        help='change (CASAA cosine) | content (SAA-style control) | oracle (GT diagnostic, not deployable) | detail (A4-D: detail 1/8 score) | detail_fused (A4: rank-fused ViT+detail)')

    # ViT 崩溃保护（Run2）
    parser.add_argument('--freeze_vit', type=int, default=0,
                        help='freeze the ViT backbone (1 = freeze; prevents ViT collapse on real data)')
    parser.add_argument('--vit_lr_ratio', type=float, default=1.0,
                        help='ViT learning-rate multiplier relative to base lr (1.0 = official protocol)')
    # Run4 主线二：ViT 深度（12 = ChangeViT 原版；4 = TinyViT4 prefix）
    parser.add_argument('--vit_depth', type=int, default=12,
                        help='ViT depth (12 = original ChangeViT; 4 = Run4 prefix-4)')
    # Run4 主线二：detail branch（resnet = 原 ResNet18；light = LightDetail 32/64/128；
    # light48 = 预注册容量 fallback 48/96/160；light_bnrelu = R4-2c adapter align（仅 audit 情况 A）；
    # psd = R4-2d PSD-Detail 0.078M；mobile_p3 = Run5 MobileNetV3-Small features 0-3；
    # depth_pyramid = Run7 CSDP（无 detail，B1/B2 token 金字塔）；
    # none_b4 = Run8 B4-SPE（无 detail，只输出 B4 token）；
    # opre = Run9 B4-OPRE（无 detail，输出 O-PRE 32×32 + B4 16×16））
    parser.add_argument('--detail_mode', type=str, default='resnet',
                        choices=['resnet', 'light', 'light48', 'light_bnrelu', 'psd', 'mobile_p3',
                                 'depth_pyramid', 'none_b4', 'opre'],
                        help='detail branch: resnet (original) | light | light48 | light_bnrelu (R4-2c) | psd (R4-2d) | mobile_p3 (Run5) | depth_pyramid (Run7) | none_b4 (Run8) | opre (Run9)')
    # Run5/Run7/Run8/Run9：head（legacy = 原 FeatureInjector+Decoder；sgdp = Semantic-Guided
    # Difference Pyramid；csdp = Run7 Cross-Depth Symmetric Difference；b4_spe = Run8 B4-SPE；
    # opre_spe = Run9 B4-OPRE 语义门控）
    parser.add_argument('--head_mode', type=str, default='legacy',
                        choices=['legacy', 'sgdp', 'csdp', 'b4_spe', 'opre_spe'],
                        help='downstream head: legacy (original FI+decoder) | sgdp (Run5 R5-2) | csdp (Run7 R7-1) | b4_spe (Run8 R8-1) | opre_spe (Run9 R9-1)')
    parser.add_argument('--opre_gate', type=int, default=1,
                        help='Run9 O-PRE semantic gate: 1 = gated residual（主模型）；0 = NOGATE 消融')
    parser.add_argument('--mobile_pretrained_weight_path', type=str, default=None,
                        help='MobileNetV3-Small ImageNet weights (--detail_mode mobile_p3)')

    # STRFusion Run1 / Run11 TASS / CASA-TViM-STR（融合主线；changevit 路径不受影响）
    parser.add_argument('--arch', type=str, default='changevit',
                        choices=['changevit', 'strfusion', 'str_tass', 'casa_tvim_str'],
                        help='model architecture: changevit | strfusion | str_tass | casa_tvim_str (TinyViM-S-Slim + CAACP-SS2D + TAR/DCR)')
    parser.add_argument('--caacp', type=int, default=0,
                        help='casa_tvim_str CAACP-SS2D gate: 1 = change-aware context pooling; 0 = official uniform pool')
    parser.add_argument('--caacp_score_mode', type=str, default='rank', choices=['rank', 'cp'],
                        help='casa_tvim_str CAACP score formula: rank (Run1 w~eps+rank) | cp (Run2 w~1+s*r confidence-preserving)')
    parser.add_argument('--caacp_residual_mode', type=str, default='current', choices=['current', 'avg_anchor'],
                        help='casa_tvim_str CAACP residual anchor: current (res=x-Up(c)) | avg_anchor (Run3 E4 RA-CAACP: res=x-Up(c_avg), protects dense high-freq residual)')
    parser.add_argument('--tinyvim_pretrained_weight_path', type=str, default=None,
                        help='TinyViM-S 1000e checkpoint (tinyvim_s_1000e.pth; model_ema weights)')
    parser.add_argument('--rep_mode', type=str, default='full', choices=['plain', 'full'],
                        help='casa_tvim_str STR rep mode: plain | full')
    parser.add_argument('--backbone_lr_ratio', type=float, default=0.1,
                        help='casa_tvim_str backbone lr ratio (fixed 0.1; scheduler honors per-group lr_scale)')
    parser.add_argument('--str_dim', type=int, default=160,
                        help='decoder width D (CASA-TViM-STR fixed 96; budget dial only, no F1 sweep)')
    parser.add_argument('--frh', type=int, default=0,
                        help='casa_tvim_str FRH fine head: 1 = STRFineHead (128^2 reparam head, +768 deploy params); 0 = plain 1x1 head')
    parser.add_argument('--fs_tar', type=int, default=0,
                        help='casa_tvim_str FS-TAR (Run3 E5): 1 = stage1 TemporalRepFine3x3 (spatial-temporal signed-diff 3x3 at 1/4, +73,728 deploy params); 0 = stage1 TemporalRep1x1')
    parser.add_argument('--fine_tap', type=int, default=0, choices=[0, 1],
                        help='casa_tvim_str R4 FET1 (FineEvidenceTap1x1): 1 = 1/4 norm0 1x1 fine-evidence tap added after DCR refine '
                             '(+13,920 deploy params, zero-init gamma -> epoch-0 bitwise identity); 0 = CTRL M1 (R4CTRL arm)')
    parser.add_argument('--exact_max_steps', type=int, default=0, choices=[0, 1],
                        help='R4 exact-80K protocol: 1 = stop at exactly args.max_steps optimizer updates and fix the poly '
                             'denominator to args.max_steps; 0 = legacy (epoch-granular, max_epochs=ceil(max_steps/iters)). '
                             'Run4 scripts pass 1 explicitly so the historical epoch-granular behaviour stays reproducible.')
    parser.add_argument('--str_rep_mode', type=str, default='full', choices=['plain', 'full'],
                        help='STRFusion rep mode: plain (C0, no aux) | full (M1, TAR+DCR aux)')
    parser.add_argument('--spatial_mode', type=str, default='token', choices=['token', 'tass'],
                        help='STRTASS spatial mode: token (C0) | tass (M1, TASS residual stem)')

    # official ChangeViT normalization (BGR order, ImageNet stats x2)
    parser.add_argument('--mean', type=float, nargs=6,
                        default=[0.406, 0.456, 0.485, 0.406, 0.456, 0.485])
    parser.add_argument('--std', type=float, nargs=6,
                        default=[0.225, 0.224, 0.229, 0.225, 0.224, 0.229])
    args = parser.parse_args()

    args.resnet_pretrained = bool(args.resnet_pretrained)
    args.freeze_vit = bool(args.freeze_vit)
    args.casaa_layers = [int(s) for s in args.casaa_layers.split(',') if s.strip() != ''] \
        if args.casaa_layers else []
    # A1 (mode=saa) 固定为 content router，与 --casaa_router 无关
    if args.mode == 'saa':
        args.casaa_router = 'content'
    if args.onGPU:
        torch.cuda.set_device(args.gpu_id)

    def log(msg):
        print(msg, flush=True)

    write_header(args, log)

    trainer = ChangeViTTrainer(args, log)

    log("[TOTAL-PARAMS] " + fmt_params(measure_params(trainer.model)) + " M")
    log("[EFFECTIVE-PARAMS] " + fmt_params(measure_effective_params(trainer.model)) + " M")
    log("[TRAINABLE-PARAMS] " + fmt_params(measure_trainable_params(trainer.model)) + " M")
    try:
        flops, n_unsup = measure_flops(trainer.model, size=args.inWidth)
        log(f"[FLOPS] {fmt_flops(flops)} G   (input 2x3x{args.inWidth}x{args.inHeight}, unsupported_ops={n_unsup})")
    except Exception as e:  # fvcore 对 CASAA 动态 routing 覆盖不全时不阻塞训练
        log(f"[FLOPS] measurement failed ({type(e).__name__}: {e})")
    log("=" * 72)

    trainer.train()
    trainer.test_best()


if __name__ == "__main__":
    main()
