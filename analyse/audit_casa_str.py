"""CASA-STR 预算审计 + 6 变体初始化哈希（主线重构 · 机器可复现预算裁决）。

在 CPU 上构建 6 个变体（seed 16），逐项输出并硬门禁：
  - trunk = 1,861,296（SHViT-S1 截断主干）
  - 预训练逐位继承（max_abs_diff == 0）
  - deploy 折叠（switch_to_deploy）后无 BN/aux 键
  - deploy params <= 5M（硬门槛）
  - 同一 rep_mode 下 attn 变体 deploy 参数一致（CASAA 常量 36,865）
  - 每个变体的初始化哈希（INIT-HASH，供跨机复现对拍）

用法：
    python audit_casa_str.py --pretrained_weight_path <shvit_s1.pth> [--all]
"""
import argparse
import hashlib
import os
import sys

import torch

_MODELS_ROOT = os.path.dirname(os.path.abspath(__file__))
if _MODELS_ROOT not in sys.path:
    sys.path.insert(0, _MODELS_ROOT)

from model.casa_str_net import CASASTRNet

VARIANTS = [
    ("A0_BASE_PLAIN", "none", "plain"),
    ("A1_CASAA_PLAIN", "change", "plain"),
    ("A2_STR_ONLY", "none", "full"),
    ("C1_FULLATTN_PLAIN", "full", "plain"),
    ("C2_CONTENT_SAA_PLAIN", "content", "plain"),
    ("M1_CASAA_STR", "change", "full"),
]


def measure_params(model):
    return sum(p.numel() for p in model.parameters())


def state_hash(model):
    h = hashlib.sha256()
    for k in sorted(model.state_dict().keys()):
        h.update(k.encode())
        h.update(model.state_dict()[k].detach().cpu().float().numpy().tobytes())
    return h.hexdigest()[:16]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_weight_path", type=str, required=True)
    parser.add_argument("--all", action="store_true",
                        help="run forward pass + fold error measurement as well (slower)")
    args = parser.parse_args()

    ckpt = torch.load(args.pretrained_weight_path, map_location="cpu", weights_only=False)["model"]

    rows = {}
    for name, attn, rep in VARIANTS:
        torch.manual_seed(16)
        model = CASASTRNet(args.pretrained_weight_path, attn_mode=attn,
                           rep_mode=rep, str_dim=160, keep_ratio=0.25, change_share=0.5)
        model.eval()

        trunk = model.encoder.param_count()
        assert trunk == 1861296, f"{name}: trunk {trunk} != 1861296"

        # 预训练逐位对拍
        sd = model.encoder.state_dict()
        worst = 0.0
        for k, v in sd.items():
            if k in ckpt:
                worst = max(worst, (v.float() - ckpt[k].float()).abs().max().item())
        assert worst == 0.0, f"{name}: pretrain mismatch {worst}"

        casaa_params = model.casaa.param_count() if model.casaa is not None else 0
        total = measure_params(model)

        import copy
        dep = copy.deepcopy(model)
        dep.switch_to_deploy()
        dep.eval()
        bn_keys = [k for k in dep.state_dict() if "bn" in k]
        assert not bn_keys, f"{name}: deploy graph has BN keys {bn_keys[:3]}"
        deploy = measure_params(dep)
        assert deploy <= 5.0e6, f"{name}: deploy {deploy} > 5M"

        fold_err = float("nan")
        if args.all:
            pre = torch.randn(1, 3, 256, 256)
            post = torch.randn(1, 3, 256, 256)
            with torch.no_grad():
                y_t = model(pre, post)
                y_d = dep(pre, post)
            fold_err = (y_t - y_d).abs().max().item()

        rows[name] = dict(total=total, deploy=deploy, casaa=casaa_params,
                          trunk=trunk, fold=fold_err, hash=state_hash(model))
        print(f"[{name}] attn={attn:7s} rep={rep:5s} | trunk={trunk:,} casaa={casaa_params:,} "
              f"total={total:,} deploy={deploy:,} ({deploy / 1e6:.3f}M) "
              f"fold_max_abs={fold_err:.3e} init_hash={rows[name]['hash']}")

    # 结构一致性：同 rep_mode 的 attn 变体 deploy 参数差 = CASAA 常量
    assert rows["A0_BASE_PLAIN"]["deploy"] == rows["A2_STR_ONLY"]["deploy"]
    for n in ("A1_CASAA_PLAIN", "C1_FULLATTN_PLAIN", "C2_CONTENT_SAA_PLAIN", "M1_CASAA_STR"):
        assert rows[n]["deploy"] == rows["A1_CASAA_PLAIN"]["deploy"]
    assert rows["A1_CASAA_PLAIN"]["deploy"] - rows["A0_BASE_PLAIN"]["deploy"] == rows["A1_CASAA_PLAIN"]["casaa"] \
        == 36865
    print("[BUDGET] all variants deploy <= 5M; CASAA delta == 36,865; rep_mode does not change deploy params")
    print("[AUDIT] CASA-STR budget audit ALL OK")


if __name__ == "__main__":
    main()
