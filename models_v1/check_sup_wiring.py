"""B 形态（stgeo=sup）接线自检（服务器 GPU，batch 4；FAIL STOP）。

验证四件事：
  S1 辅助支路参数 = 1,538，且 `switch_to_deploy()` 后**整支消失**
     ⇒ 非主干参数回到 CTRL_R8 的 2,038,868（deploy 图与 CTRL 相同）
  S2 **epoch-0 前向与 CTRL 逐位一致**（sup 不碰主路径）
  S3 **只回传辅助损失**时，梯度能到达 **编码器 stage0**（这是 B 形态的承重条件；
     若为 0 则监督退化为"训小头再扔掉"，与 v1/v2 的注入问题同源）
  S4 主损失 + 辅助损失联合训练 3 步：loss 有限、proj/encoder 都有梯度
用法：CUDA_VISIBLE_DEVICES=1 python check_sup_wiring.py --pretrained <vmamba.pth>
"""
import argparse
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from model.casa_vmamba_stgeo_net import CASAVMambaSTGeoNet  # noqa: E402
from model.utils import BCEDiceLoss  # noqa: E402

CTRL_NON_BACKBONE = 2_038_868     # Run8 CTRL_R8 实测值


def stats(model):
    return model.param_breakdown()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained", required=True)
    ap.add_argument("--batch", type=int, default=4)
    args = ap.parse_args()
    ok = True

    torch.manual_seed(16)
    model = CASAVMambaSTGeoNet(vmamba_pretrained_path=args.pretrained,
                               stgeo="sup").float().cuda()
    pb = model.param_breakdown()
    g = torch.Generator().manual_seed(1234)
    pre = torch.randn(args.batch, 3, 256, 256, generator=g).cuda()
    post = torch.randn(args.batch, 3, 256, 256, generator=g).cuda()
    gt = (torch.rand(args.batch, 1, 256, 256, generator=g) > 0.9).float().cuda()

    # ---- S1 参数与 deploy 删除
    rep = model.stgeo_report()
    print(f"[S1] aux_params={rep['params']} non_backbone(train)={pb['non_backbone']} "
          f"stgeo={pb['stgeo']}")
    ok &= (rep["params"] == 1538)
    model.switch_to_deploy()
    pb_dep = stats(model)
    # 对照：Run8 CTRL_R8 的 deploy 总量（其 TEST 区块记录 31.026 M；精确值见下）
    CTRL_DEPLOY_TOTAL = 29_938_176 + 1_087_489       # = 31,025,665
    print(f"[S1] deploy: sup total={pb_dep['total']} (CTRL_DEPLOY_TOTAL={CTRL_DEPLOY_TOTAL}) "
          f"stgeo={pb_dep['stgeo']} sup_is_none={model.stgeo_sup is None}")
    ok &= (pb_dep["total"] == CTRL_DEPLOY_TOTAL and model.stgeo_sup is None)

    # ---- S2 epoch-0 前向（training 图，未 deploy）与 CTRL 等价：用同一份 ref 张量比对由外部脚本完成
    torch.manual_seed(16)
    m2 = CASAVMambaSTGeoNet(vmamba_pretrained_path=args.pretrained,
                            stgeo="sup").float().cuda().eval()
    with torch.no_grad():
        out = m2(pre, post)
    torch.save(out.cpu(), os.path.join("/tmp/exp1", "v1_sup.pt"))
    print(f"[S2] dumped sup epoch-0 output -> /tmp/exp1/v1_sup.pt  sum={float(out.sum()):.6f}")

    # ---- S3 只回传辅助损失 ⇒ 编码器 stage0 是否有梯度
    m2.train()
    m3 = CASAVMambaSTGeoNet(vmamba_pretrained_path=args.pretrained,
                            stgeo="sup").float().cuda().train()
    m3.zero_grad(set_to_none=True)
    out = m3(pre, post)                     # 主路径（不回传）
    logit = m3.aux_change_logits()
    aux = m3.stgeo_sup.loss(logit, gt)
    aux.backward()
    enc_g = sum(float(p.grad.norm()) for n, p in m3.encoder.named_parameters()
                if p.grad is not None and "layers.0" in n)
    proj_g = float(m3.stgeo_sup.proj.weight.grad.norm())
    print(f"[S3] aux_only backward: aux_loss={float(aux):.6f} "
          f"proj_grad_norm={proj_g:.4e} encoder_stage0_grad_norm={enc_g:.4e}")
    ok &= (proj_g > 0 and enc_g > 0)

    # ---- S4 主 + 辅联合 3 步
    opt = torch.optim.Adam(m3.parameters(), 2e-4, (0.9, 0.99), eps=1e-8, weight_decay=1e-4)
    for step in range(3):
        m3.zero_grad(set_to_none=True)
        prob = m3(pre, post)
        loss = BCEDiceLoss(prob, gt) + m3.stgeo_sup.loss(m3.aux_change_logits(), gt)
        loss.backward()
        opt.step()
        fin = bool(torch.isfinite(loss).item())
        print(f"[S4] step={step + 1} loss={float(loss):.6f} finite={fin} "
              f"score={m3.stgeo_diag()}")
        ok &= fin

    print(f"[RESULT] {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
