"""Verify signed frequency paths, matched initialization, gradients and folding."""
import argparse
import copy
import json
from pathlib import Path
import torch
import sys
from model.casa_net import CASANet
from model.layers.candidate_context import TemporalWavelet, MODEL_VARIANTS
from model.layers.moe_context import haar_split, haar_merge
from train_CASA import training_losses, make_optimizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(2333)
    x = torch.randn(2,64,64,64)
    low, high = haar_split(x)
    assert torch.allclose(haar_merge(low,high),x,atol=1e-6)
    assert torch.equal(haar_split(torch.ones_like(x))[0]/2,torch.ones_like(low))
    wave = TemporalWavelet()
    wave.eta.data.fill_(1)
    a = [torch.randn(2,64,s,s) for s in (64,32,16,8)]
    b = [torch.randn_like(f) for f in a]
    outputs = wave(a,b)
    assert [f.shape[-1] for f in outputs] == [64,32,16,8]
    assert all(torch.count_nonzero(f)==0 for f in wave(a,a))
    assert max(float(haar_split(f)[0].detach().abs().max()) for f in outputs[:2]) < 2e-6
    image = torch.randn(2,6,256,256)
    target = (torch.rand(2,1,256,256)>.8).float()
    report, reference = {}, None
    for variant in MODEL_VARIANTS:
        torch.manual_seed(2333)
        model = CASANet(module=variant)
        shared = {n:p.detach().clone() for n,p in model.named_parameters() if not n.startswith('context.')}
        if reference is None:
            reference=shared
        else:
            assert all(torch.equal(p,reference[n]) for n,p in shared.items())
        model.eval()
        with torch.no_grad():
            fa,fb=model.encoder(image[:,:3])[1:],model.encoder(image[:,3:])[1:]
            if variant=='none':
                expected=model.decoder.forward_pyramid(model.tar(fa,fb))
                actual,_=model._decode(image[:,:3],image[:,3:])
                assert all(torch.equal(u,v) for u,v in zip(expected,actual))
            else:
                xx,norm,g,bank,w=model.context.prepare(fa,fb)
                assert bank[0].shape==(2,224,32)
                assert torch.equal(model.context.weights(3,xx[3],None),torch.full((2,4,8,8),.25))
        optimizer=make_optimizer(model)
        model.train()
        for step in range(3):
            optimizer.zero_grad(set_to_none=True)
            logits,aux=model(image[:,:3],image[:,3:],return_aux=True)
            loss,main,guide=training_losses(logits,aux,target)
            loss.backward()
            assert torch.isfinite(loss)
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
            if variant!='none' and step==2:
                for name in ('retrieval.qk.weight','retrieval.v.weight','wavelet.high.0.0.weight'):
                    assert dict(model.context.named_parameters())[name].grad.abs().sum()>0,name
            optimizer.step()
        model.eval()
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'train_scripts/CASA-MoE/Run2'))
        from health_probe import inspect
        probe=inspect(model,image,target)
        assert all(torch.isfinite(torch.tensor(v)) for v in probe['task_grad_rms'].values())
        deployed=copy.deepcopy(model).switch_to_deploy().eval()
        with torch.no_grad():
            before=model(image[:,:3],image[:,3:]); after=deployed(image[:,:3],image[:,3:])
            error=max(float((u-v).abs().max()) for u,v in zip(before,after))
            assert error<2e-4,(variant,error)
        report[variant]=dict(passed=True,fold_max_diff=error,
            params=sum(p.numel() for p in model.parameters()),deploy_params=sum(p.numel() for p in deployed.parameters()))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2))
    print(json.dumps(report))


if __name__=='__main__':
    main()
