"""Verify signed frequency paths, matched initialization, gradients and folding."""
import argparse
import copy
import json
from pathlib import Path
import torch
import sys
from model.casa_net import CASANet
from model.layers.candidate_context import TemporalWavelet, PairedRetrieval, MODEL_VARIANTS
from model.layers.moe_context import haar_split, haar_merge
from train_CASA import training_losses, make_optimizer


def check_shared_readout():
    retrieval = PairedRetrieval().double()
    with torch.no_grad():
        retrieval.gamma.uniform_(-.4, .4)
    x = torch.randn(2, 64, 8, 8, dtype=torch.double, requires_grad=True)
    prototype = torch.randn(2, 224, 64, dtype=torch.double, requires_grad=True)
    support = torch.rand(2, 224, dtype=torch.double).clamp_min(.01).requires_grad_()
    logits = torch.randn(2, 4, 8, 8, dtype=torch.double, requires_grad=True)
    route = logits.softmax(1)[:, :2]
    bank = retrieval.prepare(prototype, support)
    merged, _, _ = retrieval(x, bank, 1, route)
    separate, _, _ = retrieval(x, bank, 1, route, separate=True)
    torch.testing.assert_close(merged, separate, atol=1e-10, rtol=1e-9)
    upstream = torch.randn_like(merged)
    parameters = (x, prototype, support, logits, *retrieval.parameters())
    gm = torch.autograd.grad(merged, parameters, upstream, retain_graph=True)
    gs = torch.autograd.grad(separate, parameters, upstream)
    for first, second in zip(gm, gs):
        torch.testing.assert_close(first, second, atol=1e-9, rtol=1e-8)
    return dict(passed=True, output_max_diff=float((merged-separate).detach().abs().max()),
                gradient_max_diff=max(float((u-v).abs().max()) for u,v in zip(gm,gs)))


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
    report, reference = dict(shared_readout=check_shared_readout()), None
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
                assert torch.equal(model.context.weights(3,xx[3],model.tar(fa,fb)[3],None),torch.full((2,4,8,8),.25))
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
            if variant == 'moe' and step == 2:
                for layer in model.context.router:
                    assert layer[0].weight.grad[:, 64:128].abs().sum() > 0
            optimizer.step()
        model.eval()
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'train_scripts/CASA-MoE/Run3'))
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
