"""Matched geometry, isolated losses and paired initialization."""
import torch
from model.casa_net import CASANet
from model.layers.hyperbolic_context import HyperbolicContext,MatchedEuclideanContext
from train_CASA import auxiliary_weights,training_losses,guide_loss,loss_function


def main():
    torch.set_num_threads(4);torch.manual_seed(2333)
    for variant in ("none", "hyperbolic", "matched", "qkv"):
        assert auxiliary_weights(variant) == dict(guide=.1, geometric_retrieval=0.)
    h,e=HyperbolicContext(),MatchedEuclideanContext();e.load_state_dict(h.state_dict())
    x=torch.randn(1,64,8,8)
    qh,qe=h.project_queries(x),e.project_queries(x)
    assert torch.equal(qh,qe)
    assert sum(p.numel() for p in h.parameters())==sum(p.numel() for p in e.parameters())==11477
    assert not torch.allclose(h.squared_distance(qh.flatten(2,3),qh.flatten(2,3)),e.squared_distance(qe.flatten(2,3),qe.flatten(2,3)))
    target=torch.randint(0,2,(1,1,32,32)).float()
    outputs=tuple(torch.randn_like(target) for _ in range(4))
    guides=tuple(torch.randn(1,1,n,n,requires_grad=True) for n in (8,4,2,1))
    retrieval=tuple(torch.randn(1,1,n,n,requires_grad=True) for n in (8,4,2,1))
    aux=dict(guide_logits=guides,geometric_logits=retrieval)
    default, default_main, default_aux = training_losses(outputs, aux, target)
    torch.testing.assert_close(default, default_main + .1 * guide_loss(guides, target))
    unused = torch.autograd.grad(default, retrieval, allow_unused=True)
    assert all(value is None for value in unused)
    a,main,_=training_losses(outputs,aux,target,.1,.1)
    b,_,_=training_losses(outputs,aux,target,.1,0.)
    torch.testing.assert_close(a-b,.1*guide_loss(retrieval,target))
    torch.testing.assert_close(b,main+.1*guide_loss(guides,target))
    none_grad=torch.autograd.grad(b,retrieval,allow_unused=True)
    assert all(g is None or not g.any() for g in none_grad)
    op=HyperbolicContext()
    op.gamma_context.data.fill_(.2);op.gamma_detail.data.fill_(.2)
    value=torch.randn(1,64,1,1)
    constant=[value.expand(1,64,n,n).clone() for n in (16,8,4,4)]
    enhanced,_=op(constant)
    assert all(torch.allclose(a,b,atol=3e-6) for a,b in zip(enhanced,constant))
    out,_=op([torch.zeros_like(v) for v in constant])
    assert all(not z.any() for z in out)
    torch.manual_seed(11);m1=CASANet(module='hyperbolic',context_seed=22)
    torch.manual_seed(11);m2=CASANet(module='matched',context_seed=22)
    assert all(torch.equal(a,b) for a,b in zip(m1.state_dict().values(),m2.state_dict().values()))
    torch.manual_seed(11);m3=CASANet(module='hyperbolic',context_seed=23)
    assert torch.equal(m1.encoder.features[0][0].weight,m3.encoder.features[0][0].weight)
    assert not torch.equal(m1.context.qk.weight,m3.context.qk.weight)
    print('Matched H/E, shared content, isolated loss and paired initialization PASS')


if __name__=='__main__':main()
