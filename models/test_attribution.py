"""Matched geometry, loss isolation, refinement and seed controls."""
import torch
from torch.nn import functional as F
from model.casa_net import CASANet
from model.layers.hyperbolic_context import HyperbolicContext,MatchedEuclideanContext,class_assignments,refined_assignments
from model.layers.pyramid_context import feature_regions,pool_features
from train_CASA import training_losses,guide_loss,loss_function


def main():
    torch.set_num_threads(4);torch.manual_seed(2333)
    feature=torch.randn(2,64,8,8,dtype=torch.double,requires_grad=True)
    p=torch.rand(2,1,8,8,dtype=torch.double,requires_grad=True)
    w0,r0=class_assignments(p,2)
    exact=refined_assignments(feature,p,2,0.)
    assert torch.equal(w0,exact[0]) and torch.equal(r0,exact[1])
    w1,r1=refined_assignments(feature,p,2,1.)
    blocks=feature_regions(feature,2)
    initial=torch.einsum('brtp,brpc->brtc',w0,blocks)
    cosine=F.cosine_similarity(blocks[:,:,None],initial[:,:,:,None],dim=-1,eps=1e-6)
    reference=w0*cosine.exp();reference=reference/reference.sum(-1,keepdim=True)
    torch.testing.assert_close(w1,reference,rtol=1e-12,atol=1e-12)
    membership=torch.stack((feature_regions(p,2).squeeze(-1),1-feature_regions(p,2).squeeze(-1)),2)
    torch.testing.assert_close(r1,(w1*membership).sum(-1).flatten(1),rtol=1e-12,atol=1e-12)
    gradients=torch.autograd.grad(pool_features(feature,w1,2).square().mean(),(feature,p))
    assert all(torch.isfinite(x).all() and x.abs().sum()>0 for x in gradients)
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
    a,main,_=training_losses(outputs,aux,target,.1,.1)
    b,_,_=training_losses(outputs,aux,target,.1,0.)
    torch.testing.assert_close(a-b,.1*guide_loss(retrieval,target))
    torch.testing.assert_close(b,main+.1*guide_loss(guides,target))
    none_grad=torch.autograd.grad(b,retrieval,allow_unused=True)
    assert all(g is None or not g.any() for g in none_grad)
    for beta in (0.,1.):
        op=HyperbolicContext(refinement_beta=beta)
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
    print('beta0 exact control, beta1 formula/support/gradients, matched H/E, isolated loss and seed controls PASS')


if __name__=='__main__':main()
