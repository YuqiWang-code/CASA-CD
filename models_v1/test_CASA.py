"""Native pyramid, matching common initialization, task gradients and folding."""
import torch

from model.casa_net import CASANet
from train_CASA import context_decay_names, training_losses, make_optimizer, verify_deployment


def main():
    torch.set_num_threads(4)
    torch.manual_seed(2333)
    a,b=torch.randn(1,3,256,256),torch.randn(1,3,256,256)
    target=torch.randint(0,2,(1,1,256,256)).float()
    reference=None
    for variant in ('none','hyperbolic','qkv'):
        torch.manual_seed(2333)
        model=CASANet(module=variant).eval()
        count=sum(p.numel() for p in model.parameters())
        with torch.no_grad():
            encoded=model.encoder(a)[1:]
            assert [tuple(x.shape[1:]) for x in encoded]==[(24,64,64),(32,32,32),(96,16,16),(320,8,8)]
            logits,aux=model(a,b,return_aux=True)
            outputs=model(a,b)
            assert len(logits)==len(outputs)==len(aux['guide_logits'])==4
            assert all(x.shape==target.shape for x in logits)
            assert all(torch.equal(z.sigmoid(),p) for z,p in zip(logits,outputs))
            assert torch.equal(model.predict(a,b),outputs[0])
            assert all(torch.isfinite(x).all() for x in model(a,a))
            if reference is None:
                reference=outputs[0]
            else:
                assert torch.equal(reference,outputs[0]), 'common initialization differs'
        model.train()
        optimizer=make_optimizer(model)
        context_ids={id(p) for p in model.context.parameters()}
        projection_ids={id(p) for n,p in model.named_parameters() if n in context_decay_names(model)}
        grouped=[id(p) for group in optimizer.param_groups for p in group['params']]
        assert len(grouped)==len(set(grouped))==len(list(model.parameters()))
        for group in optimizer.param_groups:
            assert group['lr_scale']==1.
            assert all(group['weight_decay']==(0. if id(p) in context_ids and id(p) not in projection_ids
                                               else 1e-4) for p in group['params'])
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            logits,aux=model(a,b,return_aux=True)
            loss,main,guide=training_losses(logits,aux,target)
            assert torch.allclose(loss,main+.2*guide)
            loss.backward()
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            optimizer.step()
        assert model.context.guide.weight.grad.abs().sum()>0
        if variant!='none':
            for layer in (model.context.qk,model.context.v,model.context.out,
                          model.context.local_dw,model.context.local_out):
                assert layer.weight.grad.abs().sum()>0
            assert (model.context.gamma_context.abs().sum(-1)>0).all()
            assert (model.context.gamma_detail.abs().sum(-1)>0).all()
        model.eval()
        _,report=verify_deployment(model,torch.cat((a,b),1))
        assert max(report['per_scale_max_diff'])<2e-4 and report['binary_disagreement']==0
        deployed=model.switch_to_deploy()
        print(variant,'train/deploy params',count,sum(p.numel() for p in deployed.parameters()),
              'guide/task gradients and folding PASS')


if __name__=='__main__':
    main()
