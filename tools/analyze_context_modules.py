"""Inspect trained context branches on fixed validation samples."""
import argparse
from contextlib import contextmanager
import json
import math
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
from torch.nn import functional as F


def rms(x):
    return float(x.detach().square().mean().sqrt())


def auc(score, target):
    score, target = score.detach().cpu().numpy().ravel(), target.detach().cpu().numpy().ravel()
    positive, negative = int(target.sum()), int((~target).sum())
    if not positive or not negative:
        return None
    _, inverse, counts = np.unique(score, return_inverse=True, return_counts=True)
    end = np.cumsum(counts)
    ranks = end - (counts - 1) / 2
    return float((ranks[inverse][target].sum() - positive * (positive + 1) / 2) / (positive * negative))


def sample_indices(dataset, count):
    candidates = np.linspace(0, len(dataset) - 1, min(128, len(dataset)), dtype=int)
    fractions = []
    for index in candidates:
        _, target = dataset[int(index)]
        fractions.append((float(target.float().mean()), int(index)))
    fractions.sort()
    return [fractions[i][1] for i in np.linspace(0, len(fractions)-1, count, dtype=int)]


@contextmanager
def branch_mode(model, mode):
    hooks, methods = [], []
    for op in model.stage_modules:
        if mode in ('no_global', 'no_local'):
            def ablate(layer, inputs, op=op):
                x = inputs[0]
                local_dim = op.local.in_channels
                global_part, local_part = x[:, :-local_dim], x[:, -local_dim:]
                if mode == 'no_global':
                    global_part = torch.zeros_like(global_part)
                else:
                    local_part = torch.zeros_like(local_part)
                return (torch.cat((global_part, local_part), 1),)
            hooks.append(op.out.register_forward_pre_hook(ablate))
        elif mode == 'uniform':
            original = op.attention
            methods.append((op, original))
            op.attention = lambda q, k: q.new_full((q.shape[0], q.shape[1], k.shape[1]), 1 / k.shape[1])
    try:
        yield
    finally:
        for hook in hooks:
            hook.remove()
        for op, method in methods:
            op.attention = method


def confusion(prediction, target):
    p, t = prediction > .5, target > .5
    return np.array([int((p&t).sum()), int((p&~t).sum()), int((~p&t).sum())], dtype=np.int64)


def scores(counts):
    tp, fp, fn = (int(x) for x in counts)
    return dict(tp=tp, fp=fp, fn=fn, f1=2*tp/max(2*tp+fp+fn, 1), iou=tp/max(tp+fp+fn, 1))


def module_statistics(op, a, b, target, mode):
    from model.layers.context_compression import pool_tokens, prototype_weights, token_map
    from model.layers.poincare_ops import aligned_distance, logmap0, displacement, weighted_centroid
    h, w = a.shape[-2:]
    if mode == 'hyperbolic':
        xa, xb = op.encode(a), op.encode(b)
        qa, qb = (op.mobius_linear(x, op.metric) for x in (xa, xb))
        distance = aligned_distance(qa, qb)
        weights = prototype_weights((distance/(1+distance)).reshape(1,1,h,w))
        ka, kb = (op.compress(x, weights, h, w) for x in (qa,qb))
        va, vb = (op.mobius_linear(x, op.value) for x in (xa,xb))
        da, db = displacement(va,vb), displacement(vb,va)
        values = op.compress(da,weights,h,w)
        p = op.attention(qa,kb)
        global_a = logmap0(weighted_centroid(p,values))
        uniform = logmap0(weighted_centroid(torch.full_like(p,1/p.shape[-1]),values))
        local = logmap0(da)
        euclidean = torch.linalg.vector_norm(qa-qb,dim=-1)
        radial = (qa.norm(dim=-1)-qb.norm(dim=-1)).abs()
        directional = 1-F.cosine_similarity(qa,qb,dim=-1)
        cancellations = values.norm(dim=-1).mean()/da.norm(dim=-1).mean().clamp_min(1e-8)
    else:
        na, nb = op.normalize(a), op.normalize(b)
        distance = (na-nb).abs().mean(1).flatten(1)
        weights = prototype_weights((distance/(1+distance)).reshape(1,1,h,w))
        qa,qb=(op.q(x).flatten(2).transpose(1,2) for x in (na,nb))
        ka,kb=(pool_tokens(op.k(x),weights) for x in (na,nb))
        delta=op.v(a)-op.v(b)
        values=pool_tokens(torch.cat((delta,delta.abs()),1),weights)
        p=op.attention(qa,kb)
        global_a=p@values
        uniform=values.mean(1,keepdim=True).expand_as(global_a)
        local=delta.flatten(2).transpose(1,2)
        cancellations=values[...,:local.shape[-1]].abs().mean()/local.abs().mean().clamp_min(1e-8)
        euclidean=radial=directional=None
    pooled_target=F.adaptive_avg_pool2d(target,(h,w)).flatten(1)
    pure=(pooled_target>=.5)|(pooled_target<=.05)
    binary=pooled_target>=.5
    pair=kb.reshape(1,-1,2,kb.shape[-1])
    report=dict(assignment_tv=float((weights[:,:,0]-weights[:,:,1]).abs().sum(-1).mean()/2),
        key_pair_relative_difference=rms(pair[:,:,0]-pair[:,:,1])/max(rms(kb),1e-8),
        attention_entropy=float(-(p*p.clamp_min(1e-9).log()).sum(-1).mean()/math.log(p.shape[-1])),
        query_attention_tv=float((p-p.mean(1,keepdim=True)).abs().sum(-1).mean()/2),
        uniform_output_relative_difference=rms(global_a-uniform)/max(rms(global_a),1e-8),
        compression_retention=float(cancellations),global_value_rms=rms(global_a),local_value_rms=rms(local),
        change_auc=auc(distance[pure],binary[pure]))
    if mode=='hyperbolic':
        report.update(euclidean_auc=auc(euclidean[pure],binary[pure]),radial_auc=auc(radial[pure],binary[pure]),
                      angular_auc=auc(directional[pure],binary[pure]))
    change=distance.reshape(1,1,h,w)
    return report, change


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models-dir',type=Path,required=True)
    parser.add_argument('--run',default='Run6')
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--samples',type=int,default=12)
    args=parser.parse_args()
    sys.path.insert(0,str(args.models_dir.resolve()))
    from model.casa_net import CASANet
    from train_CASA import make_loader,loss_function
    torch.set_num_threads(4)
    device=torch.device('cuda:0')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    root=Path('/share_datasets/yqwang/checkpoints/CASA-CD/CASA')/args.run
    results=[]
    for dataset in ('LEVIR','SYSU','WHU','CDD'):
        data=make_loader(Path('/share_datasets/CD')/f'{dataset}-CD-256','val',1,0).dataset
        indices=sample_indices(data,args.samples)
        samples=[data[i] for i in indices]
        for mode,variant in (('hyperbolic','M1_HYPER_QKV'),('qkv','M2_CA_QKV')):
            model=CASANet(module=mode).to(device).eval()
            model.load_state_dict(torch.load(root/variant/f'{dataset}-CD-256/best_model.pth',map_location=device,weights_only=True),strict=True)
            directory=args.output_dir/dataset/mode
            directory.mkdir(parents=True,exist_ok=True)
            totals={name:np.zeros(3,dtype=np.int64) for name in ('full','no_global','no_local','uniform')}
            records=[]
            for number,((image,target),index) in enumerate(zip(samples,indices)):
                image=image[None].float().to(device);target=target[None].float().to(device)
                with torch.no_grad():
                    a=model.encoder_fusion(*model.encoder(image[:,:3])); b=model.encoder_fusion(*model.encoder(image[:,3:]))
                    record=dict(filename=data.file_list[index],change_fraction=float(target.mean()),stages={})
                    for stage,(op,x,y) in enumerate(zip(model.stage_modules,a,b),1):
                        record['stages'][f'stage{stage}'],change=module_statistics(op,x,y,target,mode)
                        op.collect_diag=True;op(x,y);op.collect_diag=False
                        record['stages'][f'stage{stage}'].update(op.last_diag)
                        if stage==1:
                            heat=F.interpolate(change,(256,256),mode='bilinear',align_corners=False)[0,0].cpu().numpy()
                            heat=np.uint8(np.clip((heat-heat.min())/max(float(heat.max()-heat.min()),1e-8),0,1)*255)
                            cv2.imwrite(str(directory/f'{number:02d}_distance.png'),cv2.applyColorMap(heat,cv2.COLORMAP_TURBO))
                    for branch in totals:
                        with branch_mode(model,branch):prediction=model(image[:,:3],image[:,3:])[0]
                        totals[branch]+=confusion(prediction,target)
                        if branch=='full':
                            p=prediction[0,0].cpu().numpy()>.5;t=target[0,0].cpu().numpy()>.5
                            error=np.zeros((256,256,3),np.uint8);error[p&t]=255;error[p&~t]=(0,0,255);error[~p&t]=(255,0,0)
                            cv2.imwrite(str(directory/f'{number:02d}_prediction.png'),np.uint8(p)*255)
                            cv2.imwrite(str(directory/f'{number:02d}_error.png'),error)
                    for name,path in (('A',data.pre_images[index]),('B',data.post_images[index]),('label',data.gts[index])):
                        cv2.imwrite(str(directory/f'{number:02d}_{name}.png'),cv2.imread(path))
                records.append(record)
            image,target=samples[len(samples)//2]
            image=image[None].float().to(device);target=target[None].float().to(device)
            captured={};hooks=[]
            for i,op in enumerate(model.stage_modules,1):
                def hook(layer,inputs,output,i=i):
                    captured.setdefault(i,[]).append(inputs[0]);inputs[0].retain_grad()
                hooks.append(op.out.register_forward_hook(hook))
            model.zero_grad(set_to_none=True)
            loss_function(model(image[:,:3],image[:,3:]),target).backward()
            gradients={}
            for i,op in enumerate(model.stage_modules,1):
                dim=op.local.in_channels
                gradients[f'stage{i}']=dict(parameters={name:rms(p.grad) if p.grad is not None else None for name,p in op.named_parameters()},
                    global_input_rms=rms(captured[i][0][:,:-dim]),local_input_rms=rms(captured[i][0][:,-dim:]),
                    global_task_grad_rms=rms(captured[i][0].grad[:,:-dim]),local_task_grad_rms=rms(captured[i][0].grad[:,-dim:]))
            for hook in hooks:hook.remove()
            entry=dict(dataset=dataset,module=mode,sample_count=len(samples),split='val',
                subset_scores={name:scores(total) for name,total in totals.items()},samples=records,task_gradients=gradients)
            (directory/'analysis.json').write_text(json.dumps(entry,indent=2))
            results.append(entry)
            print(dataset,mode,entry['subset_scores'],flush=True)
            del model
            torch.cuda.empty_cache()
    (args.output_dir/'analysis.json').write_text(json.dumps(results,indent=2))
    print('[ANALYSIS-COMPLETE]',flush=True)


if __name__=='__main__':main()
