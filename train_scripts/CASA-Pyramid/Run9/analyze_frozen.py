"""Full-validation branch, radial and query-dependence interventions."""
import argparse
from contextlib import contextmanager
import csv
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

MODES = ('full', 'global_off', 'local_off', 'both_off', 'mean_attention', 'fixed_radius', 'shuffle_radius')


def score(counts):
    tp, fp, fn, tn = counts
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, F1=2*tp/max(2*tp+fp+fn,1),
                precision=tp/max(tp+fp,1), recall=tp/max(tp+fn,1), IoU=tp/max(tp+fp+fn,1))


def confusion(prediction, truth):
    return [int(x.sum()) for x in (prediction & truth, prediction & ~truth,
                                    ~prediction & truth, ~prediction & ~truth)]


@contextmanager
def altered(context, mode):
    original_point, original_logits = context.geometric_point, context.attention_logits
    global_gamma = context.gamma_context.detach().clone()
    local_gamma = context.gamma_detail.detach().clone()
    if mode in ('global_off','both_off'):
        context.gamma_context.zero_()
    if mode in ('local_off','both_off'):
        context.gamma_detail.zero_()
    if mode == 'mean_attention':
        def mean_logits(*args):
            probability = original_logits(*args).softmax(-1)
            return probability.mean(-2,keepdim=True).expand_as(probability).clamp_min(1e-30).log()
        context.attention_logits = mean_logits
    if mode in ('fixed_radius','shuffle_radius'):
        def point(projected):
            from model.layers.lorentz_ops import lift
            value = original_point(projected)
            space = value[...,1:]
            shape = space.shape
            flat = space.reshape(shape[0],shape[1],-1,shape[-1])
            parts = flat.split(context.tokens_per_scale,2) if flat.shape[2] == 224 and space.ndim == 4 else (flat,)
            adjusted=[]
            for part in parts:
                radius=torch.asinh(part.norm(dim=-1,keepdim=True))
                if mode == 'fixed_radius':
                    new_radius=radius.mean(2,keepdim=True).expand_as(radius)
                else:
                    generator=torch.Generator(device=radius.device).manual_seed(2333+part.shape[2])
                    permutation=torch.randperm(part.shape[2],generator=generator,device=radius.device)
                    new_radius=radius[:,:,permutation]
                adjusted.append(F.normalize(part,dim=-1,eps=1e-6)*new_radius.sinh())
            return lift(torch.cat(adjusted,2).reshape(shape))
        context.geometric_point = point
    try:
        yield
    finally:
        context.geometric_point,context.attention_logits=original_point,original_logits
        context.gamma_context.copy_(global_gamma)
        context.gamma_detail.copy_(local_gamma)


def components(mask):
    return cv2.connectedComponentsWithStats(mask.astype(np.uint8),connectivity=8)[1:3]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models-dir',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--dataset-root',type=Path,required=True)
    parser.add_argument('--module',choices=('hyperbolic','matched'),required=True)
    parser.add_argument('--refinement-beta',type=float,choices=(0.,1.),required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--samples',type=int,default=0)
    parser.add_argument('--batch-size',type=int,default=8)
    args=parser.parse_args()
    sys.path.insert(0,str(args.models_dir.resolve()))
    from model.casa_net import CASANet
    from train_CASA import make_loader,mask_boundary
    from model.layers.pyramid_context import source_bias
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=True
    torch.backends.cudnn.allow_tf32=True
    torch.backends.cuda.matmul.allow_tf32=False
    device=torch.device('cuda:0')
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    model=CASANet(module=args.module,refinement_beta=args.refinement_beta)
    model.load_state_dict(state.get('state_dict',state))
    model=model.to(device).eval()
    source=make_loader(args.dataset_root,'val',args.batch_size,0).dataset
    indices=np.linspace(0,len(source)-1,min(args.samples,len(source)),dtype=int).tolist() if args.samples else list(range(len(source)))
    areas=[]
    for index in indices:
        target=cv2.imread(source.gts[index],0)>=128
        if target.shape!=(256,256):target=cv2.resize(target.astype(np.uint8),(256,256),interpolation=cv2.INTER_NEAREST).astype(bool)
        _,stats=components(target)
        areas.extend(stats[1:,cv2.CC_STAT_AREA].tolist())
    threshold=int(np.quantile(areas,.25)) if areas else 0
    loader=DataLoader(Subset(source,indices),batch_size=args.batch_size,shuffle=False,num_workers=1)
    counts={mode:[0,0,0,0] for mode in MODES}
    group_counts={mode:{} for mode in MODES}
    objects={mode:dict(all=[0,0,0],small=[0,0,0]) for mode in MODES}
    corrected=dict(corrected_fn=0,corrected_fp=0,new_fn=0,new_fp=0)
    routes={str(index+1):{} for index in range(4)}
    original_logits=model.context.attention_logits
    enabled=False
    def capture_logits(query,key,support,scale):
        logits=original_logits(query,key,support,scale)
        if enabled:
            prior=support.log()[:,None,None].expand_as(logits)
            bias=source_bias(model.context.scale_bias,scale,model.context.tokens_per_scale).expand_as(logits)
            distance=logits-prior-bias
            for name,part in dict(distance=distance,log_support=prior,scale_prior=bias,total=logits).items():
                item=routes[str(scale+1)].setdefault(name,dict(count=0,sum=0.,square=0.,mass_sum=[0.]*4,queries=0))
                item['count']+=part.numel();item['sum']+=float(part.sum());item['square']+=float(part.square().sum())
                probability=part.softmax(-1)
                masses=torch.stack([p.sum(-1) for p in probability.split(model.context.tokens_per_scale,-1)],-1).sum((0,1,2))
                item['mass_sum']=[a+b for a,b in zip(item['mass_sum'],masses.tolist())]
                item['queries']+=part.shape[0]*part.shape[1]*part.shape[2]
        return logits
    model.context.attention_logits=capture_logits
    per_image=[];seen=0
    with torch.inference_mode():
        for image,target in loader:
            image,target=image.to(device).float(),target.to(device).float()
            truth=target>.5
            predictions={}
            for mode in MODES:
                enabled=mode=='full'
                with altered(model.context,mode):
                    predictions[mode]=model.predict(image[:,:3],image[:,3:])>.5
                batch=confusion(predictions[mode],truth)
                counts[mode]=[a+b for a,b in zip(counts[mode],batch)]
            enabled=False
            full,off=predictions['full'],predictions['global_off']
            corrected['corrected_fn']+=int((truth & ~off & full).sum())
            corrected['corrected_fp']+=int((~truth & off & ~full).sum())
            corrected['new_fn']+=int((truth & off & ~full).sum())
            corrected['new_fp']+=int((~truth & ~off & full).sum())
            edge=mask_boundary(target,1).bool()
            mixed=F.interpolate(((F.adaptive_avg_pool2d(target,(64,64))>0)&(F.adaptive_avg_pool2d(target,(64,64))<1)).float(),(256,256),mode='nearest').bool()
            groups=dict(boundary=edge,interior=~mask_boundary(target,2).bool(),mixed_occupancy=mixed)
            for mode,prediction in predictions.items():
                for name,mask in groups.items():
                    value=confusion(prediction&mask,truth&mask)
                    value[-1]=int((~prediction & ~truth & mask).sum())
                    item=group_counts[mode].setdefault(name,[0,0,0,0])
                    group_counts[mode][name]=[a+b for a,b in zip(item,value)]
            target_np=truth[:,0].cpu().numpy()
            predicted_np={mode:p[:,0].cpu().numpy() for mode,p in predictions.items()}
            for index,gt in enumerate(target_np):
                labels,stats=components(gt);sizes=stats[1:,cv2.CC_STAT_AREA]
                small=sizes<=threshold
                record=dict(name=source.file_list[indices[seen+index]])
                for mode in MODES:
                    pred=predicted_np[mode][index]
                    hits=np.bincount(labels.ravel(),weights=pred.ravel(),minlength=len(stats))[1:]
                    for name,selected in [('all',np.ones_like(sizes,dtype=bool)),('small',small)]:
                        item=objects[mode][name]
                        item[0]+=int(selected.sum());item[1]+=int(((hits>=.1*sizes)&selected).sum());item[2]+=int(((hits>=.5*sizes)&selected).sum())
                    image_counts=confusion(torch.as_tensor(pred),torch.as_tensor(gt))
                    for name,value in score(image_counts).items():record[mode+'_'+name]=value
                per_image.append(record)
            seen+=len(image)
            if seen%512<len(image):print('VAL',seen,'/',len(indices),flush=True)
    for stages in routes.values():
        for item in stages.values():
            item.update(mean=item['sum']/item['count'],std=max(0,item['square']/item['count']-(item['sum']/item['count'])**2)**.5,
                        source_mass=[x/item['queries'] for x in item['mass_sum']])
    for values in objects.values():
        for name,(number,low,high) in values.copy().items():
            values[name]=dict(components=number,coverage10_recall=low/number if number else None,coverage50_recall=high/number if number else None)
    report=dict(checkpoint=str(args.checkpoint),models_dir=str(args.models_dir),dataset=str(args.dataset_root),split='val',samples=len(indices),
                module=args.module,refinement_beta=args.refinement_beta,
                segmentation={mode:score(value) for mode,value in counts.items()},pixel_groups={m:{g:score(v) for g,v in values.items()} for m,values in group_counts.items()},
                component_coverage=objects,small_component_max_area=threshold,component_area_quartiles=np.quantile(areas,[0,.25,.5,.75,1]).tolist() if areas else [],
                corrections=corrected,route_components=routes,
                definitions=dict(fixed_radius='Each query stage and each source-key scale equalized per sample/head, preserving directions.',shuffle_radius='Fixed seeded within-scale radius permutation, preserving source marginals and directions.',
                mean_attention='Per sample/head and query stage, replace A with mean across queries; values unchanged.',component_coverage='Fraction of 8-connected GT components with >=10% or >=50% of their foreground predicted change; small cutoff is val lower quartile, fixed before model inference.'),
                limitations=['Frozen-weight interventions are not retrained module gains.','No scene IDs supplied; per-image paired records are saved, without claiming scene-independent confidence intervals.'])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    with args.output.with_suffix('.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(per_image[0]));writer.writeheader();writer.writerows(per_image)
    print(json.dumps(dict(module=args.module,refinement_beta=args.refinement_beta,
                          segmentation=report['segmentation'])),flush=True)


if __name__=='__main__':main()
