"""Fixed validation interventions using the archived Run4 implementation."""
import json
from pathlib import Path
import sys
import tarfile
import time

import torch
from torch.nn import functional as F

BASE = Path('/home/yqwang/outputs/CASA-CD/CASA-Pyramid')
OUT = BASE / 'Run5/analysis'
SOURCE = OUT / 'run4_source'
OUT.mkdir(parents=True, exist_ok=True)
with tarfile.open(BASE / 'Run4/source_code.tar.gz') as archive:
    for member in archive.getmembers():
        if member.isfile() and member.name.startswith('models/'):
            destination = SOURCE / member.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(archive.extractfile(member).read())
sys.path.insert(0, str(SOURCE / 'models'))
from model.casa_net import CASANet
from model.layers.pyramid_context import (channel_norm, assignments, pool_features,
    project_tokens, merge_heads, highpass, feature_regions, source_bias)
from model.layers.lorentz_ops import lift, pairwise_squared_distance
from model.layers.euclidean_context import EuclideanContext
from train_CASA import make_loader, SpatialMetrics


def rms(x):
    return float(x.square().mean().sqrt())


@torch.inference_mode()
def contexts(op, tar, target, record):
    x = [channel_norm(op.norm, t) for t in tar]
    guides = [op.guide(t).sigmoid() for t in x]
    compressed = [assignments(p, g) for p, g in zip(guides, op.grid_sizes)]
    prototypes = [pool_features(t, w, g) for t, (w, _), g in zip(x, compressed, op.grid_sizes)]
    queries = [op.project_queries(t) for t in x]
    keys = [op.pool_keys(q, p, w, g) for q, p, (w, _), g in zip(queries, prototypes, compressed, op.grid_sizes)]
    key = torch.cat(keys, 2)
    reliability = torch.cat([r for _, r in compressed], 1)
    values = project_tokens(op.v, torch.cat(prototypes, 1), op.heads)
    modes = {name: [] for name in ('baseline', 'no_context', 'no_local', 'self_only',
                                  'retrieval_only', 'uniform_attention', 'alternate_geometry')}
    measurements = []
    for s, (original, z, q, g) in enumerate(zip(tar, x, queries, op.grid_sizes)):
        h, w = z.shape[-2:]
        q = q.flatten(2, 3)
        attention = op.attention(q, key, reliability, s)
        read = merge_heads(attention @ values, h, w)
        own = op.v(z)
        local = (op.gamma_detail[s][None, :, None, None] * op.local_out(op.local_dw(highpass(z)))
                 if s < 2 else torch.zeros_like(z))
        gamma = op.gamma_context[s][None, :, None, None]
        update = gamma * op.out(read - own)
        uniform = torch.cat([torch.full_like(attention[..., :n], 1 / (4*n))
                             for n in op.tokens_per_scale], -1)
        if op.geometry == 'hyperbolic':
            other_distance = EuclideanContext.squared_distance(q[..., 1:], key[..., 1:])
        else:
            other_distance = pairwise_squared_distance(lift(q), lift(key))
        temperature = F.softplus(op.raw_temperature)[None, :, None, None]
        other = (-other_distance / (2 * temperature.square()) + reliability.log()[:, None, None]
                 + source_bias(op.scale_bias, s, op.tokens_per_scale)).softmax(-1)
        modes['baseline'].append(original + update + local)
        modes['no_context'].append(original + local)
        modes['no_local'].append(original + update)
        modes['self_only'].append(original - gamma * op.out(own) + local)
        modes['retrieval_only'].append(original + gamma * op.out(read) + local)
        modes['uniform_attention'].append(original + gamma * op.out(merge_heads(uniform @ values,h,w)-own)+local)
        modes['alternate_geometry'].append(original + gamma * op.out(merge_heads(other @ values,h,w)-own)+local)
        if record:
            label = F.adaptive_avg_pool2d(target, (h,w))
            labels = feature_regions(label, g).squeeze(-1)
            weights, _ = compressed[s]
            squared, _ = assignments(guides[s], g)
            squared = squared.square() / squared.square().sum(-1, keepdim=True)
            sparse = (labels.sum(-1) > 0) & (labels.mean(-1) <= .125)
            positive_purity = (weights[:, :, 0] * labels).sum(-1)
            square_purity = (squared[:, :, 0] * labels).sum(-1)
            measurements.append(dict(stage=s+1, retrieved_rms=rms(read), self_rms=rms(own),
                retrieved_self_cos=float(F.cosine_similarity(read.flatten(1), own.flatten(1)).mean()),
                retrieved_spatial_rms=rms(read-read.mean((-2,-1),keepdim=True)),
                update_ratio=rms(update)/max(rms(original),1e-8),
                original_vs_uniform_read_rms=rms(merge_heads((attention-uniform)@values,h,w)),
                geometric_read_difference_rms=rms(merge_heads((attention-other)@values,h,w)),
                geometric_top1_agreement=float((attention.argmax(-1)==other.argmax(-1)).float().mean()),
                guide_brier=float((guides[s]-label).square().mean()),
                sparse_regions=int(sparse.sum()),
                positive_purity_sum=float(positive_purity[sparse].sum()),
                squared_positive_purity_sum=float(square_purity[sparse].sum()),
                region_mean_positive_sum=float(labels.mean(-1)[sparse].sum())))
    return modes, measurements


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = False
    report = dict(split='val', selection='128 equally spaced list indices, fixed before inference',
                  source='CASA-Pyramid/Run4/source_code.tar.gz',
                  scope='Same-checkpoint interventions; not retrained ablations or independent test results', results={})
    started = time.time()
    for dataset in ('LEVIR-CD-256','SYSU-CD-256','WHU-CD-256','CDD-CD-256'):
        full = make_loader(Path('/share_datasets/CD')/dataset, 'val', 8, 1)
        indices = torch.linspace(0,len(full.dataset)-1,128).long().tolist()
        loader = torch.utils.data.DataLoader(torch.utils.data.Subset(full.dataset, indices),
                                            batch_size=8, num_workers=1)
        for mode, variant in (('none','M0_CTRL'),('qkv','M2_CA_QKV'),('hyperbolic','M1_HYPER_QKV')):
            model = CASANet(module=mode).cuda().eval()
            checkpoint = Path('/share_datasets/yqwang/checkpoints/CASA-CD/CASA-Pyramid/Run4')/variant/dataset/'best_model.pth'
            model.load_state_dict(torch.load(checkpoint,map_location='cuda',weights_only=True))
            counters, spatial, measurements = {}, {}, []
            for batch, (images, target) in enumerate(loader):
                images, target = images.cuda().float(), target.cuda().float()
                tar = model.tar(model.encoder(images[:,:3])[1:],model.encoder(images[:,3:])[1:])
                if mode == 'none':
                    variants, measures = {'baseline':tar}, []
                else:
                    variants, measures = contexts(model.context,tar,target,batch<4)
                    if batch==0:
                        reference,_=model.context(tar)
                        assert all(torch.allclose(a,b,atol=2e-5,rtol=1e-5) for a,b in zip(reference,variants['baseline']))
                measurements.extend(measures)
                for condition, features in variants.items():
                    decoded = model.decoder.forward_pyramid(features)[0]
                    pred = F.interpolate(model.head(decoded),target.shape[-2:],mode='bilinear',align_corners=False)>0
                    truth = target.bool()
                    counts = torch.stack(((pred&truth).sum(),(pred&~truth).sum(),(~pred&truth).sum()))
                    if condition not in counters:
                        counters[condition] = torch.zeros_like(counts)
                        spatial[condition] = SpatialMetrics(target.device)
                    counters[condition] += counts
                    spatial[condition].update(pred,target)
            results = {}
            for condition,counts in counters.items():
                tp,fp,fn=counts.cpu().tolist()
                results[condition]=dict(F1=2*tp/(2*tp+fp+fn),precision=tp/(tp+fp),recall=tp/(tp+fn),
                                        spatial=spatial[condition].scores())
            report['results'][f'{dataset}/{mode}']=dict(indices=indices,metrics=results,measurements=measurements)
            report['elapsed_seconds']=time.time()-started
            (OUT/'run4_validation_interventions.json').write_text(json.dumps(report,indent=2))
            print(json.dumps(dict(dataset=dataset,module=mode,F1={k:round(v['F1']*100,4) for k,v in results.items()})),flush=True)
            del model
            torch.cuda.empty_cache()
    print('ANALYSIS_COMPLETE',flush=True)


if __name__=='__main__':
    main()
