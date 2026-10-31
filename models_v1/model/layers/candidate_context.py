"""Matched H/E retrieval, temporal Haar evidence and decoder-conditioned routing."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from model.layers.lorentz_ops import acosh1p_squared
from model.layers.pyramid_context import channel_norm, feature_regions, pool_features, rms
from model.layers.moe_context import haar_split, haar_merge

MODEL_VARIANTS = ('none', 'static', 'moe')
EXPERTS = ('hyperbolic', 'qkv', 'frequency', 'tar')


class TemporalStem(nn.Module):
    def __init__(self):
        super().__init__()
        self.project = nn.ModuleList(nn.Conv2d(c, 64, 1, bias=False) for c in (24, 32, 96, 320))
        self.fuse = nn.ModuleList(nn.Sequential(nn.Conv2d(128, 64, 1, bias=False),
                                               nn.BatchNorm2d(64), nn.SiLU()) for _ in range(4))

    def forward(self, a, b):
        pa, pb = [p(x) for p, x in zip(self.project, a)], [p(x) for p, x in zip(self.project, b)]
        x = [f(torch.cat(((v-u).abs(), (v+u)/2), 1)) for f, u, v in zip(self.fuse, pa, pb)]
        return x, pa, pb

    def switch_to_deploy(self):
        from torch.nn.utils.fusion import fuse_conv_bn_eval
        for f in self.fuse:
            f[0] = fuse_conv_bn_eval(f[0], f[1])
            f[1] = nn.Identity()


class TemporalWavelet(nn.Module):
    def __init__(self):
        super().__init__()
        self.high = nn.ModuleList(nn.Sequential(nn.Conv2d(192, 192, 1, bias=False), nn.SiLU(),
            nn.Conv2d(192, 192, 3, padding=1, groups=192, bias=False)) for _ in range(2))
        self.out = nn.ModuleList(nn.Conv2d(64, 64, 1, bias=False) for _ in range(2))
        self.low = nn.ModuleList(nn.Sequential(nn.Conv2d(64, 64, 1, bias=False), nn.SiLU(),
            nn.Conv2d(64, 64, 3, padding=1, groups=64, bias=False)) for _ in range(2))
        self.eta = nn.Parameter(torch.zeros(4, 64))

    def forward(self, pa, pb):
        # Linearity gives exactly the signed difference of the two DWTs.
        bands = [haar_split(b-a) for a, b in zip(pa[:3], pb[:3])]
        outputs = []
        for s in range(2):
            low, high = bands[s]
            packed = high.transpose(1, 2).flatten(1, 2)
            update = packed + self.high[s](packed)
            high = update.reshape(update.shape[0], 3, 64, *low.shape[-2:]).transpose(1, 2)
            outputs.append(self.out[s](haar_merge(torch.zeros_like(low), high)))
        outputs += [layer(bands[s+1][0].abs()/2) for s, layer in enumerate(self.low)]
        return [x * self.eta[s][None, :, None, None] for s, x in enumerate(outputs)]


class PairedRetrieval(nn.Module):
    def __init__(self):
        super().__init__()
        self.qk = nn.Conv2d(64, 32, 1, bias=False)
        self.v = nn.Conv2d(64, 32, 1, bias=False)
        self.out = nn.Conv2d(32, 64, 1, bias=False)
        self.scale_bias = nn.Parameter(torch.zeros(4, 4))
        self.raw_temperature = nn.Parameter(torch.full((2,), math.log(math.expm1(.5))))
        self.gamma = nn.Parameter(torch.zeros(4, 64))
        self.counts = (128, 32, 32, 32)

    def prepare(self, prototype, support):
        return (F.linear(prototype, self.qk.weight[:, :, 0, 0])/math.sqrt(32),
                F.linear(prototype, self.v.weight[:, :, 0, 0]), support)

    def forward(self, x, bank, scale, route_weights, diagnose=False, separate=False):
        """Return weighted H/E update, optional individual residuals and diagnostics."""
        key, value, support = bank
        query = self.qk(x).flatten(2).transpose(1, 2)/math.sqrt(32)
        q2, k2 = query.square().sum(-1, keepdim=True), key.square().sum(-1).unsqueeze(1)
        gram = query @ key.transpose(1, 2)
        euclidean = (q2+k2-2*gram).clamp_min(0)
        time_diff = (1+q2).sqrt() - (1+k2).sqrt()
        hyperbolic = acosh1p_squared((.5*(euclidean-time_diff.square())).clamp_min(0))
        bias = torch.cat([self.scale_bias[scale, s].expand(n)-math.log(n)
                          for s, n in enumerate(self.counts)])
        temp = F.softplus(self.raw_temperature)
        attentions, stats = [], {}
        for i, distance in enumerate((hyperbolic, euclidean)):
            scaled = distance/(2*temp[i].square())
            attention = (-scaled + support.log()[:, None] + bias).softmax(-1)
            attentions.append(attention)
            if diagnose:
                with torch.no_grad():
                    stats[EXPERTS[i]] = dict(temperature=float(temp[i]), scaled_distance_mean=float(scaled.mean()),
                        scaled_distance_std=float(scaled.std(unbiased=False)),
                        scale_mass=[float(a.sum(-1).mean()) for a in attention.split(self.counts, -1)],
                        entropy=float(-(attention*attention.clamp_min(1e-9).log()).sum(-1).mean()/math.log(224)))
        local_value = self.v(x)
        gamma = self.gamma[scale][None, :, None, None]
        h, e = route_weights.split(1, 1)
        residuals = None
        if separate or diagnose:
            residuals = []
            for i, attention in enumerate(attentions):
                retrieved = (attention @ value).transpose(1, 2).reshape_as(local_value)
                residuals.append(gamma * self.out(retrieved-local_value))
                if diagnose:
                    with torch.no_grad():
                        stats[EXPERTS[i]]['retrieved_spatial_std'] = float(
                            retrieved.flatten(2).std(-1, unbiased=False).mean())
            update = h*residuals[0] + e*residuals[1]
        else:
            mixed_attention = (h.flatten(2).transpose(1, 2)*attentions[0]
                               + e.flatten(2).transpose(1, 2)*attentions[1])
            retrieved = (mixed_attention @ value).transpose(1, 2).reshape_as(local_value)
            update = gamma * self.out(retrieved-(h+e)*local_value)
        if diagnose:
            with torch.no_grad():
                radius = query.norm(dim=-1).asinh()
                stats.update(radius_mean=float(radius.mean()), radius_cv=float(radius.std(unbiased=False)/radius.mean().clamp_min(1e-8)),
                             attention_he_tv=float((attentions[0]-attentions[1]).abs().sum(-1).mean()/2),
                             residual_he_cosine=float(F.cosine_similarity(residuals[0].flatten(1), residuals[1].flatten(1)).mean()))
        return update, residuals, stats


class CandidateContext(nn.Module):
    geometry = 'paired_lorentz_euclidean'
    route_names = EXPERTS
    grid_sizes = (8, 4, 4, 4)

    def __init__(self, variant):
        super().__init__()
        self.variant = variant
        self.stem = TemporalStem()
        self.norm = nn.LayerNorm(64)
        self.guide = nn.Conv2d(64, 1, 1)
        if variant != 'none':
            self.retrieval = PairedRetrieval()
            self.wavelet = TemporalWavelet()
        if variant == 'static':
            self.static_logits = nn.Parameter(torch.zeros(4, 4))
        if variant == 'moe':
            self.router = nn.ModuleList(nn.Sequential(nn.Conv2d(128 if s == 3 else 192, 16, 1), nn.SiLU(),
                nn.Conv2d(16, 16, 3, padding=1, groups=16), nn.SiLU(), nn.Conv2d(16, 4, 1)) for s in range(4))
            for layer in self.router:
                nn.init.zeros_(layer[-1].weight)
                nn.init.zeros_(layer[-1].bias)
        self.collect_diag = False
        self.capture = False
        self.last_diag = {}
        self.captured = {}

    def prepare(self, a, b):
        x, pa, pb = self.stem(a, b)
        normalized = [channel_norm(self.norm, f) for f in x]
        guide = [self.guide(f) for f in normalized]
        bank, waves = None, None
        if self.variant != 'none':
            prototypes, supports = [], []
            for f, g, grid in zip(normalized, guide, self.grid_sizes):
                p = feature_regions(g.sigmoid(), grid).squeeze(-1)
                membership = torch.stack((p, 1-p), 2)
                weights = (membership+1e-6).square()
                weights = weights/weights.sum(-1, keepdim=True)
                prototypes.append(pool_features(f, weights, grid))
                supports.append((weights*membership).sum(-1).flatten(1).clamp_min(1e-6))
            bank = self.retrieval.prepare(torch.cat(prototypes, 1), torch.cat(supports, 1))
            waves = self.wavelet(pa, pb)
        if self.capture:
            self.captured = {}
        return x, normalized, guide, bank, waves

    def weights(self, scale, x, tar, coarse):
        if self.variant == 'none':
            g = x.new_zeros(x.shape[0], 4, *x.shape[-2:]); g[:, 3] = 1
            return g
        if self.variant == 'static':
            return self.static_logits[scale].softmax(0)[None, :, None, None].expand(x.shape[0], -1, *x.shape[-2:])
        norm = lambda f: F.layer_norm(f.permute(0, 2, 3, 1), (64,)).permute(0, 3, 1, 2)
        inputs = [norm(x), norm(tar)]
        if coarse is not None:
            inputs.append(norm(coarse))
        inputs = torch.cat(inputs, 1)
        return self.router[scale](inputs).softmax(1)

    def mix(self, scale, x, normalized, tar, coarse, bank, waves):
        if self.variant == 'none':
            return tar
        g = self.weights(scale, x, tar, coarse)
        retrieval_update, residuals, geometry = self.retrieval(
            normalized, bank, scale, g[:, :2], diagnose=self.collect_diag, separate=self.capture)
        rw = waves[scale]
        h, e, w, t = g.split(1, 1)
        z = (h+e+w)*x + t*tar + retrieval_update + w*rw
        if self.capture:
            rh, re = residuals
            self.captured[scale] = dict(routes=g, residuals=(rh, re, rw), tar=tar, x=x)
        if self.collect_diag:
            with torch.no_grad():
                rh, re = residuals
                deltas = (rh, re, rw, tar-x)
                base = rms(x).clamp_min(1e-8)
                self.last_diag[f'stage{scale+1}'] = dict(finite=bool(torch.isfinite(z).all()), contexts=224,
                    route_mass=dict(zip(EXPERTS, g.mean((0, 2, 3)).tolist())),
                    route_variance=dict(zip(EXPERTS, g.flatten(2).var(-1, unbiased=False).mean(0).tolist())),
                    route_entropy=float(-(g*g.clamp_min(1e-9).log()).sum(1).mean()/math.log(4)),
                    expert_rms={n: float(rms(d)) for n, d in zip(EXPERTS, deltas)},
                    effective_rms={n: float(rms(g[:, i:i+1]*d)) for i, (n, d) in enumerate(zip(EXPERTS, deltas))},
                    residual_to_feature_rms=float(rms(z-x)/base), feature_rms=float(base), tar_rms=float(rms(tar)),
                    wavelet_ll_rms=float(rms(haar_split(rw)[0])) if scale < 2 else None,
                    routed_wavelet_ll_rms=float(rms(haar_split(w*rw)[0])) if scale < 2 else None,
                    geometry=geometry)
        return z

    def metadata(self):
        return dict(variant=self.variant, active_experts=list(EXPERTS) if self.variant != 'none' else ['tar'],
            routing_mode={'none': 'tar_only', 'static': 'learned_per_scale_constant', 'moe': 'coarse_to_fine_spatial'}[self.variant],
            execution='dense', context_grid_sizes=list(self.grid_sizes), context_prototypes_per_scale=[128,32,32,32],
            joint_contexts=224, heads=1, qkv_dim=32, value_geometry='euclidean',
            key_mode='shared_pooled_content', hyperbolic_radius='asinh(projection_norm)',
            shared_he=['qk','v','out','gamma','scale_bias','guide','norm','prototypes'],
            frequency_transform='signed_temporal_haar_H1_H2_L2_L3', guide_input='common_X',
            router_inputs=['LN(X)', 'LN(TAR)', 'LN(coarse_D) at S1-S3'] if self.variant == 'moe' else [],
            retrieval_readout='route_weighted_shared_AV_out' if self.variant != 'none' else 'inactive',
            context_gamma_shape=[4,64], wavelet_gamma_shape=[4,64], initializer_seed=2333)
