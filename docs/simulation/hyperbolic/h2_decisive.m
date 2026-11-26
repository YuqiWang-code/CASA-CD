%% H2 -- Content gain vs geometry gain: a decisive factorial experiment
% History motivating this test:
%   * Run14 (12 H/E pairs, mean -0.167pp) + Run15 (+0.334 / -0.009pp):
%     swapping the Euclidean distance for the geodesic one changes nothing.
%   * Run1: swapping the KV *content* from the prototype to the difference
%     prototype |P-Q| gave +0.72pp (WHU) / +0.93pp (SYSU), single seed.
% So the question is not "is hyperbolic better" but "is the gain content or
% geometry". Design: 2x2 factorial {content: prototype, difference} x
% {form: Euclidean, geodesic} on identical synthetic scenes, plus the scalar
% control the captain asked for:
%   r-only arm  r = asinh(||delta||)   (radius = a monotone function of ||delta||)
% Protocol: prototypes are fitted on a random half of each cell and every
% score/support is evaluated on the held-out half, so the local prior is not
% a label leak.
here = fileparts(mfilename('fullpath')); if isempty(here), here = pwd; end
addpath(here); H = hyper_setup();
figdir = fullfile(here,'figures'); if ~exist(figdir,'dir'), mkdir(figdir); end
% sanity: the elementwise geodesic formula used for large point sets must
% agree with the toolkit's cross-distance routine
Pp = H.pball(randn(50,32)); Qq = H.pball(randn(50,32));
fprintf('H2 check |gdiag - geo_cdist(diag)| = %.2e\n', ...
        max(abs(gdiag(Pp,Qq) - diag(H.cdist(Pp,Qq)))));

d  = 32;                 % feature channels (model: 64; halved for speed)
Ns = [64 32 16 8];       % Q grid per scale (model: 5440 = 4096+1024+256+64)
Ks = [8 4 4 4];          % KV grid per scale (model: 224 = 2*128+2*32+2*32+2*32)
seeds = 1:3;
imbs  = [0.01 0.03 0.05];
regimes = {'coherent','heterogeneous'};
arms = {'euc-den','hyp-den','delta-euc','delta-hyp','r-only','geoDist-proto', ...
        'eucDist-proto','diff-euc','diff-geo','geo+sup','euc+sup','hyp-den+r'};
na = numel(arms);

AUC = nan(numel(regimes), numel(imbs), numel(seeds), na);
AP  = AUC;
for r = 1:numel(regimes)
    for q = 1:numel(imbs)
        for s = seeds
            [Z1,Z2,Y] = gen_scene(d, Ns, imbs(q), regimes{r}, s);
            F = arm_features(H, Z1, Z2, Y, Ns, Ks);
            y = F.y;
            for a = 1:na
                rng(9000 + 100*s + a);
                yh = probe_cv(F.(strrep(strrep(arms{a},'-','_'),'+','_')), y, 1e-2, 5);
                AUC(r,q,s,a) = H.auc(yh, y);
                AP(r,q,s,a)  = pr_auc(yh, y);
            end
            if r==1 && q==2 && s==1, S1 = F.diag; end %#ok<SAGROW>
        end
    end
end
m = squeeze(mean(AUC,3));   sd = squeeze(std(AUC,0,3));  mA = squeeze(mean(AP,3));
fprintf('\nH2 ===== AUC (5-fold CV logistic probe, held-out half), mean of %d seeds =====\n', numel(seeds));
fprintf('%-16s | %-24s | %-24s\n','arm','coherent 1%/3%/5%','heterogeneous 1%/3%/5%');
for a = 1:na
    fprintf('%-16s | %s| %s\n', arms{a}, ...
        sprintf('%6.3f %6.3f %6.3f  ', m(1,:,a)), sprintf('%6.3f %6.3f %6.3f  ', m(2,:,a)));
end
fprintf('\nH2 ===== mean AP (imbalance-aware), 3%% positives =====\n');
for a = 1:na
    fprintf('  %-16s coh %.3f   het %.3f\n', arms{a}, mA(1,2,a), mA(2,2,a));
end

% ---- paired contrasts on identical scenes ----
ci = @(a1,a2) squeeze(AUC(:,:,:,a1) - AUC(:,:,:,a2));
D = struct('hyp_vs_euc_den',ci(2,1), 'hyp_vs_euc_dist',ci(6,7), ...
           'content_diff_geo',ci(9,6), 'content_diff_euc',ci(8,7), ...
           'hyp_vs_scalar',ci(4,5), 'geo_vs_scalar',ci(6,5), ...
           'sup_geo',ci(10,6), 'sup_euc',ci(11,7), 'hypden_vs_scalar',ci(12,5));
fprintf('\nH2 ===== paired differences (mean +- sd over %d scenes, pp) =====\n', ...
        numel(AUC(:,:,:,1)));
fn = fieldnames(D);
for i = 1:numel(fn)
    v = D.(fn{i})*100;
    fprintf('  %-18s %+0.2f +- %0.2f pp   (coh %+0.2f | het %+0.2f)\n', fn{i}, ...
        mean(v(:)), std(v(:)), mean(mean(v(1,:,:))), mean(mean(v(2,:,:))));
end
[h1,p1] = ttest(D.hyp_vs_euc_dist(:));
[h2,p2] = ttest(D.content_diff_geo(:));
[h3,p3] = ttest(D.hyp_vs_scalar(:));
[h4,p4] = ttest(D.content_diff_euc(:));
[h5,p5] = ttest(D.hyp_vs_euc_den(:));
fprintf('\nH2 decisive tests (paired t, n=%d scenes):\n', numel(D.hyp_vs_euc_dist));
fprintf('  geodesic vs euclid, same content : %+0.2f pp, p=%.2g -> %s\n', ...
        mean(D.hyp_vs_euc_dist(:))*100, p1, tern(h1,'SIG','not sig'));
fprintf('  difference vs prototype content  : %+0.2f pp, p=%.2g -> %s\n', ...
        mean(D.content_diff_geo(:))*100, p2, tern(h2,'SIG','not sig'));
fprintf('  (content effect, euclid form)    : %+0.2f pp, p=%.2g -> %s\n', ...
        mean(D.content_diff_euc(:))*100, p4, tern(h4,'SIG','not sig'));
fprintf('  full hyperbolic vs full euclid   : %+0.2f pp, p=%.2g -> %s\n', ...
        mean(D.hyp_vs_euc_den(:))*100, p5, tern(h5,'SIG','not sig'));
fprintf('  full hyperbolic vs scalar r only : %+0.2f pp, p=%.2g -> %s\n', ...
        mean(D.hyp_vs_scalar(:))*100, p3, tern(h3,'SIG','not sig'));
fprintf('  [hyp-den+r] vs [r only]          : %+0.2f pp\n', mean(D.hypden_vs_scalar(:))*100);
fprintf('  support prior on top of geodesic : %+0.2f pp\n', mean(D.sup_geo(:))*100);

% ---- why the geodesic loses resolution: saturation of the direct lift ----
fprintf('\nH2 ===== saturation diagnostic (single scene, 3%%) =====\n');
fprintf('  median radius of pi(Q)          : %.3f  (0 = centre, 1 = horizon)\n', S1.medR);
fprintf('  median geodesic d(Q,proto)      : %.3f ; 99th pct %.3f ; asinh(||z||) at same points %.3f\n', ...
        S1.medG, S1.p99G, S1.medAsinh);
fprintf('  fraction of positions at r>0.95 : %.3f  (metric resolution collapses there)\n', S1.frac95);
fprintf('  Spearman(euclid contrast, geo contrast) = %.3f (1 = same ranking)\n', S1.spEG);
fprintf('  Spearman(geo contrast, radius only)     = %.3f\n', S1.spGR);
fprintf('  Fisher ratio (scale-1 column): eucDist %.3f | geoDist %.3f\n', S1.fishE, S1.fishG);

% ---- H2c: dimension-matched single-feature AUC (1-D, no probe), per scale.
% This is the clean geometry test: same content, same dimensionality, only the
% distance form differs. The 128-d / 4-d probe arms above mix scales.
fprintf('\nH2c ==== 1-D prototype contrast AUC per scale (3%% pos, held-out half) ====\n');
AE = zeros(1,4); AG = zeros(1,4); ns = 0; PR = zeros(1,4);
for r2 = 1:2
    for s = seeds
        [Za,Zb,Ya] = gen_scene(d, Ns, 0.03, regimes{r2}, s);
        [aeV, agV] = single_auc(H, Za, Zb, Ya, Ns, Ks, 0, s);
        AE = AE + aeV; AG = AG + agV; ns = ns + 1;
        for s2 = 1:4, PR(s2) = PR(s2) + mean(double(Ya{s2}(:)) > 0.5); end
    end
end
AE = AE/ns; AG = AG/ns; PR = PR/ns;
fprintf('  positive-cell rate: %.3f %.3f %.3f %.3f  (mean-pooled cell labels)\n', PR);
fprintf('  scale (grid)     :  8x8    4x4    2x2    1x1\n');
fprintf('  Euclidean   AUC  : %6.3f %6.3f %6.3f %6.3f\n', AE);
fprintf('  Geodesic    AUC  : %6.3f %6.3f %6.3f %6.3f\n', AG);
fprintf('  geo-euc (pp)     : %+6.2f %+6.2f %+6.2f %+6.2f  | mean %+0.2f pp\n', ...
        (AG-AE)*100, mean(AG-AE)*100);
% ---- H2b: when does the geodesic form actually help? prototype noise sweep
% The two distance forms differ only in how they weight the norm of the
% prototype. With well estimated prototypes the Euclidean contrast is fine;
% with noisy prototypes the bounded geodesic contrast is more robust.
fprintf('\nH2b ==== paired geo-euc gap vs prototype noise (3%% pos, scale 1) ====\n');
sg  = [0 0.05 0.10 0.20 0.40 0.80];
gap = zeros(numel(sg), 2, numel(seeds));
for r2 = 1:2
    for s = seeds
        [Za,Zb,Ya] = gen_scene(d, Ns, 0.03, regimes{r2}, s);
        for k = 1:numel(sg)
            [ae, ag] = sweep_noise(H, Za{1}, Zb{1}, double(Ya{1}), sg(k), s, Ns(1), Ks(1));
            gap(k,r2,s) = (ag-ae)*100;
        end
    end
end
for k = 1:numel(sg)
    g = squeeze(gap(k,:,:));
    fprintf('  sigma_proto=%.2f : geo-euc = %+0.2f pp (coh %+0.2f | het %+0.2f)\n', ...
        sg(k), mean(g(:)), mean(g(1,:)), mean(g(2,:)));
end

% ------------------------------------------------------------------ figures --
f = figure('Position',[60 60 1240 420]);
[Z1,Z2,Y] = gen_scene(d, Ns, 0.03, 'heterogeneous', 1);
Dlt = Z2{1}-Z1{1}; yy = Y{1};
subplot(1,3,1);
P = H.pball(Dlt); th = linspace(0,2*pi,300);
plot(cos(th),sin(th),'k-','LineWidth',0.8); hold on;
plot(P(yy==0,1),P(yy==0,2),'.','Color',[.72 .72 .72],'MarkerSize',6);
plot(P(yy==1,1),P(yy==1,2),'.','Color',[.85 .1 .1],'MarkerSize',9);
axis equal off; title('Poincare disk of \Delta (red = changed, 3%)');
subplot(1,3,2);
rr = sqrt(sum(Dlt.^2,2));
e = linspace(0,max(rr)*1.02,40);
histogram(rr(yy==0),e,'FaceColor',[.7 .7 .7],'EdgeColor','none'); hold on;
histogram(rr(yy==1),e,'FaceColor',[.85 .1 .1],'EdgeColor','none');
xlabel('||\Delta||'); ylabel('count');
title(sprintf('change strength: hyp radius %.2f vs %.2f', ...
    mean(asinh(rr(yy==1))), mean(asinh(rr(yy==0)))));
subplot(1,3,3);
bar(squeeze(m(:,2,:)).'); set(gca,'XTick',1:na,'XTickLabel',arms,'XTickLabelRotation',70,'FontSize',7);
ylabel('AUC (3% pos)'); grid on; ylim([0.5 1]);
title('content arms beat geometry arms');
H.savefig(figdir,'fig_h2_arms');

f = figure('Position',[60 60 900 340]);
subplot(1,2,1);
v = cat(2, squeeze(AUC(:,2,:,6)), squeeze(AUC(:,2,:,7)), ...
           squeeze(AUC(:,2,:,2)), squeeze(AUC(:,2,:,1)));
v = reshape(v,[],4);
boxplot(v, {'geoDist','eucDist','hyp-den','euc-den'}); grid on;
ylabel('AUC at 3% pos'); title('geometry main effect \approx 0');
subplot(1,2,2);
v2 = cat(2, squeeze(AUC(:,2,:,9)), squeeze(AUC(:,2,:,8)), ...
            squeeze(AUC(:,2,:,6)), squeeze(AUC(:,2,:,5)));
v2 = reshape(v2,[],4);
boxplot(v2, {'diff-geo','diff-euc','proto-geo','r-only'}); grid on;
ylabel('AUC at 3% pos'); title('content effect >> geometry effect');
H.savefig(figdir,'fig_h2_effect');

res_h2 = struct('AUC',AUC,'AP',AP,'arms',{arms},'m',m,'sd',sd,'regimes',{regimes}, ...
  'imbs',imbs,'D',D,'p',[p1 p2 p3 p4 p5]);
save(fullfile(here,'res_h2.mat'),'res_h2');
fprintf('H2 done.\n');

% =============================================================== generators ==
function [Z1,Z2,Y] = gen_scene(d, Ns, imb, regime, seed)
rng(1000 + seed + round(1000*imb) + 77*strcmp(regime,'heterogeneous'));
N = Ns(1);
npos = round(imb*N^2); area = 0; rects = {}; dirs = [];
while area < npos
    w = 2 + floor(exp(rand*log(22)));        % 2..24 px: mixed object scales
    h = 2 + floor(exp(rand*log(22)));
    if w >= N || h >= N, continue; end
    rects{end+1} = [randi([1 N-w]), randi([1 N-h]), w, h]; %#ok<AGROW>
    area = area + w*h;
    v = randn(d,1); dirs(:,end+1) = v/norm(v); %#ok<AGROW>
end
if strcmp(regime,'coherent')
    dirs = dirs(:,1) + 0.25*randn(d,size(dirs,2));
    dirs = dirs./vecnorm(dirs);
end
u1 = smoothnoise(N); u1 = u1/std(u1(:));
B  = randn(d,d)/sqrt(d);
Z1f = (u1(:)*ones(1,d)).*randn(1,d) + 0.6*randn(N*N,d);
Z1f = Z1f*(0.9*eye(d) + 0.1*B);
Dl = zeros(N*N,d); mask = false(N*N,1);
for k = 1:numel(rects)
    rr = rects{k}; sub = false(N,N);
    sub(rr(1):(rr(1)+rr(3)-1), rr(2):(rr(2)+rr(4)-1)) = true; sub = sub(:);
    wgt = 0.35 + 0.65*rand(nnz(sub),1);       % sub-cell magnitude structure
    Dl(sub,:) = Dl(sub,:) + wgt*dirs(:,k)';
    mask = mask | sub;
end
Z1f = Z1f + 0.55*randn(N*N,d);
Z2f = Z1f + Dl + 0.55*randn(N*N,d);           % unchanged pixels get noise only
Z1 = cell(1,4); Z2 = cell(1,4); Y = cell(1,4);
Z1{1} = Z1f; Z2{1} = Z2f; Y{1} = mask;
for s = 2:4
    Z1{s} = pool2(Z1{s-1}, Ns(s-1)); Z2{s} = pool2(Z2{s-1}, Ns(s-1));
    Y{s}  = pool2(double(Y{s-1}), Ns(s-1)) > 0.5;
end
end

function B = pool2(A, N)
[~,d] = size(A); n = N/2;
A = reshape(A, N, N, d);
B = (A(1:2:end,1:2:end,:) + A(2:2:end,1:2:end,:) + ...
     A(1:2:end,2:2:end,:) + A(2:2:end,2:2:end,:))/4;
B = reshape(B, n*n, d);
end

function u = smoothnoise(N)
k = exp(-((-(4:4)).^2)'/(2*2^2)); K = k*k'; K = K/sum(K(:));
u = conv2(randn(N), K, 'same');
end

function F = arm_features(H, Z1, Z2, Y, Ns, Ks)
V = zeros(sum(Ns.^2),4); euc = V; geo = V; r = V; dm = V; dmg = V; sp = V;
eDen = []; hDen = []; dEuc = []; dHyp = []; yall = zeros(sum(Ns.^2),1);
kTe = false(sum(Ns.^2),1); off = 0;
for s = 1:4
    n = Ns(s); K = Ks(s); cs = n/K;
    Z1s = Z1{s}; Z2s = Z2{s}; Ys = double(Y{s}); Dlt = Z2s - Z1s;
    tr = rand(n*n,1) < 0.5;                   % fit on tr, score on ~tr
    g1 = mean(Z2s(Ys==1 & tr,:),1); if any(isnan(g1)), g1 = mean(Z2s,1); end
    g0 = mean(Z2s(Ys==0 & tr,:),1);
    C = repmat(g1,K*K,1); Bk = repmat(g0,K*K,1);
    sup = zeros(K*K,1); mg = zeros(K*K,size(Z2s,2));
    for ci = 1:K
        for cj = 1:K
            k = (ci-1)*K + cj;
            idx = resh((ci-1)*cs + (1:cs), (cj-1)*cs + (1:cs), n);
            itr = idx(tr(idx)); yy = Ys(itr);
            if sum(yy==1) >= 3, C(k,:)  = mean(Z2s(itr(yy==1),:),1); end
            if sum(yy==0) >= 3, Bk(k,:) = mean(Z2s(itr(yy==0),:),1); end
            if sum(yy==1) >= 3, mg(k,:) = mean(Dlt(itr(yy==1),:),1); end
            % support prior: feature-only soft assignment on the fit half
            e1 = sum((Z2s(itr,:)-C(k,:)).^2,2); e0 = sum((Z2s(itr,:)-Bk(k,:)).^2,2);
            w  = 1./(1+exp((e1-e0)/2));
            sup(k) = mean(w);
        end
    end
    mgn = mg ./ max(vecnorm(mg,2,2),1e-9);
    kidx = zeros(n*n,1);
    for i = 1:n
        for j = 1:n
            kidx((i-1)*n+j) = (ceil(i/cs)-1)*K + ceil(j/cs);
        end
    end
    q = Z2s; I = off + (1:n*n);
    euc(I,s) = sum((q-Bk(kidx,:)).^2,2) - sum((q-C(kidx,:)).^2,2);
    pq = H.pball(q);
    dCall = H.cdist(pq, H.pball(C)); dC = dCall(sub2ind(size(dCall),(1:n*n)', kidx));
    dBall = H.cdist(pq, H.pball(Bk)); dB = dBall(sub2ind(size(dBall),(1:n*n)', kidx));
    geo(I,s) = dB.^2 - dC.^2;
    r(I,s)   = asinh(vecnorm(Dlt,2,2));
    Mq = mgn(kidx,:);
    dm(I,s)  = sum(Dlt.*Mq,2);
    pD = H.pball(Dlt);
    dmg(I,s) = -gdiag(pD, H.pball(Mq));
    sp(I,s)  = log(sup(kidx)+1e-3);
    eDen = [eDen; q]; hDen = [hDen; pq]; %#ok<AGROW>
    dEuc = [dEuc; Dlt]; dHyp = [dHyp; pD]; %#ok<AGROW>
    yall(I) = double(Ys(:) > 0.5);
    kTe(I)  = ~tr;
    if s == 1
        diag_s = struct('medR',median(sqrt(sum(pq.^2,2))), ...
                      'medG',median(dC), 'p99G',prctile(dC,99), ...
                      'medAsinh',median(asinh(vecnorm(q,2,2))), ...
                      'frac95',mean(sqrt(sum(pq.^2,2))>0.95), ...
                      'spEG',sp_corr(euc(I,s), geo(I,s)), ...
                      'spGR',sp_corr(geo(I,s), r(I,s)), ...
                      'fishE',H.fisher(euc(I,s), double(Ys(:)>0.5)), ...
                      'fishG',H.fisher(geo(I,s), double(Ys(:)>0.5)));
    end
    off = off + n*n;
end
F = struct();
ii = find(kTe);
F.euc_den = eDen(ii,:);     F.hyp_den = hDen(ii,:);
F.delta_euc = dEuc(ii,:);   F.delta_hyp = dHyp(ii,:);
F.r_only = r(ii,:);
F.geoDist_proto = geo(ii,:);  F.eucDist_proto = euc(ii,:);
F.diff_euc = dm(ii,:);        F.diff_geo = dmg(ii,:);
F.geo_sup = [geo(ii,:), sp(ii,:)];  F.euc_sup = [euc(ii,:), sp(ii,:)];
F.hyp_den_r = [hDen(ii,:), r(ii,:)];
F.y = yall(ii);
F.diag = diag_s;
end

function [C, Bk, kidx, nn] = fit_protos(Z2s, Ys, ngrid, K, sg, seed)
% cell prototypes (change C / background Bk) fitted on a random half of each
% cell; optional multiplicative prototype noise sg (in units of feature std)
nn = ngrid*ngrid; cs = ngrid/K;
rng(500+seed); tr = rand(nn,1) < 0.5;
g1 = mean(Z2s(Ys==1 & tr,:),1); if any(isnan(g1)), g1 = mean(Z2s,1); end
g0 = mean(Z2s(Ys==0 & tr,:),1);
C = repmat(g1,K*K,1); Bk = repmat(g0,K*K,1);
for ci = 1:K
    for cj = 1:K
        k = (ci-1)*K + cj;
        idx = resh((ci-1)*cs + (1:cs), (cj-1)*cs + (1:cs), ngrid);
        itr = idx(tr(idx)); yy = Ys(itr);
        if sum(yy==1) >= 3, C(k,:)  = mean(Z2s(itr(yy==1),:),1); end
        if sum(yy==0) >= 3, Bk(k,:) = mean(Z2s(itr(yy==0),:),1); end
    end
end
if sg > 0
    s0 = sg*std(Z2s(:));
    C  = C  + s0*randn(size(C));
    Bk = Bk + s0*randn(size(Bk));
end
kidx = zeros(nn,1);
for i = 1:ngrid
    for j = 1:ngrid, kidx((i-1)*ngrid+j) = (ceil(i/cs)-1)*K + ceil(j/cs); end
end
end

function [sE, sG, y] = proto_contrast(H, Z2s, C, Bk, kidx, mask)
% 1-D prototype contrasts, Euclidean vs geodesic, on the requested positions
tt = find(mask); kk = kidx(tt); q = Z2s(tt,:); y = double(0);
sE = sum((q-Bk(kk,:)).^2,2) - sum((q-C(kk,:)).^2,2);
pq = H.pball(q);
dC = H.cdist(pq, H.pball(C)); dC = dC(sub2ind(size(dC),(1:numel(tt))', kk));
dB = H.cdist(pq, H.pball(Bk)); dB = dB(sub2ind(size(dB),(1:numel(tt))', kk));
sG = dB.^2 - dC.^2;
end

function [aeV, agV] = single_auc(H, Z1, Z2, Y, Ns, Ks, sg, seed)
% dimension-matched (1-D, no probe) AUC of the two prototype contrasts,
% per scale, on the held-out half of each cell
aeV = zeros(1,4); agV = zeros(1,4);
for s = 1:4
    Ys = double(Y{s});
    [C, Bk, kidx, nn] = fit_protos(Z2{s}, Ys, Ns(s), Ks(s), sg, seed+s);
    rng(900+seed+s); te = rand(nn,1) < 0.5;
    [sE, sG, ~] = proto_contrast(H, Z2{s}, C, Bk, kidx, te);
    y = double(Ys(te) > 0.5);
    aeV(s) = H.auc(sE, y); agV(s) = H.auc(sG, y);
end
end

function [ae, ag] = sweep_noise(H, Z1s, Z2s, Ys, sg, seed, ngrid, K)
% AUC of the prototype contrast at scale 1 as a function of prototype noise
[C, Bk, kidx, nn] = fit_protos(Z2s, Ys, ngrid, K, sg, seed);
rng(900+seed); te = rand(nn,1) < 0.5;
[sE, sG, ~] = proto_contrast(H, Z2s, C, Bk, kidx, te);
y = double(Ys(te) > 0.5);
ae = H.auc(sE, y); ag = H.auc(sG, y);
end

function d = gdiag(P, Q)
% elementwise Poincare geodesic distance (same formula as H.cdist, evaluated
% only on the paired diagonal: avoids an N x N matrix for N = 4096)
a = 1 - sum(P.^2,2); b = 1 - sum(Q.^2,2); S = sum((P-Q).^2,2);
d = acosh(1 + 2*S./max(a.*b,1e-24));
end

function c = sp_corr(a,b)
ra = rank_of(a(:)); rb = rank_of(b(:));
cc = corrcoef(ra,rb); c = cc(1,2);
end

function r = rank_of(x)
[~,ix] = sort(x); r = zeros(size(x)); r(ix) = 1:numel(x);
end

function idx = resh(rows, cols, n)
[R,Cc] = meshgrid(rows, cols);
idx = (R(:)-1)*n + Cc(:);
end

function yh = probe_cv(X, y, lam, nfold)
n = size(X,1); y = y(:); yh = zeros(n,1); perm = randperm(n);
fe = ceil(n/nfold);
for f = 1:nfold
    te = perm((f-1)*fe+1 : min(f*fe,n)); tr = setdiff(perm, te);
    mu = mean(X(tr,:),1); sd = std(X(tr,:),0,1); sd(sd==0) = 1;
    Xtr = (X(tr,:)-mu)./sd; Xte = (X(te,:)-mu)./sd;
    Xtr(~isfinite(Xtr)) = 0; Xte(~isfinite(Xte)) = 0;
    w = fit_logreg(Xtr, y(tr), lam);
    yh(te) = [ones(numel(te),1), Xte]*w;
end
end

function w = fit_logreg(X, y, lam)
[n,p] = size(X); Xa = [ones(n,1) X]; w = zeros(p+1,1);
for it = 1:40
    z = Xa*w; mu = 1./(1+exp(-z)); s = mu.*(1-mu);
    g  = Xa'*(mu-y) + lam*[0; w(2:end)];
    Hs = (Xa.*s)'*Xa + lam*eye(p+1); Hs(1,1) = Hs(1,1) + 1e-8;
    st = Hs\g; w = w - st;
    if norm(st) < 1e-7*max(1,norm(w)), break; end
end
end

function a = pr_auc(score, y)
[~,ix] = sort(score,'descend'); y = double(y(ix)); y = y(:);
tp = cumsum(y); P = sum(y);
a = sum((tp./(1:numel(y))').*y)/max(P,1);
end

function s = tern(c,a,b), if c, s = a; else, s = b; end, end
