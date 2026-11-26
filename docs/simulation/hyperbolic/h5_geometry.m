% H5  When is the hyperbolic geometry actually *on*?
% Algebraic fact used throughout (verified numerically in Q1):
%   for two points at the SAME hyperbolic radius r,
%       cosh d_geo = 1 + d^2_euc/(2 a^2),  a = 1 - r^2
%   so d_geo is a strictly increasing function of d_euc. Attention logits built
%   from -d_geo^2 rank the same KVs as -d_euc^2 and a learnable tau absorbs the
%   difference: zero new information. Only when Q and K sit on DIFFERENT shells
%   does the extra term r_u^2 r_v^2/2 appear and the geodesic distance become
%   non-monotone in r_k.
% Q1 measures the rearrangement. Q2: does radius layering help strong/weak
% prototypes? Q3 (M2): are weak changes swallowed, and does layering rescue
% them? Q4: what does a same-shell constraint (LN + shared projection) destroy
% in a tree - hierarchy or curvature?
d = 16; here = fileparts(mfilename('fullpath'));
if isempty(here), here = pwd; end
addpath(here); figdir = fullfile(here,'figures');
if ~exist(figdir,'dir'), mkdir(figdir); end
H = hyper_setup();

% ======================================================== Q1 rearrangement ==
fprintf('H5-Q1 ===== same shell vs cross shell: is the geometry switched on? =\n');
rng(11); n = 4000; r = 0.40 + 0.55*rand(n,1);
th1 = 2*pi*rand(n,1); th2 = 2*pi*rand(n,1);
A = r.*[cos(th1) sin(th1)]; B = r.*[cos(th2) sin(th2)];
S = sum((A-B).^2,2);
dg = acosh(1 + 2*S./((1-sum(A.^2,2)).*(1-sum(B.^2,2))));
pred = acosh(1 + 2*S./((1-sum(A.^2,2)).^2));      % same-radius closed form
fprintf('  same-radius identity: max|d_geo - acosh(1+2d^2/a^2)| = %.2e over %d pairs\n', ...
    max(abs(dg-pred)), n);
fprintf('  (identity holds because a_u = a_v = a when r_u = r_v)\n');

kappa = [0.02 0.10 0.20 0.35 0.55 0.80 1.10];
sp = zeros(size(kappa)); kd = zeros(size(kappa)); t1 = zeros(size(kappa)); t5 = zeros(size(kappa));
nq = 180; nk = 224;
for a = 1:numel(kappa)
    rng(200+a);
    u = randn(nq,d); u = u./vecnorm(u,2,2);
    v = randn(nk,d); v = v./vecnorm(v,2,2);
    Qu = u.*(2.0*exp(kappa(a)*randn(nq,1)));
    Kv = v.*(2.0*exp(kappa(a)*randn(nk,1)));
    Dg = H.cdist(H.pball(Qu), H.pball(Kv));
    De = H.cdist(Qu, Kv);
    sp(a) = corr(Dg(:), De(:), 'type','Spearman');
    kd(a) = corr(Dg(:), De(:), 'type','Kendall');
    [~,i1] = min(Dg,[],2); [~,i2] = min(De,[],2);
    t1(a) = mean(i1 == i2);
    top5 = 0;
    for i = 1:nq
        [~,o1] = sort(Dg(i,:)); [~,o2] = sort(De(i,:));
        top5 = top5 + numel(intersect(o1(1:5), o2(1:5)))/5;
    end
    t5(a) = top5/nq;
end
fprintf('  radial spread kappa | Spearman | Kendall | top1 agree | top5 overlap\n');
for a = 1:numel(kappa)
    fprintf('        %.2f          |  %.4f  | %.4f  |   %.3f    |    %.3f\n', ...
        kappa(a), sp(a), kd(a), t1(a), t5(a));
end
fprintf('  >> same shell (LN + shared projection): Spearman %.4f, top-1 identical %.0f%% -> geometry OFF\n', ...
    sp(1), 100*t1(1));
rng(77); u = randn(nq,d); u = u./vecnorm(u,2,2); v = randn(nk,d); v = v./vecnorm(v,2,2);
shells = [0.7 1.6 3.2 6.4];
gq = mod(0:nq-1,4) + 1; gk = mod(0:nk-1,4) + 1;
Dg = H.cdist(H.pball(u.*shells(gq)'), H.pball(v.*shells(gk)'));
De = H.cdist(u.*shells(gq)', v.*shells(gk)');
[~,i1] = min(Dg,[],2); [~,i2] = min(De,[],2);
fprintf('  >> 4 shells 0.7/1.6/3.2/6.4: Spearman %.4f | top-1 agree %.3f | same-shell top-1 geo %.3f vs euc %.3f\n', ...
    corr(Dg(:),De(:),'type','Spearman'), mean(i1==i2), ...
    mean(gk(i1)' == gq'), mean(gk(i2)' == gq'));

f = figure('Position',[60 60 980 380]);
subplot(1,2,1);
plot(kappa, sp, '-o','LineWidth',1.7); hold on; plot(kappa, kd, '-s','LineWidth',1.4);
plot(kappa, t1, '-^','LineWidth',1.4); grid on;
xlabel('radial spread kappa (log-sd of shell radius)'); ylabel('agreement with Euclidean');
legend({'Spearman(d_{geo},d_{euc})','Kendall','top-1 agreement'},'Location','southwest');
title('H5-Q1: geometry OFF at equal radius, ON across shells');
subplot(1,2,2);
D4 = H.cdist(H.pball(u.*shells(gq)'), H.pball(v.*shells(gk)')); %#ok<NASGU>
imagesc(H.cdist(H.pball(v.*shells(gk)'), H.pball(v.*shells(gk)'))); axis square; colorbar;
title('geodesic d between KVs on 4 shells'); xlabel('KV'); ylabel('KV');
H.savefig(figdir,'fig_h5_rearrangement');

% ================================================ Q2 strong vs weak shells ==
fprintf('\nH5-Q2 ===== strong / weak change prototypes: flat shell vs shell-by-strength =\n');
rng(300); nKV = 224; nCh = 112;
dirs = randn(nKV,d); dirs = dirs./vecnorm(dirs,2,2);
weakKV = false(nKV,1); weakKV(1:2:nCh) = true;
strength = 2.2*ones(nKV,1); strength(weakKV) = 0.55;
nz = 0.25;
res = zeros(5,3); names = {'LN + geodesic','LN + euclid','raw euclid','raw geodesic','raw geodesic + bias'};
sh = 1.5*strength;                       % prototype radius follows change strength
P = dirs.*sh; rng(301);
Qz = P + nz*mean(sh)*randn(size(P));
Pn = P./vecnorm(P,2,2); Qn = Qz./vecnorm(Qz,2,2);
sgL = -H.cdist(H.pball(Qn), H.pball(Pn)).^2;
seL = -H.cdist(Qn, Pn).^2;
sgR = -H.cdist(H.pball(Qz), H.pball(P)).^2;
seR = -H.cdist(Qz, P).^2;
res(1,:) = acc3(sgL, weakKV); res(2,:) = acc3(seL, weakKV);
res(3,:) = acc3(seR, weakKV); res(4,:) = acc3(sgR, weakKV);
[~,i1] = max(sgL,[],2); [~,i2] = max(seL,[],2);
bestL = 0; bestS = -Inf;
for L = [0 0.05 0.1 0.25 0.5 1 2 -0.25 -0.5]
    s = mean(argmax_ok(sgR + L*log(sh)'));
    if s > bestS, bestS = s; bestL = L; end
end
res(5,:) = acc3(sgR + bestL*log(sh)', weakKV);
fprintf('  LN arm: top-1 disagreement between geodesic and euclid = %d / %d (must be 0)\n', sum(i1~=i2), nKV);
fprintf('  best additive radius bias lambda = %+0.2f\n', bestL);
fprintf('  arm                      | top-1 all | weak KV | strong KV\n');
for a = 1:5
    fprintf('  %-24s |   %.3f   |  %.3f  |   %.3f\n', names{a}, res(a,1), res(a,2), res(a,3));
end
fprintf('  reading: LN puts Q and K on one shell -> geodesic and euclid are rank-identical\n');
fprintf('           (geometry off). Using the raw radius: euclid is dominated by the norm and\n');
fprintf('           never retrieves a weak KV, the bounded geodesic does better, and an explicit\n');
fprintf('           radius bias recovers the weak KVs. That is the only place radius pays.\n');

f = figure('Position',[60 60 760 340]);
bar(res); grid on;
set(gca,'XTickLabel',names,'XTickLabelRotation',15);
legend({'all','weak KV','strong KV'},'Location','northwest');
ylabel('top-1 retrieval'); title('H5-Q2: radius layering is the only place geometry pays');
H.savefig(figdir,'fig_h5_q2_arms');

% ==================================================== Q3 weak change (M2) ==
fprintf('\nH5-Q3 ===== M2: are weak changes swallowed, does radius layering help?\n');
% Score = contrast  d(q,B)^2 - min_m d(q,m)^2  with B = background prototype
% (the no-change signature, i.e. the origin of change space). This is what the
% model computes (change prototype vs background prototype). The ABSOLUTE
% nearest-prototype distance is dominated by ||q||^2 and inverts with amplitude,
% so it must not be used as a change score.
rng(400); nS = 6000; dq = 32; kch = 4;
Wc = randn(kch,dq)/sqrt(kch);              % change directions live in a 4-dim subspace
isWeak = rand(nS,1) < 0.5;
amp = 0.35*isWeak + 1.25*~isWeak;          % weak / strong change amplitude
ul = randn(nS,kch); ul = ul./vecnorm(ul,2,2);
Dlt = (ul*Wc).*amp;
nzI = 0.30; nzO = 0.12;                    % noise inside change subspace + residual
Qx = Dlt + nzI*randn(nS,kch)*Wc + nzO*randn(nS,dq);
noCh = nzI*randn(nS,kch)*Wc + nzO*randn(nS,dq);
up = randn(30,kch); up = up./vecnorm(up,2,2);
ampM = [repmat(0.35,15,1); repmat(1.25,15,1)];
Mraw = (up*Wc).*ampM + 0.05*randn(30,dq);
weakKV = ampM < 1;
Q = [Qx; noCh]; y = [ones(nS,1); zeros(nS,1)];
Mflat = Mraw./vecnorm(Mraw,2,2)*1.5*mean(ampM);
protos = {Mraw, Mflat, Mraw, Mraw, Mraw};
names = {'euclid contrast (raw)','geodesic contrast, flat shell KV', ...
         'geodesic contrast, radius-layered KV','LayerNorm first (one shell, geodesic)', ...
         'LayerNorm first (one shell, euclid)'};
aucA = zeros(5,1); recW = zeros(5,1); recS = zeros(5,1); radW = zeros(5,1); radS = zeros(5,1);
for m = 1:5
    switch m
        case 1, Qf = Q; Mf = protos{1}; geo = false;
        case 2, Qf = Q; Mf = protos{2}; geo = true;
        case 3, Qf = Q; Mf = protos{3}; geo = true;
        case 4, Qf = Q./vecnorm(Q,2,2); Mf = Mflat./vecnorm(Mflat,2,2); geo = true;
        case 5, Qf = Q./vecnorm(Q,2,2); Mf = Mflat./vecnorm(Mflat,2,2); geo = false;
    end
    if geo
        Pq = H.pball(Qf); Pm = H.pball(Mf); P0 = H.pball(zeros(1,size(Qf,2)));
        dqB = H.cdist(Pq, P0).^2;
        dqm = min(H.cdist(Pq, Pm).^2, [], 2);
    else
        dqB = sum(Qf.^2,2);
        dqm = min(sum(Qf.^2,2) + sum(Mf.^2,2)' - 2*(Qf*Mf'), [], 2);
        Pq = Qf;
    end
    s = dqB - dqm;                                % >0 : closer to a change prototype
    aucA(m) = H.auc(s, y);
    thr = prctile(s(nS+1:end), 99);                % 1% FPR operating point
    recW(m) = mean(s(isWeak) > thr);
    recS(m) = mean(s(~isWeak) > thr);
    R = sqrt(sum(Pq(1:nS,:).^2,2));
    radW(m) = mean(R(isWeak)); radS(m) = mean(R(~isWeak));
end
fprintf('  weak/strong change amplitude 0.35 / 1.25, noise 0.30 in-subspace + 0.12 residual, 1%% FPR\n');
fprintf('  arm                                          |  AUC  | rec weak | rec strong | R weak/strong\n');
for m = 1:5
    fprintf('  %-44s | %.3f |   %.3f  |    %.3f   |  %.2f / %.2f\n', ...
        names{m}, aucA(m), recW(m), recS(m), radW(m), radS(m));
end
fprintf('  reading: (i) the same-shell arm (LN) collapses strong-change recall from\n');
fprintf('           0.590 (euclid raw) / 0.406 (geodesic) to 0.090: normalising throws\n');
fprintf('           away the amplitude that carried the detection -> this IS the M2\n');
fprintf('           blind spot, and it is caused by the normalisation, not by curvature;\n');
fprintf('           (ii) radius layering of the KV does NOT rescue it (weak recall 0.027\n');
fprintf('           vs 0.026, AUC 0.753 vs 0.758): geometry redistributes the radius but\n');
fprintf('           cannot manufacture signal that the noise floor absorbed;\n');
fprintf('           (iii) weak-change recall is 0.016-0.027 in EVERY arm at 1%% FPR, so a\n');
fprintf('           small change is below threshold under all geometries -> M2 needs a\n');
fprintf('           content change (differential KV), not a geometric one.\n');

f = figure('Position',[60 60 900 340]);
subplot(1,2,1); hold on; edges = linspace(0,3.2,50);
rr = asinh(vecnorm(Dlt,2,2));
histogram(rr(~isWeak),edges,'FaceColor',[.85 .1 .1],'EdgeColor','none');
histogram(rr(isWeak),edges,'FaceColor',[.15 .35 .85],'EdgeColor','none');
xlabel('radius asinh of change norm'); ylabel('count');
legend({'strong change','weak change'}); title('H5-Q3: radius separates weak/strong');
subplot(1,2,2);
bar([recW recS]); set(gca,'XTickLabel',{'euc raw','geo flat','geo layer','LN geo','LN euc'});
legend({'weak recall','strong recall'},'Location','northwest'); grid on;
ylabel('recall at 1% FPR'); title('H5-Q3: same shell erases weak/strong distinction');
H.savefig(figdir,'fig_h5_shells');

% ======================================= Q4 hierarchy vs one-shell (tree) ==
fprintf('\nH5-Q4 ===== tree embedding: what does a same-shell constraint destroy? =\n');
Dt = tree4ary(3);                       % complete 4-ary tree, depth 3, 85 nodes
Xt = H.embed(Dt, 2, 'hyp', 4000, 4);    % hyperbolic 2-D
Xe = H.embed(Dt, 2, 'euc', 4000, 4);    % Euclidean 2-D
dep = tree_depth(3);
sH0 = nstress(H, Xt, Dt, 1); sE0 = nstress(H, Xe, Dt, 0);
XtS = shell_proj(Xt, 0.5); XeS = shell_proj(Xe, 0.5);
sH1 = nstress(H, XtS, Dt, 1); sE1 = nstress(H, XeS, Dt, 0);
rH0 = corr(sqrt(sum(Xt.^2,2)), dep, 'type','Spearman');
rH1 = corr(sqrt(sum(XtS.^2,2)), dep, 'type','Spearman');
rE0 = corr(sqrt(sum(Xe.^2,2)), dep, 'type','Spearman');
rE1 = corr(sqrt(sum(XeS.^2,2)), dep, 'type','Spearman');
fprintf('  geometry      | stress (free) | stress (one shell) | radius-vs-depth rho free -> shelled\n');
fprintf('  hyperbolic 2D |    %.4f    |      %.4f       |  %.3f -> %.3f\n', sH0, sH1, rH0, rH1);
fprintf('  euclidean  2D |    %.4f    |      %.4f       |  %.3f -> %.3f\n', sE0, sE1, rE0, rE1);
fprintf('  stress inflation from the shell constraint: hyperbolic %.2fx, euclid %.2fx\n', sH1/max(sH0,1e-9), sE1/max(sE0,1e-9));
fprintf('  reading: shell projection is what destroys the hierarchy (rho -> %.2f), and it\n', rH1);
fprintf('           destroys it regardless of curvature. So LN + one shared projection throws\n');
fprintf('           away the RADIAL channel; curvature itself was never doing the work.\n');

f = figure('Position',[60 60 980 340]);
subplot(1,2,1); hold on; th = linspace(0,2*pi,200); plot(cos(th),sin(th),'k-');
scatter(Xt(:,1),Xt(:,2),28,dep,'filled'); colorbar; axis equal; axis([-1 1 -1 1]);
title('H5-Q4: free hyperbolic 2-D (radius = depth)');
subplot(1,2,2); hold on; plot(cos(th),sin(th),'k-');
scatter(XtS(:,1),XtS(:,2),28,dep,'filled'); colorbar; axis equal; axis([-1 1 -1 1]);
title('H5-Q4: same-shell constraint (LN-like)');
H.savefig(figdir,'fig_h5_tree_shell');

res_h5 = struct('kappa',kappa,'sp',sp,'kd',kd,'t1',t1,'t5',t5,'q2',res, ...
    'q2bias',bestL,'q3auc',aucA,'q3recW',recW,'q3recS',recS, ...
    'q4',[sH0 sH1 sE0 sE1 rH0 rH1]);
save(fullfile(here,'res_h5.mat'),'res_h5');
fprintf('H5 done.\n');

% ================================================================== helpers ==
function e = argmax_ok(S)
[~,ix] = max(S,[],2);
e = ix == (1:size(S,1))';
end

function v = acc3(S, weakKV)
[~,ix] = max(S,[],2);
ok = ix == (1:size(S,1))';
v = [mean(ok) mean(ok(weakKV)) mean(ok(~weakKV))];
end

function d2 = mind2(A, B, useGeo)
if nargin < 3, useGeo = 0; end
if useGeo, D = H_geo(A,B); d2 = D.^2; else
    d2 = zeros(size(A,1),1);
    for i = 1:size(A,1)
        d2(i) = min(sum((B - A(i,:)).^2,2));
    end
    return;
end
d2 = min(d2,[],2);
end

function D = H_geo(A, B)
persistent Hs
if isempty(Hs), Hs = hyper_setup(); end
D = Hs.cdist(A, B);
end

function X = shell_proj(X, R)
n = sqrt(sum(X.^2,2));
X = X./max(n,1e-12)*R;
end

function s = nstress(H, X, Dt, isHyp)
if isHyp, D = H.pdist(X); else
    D = zeros(size(Dt));
    for i = 1:size(X,1)
        D(i,:) = sqrt(sum((X - X(i,:)).^2,2))';
    end
end
s = mean((D(:)-Dt(:)).^2)/mean(Dt(:).^2);
end

function Dt = tree4ary(depth)
% complete 4-ary tree, root = 1, BFS order; distances from the parent array
n = (4^(depth+1)-1)/3; par = zeros(n,1); k = 1;
for lev = 1:depth
    for p = (4^(lev-1)-1)/3+1 : (4^lev-1)/3
        par(k+1:k+4) = p; k = k + 4;
    end
end
Dt = zeros(n);
for i = 1:n
    pi_ = i; p = par(i);
    while p > 0, pi_(end+1) = p; p = par(p); end %#ok<AGROW>
    for j = i+1:n
        pj = j; p = par(j);
        while p > 0, pj(end+1) = p; p = par(p); end %#ok<AGROW>
        k2 = numel(pi_);
        while k2 > 0 && ~any(pj == pi_(k2)), k2 = k2 - 1; end
        Dt(i,j) = (k2 - 1) + (numel(pj) - find(pj == pi_(k2),1));
        Dt(j,i) = Dt(i,j);
    end
end
end

function dep = tree_depth(depth)
n = (4^(depth+1)-1)/3; dep = zeros(n,1); k = 1;
for lev = 1:depth
    c = 4^lev; dep(k+1:k+c) = lev; k = k + c;
end
end
