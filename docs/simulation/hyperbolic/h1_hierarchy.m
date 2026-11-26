%% H1 -- Is the 4-scale hierarchy separable on the hyperbolic radius?
% Synthetic trees that mirror the real KV structure:
%   Tree A (current design): root -> 4 scale nodes (RF 4/8/16/32) -> cells,
%           grids 8/4/4/4 -> 128+32+32+32 = 224 leaves, depth 2.
%   Tree B (counterfactual quadtree KV): complete 4-ary tree, depth 3.
% Questions:
%   Q1 how many Euclidean dims are needed to reach the hyperbolic-dim-2
%      distortion of the same tree metric (exact classical MDS curve vs
%      stress-minimising Poincare embedding)?
%   Q2 is the hyperbolic radius monotone in depth, i.e. does it separate scale?
%   Q3 with the model's ACTUAL lift (direct Lorentz lift, radius = asinh||z||)
%      does radius encode scale, or only the feature norm?
here = fileparts(mfilename('fullpath')); if isempty(here), here = pwd; end
addpath(here); H = hyper_setup();
figdir = fullfile(here,'figures'); if ~exist(figdir,'dir'), mkdir(figdir); end
rng(0);

grids = [8 4 4 4];  strides = [4 8 16 32];
nLeaf = 2*grids.^2;                 % 128 32 32 32 -> 224 KV
fprintf('H1: KV per scale = %s  (total %d)\n', mat2str(nLeaf), sum(nLeaf));

% ---------------------------------------------------------------- tree A ----
parA = [0 1 1 1 1];  lvlA = [0 1 1 1 1];  scA = [0 1 2 3 4];
for s = 1:4
    parA = [parA, (1+s)*ones(1,nLeaf(s))];       %#ok<AGROW>
    lvlA = [lvlA, 2*ones(1,nLeaf(s))];           %#ok<AGROW>
    scA  = [scA , s*ones(1,nLeaf(s))];           %#ok<AGROW>
end
DA = tree_dist(parA);

% ---------------------------------------------------------------- tree B ----
parB = 0; lvlB = 0; frontier = 1;
for d = 1:3
    nxt = [];
    for u = frontier
        for c = 1:4
            parB(end+1) = u; lvlB(end+1) = d; nxt(end+1) = numel(parB); %#ok<AGROW>
        end
    end
    frontier = nxt;
end
DB = tree_dist(parB);
fprintf('H1: tree A n=%d (distinct d=%s) | tree B n=%d (distinct d=%s)\n', ...
        numel(parA), mat2str(unique(DA(:))'), numel(parB), mat2str(unique(DB(:))'));

% ------------------------------------------- Q1: Euclidean dims vs hyperbolic
kmax = 48;
[stressA_e, ~] = mds_curve(DA, kmax);
[stressB_e, ~] = mds_curve(DB, kmax);
hypDims = [2 4 8];
sA_h = zeros(1,numel(hypDims)); sB_h = zeros(1,numel(hypDims));
XA2 = []; XB2 = [];
for i = 1:numel(hypDims)
    if i == 1
        [XA2, sA_h(i)] = H.embed(DA, hypDims(i), 'hyp', 700, 1);
        [XB2, sB_h(i)] = H.embed(DB, hypDims(i), 'hyp', 700, 1);
    else
        [~, sA_h(i)] = H.embed(DA, hypDims(i), 'hyp', 700, 1);
        [~, sB_h(i)] = H.embed(DB, hypDims(i), 'hyp', 700, 1);
    end
end
% radial (Sarkar-style) initialisation: does a depth-aligned solution exist?
X0A = radial_init(parA, 2);  X0B = radial_init(parB, 2);
[XA2r, sA_h_r] = H.embed(DA, 2, 'hyp', 700, 1, X0A);
[XB2r, sB_h_r] = H.embed(DB, 2, 'hyp', 700, 1, X0B);
fprintf('H1 Q1 stress (lower=better):\n');
fprintf('  tree A  euc dim1..4 = %s | hyp d2=%.4f d4=%.4f d8=%.4f | hyp2-radial=%.4f\n', ...
        mat2str(stressA_e(1:4)',3), sA_h(1), sA_h(2), sA_h(3), sA_h_r);
fprintf('  tree B  euc dim1..4 = %s | hyp d2=%.4f d4=%.4f d8=%.4f | hyp2-radial=%.4f\n', ...
        mat2str(stressB_e(1:4)',3), sB_h(1), sB_h(2), sB_h(3), sB_h_r);
fprintf('  euc tail: tree A dim8/16/32 = %s | tree B = %s\n', ...
        mat2str(stressA_e([8 16 32])',3), mat2str(stressB_e([8 16 32])',3));
dA_need = min_dim(stressA_e, sA_h(1));   % euclid dims to match hyp-2
dB_need = min_dim(stressB_e, sB_h(1));
dA_r = min_dim(stressA_e, sA_h_r);       dB_r = min_dim(stressB_e, sB_h_r);
fprintf('H1 Q1 Euclid dims needed to match hyp-2: tree A = %d (radial %d), tree B = %d (radial %d)\n', ...
        dA_need, dA_r, dB_need, dB_r);

% structural fact: where does "scale" live in the tree metric at all?
pairCross = DA( lvlA==2 & scA==1, lvlA==2 & scA==2 );
fprintf('H1 Q1b tree A: every leaf sits at depth 2 (root-leaf d=%d for all 4 scales);\n', ...
        DA(1, 6));
fprintf('       leaf-leaf distance across scales = %s, within one scale = %s\n', ...
        mat2str(unique(pairCross(:))'), ...
        mat2str(unique(DA(lvlA==2 & scA==1, lvlA==2 & scA==1))'));

% ------------------------------------------------- Q2: radius vs depth/scale
rA = sqrt(sum(XA2.^2,2));  rB = sqrt(sum(XB2.^2,2));
rBr = sqrt(sum(XB2r.^2,2));  rAr = sqrt(sum(XA2r.^2,2));
spA = spearman(lvlA(:), rA);   spB  = spearman(lvlB(:), rB);
spAr = spearman(lvlA(:), rAr); spBr = spearman(lvlB(:), rBr);
auc_scale_A  = min_pair_auc(rA,  scA);
auc_scale_A2 = min_pair_auc(rAr, scA);
fprintf('H1 Q2 radius-vs-depth Spearman: tree A %.3f (random init) / %.3f (radial init);\n', spA, spAr);
fprintf('       tree B %.3f (random init) / %.3f (radial init, by construction)\n', spB, spBr);
fprintf('H1 Q2 min pairwise AUC of radius for scale id (tree A) = %.3f (random) / %.3f (radial)\n', ...
        auc_scale_A, auc_scale_A2);
fprintf('  radius per scale (tree A, random): %s | (radial): %s\n', ...
        mat2str(round(mean_by(rA, scA)',3)), mat2str(round(mean_by(rAr, scA)',3)));
fprintf('  radius per depth (tree B, radial): %s\n', mat2str(round(mean_by(rBr, lvlB)',3)));

% ------------------------------------------- Q3: model's actual direct lift
% Features z per (scale, cell); deeper scale = more context = larger norm.
ns = 400; mu = [1.0 1.4 1.9 2.5]; sd = [0.35 0.45 0.60 0.80];
Z = []; lab = [];
for s = 1:4
    Z = [Z; mu(s) + sd(s)*randn(ns,32)];  %#ok<AGROW>
    lab = [lab; s*ones(ns,1)];            %#ok<AGROW>
end
r_dir = asinh(sqrt(sum(Z.^2,2)));         % radius of the direct Lorentz lift
nm    = sqrt(sum(Z.^2,2));
auc_r = min_pair_auc(r_dir, lab);  auc_n = min_pair_auc(nm, lab);
fprintf('H1 Q3 direct lift: max|radius - asinh||z||| = %.2e\n', ...
        max(abs(r_dir - asinh(nm))));
fprintf('H1 Q3 min pairwise AUC for scale: radius=%.3f  raw ||z||=%.3f  (identical ranking: %d)\n', ...
        auc_r, auc_n, isequal(rank_of(r_dir), rank_of(nm)));

% ------------------------------------------------------------------ figures --
f = figure('Position',[80 80 980 420]);
subplot(1,2,1);
draw_disk(XA2, scA, lines(4), {'scale RF4','scale RF8','scale RF16','scale RF32'});
title(sprintf('Tree A: current KV (depth 2), Poincare d=2, stress=%.3f', sA_h(1)));
subplot(1,2,2);
draw_disk(XB2r, lvlB, lines(3), {'depth 1','depth 2','depth 3'});
title(sprintf('Tree B: quadtree KV (depth 3), Poincare d=2, radial init, stress=%.3f', sB_h_r));
H.savefig(figdir,'fig_h1_disk');

f = figure('Position',[80 80 980 400]);
subplot(1,2,1);
plot(1:kmax, stressA_e, '-o', 'LineWidth',1.4); hold on;
plot(1:kmax, stressB_e, '-s', 'LineWidth',1.4);
for i = 1:3
    plot(hypDims(i), sA_h(i), 'o', 'MarkerSize',9, 'MarkerFaceColor','auto');
    plot(hypDims(i), sB_h(i), 's', 'MarkerSize',9, 'MarkerFaceColor','auto');
end
set(gca,'YScale','log'); grid on; xlabel('embedding dimension');
ylabel('normalised stress'); legend({'euc tree A','euc tree B','hyp tree A','hyp tree B'}, ...
       'Location','northeast'); title('Distortion vs dimension');
subplot(1,2,2);
hold on;
for s = 0:4
    r = rA(scA==s);
    if isempty(r), continue; end
    plot(s*ones(size(r))+0.08*randn(size(r)), r, '.', 'Color',[.6 .6 .6]);
    plot(s, mean(r), 'kp', 'MarkerSize',10, 'MarkerFaceColor','k');
end
xlabel('node type (0=root, 1..4 = scale RF 4/8/16/32)'); ylabel('hyperbolic radius');
title('Tree A: radius tracks cell crowding, not scale'); grid on; xlim([-0.5 4.5]);
H.savefig(figdir,'fig_h1_dims');

res_h1 = struct('stressA_e',stressA_e,'stressB_e',stressB_e,'hypDims',hypDims, ...
  'stressA_h',sA_h,'stressB_h',sB_h,'stressA_r',sA_h_r,'stressB_r',sB_h_r, ...
  'dimA',dA_need,'dimB',dB_need,'dimA_r',dA_r,'dimB_r',dB_r, ...
  'spearA',spA,'spearB',spB,'spearA_r',spAr,'spearB_r',spBr, ...
  'aucScaleA',auc_scale_A,'aucScaleA_r',auc_scale_A2,'aucRadius',auc_r,'aucNorm',auc_n, ...
  'nA',numel(parA),'nB',numel(parB),'rByDepthB',mean_by(rBr,lvlB));
save(fullfile(here,'res_h1.mat'),'res_h1');
fprintf('H1 done.\n');

% ------------------------------------------------------------------ helpers --
function D = tree_dist(par)
n = numel(par); adj = sparse(n,n);
for i = 2:n, adj(i,par(i)) = 1; adj(par(i),i) = 1; end
D = inf(n);
for s = 1:n
    d = -ones(n,1); d(s) = 0; q = s;
    while ~isempty(q)
        nq = [];
        for u = q(:)'
            nb = find(adj(u,:));
            for v = nb
                if d(v) < 0, d(v) = d(u)+1; nq(end+1) = v; end %#ok<AGROW>
            end
        end
        q = nq;
    end
    D(s,:) = d';
end
end

function [stress, Dk] = mds_curve(D, kmax)
n = size(D,1); J = eye(n) - ones(n)/n; B = -0.5*J*(D.^2)*J;
[V,E] = eig((B+B')/2); ev = diag(E); [ev,ix] = sort(ev,'descend'); V = V(:,ix);
evp = max(ev,0); den = norm(D,'fro'); stress = zeros(kmax,1); Dk = [];
for k = 1:kmax
    Z = V(:,1:k).*sqrt(evp(1:k))';
    Dk = sqrt(max(sum(Z.^2,2)+sum(Z.^2,2)'-2*Z*Z',0));
    stress(k) = norm(Dk-D,'fro')/den;
end
end

function d = min_dim(stress, target)
k = find(stress <= target, 1);
if isempty(k), d = NaN; else, d = k; end
end

function rho = spearman(a,b)
ra = rank_of(a(:)); rb = rank_of(b(:));
rho = corrcoef(ra,rb); rho = rho(1,2);
end

function X0 = radial_init(par, dim)
% Sarkar-style radial layout: radius linear in depth, children share the
% parent's angular wedge. Used as an init to test whether a depth-aligned
% low-stress solution exists at all.
n = numel(par); dep = zeros(n,1);
for i = 2:n, dep(i) = dep(par(i)) + 1; end
lo = -pi*ones(n,1); hi = pi*ones(n,1);
for i = 1:n
    ch = find(par == i);
    if isempty(ch), continue; end
    e = linspace(lo(i), hi(i), numel(ch)+1);
    for k = 1:numel(ch), lo(ch(k)) = e(k); hi(ch(k)) = e(k+1); end
end
th = (lo+hi)/2; r = 0.30*dep;
X0 = [r.*cos(th), r.*sin(th), zeros(n, dim-2)];
end

function r = rank_of(x)
[~,ix] = sort(x); r = zeros(size(x)); r(ix) = 1:numel(x);
end

function m = mean_by(x, g)
m = zeros(1, max(g));
for k = 1:max(g), m(k) = mean(x(g==k)); end
end

function a = min_pair_auc(score, lab)
u = unique(lab); a = inf; any_ = false;
for i = 1:numel(u)
    for j = i+1:numel(u)
        m = lab==u(i) | lab==u(j);
        y = double(lab(m)==u(i));
        s = score(m);
        % AUC that score ranks group i above group j
        n1 = sum(y); n0 = numel(y)-n1;
        [~,ix] = sort(s); ys = y(ix); r = 1:numel(s);
        a1 = (sum(r(ys==1)) - n1*(n1+1)/2)/(n1*n0);
        a = min(a, max(a1,1-a1)); any_ = true;
    end
end
if ~any_, a = NaN; end
end

function draw_disk(X, g, cmap, names)
hold on;
th = linspace(0,2*pi,200); plot(cos(th),sin(th),'k-','LineWidth',0.8);
plot([-1 1],[0 0],'k:','LineWidth',0.4); plot([0 0],[-1 1],'k:','LineWidth',0.4);
h = [];
for k = 1:max(g)
    idx = (g==k);
    if ~any(idx), continue; end
    h(end+1) = plot(X(idx,1),X(idx,2),'o','MarkerSize',5, ...
        'MarkerFaceColor',cmap(k,:),'MarkerEdgeColor','none'); %#ok<AGROW>
end
axis equal; axis([-1.05 1.05 -1.05 1.05]); axis off;
if ~isempty(h), legend(h, names(1:numel(h)), 'Location','southeast','FontSize',7); end
end
