% H3  Class-conditional KV prototypes (224 = 2 x [64 16 16 16] cells).
% Question: for "same cell, different scale" discrimination, is the geodesic
% distance more robust than the Euclidean one under noise + scale perturbation,
% and how much of any gain comes from the log(support) prior rather than from
% the geometry?
%
% Faithful-to-model setup: the 224 prototypes are *learned* in the real model,
% so here they are estimated by pooling cell statistics over 10 training
% scenes (stable prototypes), while queries come from 2 held-out scenes
% (noisy, magnitude perturbed). Three arms:
%   A  geodesic + prior      -d_geo^2(q,p) + lam*log(sup)
%   B  geodesic, no prior    -d_geo^2(q,p)
%   C  Euclidean + prior     -||q-p||^2 + lam*log(sup)
% lam is tuned on a validation split of the queries, never on the test split.
d = 32; Ns = [64 32 16 8]; Ks = [8 4 4 4]; imb = 0.05; Ntr = 10;
here = fileparts(mfilename('fullpath'));
if isempty(here), here = pwd; end
addpath(here); figdir = fullfile(here,'figures');
if ~exist(figdir,'dir'), mkdir(figdir); end
H = hyper_setup();

% ---------------------------------------------------------------- prototypes
nKV = 2*sum(Ks.^2);                       % 224
tagS = zeros(nKV,1); tagC = zeros(nKV,1); tagK = zeros(nKV,1); tagBox = zeros(nKV,4);
sumC = cell(1,4); sumB = cell(1,4); nC = cell(1,4); nB = cell(1,4); gC = cell(1,4); gB = cell(1,4);
for s = 1:4
    sumC{s} = zeros(Ks(s)^2,d); sumB{s} = zeros(Ks(s)^2,d);
    nC{s} = zeros(Ks(s)^2,1);   nB{s} = zeros(Ks(s)^2,1);
    gC{s} = zeros(1,d); gB{s} = zeros(1,d);
end
for t = 1:Ntr
    [~,Zb,Y] = scene(d,Ns,imb,t);
    for s = 1:4
        [c1,n1,c0,n0,bx] = cellstats(Zb{s}, double(Y{s}), Ns(s), Ks(s));
        sumC{s} = sumC{s} + c1; nC{s} = nC{s} + n1;
        sumB{s} = sumB{s} + c0; nB{s} = nB{s} + n0;
        gC{s} = gC{s} + sum(c1,1); gB{s} = gB{s} + sum(c0,1);
        if t == 1, boxS{s} = bx; end
    end
end
P = zeros(nKV,d); sup = zeros(nKV,1); kk = 0;
for s = 1:4
    Cs = sumC{s}./max(nC{s},1); Bs = sumB{s}./max(nB{s},1);
    gc = gC{s}/max(sum(nC{s}),1); gb = gB{s}/max(sum(nB{s}),1);
    Cs(nC{s}<3,:) = repmat(gc,sum(nC{s}<3),1);  % fall back to global class mean
    Bs(nB{s}<3,:) = repmat(gb,sum(nB{s}<3),1);
    tot = nC{s} + nB{s};
    for i = 1:Ks(s)^2
        kk = kk + 1;
        P(kk,:) = Cs(i,:); sup(kk) = nC{s}(i)/max(tot(i),1);
        tagS(kk) = s; tagC(kk) = 1; tagK(kk) = i; tagBox(kk,:) = boxS{s}(i,:);
    end
    for i = 1:Ks(s)^2
        kk = kk + 1;
        P(kk,:) = Bs(i,:); sup(kk) = nB{s}(i)/max(tot(i),1);
        tagS(kk) = s; tagC(kk) = 0; tagK(kk) = i; tagBox(kk,:) = boxS{s}(i,:);
    end
end
fprintf('H3 prototypes: %d KV (%d change / %d background), support range %.3f..%.3f\n', ...
    nKV, sum(tagC==1), sum(tagC==0), min(sup), max(sup));

% --------------------------------------------------------------- test queries
Qm = zeros(0,d); qs = zeros(0,1); qk = zeros(0,1); qbx = zeros(0,4);
for t = 1:2
    [~,Zb,Y] = scene(d,Ns,imb,100+t);
    for s = 1:4
        [c1,n1,~,~,bx] = cellstats(Zb{s}, double(Y{s}), Ns(s), Ks(s));
        for i = 1:Ks(s)^2
            if n1(i) >= 2
                Qm(end+1,:) = c1(i,:); qs(end+1,1) = s; qk(end+1,1) = i; %#ok<SAGROW>
                qbx(end+1,:) = bx(i,:); %#ok<SAGROW>
            end
        end
    end
end
nq = size(Qm,1);
qKV = zeros(nq,1);
for i = 1:nq, qKV(i) = sum(Ks(1:qs(i)-1).^2) + qk(i); end
fprintf('H3 queries: %d held-out change-cell statistics\n', nq);

PG = H.pball(P);                        % Poincare image of prototypes
sig = [0 0.05 0.10 0.20 0.35];
lams = [0 0.1 0.25 0.5 1 2 -0.25 -0.5 -1];
acc = zeros(numel(sig),4); lamBest = zeros(numel(sig),2);
alias = zeros(numel(sig),4); classErr = zeros(numel(sig),4);
for a = 1:numel(sig)
    rng(7000+a);
    al = 0.75 + 0.55*rand(nq,1);              % scale perturbation of magnitude
    QQ = zeros(nq, d); M0 = P(tagC==1,:);
    for i = 1:nq
        QQ(i,:) = al(i)*M0(qKV(i),:) + sig(a)*std(P(:))*randn(1,d);
    end
    qg = H.pball(QQ);
    sGeo = -H.cdist(qg, PG).^2;               % nq x 224, higher = better
    sEuc = -H.cdist(QQ, P).^2;
    sp   = log(sup + 1e-3)';
    if a == 1
        nrm = zeros(1,4);
        for s = 1:4, nrm(s) = mean(sqrt(sum(P(tagS==s,:).^2,2))); end
        [~,i1] = max(sGeo,[],2); [~,i2] = max(sEuc,[],2);
        fprintf('  KV norm per scale: %.2f %.2f %.2f %.2f\n', nrm);
        fprintf('  Spearman(euc,geo) over all %d pairs = %.4f ; argmax disagreements = %d/%d\n', ...
            numel(sGeo), corr(sGeo(:), sEuc(:), 'type','Spearman'), sum(i1~=i2), nq);
        fprintf('  mean |pi(P)| radius = %.4f (1 = horizon)\n', mean(sqrt(sum(PG.^2,2))));
    end
    va = 1:2:nq; te = 2:2:nq;                 % lam tuned on val, scored on test
    for arm = 1:2
        best = -Inf; bl = 0;
        for L = lams
            sc = tern(arm==1, sGeo, sEuc) + L*sp;
            ac = mean(pick(sc(va,:), tagC, tagBox, qbx(va,:), qKV(va)) == 1);
            if ac > best, best = ac; bl = L; end
        end
        lamBest(a,arm) = bl;
    end
    S = {sGeo + lamBest(a,1)*sp, sGeo, sEuc + lamBest(a,2)*sp, sEuc};
    for arm = 1:4
        ok = pick(S{arm}(te,:), tagC, tagBox, qbx(te,:), qKV(te));
        acc(a,arm) = mean(ok == 1);
        alias(a,arm) = mean(ok == 2);
        classErr(a,arm) = mean(ok == 3);
    end
end
fprintf('\nH3 ===== 224-way retrieval of the change prototype (noise x arms) =====\n');
fprintf('  sigma_q | geo+pr  geo   euc+pr  euc  | lam(geo,euc) | cross-scale  | background-KV\n');
for a = 1:numel(sig)
    fprintf('   %.2f   | %.3f  %.3f  %.3f  %.3f | %+0.2f %+0.2f  | %.3f %.3f %.3f %.3f | %.3f %.3f %.3f %.3f\n', ...
        sig(a), acc(a,:), lamBest(a,1), lamBest(a,2), alias(a,:), classErr(a,:));
end
dGeoEuc = acc(:,1) - acc(:,3); dGeoNoPr = acc(:,2) - acc(:,4);
fprintf('  geo+prior - euc+prior   : mean %+0.2f pp (max %+0.2f at sigma=%.2f)\n', ...
    100*mean(dGeoEuc), 100*max(dGeoEuc), sig(find(dGeoEuc==max(dGeoEuc),1)));
fprintf('  geo(no prior)-euc(no pr): mean %+0.2f pp\n', 100*mean(dGeoNoPr));
fprintf('  prior effect, euc arm   : mean %+0.2f pp ; geo arm %+0.2f pp\n', ...
    100*mean(acc(:,3)-acc(:,4)), 100*mean(acc(:,1)-acc(:,2)));
fprintf('  background-KV confusion at sigma=0.20: %.3f %.3f %.3f %.3f\n', classErr(sig==0.20,:));

% ------------------------------------------------------------------- figures
f = figure('Position',[60 60 1100 460]);
subplot(1,2,1); imagesc(-H.cdist(PG,PG)); axis square; colorbar;
title('geodesic distance between the 224 KV'); xlabel('KV index (scale 1..4)'); ylabel('KV index');
subplot(1,2,2); imagesc(-H.cdist(P,P)); axis square; colorbar;
title('Euclidean distance between the 224 KV'); xlabel('KV index'); ylabel('KV index');
H.savefig(figdir,'fig_h3_distmat');

f = figure('Position',[60 60 760 340]);
plot(sig, acc(:,1), '-o', 'LineWidth',1.6); hold on;
plot(sig, acc(:,2), '-s', 'LineWidth',1.6);
plot(sig, acc(:,3), '-^', 'LineWidth',1.6);
grid on; xlabel('query noise sigma (fraction of feature std)');
ylabel('224-way top-1 accuracy'); ylim([0 1]);
legend({'geo + support prior','geo, no prior','euclid + prior'},'Location','southwest');
title('H3: prototype retrieval under noise');
H.savefig(figdir,'fig_h3_retrieval');

res_h3 = struct('sig',sig,'acc',acc,'alias',alias,'lamBest',lamBest, ...
    'nq',nq,'nKV',nKV,'dGeoEuc',dGeoEuc,'classErr',classErr);
save(fullfile(here,'res_h3.mat'),'res_h3');
fprintf('H3 done.\n');

% ================================================================== helpers ==
function ok = pick(sc, tagC, tagBox, qbx, qKV)
% classify each query's top-1: 1 = its own change KV, 2 = change KV at another
% scale whose box overlaps the query box (cross-scale aliasing), 3 = background
% KV, 0 = a change KV of another region
[~,ix] = max(sc, [], 2);
ok = zeros(size(qbx,1),1);
for i = 1:size(qbx,1)
    j = ix(i);
    if tagC(j) ~= 1
        ok(i) = 3;
    elseif j == qKV(i)
        ok(i) = 1;
    elseif overlap(tagBox(j,:), qbx(i,:))
        ok(i) = 2;
    end
end
end

function t = overlap(a, b)
t = a(1) <= b(2) && b(1) <= a(2) && a(3) <= b(4) && b(3) <= a(4);
end

function [c1,n1,c0,n0,bx] = cellstats(Z, Y, ngrid, K)
% per-cell class means and counts, plus each cell's box in scale-1 units
cs = ngrid/K; [~,d] = size(Z);
c1 = zeros(K*K,d); c0 = c1; n1 = zeros(K*K,1); n0 = n1; bx = zeros(K*K,4);
u = 64/ngrid;                                % scale-1 positions per feature step
for ci = 1:K
    for cj = 1:K
        k = (ci-1)*K + cj;
        r = (ci-1)*cs + (1:cs); c = (cj-1)*cs + (1:cs);
        idx = reshape(bsxfun(@plus, ngrid*(c-1)', r), [], 1);
        y = Y(idx); z = Z(idx,:);
        n1(k) = sum(y==1); n0(k) = sum(y==0);
        if n1(k) > 0, c1(k,:) = mean(z(y==1,:),1); end
        if n0(k) > 0, c0(k,:) = mean(z(y==0,:),1); end
        bx(k,:) = [(ci-1)*cs*u + 1, ci*cs*u, (cj-1)*cs*u + 1, cj*cs*u];
    end
end
end

function s = tern(c,a,b), if c, s = a; else, s = b; end, end

function [Z1,Z2,Y] = scene(d, Ns, imb, seed)
rng(1000 + seed + round(1000*imb));
N = Ns(1); npos = round(imb*N^2); area = 0; rects = {}; dirs = [];
while area < npos
    w = 2 + floor(exp(rand*log(22))); h = 2 + floor(exp(rand*log(22)));
    if w >= N || h >= N, continue; end
    rects{end+1} = [randi([1 N-w]), randi([1 N-h]), w, h]; %#ok<AGROW>
    area = area + w*h;
    v = randn(d,1); dirs(:,end+1) = v/norm(v); %#ok<AGROW>
end
u1 = smoothnoise(N); u1 = u1/std(u1(:));
Z1f = (u1(:)*ones(1,d)).*randn(1,d) + 0.6*randn(N*N,d);
Dl = zeros(N*N,d); mask = false(N*N,1);
for k = 1:numel(rects)
    rr = rects{k}; sub = false(N,N);
    sub(rr(1):(rr(1)+rr(3)-1), rr(2):(rr(2)+rr(4)-1)) = true; sub = sub(:);
    wgt = 0.35 + 0.65*rand(nnz(sub),1);
    Dl(sub,:) = Dl(sub,:) + wgt*dirs(:,k)';
    mask = mask | sub;
end
Z1f = Z1f + 0.55*randn(N*N,d);
Z2f = Z1f + Dl + 0.55*randn(N*N,d);
Z1 = cell(1,4); Z2 = cell(1,4); Y = cell(1,4);
gn = [1.0 1.35 1.8 2.4];      % per-scale feature gain: real multi-scale features
                              % have different activation magnitudes, so KV norms
                              % differ across scales (the only way the Euclidean
                              % and geodesic rankings can diverge at all)
Z1{1} = Z1f; Z2{1} = Z2f; Y{1} = mask;
for s = 2:4
    Z1{s} = pool2(Z1{s-1}, Ns(s-1)); Z2{s} = pool2(Z2{s-1}, Ns(s-1));
    Y{s}  = pool2(double(Y{s-1}), Ns(s-1)) > 0.5;
end
for s = 1:4
    Z1{s} = gn(s)*Z1{s}; Z2{s} = gn(s)*Z2{s};
end
end

function B = pool2(A, N)
[~,d] = size(A); n = N/2; A = reshape(A, N, N, d);
B = (A(1:2:end,1:2:end,:) + A(2:2:end,1:2:end,:) + ...
     A(1:2:end,2:2:end,:) + A(2:2:end,2:2:end,:))/4;
B = reshape(B, n*n, d);
end

function u = smoothnoise(N)
k = exp(-((-(4:4)).^2)'/(2*2^2)); K = k*k'; K = K/sum(K(:));
u = conv2(randn(N), K, 'same');
end
