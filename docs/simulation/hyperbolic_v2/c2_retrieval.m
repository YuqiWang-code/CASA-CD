% c2_retrieval.m — 修正版 H3/H5：距离族、归一化、检索、支持度与"算子等价性"
%
%   对应执行方案 §1.2 / §1.5(H3) / §1.6 的修正项：
%     1. 欧氏对照一律用 S.euc_dist；另加 lorentz 一路，验证三者在球内点上一致。
%     2. "受控自检索（原型加噪）"与"真独立查询（新场景）"分成两个任务，分别报告。
%     3. 支持度先验分三档：none / oracle（GT 占比，单独标注为上界）/ predicted（带噪）；
%        λ 只在独立开发场景上选，再到测试场景上评。
%     4. 除 top-1 外，报"排序一致但算子不同"的量：attention 的 JS 散度、V 聚合相对误差。
%     5. 跨尺度错误只在"受控期望尺度"任务里计分，自由检索单列。
%
%   数据模型（所有臂共用同一 realization，种子固定）：
%     每场景每尺度两个类方向 M；原型 = alpha_s·normalize(M_c + 噪声)，alpha_s 取
%     v1 H3 实测的每尺度 KV 范数 [0.07 0.09 0.22 0.48]；查询像素同构。
%     归一化口径 raw / l2 / ln / lnproj 之后再统一把中位半径缩到 0.50，再加半径发散 κ。
%
%   输出：results/log_c2.txt、results/res_c2.mat、figures/fig_c2_01*、fig_c2_02*

clear; clc;
here = fileparts(mfilename('fullpath'));
addpath(here);
S = hyper_setup2(); S.zh();
outdir = fullfile(here,'results'); figdir = fullfile(here,'figures');
if ~exist(outdir,'dir'), mkdir(outdir); end
if ~exist(figdir,'dir'), mkdir(figdir); end
logline('reset');

d = 16;
nTok   = [128 32 32 32];                 % 与模块 FLAT_GRIDS=(8,4,4,4) 一致，合计 224
nScale = numel(nTok);
nKey   = sum(nTok);
scaleTag = repelem((1:nScale), nTok);
classTag = zeros(1,nKey);
for s = 1:nScale
    b = sum(nTok(1:s-1));
    classTag(b + (1:nTok(s))) = [ones(1,nTok(s)/2), zeros(1,nTok(s)/2)];
end
alphaN = [0.07 0.09 0.22 0.48];          % v1 H3 实测每尺度 KV 范数
tau    = 0.4696;                          % 模块 FROZEN_TAU
modes  = {'raw','l2','ln','lnproj'};
kappaV = [0 0.05 0.10 0.20 0.40 0.80];
noiseP = 1.20; noiseQ = 1.50;             % 原型/查询的方向噪声（相对幅度，见下）
nPix   = 40;                              % 每场景每尺度的查询像素数
nScene = 16; nTrain = 10; nQ = 4;         % nQ：前 2 个查询场景做 dev，后 2 个做 test
nDev = 2; nTest = nQ - nDev;
W = (rand(RandStream('mt19937ar','Seed',99), d, d)*2-1)/sqrt(d);   % 固定投影（复刻初始化）

logline('===== C2 修正版 H3/H5：检索与算子等价性 =====');
logline('键数 %d（尺度 token 数 %s）；原型 %d 场景；查询 %d 场景（dev %d + test %d）× %d 尺度 × %d 像素', ...
    nKey, mat2str(nTok), nScene, nQ, nDev, nTest, nScale, nPix);
logline('欧氏一律用 S.euc_dist；Poincare 只喂球内点，越界报错（C1-A3 已验）。');
logline('tau=%.4f（模块冻结值）；值向量取键的切向坐标。', tau);
logline('');

% ---------- 生成原始特征（一次，所有臂共用） ----------
% 关键：类方向 M 是全局的（同一 (尺度,类别) 在所有场景共用），
%       场景之间只差噪声 realization —— 否则跨场景检索在定义上就不成立。
rs = RandStream('mt19937ar','Seed',20261126);
M = cell(nScale,2);
for s2 = 1:nScale
    M{s2,1} = unit(randn(rs,1,d));
    M{s2,2} = unit(randn(rs,1,d));          % 两类的方向近似正交
end
[ZP, ZQ, YQ, SQ] = gen_scenes(rs, d, nScene, nQ, nTok, nScale, nPix, alphaN, noiseP, noiseQ, M, 1.8);
logline('原型幅度（切向范数）每尺度中位：%s', sprintf('%.3f ', ...
    arrayfun(@(s) median(sqrt(sum(reshape(ZP(1, scaleTag==s, :),[],d).^2,2))), 1:nScale)));

% ---------- 表1：模式 × kappa ----------
rows = {};
RES = struct();
for mi = 1:numel(modes)
    for ki = 1:numel(kappaV)
        kap = kappaV(ki);
        P = apply_mode(ZP, modes{mi}, W);
        Q = apply_mode(ZQ, modes{mi}, W);
        scP = sinh(0.5)/median(sinh(S.radius(P(:,:))));
        scQ = sinh(0.5)/median(sinh(S.radius(Q(:,:))));
        P = P*scP; Q = Q*scQ;
        % 半径发散：重置同一 seed，保证四种模式拿到完全相同的乘子
        rsM = RandStream('mt19937ar','Seed',2026);
        multP = exp(kap*randn(rsM, size(P,1),1));
        multQ = exp(kap*randn(rsM, size(Q,1),1));
        P = bsxfun(@times, P, multP); Q = bsxfun(@times, Q, multQ);

        rP = S.radius(P); rQ = S.radius(Q);
        acc = eval_retrieval(S, P, Q, classTag, scaleTag, nTrain, nQ, nPix, tau, YQ, SQ);
        cvK = std(sinh(rP))/mean(sinh(rP));
        cvQ = std(sinh(rQ))/mean(sinh(rQ));
        rows(end+1,:) = {modes{mi}, kap, cvK, cvQ, acc.top1_class, acc.top1_scale, ...
                         acc.spearman, acc.mismatch, acc.js, acc.vrel, acc.entropyG};
        RES(mi,ki).acc = acc;
    end
end

logline('[表1] 独立查询：归一化 × 半径发散（每格 %d 查询场景 × %d 原型场景 × %d 查询）', ...
    nQ, nTrain, nScale*nPix);
logline('%-8s %6s %7s %7s %9s %10s %10s %9s %8s %8s', ...
    '模式','kappa','cv_K','cv_Q','top1类','top1受控尺','Spearman','argmax异','JS','V相对');
for i = 1:size(rows,1)
    logline('%-8s %6.2f %7.3f %7.3f %9.3f %10.3f %10.4f %9.3f %8.4f %8.4f', ...
        rows{i,1}, rows{i,2}, rows{i,3}, rows{i,4}, rows{i,5}, rows{i,6}, ...
        rows{i,7}, rows{i,8}, rows{i,9}, rows{i,10});
end

% ---------- 表2：三种距离的一致性 ----------
logline('');
logline('[表2] 同一批球内点上三种距离（kappa=0，ln 模式，2 查询场景 × 224 键）');
P0 = apply_mode(ZP,'ln',W); Q0 = apply_mode(ZQ,'ln',W);
P0 = P0*(sinh(0.5)/median(sinh(S.radius(P0(:,:)))));
Q0 = Q0*(sinh(0.5)/median(sinh(S.radius(Q0(:,:)))));
dg = []; dlz = []; de = [];
Pk0 = reshape(P0(1,:,:), [], d);
for kq = 1:2
    Qk = reshape(Q0(kq,:,:), [], d);
    dg  = [dg;  reshape(S.geo_dist(S.pball(Qk), S.pball(Pk0)), [], 1)];
    dlz = [dlz; reshape(S.lorentz_dist(S.lift(Qk), S.lift(Pk0)), [], 1)];
    tmp = S.euc_dist(Qk, Pk0);
    de  = [de;  reshape(tmp, [], 1)];
end
logline('  |Poincare − Lorentz| 最大 %.3e（应为 0；两种实现同一距离）', max(abs(dg-dlz)));
logline('  Spearman(Poincare, 欧氏) = %.6f；Spearman(Poincare, Lorentz) = %.6f', ...
    S.spearman(dg,de), S.spearman(dg,dlz));

% ---------- 表3：自检索 vs 独立查询 ----------
logline('');
logline('[表3] 受控自检索（原型加噪）vs 真独立查询（新场景）：kappa=0.2，ln 模式');
kap = 0.2;
Pz = apply_mode(ZP,'ln',W); Qz = apply_mode(ZQ,'ln',W);
Pz = Pz*(sinh(0.5)/median(sinh(S.radius(Pz(:,:)))));
Qz = Qz*(sinh(0.5)/median(sinh(S.radius(Qz(:,:)))));
rsM = RandStream('mt19937ar','Seed',2026);
Pz = bsxfun(@times, Pz, exp(kap*randn(rsM,size(Pz,1),1)));
Qz = bsxfun(@times, Qz, exp(kap*randn(rsM,size(Qz,1),1)));
nSelf = 5; sSelf = nan(nSelf,2);
for si2 = 1:nSelf
    Qs = make_selfquery(RandStream('mt19937ar','Seed',320+si2), Pz, nTok, nScale, nPix, noiseQ);
    aS = eval_retrieval(S, Pz, Qs.Z, classTag, scaleTag, nTrain, 2, nPix, tau, Qs.Y, Qs.S);
    sSelf(si2,:) = [aS.top1_class, aS.top1_scale];
end
aInd = eval_retrieval(S, Pz, Qz, classTag, scaleTag, nTrain, nQ, nPix, tau, YQ, SQ);
logline('  受控自检索（%d 次独立抽样）：top1类 %.3f ± %.3f  受控尺度 %.3f ± %.3f', ...
    nSelf, mean(sSelf(:,1)), std(sSelf(:,1)), mean(sSelf(:,2)), std(sSelf(:,2)));
logline('  真独立查询（新场景）  ：top1类 %.3f        受控尺度 %.3f', ...
    aInd.top1_class, aInd.top1_scale);
logline('  说明：类方向 M 是全局的（同一 (尺度,类别) 全场景共用），场景之间只差噪声 realization。');
logline('        受控自检索的查询 = "已被噪声污染的原型" 再加一次噪声，等效噪声更大（1.5 与约 1.9），');
logline('        所以它命中率更低；它既不是上界也不是下界，只是"同一对象再加噪"这一特定工况。');
logline('        v1 把这一行当"上界"是错的：v1 的 0.484 vs 0.491 正是这个混淆，不能用来支持任何几何结论。');

% ---------- 表4：支持度先验 ----------
logline('');
logline('[表4] 支持度先验三档；lambda 只在 %d 个开发场景上选，再到 %d 个测试场景评', nDev, nTest);lamGrid = [0 0.25 0.5 1 2];
sup = eval_support(S, ZP, ZQ, W, classTag, scaleTag, nTrain, nQ, nDev, nPix, tau, lamGrid, YQ, SQ);
for k = 1:numel(sup.kind)
    logline('  %-10s: lambda=%.2f  top1类=%.3f  (oracle 臂只作上限)', sup.kind{k}, sup.lam(k), sup.top1(k));
end

% ---------- 图 ----------
mk = {'o-','s-','d-','^-'}; cl = lines(4);
fh = figure('Visible','off','Position',[30 40 1180 430]);
subplot(1,2,1); hold on;
for mi = 1:numel(modes)
    y = cell2mat(rows(strcmp(rows(:,1),modes{mi}),5));
    plot(kappaV, y, mk{mi}, 'Color', cl(mi,:), 'MarkerSize',5, 'LineWidth',1.3);
end
grid on; xlabel('半径发散度 \kappa'); ylabel('top-1（类正确）');
legend(modes,'Location','southwest','FontSize',8); legend boxoff;
title('(a) 独立查询：类正确 top-1','FontSize',10.5);
subplot(1,2,2); hold on;
for mi = 1:numel(modes)
    y = max(1 - cell2mat(rows(strcmp(rows(:,1),modes{mi}),7)), 1e-12);
    semilogy(kappaV, y, mk{mi}, 'Color', cl(mi,:), 'MarkerSize',5, 'LineWidth',1.3);
end
grid on; xlabel('半径发散度 \kappa'); ylabel('1 - Spearman(测地, 欧氏)');
legend(modes,'Location','northwest','FontSize',8); legend boxoff;
title('(b) 排序差异（对数轴；10^{-5} 量级 = 排序恒等）','FontSize',10.5);
S.savefig('fig_c2_01_top1_kappa', figdir); close(fh);

fh = figure('Visible','off','Position',[30 40 1180 430]);
subplot(1,2,1); hold on;
for mi = 1:numel(modes)
    y = cell2mat(rows(strcmp(rows(:,1),modes{mi}),9));
    plot(kappaV, y, mk{mi}, 'Color', cl(mi,:), 'MarkerSize',5, 'LineWidth',1.3);
end
grid on; xlabel('半径发散度 \kappa'); ylabel('JS 散度');
legend(modes,'Location','northwest','FontSize',8); legend boxoff;
title('(a) 排序相同 ≠ 注意力相同','FontSize',10.5);
subplot(1,2,2); hold on;
for mi = 1:numel(modes)
    y = cell2mat(rows(strcmp(rows(:,1),modes{mi}),10));
    plot(kappaV, y, mk{mi}, 'Color', cl(mi,:), 'MarkerSize',5, 'LineWidth',1.3);
end
grid on; xlabel('半径发散度 \kappa'); ylabel('V 聚合相对误差');
legend(modes,'Location','northwest','FontSize',8); legend boxoff;
title('(b) 下游聚合的差异','FontSize',10.5);
S.savefig('fig_c2_02_operator', figdir); close(fh);

% =========================================================================
% 实验 B：边界分辨力 —— 曲率唯一真正能起作用的地方
%   两个类别是两个同心壳（方向分布相同，只有半径不同）：内壳 r=a，外壳 r=a·ratio。
%   注意"欧氏"取 z 空间（模块 MatchedEuclidean 的真实口径），另加 euc_p 在
%   Poincare 坐标里算欧氏，只作诊断，说明"饱和"到底来自哪里。
% =========================================================================
logline('');
logline('[表5] 边界分辨力：内壳 r=a、外壳 r=a·ratio（方向分布相同，只有半径不同）');
logline('      raw = 保留半径；l2 = 半径被压平；euc_p = 在 Poincare 坐标里算欧氏（仅诊断用）。');
logline('      解析：cosh d − 1 = 2sinh²(Δρ/2) + sinhρ₁sinhρ₂(1−cosθ)；');
logline('            |Δz|²    = (sinhρ₁−sinhρ₂)² + 2sinhρ₁sinhρ₂(1−cosθ)。');
logline('            角度项完全相同，只有径向项不同：测地 ≈ (Δρ)²/2，欧氏 ≈ (Δρ·coshρ)²。');
logline('            所以小半径处两种几何几乎一样；大半径处欧氏放大径向、测地压缩径向。');
logline('%-6s %-7s %-8s %-6s %9s %9s', 'a','ratio','模式','距离','top1类','AUC');
aList = [0.2 1 3 6];
resB = nan(numel(aList), 5, 2);
armB = {'raw','geo'; 'raw','euc'; 'raw','euc_p'; 'l2','geo'; 'l2','euc'};
for si = 1:numel(aList)
    a = aList(si); ratio = 1.5;
    [Pb, Qb, Yb] = gen_shells(RandStream('mt19937ar','Seed',777), d, nKey, nScale*nPix, ...
                               nQ, a, ratio, 0.35);
    for ai = 1:5
        Pm = Pb; Qm = Qb;
        if strcmp(armB{ai,1},'l2')
            Pm = reshape(unit(reshape(Pm,[],d))*0.30, size(Pb));
            Qm = reshape(unit(reshape(Qm,[],d))*0.30, size(Qb));
        end
        [top1, aucv] = eval_shells(S, Pm, Qm, Yb, nQ, tau, armB{ai,2});
        resB(si, ai, 1) = top1; resB(si, ai, 2) = aucv;
        logline('%-6.2f %-7.1f %-8s %-6s %9.3f %9.3f', a, ratio, armB{ai,1}, armB{ai,2}, top1, aucv);
    end
end
logline('      读法：a≤0.2 时 raw 的三路几乎相同（曲率还没起作用）；');
logline('            a≥3 时 raw+测地的径向分辨力被 log 压缩吃掉（AUC→0.5），');
logline('            而 raw+欧氏在 z 空间仍按 sinhρ 线性放大径向差（AUC=1）；');
logline('            l2 一路无论哪种几何都退化到随机——半径信息被归一化抹掉。');
logline('      这一条和 v1 的说法相反：不是"双曲更能分层半径"，而是在大半径处双曲把半径压平了。');

fh = figure('Visible','off','Position',[30 40 980 420]);
cols = lines(5); lb = {'raw+测地','raw+欧氏(z)','raw+欧氏(Poincare)','l2+测地','l2+欧氏'};
subplot(1,2,1); hold on;
for ai = 1:5
    plot(aList, resB(:,ai,1), mk{min(ai,4)}, 'Color', cols(ai,:), 'MarkerSize',5,'LineWidth',1.3);
end
set(gca,'XScale','log'); grid on; xlabel('内壳半径 a'); ylabel('top-1（壳分类正确）');
legend(lb,'Location','southwest','FontSize',7); legend boxoff;
title('(a) 壳分类 top-1（ratio=1.5）','FontSize',10.5);
subplot(1,2,2); hold on;
for ai = 1:5
    plot(aList, resB(:,ai,2), mk{min(ai,4)}, 'Color', cols(ai,:), 'MarkerSize',5,'LineWidth',1.3);
end
set(gca,'XScale','log'); grid on; xlabel('内壳半径 a'); ylabel('AUC');
plot(aList, 0.5*ones(size(aList)), 'k--','LineWidth',1);
legend(lb,'Location','southwest','FontSize',7); legend boxoff;
title('(b) 壳分类 AUC（0.5=随机）','FontSize',10.5);
S.savefig('fig_c2_03_shell_resolution', figdir); close(fh);

save(fullfile(outdir,'res_c2.mat'),'rows','kappaV','modes','sSelf','aInd','sup','P0','Q0', ...
     'resB','aList','armB');
L = logline('get'); S.writefile(fullfile(outdir,'log_c2.txt'), L);
fprintf('C2 done: %d rows\n', size(rows,1));

% =======================================================================
function [ZP, ZQ, YQ, SQ] = gen_scenes(rs, d, nScene, nQ, nTok, nScale, nPix, alphaN, nP, nQnoise, M, ampQ)
if nargin < 12, ampQ = 1.8; end
% 场景：每尺度 2 个类方向；原型幅度按 v1 实测的 alphaN；
% 类标签约定：每尺度前半 = 类1（变化），后半 = 类0（背景），与 classTag 一致。
nKey = sum(nTok);
ZP = zeros(nScene, nKey, d);
ZQ = zeros(nQ, nScale*nPix, d);
YQ = zeros(nQ, nScale*nPix);
SQ = zeros(nQ, nScale*nPix);
for k = 1:(nScene + nQ)
    if k <= nScene
        for s = 1:nScale
            b = sum(nTok(1:s-1));
            for cc = [1 0]
                base = b + (cc==0)*nTok(s)/2;     % 类1 在每尺度前半
                for t = 1:nTok(s)/2
                    v = alphaN(s) * unit(M{s,cc+1} + nP*randn(rs,1,d)/sqrt(d));
                    ZP(k, base+t, :) = reshape(v,1,1,d);
                end
            end
        end
    else
        kq = k - nScene;
        for s = 1:nScale
            for j = 1:nPix
                c = double(j <= nPix/2);          % 前半 = 类1
                v = alphaN(s)*ampQ * unit(M{s,c+1} + nQnoise*randn(rs,1,d)/sqrt(d));
                idx = (s-1)*nPix + j;
                ZQ(kq, idx, :) = reshape(v,1,1,d);
                YQ(kq, idx) = c;
                SQ(kq, idx) = s;
            end
        end
    end
end
end

function Q = make_selfquery(rs, P, nTok, nScale, nPix, noise)
% 受控自检索：查询 = 同一场景某原型再加噪（明确是上界）
d = size(P,3); nSc = size(P,1);
Q.Z = zeros(2, nScale*nPix, d);
Q.Y = zeros(2, nScale*nPix); Q.S = zeros(2, nScale*nPix);
for kq = 1:2
    for s = 1:nScale
        b = sum(nTok(1:s-1));
        for j = 1:nPix
            c = (j <= nPix/2);
            t = mod(j-1, nTok(s)/2) + 1;
            idx = b + (1-c)*nTok(s)/2 + t;   % 类1 在每尺度前半（与 gen_scenes 一致）
            pv = reshape(P(kq, idx, :), 1, d);
            v = pv + noise*max(norm(pv),eps)*randn(rs,1,d)/sqrt(d);
            ii = (s-1)*nPix + j;
            Q.Z(kq, ii, :) = reshape(v,1,1,d);
            Q.Y(kq, ii) = double(c);
            Q.S(kq, ii) = s;
        end
    end
end
end

function Z = apply_mode(V, mode, W)
% V: N×d（或 nScene×nKey×d 会被 reshape 后处理）；四种归一化口径
sz = size(V);
d = sz(end);
M = reshape(V, [], d);
switch mode
    case 'raw',    Z = M;
    case 'l2',     Z = unit(M) * sqrt(d);
    case 'ln'
        mu = mean(M,2); vr = var(M,1,2);
        Z = bsxfun(@rdivide, M-mu, sqrt(vr+1e-5));
    case 'lnproj'
        mu = mean(M,2); vr = var(M,1,2);
        Z = bsxfun(@rdivide, M-mu, sqrt(vr+1e-5)) * W';
    otherwise, error('c2:mode','未知模式 %s', mode);
end
Z = reshape(Z, sz);
end

function acc = eval_retrieval(S, P, Q, classTag, scaleTag, nTrain, nQ, nPix, tau, YQ, SQ, ~)
nKey = size(P,2); d = size(P,3);
top1c = 0; top1s = 0; nTot = 0; sp = []; mism = 0; js = []; vrel = []; ent = [];
for kq = 1:nQ
    Qk = reshape(Q(kq,:,:), [], d);
    for kp = 1:nTrain
        Pk = reshape(P(kp,:,:), nKey, d);
        Dg = S.geo_dist(S.pball(Qk), S.pball(Pk));
        De = S.euc_dist(Qk, Pk);
        Lg = -Dg.^2/(2*tau^2); Le = -De.^2/(2*tau^2);
        [~, ig] = max(Lg, [], 2); [~, ie] = max(Le, [], 2);
        yq = YQ(kq,:)'; sq = SQ(kq,:)';
        top1c = top1c + mean(classTag(ig)' == yq);
        top1s = top1s + mean(scaleTag(ig)' == sq);
        nTot = nTot + 1;
        sp(end+1) = S.spearman(Dg(:), De(:));
        mism = mism + mean(ig == ie);
        Pg = S.softmax(Lg, 1); Pe = S.softmax(Le, 1);
        js(end+1) = mean(S.jsdiv(Pg, Pe));
        Vk = Pk;                          % 值 = 键的切向坐标
        Ag = Pg*Vk; Ae = Pe*Vk;
        vrel(end+1) = mean(sqrt(sum((Ag-Ae).^2,2)) ./ max(sqrt(sum(Ae.^2,2)),1e-12));
        ent(end+1) = mean(S.entropy(Pg))/log(nKey);
    end
end
acc.top1_class = top1c/nTot;
acc.top1_scale = top1s/nTot;
acc.spearman  = mean(sp);
acc.mismatch  = mism/nTot;
acc.js        = mean(js);
acc.vrel      = mean(vrel);
acc.entropyG  = mean(ent);
end

function sup = eval_support(S, ZP, ZQ, W, classTag, scaleTag, nTrain, nQ, nDev, nPix, tau, lamGrid, YQ, SQ)
% 三档：none / oracle（GT 占比）/ predicted（带噪）；λ 在开发场景上选，测试场景上评
P = apply_mode(ZP,'ln',W); Q = apply_mode(ZQ,'ln',W);
P = P*(sinh(0.5)/median(sinh(S.radius(P(:,:)))));
Q = Q*(sinh(0.5)/median(sinh(S.radius(Q(:,:)))));
rsM = RandStream('mt19937ar','Seed',2026);
kap = 0.2;
P = bsxfun(@times, P, exp(kap*randn(rsM,size(P,1),1)));
Q = bsxfun(@times, Q, exp(kap*randn(rsM,size(Q,1),1)));

% 人为让每 (查询场景, 尺度) 的变化占比不同，模拟类不平衡；oracle 用真值，predicted 加噪
rsS = RandStream('mt19937ar','Seed',5);
trueFrac = 0.15 + 0.7*rand(rsS, nQ, 4);
predFrac = min(max(trueFrac + 0.08*randn(rsS,size(trueFrac)),0.02),0.98);

kind = {'none','oracle','predicted'};
lam = zeros(1,3); top1 = zeros(1,3);
devList = 1:nDev; testList = (nDev+1):nQ;
for kk = 1:3
    if strcmp(kind{kk},'none')
        lamCand = 0;
    else
        lamCand = lamGrid;
    end
    best = -inf; lamBest = 0;
    for li = 1:numel(lamCand)
        a = acc_with_support(S, P, Q, devList, lamCand(li), kind{kk}, classTag, scaleTag, ...
                             nTrain, tau, YQ, SQ, trueFrac, predFrac);
        if a > best, best = a; lamBest = lamCand(li); end
    end
    lam(kk) = lamBest;
    top1(kk) = acc_with_support(S, P, Q, testList, lamBest, kind{kk}, classTag, scaleTag, ...
                                nTrain, tau, YQ, SQ, trueFrac, predFrac);
end
sup = struct('kind',{kind},'lam',lam,'top1',top1,'trueFrac',trueFrac,'predFrac',predFrac);
end

function a = acc_with_support(S, P, Q, kqList, lamV, kind, classTag, scaleTag, nTrain, tau, YQ, SQ, trueFrac, predFrac)
nKey = size(P,2); d = size(P,3); tot = 0; hit = 0;
for kq = kqList
    Qk = reshape(Q(kq,:,:), [], d);
    yq = YQ(kq,:)'; 
    supp = zeros(size(classTag));
    if ~strcmp(kind,'none')
        for s = 1:4
            fr = trueFrac(kq,s);
            if strcmp(kind,'predicted'), fr = predFrac(kq,s); end
            for c = 0:1
                idx = (scaleTag == s) & (classTag == c);
                supp(idx) = log(max(fr*(c==1) + (1-fr)*(c==0), 1e-3));
            end
        end
    end
    for kp = 1:nTrain
        Pk = reshape(P(kp,:,:), nKey, d);
        Dg = S.geo_dist(S.pball(Qk), S.pball(Pk));
        L = -Dg.^2/(2*tau^2) + lamV*supp;
        [~, ig] = max(L, [], 2);
        hit = hit + sum(classTag(ig)' == yq);
        tot = tot + numel(yq);
    end
end
a = hit/tot;
end

function n = mode_num(~)
n = 1;
end

% ---- 实验 B 的场景与评价 ----
function [P, Q, Y] = gen_shells(rs, d, nKey, nPixTot, nQ, a, ratio, noiseDir)
% 两个同心壳：类 0 在半径 a，类 1 在半径 a*ratio；方向共享一个基方向 + 方向噪声。
% 返回 P: 1×nKey×d（单库）；Q: nQ×nPixTot×d；Y: nQ×nPixTot（0/1 壳标签）。
u0  = unit(randn(rs,1,d));
lab = [ones(1,nKey/2), zeros(1,nKey/2)];       % 前半类1（外壳）
rad = a*ratio*lab + a*(1-lab);
dirs = bsxfun(@plus, u0, noiseDir*randn(rs,nKey,d));
dirs = unit(dirs);
P = zeros(1, nKey, d);
for i = 1:nKey
    P(1,i,:) = reshape(sinh(rad(i))*dirs(i,:), 1,1,d);
end
Q = zeros(nQ, nPixTot, d);
Y = zeros(nQ, nPixTot);
for k = 1:nQ
    ub = unit(u0 + 0.3*randn(rs,1,d));
    for j = 1:nPixTot
        c = double(j <= nPixTot/2);            % 前半类1（外壳）
        r = a*ratio*c + a*(1-c);
        v = sinh(r)*unit(ub + noiseDir*randn(rs,1,d));
        Q(k,j,:) = reshape(v,1,1,d);
        Y(k,j) = c;
    end
end
end

function [top1, aucv] = eval_shells(S, P, Q, Y, nQ, tau, distKind)
% 单库检索：top-1 键的壳标签是否与查询壳标签一致；AUC 用"到内/外壳的最小距离差"
nKey = size(P,2); d = size(P,3);
Pm = reshape(P, nKey, d);
ylab = [ones(1,nKey/2), zeros(1,nKey/2)];
hit = 0; tot = 0; sc = []; yy = [];
for k = 1:nQ
    Qk = reshape(Q(k,:,:), [], d);
    switch distKind
        case 'geo'
            D = S.geo_dist(S.pball(Qk), S.pball(Pm));
        case 'euc'
            D = S.euc_dist(Qk, Pm);                       % z 空间（模块口径）
        case 'euc_p'
            D = S.euc_dist(S.pball(Qk), S.pball(Pm));      % Poincare 坐标（诊断）
        otherwise
            error('c2:dist','未知距离 %s', distKind);
    end
    [~, ig] = min(D, [], 2);
    hit = hit + mean(ylab(ig)' == Y(k,:)');
    tot = tot + 1;
    sc = [sc; (min(D(:,ylab==0),[],2) - min(D(:,ylab==1),[],2))];
    yy = [yy; Y(k,:)'];
end
top1 = hit/tot;
aucv = S.auc(sc, yy);
end

function U = unit(U)
n = sqrt(sum(U.^2,2)); n(n<1e-12) = 1;
U = bsxfun(@rdivide, U, n);
end

function L = logline(fmt, varargin)
persistent B
if isempty(B), B = {}; end
if nargin >= 1 && ischar(fmt) && strcmp(fmt,'reset'), B = {}; L = B; return; end
if nargin >= 1 && ischar(fmt) && strcmp(fmt,'get'), L = B; return; end
if isempty(varargin), B{end+1,1} = fmt; else, B{end+1,1} = sprintf(fmt, varargin{:}); end
L = B;
end
