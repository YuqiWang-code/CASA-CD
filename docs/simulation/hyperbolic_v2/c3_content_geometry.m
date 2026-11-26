% =========================================================================
% c3_content_geometry.m  ——  修正版 H2：内容 vs 几何的 2x2 公平对照
%
% v1 的 H2 结论："差分原型内容 +22.84 pp，几何只差 −1.16~−4.28 pp"。
% 那份实验有三处 oracle 成分（执行方案 §1.5）：原型用真实标签构造、两臂评分
% 口径不一致、拟合/评估掩膜有重叠。这里重做，只换两个因子，其余全部固定：
%     content ∈ {T2 原始融合特征, D = T2 − T1 变化证据}
%     score   ∈ {euclid（z 空间）, geodesic（Poincare 测地）}
%
% 原型口径与模块一致：**每张图（每场景）每尺度各算两个原型**，用该图自己的
% token 做无监督二分（k-means(2)），再用"半径更大的一簇 = 变化"这条固定规则
% 定语义（不看 GT）。GT 只在算 AUC 时使用。
% 这同时满足执行方案对 H3 的要求"主实验应为同图内查询检索图内原型"。
%
% 生成模型（显式写清，便于判断结论适用范围）：
%   每场景每尺度有一个场景级背景方向 b0（同一图内背景共享）；token = b0 + 噪声；
%   变化 token 在 T2 上叠加 ampC·M2（M2 为全局变化方向），T1 不含叠加。
%   D = T2 − T1 把 b0 减掉，只剩变化 + 噪声差分。
%
% 输出：results/log_c3.txt、res_c3.mat、figures/fig_c3_01_fair_2x2.png
% =========================================================================
clear; clc;
here = fileparts(mfilename('fullpath'));
addpath(here);
S = hyper_setup2(); S.zh;
outdir = fullfile(here,'results'); figdir = fullfile(here,'figures');
if ~exist(outdir,'dir'), mkdir(outdir); end
if ~exist(figdir,'dir'), mkdir(figdir); end
logline('reset');
logline('===== C3 修正版 H2：内容 vs 几何的 2x2 公平对照 =====');

%% ---------------- 规模 ----------------
d      = 16;
nScale = 4;
nScene = 30;
nTok   = [16 8 8 8];
ampC   = [1.00 0.95 0.95 0.95];
ampB   = [0.55 0.52 0.52 0.52];
noiseV = [1.00 1.60 2.20 3.00];
logline('设置：d=%d，%d 尺度，%d 场景，每尺度 token %s', d, nScale, nScene, mat2str(nTok));
logline('原型：每场景每尺度无监督二分（k-means 2），"半径大的一簇=变化"；GT 只用于算 AUC');

%% ---------------- 全局变化方向 ----------------
rs = RandStream('mt19937ar','Seed',20261126);
M2 = cell(nScale,1);
for s = 1:nScale, M2{s} = unit(randn(rs,1,d)); end

cont = {'T2','D'};
dist = {'euc','geo'};
AUCs = nan(numel(noiseV),2,2);        % (噪声, content, score) 汇总
AUCsc = nan(numel(noiseV),2,2);       % 每场景 AUC 的均值（另看稳定性）
v1s  = nan(numel(noiseV),2,2);

logline('');
logline('[表1] 四臂 AUC（每场景无监督原型；GT 只用于算指标）');
logline('%-6s %-8s %-10s %8s %8s %8s', '噪声','content','score','pooled','按场景均值','最差尺度');
for vi = 1:numel(noiseV)
    [T1, T2, LAB, SCL] = gen_pairs(rs, d, nScene, nTok, nScale, ampC, ampB, noiseV(vi), M2);
    D = T2 - T1;
    for ci = 1:2
        if ci == 1, X = T2; else, X = D; end
        for di = 1:2
            scAll = []; yAll = []; scPer = nan(nScene,1); aucS = nan(1,nScale);
            for k = 1:nScene
                sc = []; yk = [];
                for s = 1:nScale
                    sel = (SCL(k,:) == s);
                    Xk = reshape(X(k,:,:), [], d);
                    Xs = Xk(sel(:),:); ys = LAB(k,sel)'; ys = ys(:);
                    P = proto_unsup(S, Xs, rs);
                    if strcmp(dist{di},'geo')
                        Dm = S.geo_dist(S.pball(Xs), S.pball(P)).^2;
                    else
                        Dm = S.euc_dist(Xs, P).^2;
                    end
                    sc = [sc; Dm(:,1)-Dm(:,2)]; %#ok<AGROW>
                    yk = [yk; ys];              %#ok<AGROW>
                    aucS(s) = S.auc(Dm(:,1)-Dm(:,2), ys);
                end
                scAll = [scAll; sc]; yAll = [yAll; yk]; %#ok<AGROW>
                scPer(k) = S.auc(sc, yk);
            end
            AUCs(vi,ci,di) = S.auc(scAll, yAll);
            AUCsc(vi,ci,di) = mean(scPer);
            logline('%-6.2f %-8s %-10s %8.4f %8.4f %8.4f', noiseV(vi), cont{ci}, dist{di}, ...
                    AUCs(vi,ci,di), AUCsc(vi,ci,di), min(aucS));
            % v1 口径：GT 原型 + 同场景评估
            scV = []; yV = [];
            for k = 1:nScene
                for s = 1:nScale
                    sel = (SCL(k,:) == s);
                    Xk = reshape(X(k,:,:), [], d);
                    Xs = Xk(sel(:),:); ys = LAB(k,sel)'; ys = ys(:);
                    P = [mean(Xs(ys==0,:),1); mean(Xs(ys==1,:),1)];
                    if strcmp(dist{di},'geo')
                        Dm = S.geo_dist(S.pball(Xs), S.pball(P)).^2;
                    else
                        Dm = S.euc_dist(Xs, P).^2;
                    end
                    scV = [scV; Dm(:,1)-Dm(:,2)]; %#ok<AGROW>
                    yV  = [yV;  ys];              %#ok<AGROW>
                end
            end
            v1s(vi,ci,di) = S.auc(scV, yV);
        end
    end
end

%% ---------------- 主效应 ----------------
logline('');
logline('[表2] 主效应（pp）：内容 = D−T2；几何 = geo−euc');
logline('%-6s %-14s %-14s %-14s %-14s', '噪声','内容@欧氏','内容@测地','几何@T2','几何@D');
for vi = 1:numel(noiseV)
    logline('%-6.2f %+14.2f %+14.2f %+14.2f %+14.2f', noiseV(vi), ...
        100*(AUCs(vi,2,1)-AUCs(vi,1,1)), 100*(AUCs(vi,2,2)-AUCs(vi,1,2)), ...
        100*(AUCs(vi,1,2)-AUCs(vi,1,1)), 100*(AUCs(vi,2,2)-AUCs(vi,2,1)));
end

%% ---------------- 成对 bootstrap（最低噪声档，按场景重采样） ----------------
vi = 1;
[T1, T2, LAB, SCL] = gen_pairs(rs, d, nScene, nTok, nScale, ampC, ampB, noiseV(vi), M2); %#ok<NASGU>
D = T2 - T1;
scPer4 = nan(nScene,4); yPer = cell(nScene,1);
kk = 0;
for ci = 1:2
    if ci == 1, X = T2; else, X = D; end
    for di = 1:2
        kk = kk + 1;
        for k = 1:nScene
            sc = []; yk = [];
            for s = 1:nScale
                sel = (SCL(k,:) == s);
                Xk = reshape(X(k,:,:), [], d);
                Xs = Xk(sel(:),:); ys = LAB(k,sel)'; ys = ys(:);
                P = proto_unsup(S, Xs, rs);
                if strcmp(dist{di},'geo')
                    Dm = S.geo_dist(S.pball(Xs), S.pball(P)).^2;
                else
                    Dm = S.euc_dist(Xs, P).^2;
                end
                sc = [sc; Dm(:,1)-Dm(:,2)]; %#ok<AGROW>
                yk = [yk; ys];              %#ok<AGROW>
            end
            scPer4(k,kk) = S.auc(sc, yk);
            yPer{k} = yk;
        end
    end
end
logline('  诊断：四臂的按场景平均 AUC = %s', mat2str(mean(scPer4,1),4));
nBoot = 2000;
ub = nan(nBoot,4);
for b = 1:nBoot
    pick = randsample(rs, nScene, nScene, true);
    for j2 = 1:4
        ub(b,j2) = mean(scPer4(pick,j2));   % 场景级 AUC 的均值差（成对）
    end
end
% 列顺序 = [(T2,euc) (T2,geo) (D,euc) (D,geo)]
bootEff = [ub(:,3)-ub(:,1), ub(:,4)-ub(:,2), ub(:,2)-ub(:,1), ub(:,4)-ub(:,3)];
lo = zeros(1,4); hi = zeros(1,4);
for j2 = 1:4
    lo(j2) = prctile_simple(bootEff(:,j2), 2.5);
    hi(j2) = prctile_simple(bootEff(:,j2), 97.5);
end
nmE = {'内容@欧氏','内容@测地','几何@T2','几何@D'};
logline('');
logline('[表3] 主效应与成对 bootstrap 95%% CI（%d 次，按场景重采样，噪声 %.2f）', nBoot, noiseV(vi));
for k2 = 1:4
    logline('  %-10s %+6.2f pp   [%+6.2f, %+6.2f]', nmE{k2}, 100*mean(bootEff(:,k2)), ...
            100*lo(k2), 100*hi(k2));
end

%% ---------------- v1 口径对照 ----------------
logline('');
logline('[表4] v1 口径（GT 造原型 + 同场景评估）与本次口径的差（pp，取噪声 1.0）');
logline('%-8s %-10s %10s %10s %10s', 'content','score','v1 口径','本次','虚高');
for ci = 1:2
    for di = 1:2
        logline('%-8s %-10s %10.4f %10.4f %+10.2f', cont{ci}, dist{di}, v1s(1,ci,di), ...
                AUCs(1,ci,di), 100*(v1s(1,ci,di)-AUCs(1,ci,di)));
    end
end
logline('  注：v1 口径的原型用 GT 分簇，本次用无监督二分；两者差就是"用 GT 造原型"的收益。');

%% ---------------- 图 ----------------
fh = figure('Visible','off','Position',[30 40 1320 470]);
subplot(1,2,1); hold on;
hb = bar(1:numel(noiseV), [AUCs(:,1,1) AUCs(:,1,2) AUCs(:,2,1) AUCs(:,2,2)], 0.82);
set(gca,'XTick',1:numel(noiseV),'XTickLabel', ...
    arrayfun(@(n) sprintf('%.1f',n),noiseV,'UniformOutput',false),'FontSize',9); grid on;
xlabel('方向噪声'); ylabel('AUC'); ylim([0.70 1.02]);
legend(hb,{'T2+欧氏','T2+测地','D+欧氏','D+测地'},'Location','north','NumColumns',2,'FontSize',7.5); legend boxoff;
title('(a) 四臂 AUC 随噪声（每场景无监督原型）','FontSize',10.5);

subplot(1,2,2); hold on;
xL = 1:numel(noiseV); cols = lines(4);
plot(xL, 100*(AUCs(:,2,1)-AUCs(:,1,1)), 'o-','Color',cols(1,:),'LineWidth',1.4,'MarkerSize',5);
plot(xL, 100*(AUCs(:,2,2)-AUCs(:,1,2)), 's-','Color',cols(2,:),'LineWidth',1.4,'MarkerSize',5);
plot(xL, 100*(AUCs(:,1,2)-AUCs(:,1,1)), '^-','Color',cols(3,:),'LineWidth',1.2,'MarkerSize',4.5);
plot(xL, 100*(AUCs(:,2,2)-AUCs(:,2,1)), 'd-','Color',cols(4,:),'LineWidth',1.2,'MarkerSize',4.5);
plot([0.6 numel(noiseV)+0.4],[0 0],'k--','LineWidth',1);
set(gca,'XTick',xL,'XTickLabel',arrayfun(@(n) sprintf('%.1f',n),noiseV,'UniformOutput',false),'FontSize',9);
grid on; xlabel('方向噪声'); ylabel('ΔAUC（pp）');
xlim([0.8 numel(noiseV)+1.1]);
lbl = {'内容@欧氏','内容@测地','几何@T2','几何@D'};
for j3 = 1:4
    yEnd = 100*( [AUCs(end,2,1)-AUCs(end,1,1), AUCs(end,2,2)-AUCs(end,1,2), ...
                  AUCs(end,1,2)-AUCs(end,1,1), AUCs(end,2,2)-AUCs(end,2,1)] );
    text(numel(noiseV)+0.12, yEnd(j3), lbl{j3}, 'FontSize',7.5,'Color',cols(j3,:), ...
         'VerticalAlignment','middle');
end
title('(b) 内容效应稳定为正，几何效应接近 0','FontSize',10.5);
S.savefig('fig_c3_01_fair_2x2', figdir); close(fh);

save(fullfile(outdir,'res_c3.mat'), 'AUCs','AUCsc','v1s','bootEff','lo','hi','noiseV','cont','dist');
logline('C3 done.');

%% ======================= local functions =======================
function P = proto_unsup(S, Xs, rs)
% 该场景该尺度的无监督二维原型：k-means(2)，半径大的一簇 = 变化
[~, C] = kmeans2(Xs, rs);
if S.radius(C(1,:)) >= S.radius(C(2,:))
    P = [C(2,:); C(1,:)];
else
    P = [C(1,:); C(2,:)];
end
end

function [T1, T2, LAB, SCL] = gen_pairs(rs, d, nScene, nTok, nScale, ampC, ampB, noise, M2)
nT = sum(nTok);
T1 = zeros(nScene, nT, d); T2 = zeros(nScene, nT, d);
LAB = zeros(nScene, nT);   SCL = zeros(nScene, nT);
for k = 1:nScene
    b = 0;
    for s = 1:nScale
        b0 = ampB(s)*unit(randn(rs,1,d));
        for t = 1:nTok(s)
            isCh = (t > nTok(s)/2);
            u1 = unit(b0 + noise*randn(rs,1,d)/sqrt(d));
            v1 = u1;
            if isCh
                amp = ampC(s)*exp(0.35*randn(rs,1));
                v2 = u1 + amp*unit(M2{s} + 0.6*randn(rs,1,d)/sqrt(d));
            else
                v2 = u1 + 0.45*ampB(s)*randn(rs,1,d)/sqrt(d);
            end
            i = b + t;
            T1(k,i,:) = reshape(v1,1,1,d);
            T2(k,i,:) = reshape(v2,1,1,d);
            LAB(k,i) = isCh; SCL(k,i) = s;
        end
        b = b + nTok(s);
    end
end
end

function [idx, C] = kmeans2(X, rs)
n = size(X,1); best = inf; C = X([1 end],:); idx = zeros(n,1);
for rep = 1:6
    if rep == 1
        C0 = X([1 max(2,round(n/3))],:);
    else
        p = randsample(rs, n, 2, false);
        C0 = X(p,:);
    end
    [ii, Cc] = kmeans2_run(X, C0);
    inertia = 0;
    for c = 1:2
        m = (ii == c-1);
        if any(m), inertia = inertia + sum(sum(bsxfun(@minus, X(m,:), Cc(c,:)).^2)); end
    end
    if inertia < best, best = inertia; C = Cc; idx = ii; end
end
end

function [idx, C] = kmeans2_run(X, C)
n = size(X,1); idx = zeros(n,1);
for it = 1:40
    D2 = zeros(n,2);
    for c = 1:2
        D2(:,c) = sum(bsxfun(@minus, X, C(c,:)).^2, 2);
    end
    [~, idx] = min(D2, [], 2); idx = idx - 1;
    for c = 1:2
        m = (idx == (c-1));
        if any(m), C(c,:) = mean(X(m,:),1); end
    end
end
end

function v = prctile_simple(X, p)
X = sort(X(:)); n = numel(X);
if n == 0, v = NaN; return; end
pos = (p/100)*(n-1) + 1;
lo = floor(pos); hi = ceil(pos);
v = X(lo) + (pos-lo)*(X(hi)-X(lo));
end

function U = unit(U)
n = sqrt(sum(U.^2,2)); n(n < eps) = 1;
U = bsxfun(@rdivide, U, n);
end

function L = logline(fmt, varargin)
persistent B
if isempty(B) || strcmp(fmt,'reset'), B = {}; end
if strcmp(fmt,'get'), L = B; return; end
if isempty(varargin), s = fmt; else, s = sprintf(fmt, varargin{:}); end
B{end+1} = s;
here = fileparts(mfilename('fullpath'));
if numel(B) == 1
    fid = fopen(fullfile(here,'results','log_c3.txt'),'w','n','UTF-8');
else
    fid = fopen(fullfile(here,'results','log_c3.txt'),'a','n','UTF-8');
end
fprintf(fid,'%s\n', s); fclose(fid);
L = s;
end
