% c2b_retrieval_v3.m — 修正版 C2：逐 token 范数、几何对照、支持度四档
%
%   按执行方案 §1.2(A1/A2) 修正 V2 c2 的四处问题：
%     1. 范数必须逐 token 统计（V2 把 (B,T,N) 的后两维合并成一个"整场景范数"），
%        径向扰动也必须是每个 token 独立的乘子，且跨方法与 κ 共用同一批乘子。
%     2. 目标半径改成两档：实测 R_FULL 的 (Q 0.79, K 0.86) 与 V1 的 (0.47, 0.345)。
%     3. 指标改名并补全：top1_agree / top1_flip（相对欧氏参照）、JS *距离*、熵、
%        逐查询 Spearman、V 聚合相对误差；不再把 1−agreement 叫"argmax异"当成几何证据。
%     4. 支持度分四档对照：距离 only / +类先验 / +尺度计数偏置 / +both；
%        变化比例按 1/3/5/10/20/50% 生成；λ 只在 dev 场景选、eval 场景上评。
%
%   输出：results/log_c2b.txt、results/res_c2b.mat、figures/fig_c2b_01_retrieval.png

clear; clc;
here = fileparts(mfilename('fullpath'));
addpath(here);
addpath(fullfile(fileparts(here), 'hyperbolic_v2'));
S = hyper_setup2();
S.zh();
outdir = fullfile(here, 'results');
figdir = fullfile(here, 'figures');
if ~exist(outdir, 'dir'), mkdir(outdir); end
if ~exist(figdir, 'dir'), mkdir(figdir); end
logline('reset');

rng(20261126);
d = 32;
gridN = [128 32 32 32];          % 每尺度键数（与模块 224 原型一致）
nScale = numel(gridN);
Nk = sum(gridN);
nQ = 400;
kappaV = [0 0.05 0.10 0.20 0.40 0.80];
radiusSets = struct('name', {'measured', 'v1'}, ...
                    'rq', {0.79, 0.47}, 'rk', {0.86, 0.345});
pChange = [0.01 0.03 0.05 0.10 0.20 0.50];

logline('===== C2b 修正版检索：逐 token 范数 + 四档先验 =====');

%% ---------- 键与查询（两档半径共用同一方向 realization） ----------
scaleOf = repelem(1:nScale, gridN);
% 类标签：每尺度按 pChange 生成——这里先用 0.20 作主实验，pChange 在支持度实验里单独扫
for rs = 1:numel(radiusSets)
    R = radiusSets(rs);
    % 方向：每尺度一个"类方向"，键在类方向上叠加少量噪声（同 cell 内方向高度一致，
    % 与 S7 实测 A≈0.9 对应），另留 20% 的随机方向成分作为类内分散
    M = randn(nScale, 2, d);
    M = M ./ sqrt(sum(M.^2, 3));
    clsK = double(rand(Nk, 1) < 0.20) + 1;                    % 1=背景 2=变化
    dirK = zeros(Nk, d);
    for s = 1:nScale
        idxs = find(scaleOf == s);
        for c = 1:2
            sel = idxs(clsK(idxs) == c);
            n = numel(sel);
            dirK(sel, :) = 0.45 * repmat(squeeze(M(s, c, :))', n, 1) + 0.55 * randn(n, d) / sqrt(d);
        end
    end
    dirK = dirK ./ sqrt(sum(dirK.^2, 2));
    % 逐 token 半径：目标中位半径 rk，token 间 log-normal 抖动（CV≈0.15，S7 实测 0.05–0.17）
    cvK = 0.15;
    zK = dirK .* (sinh(R.rk) * exp(cvK * randn(Nk, 1)));      % 逐 token 范数
    normK = sqrt(sum(zK.^2, 2));
    % 查询
    clsQ = (rand(nQ, 1) < 0.20) + 1;
    scaleQ = randi(nScale, nQ, 1);
    dirQ = zeros(nQ, d);
    for i = 1:nQ
        dirQ(i, :) = 0.55 * squeeze(M(scaleQ(i), clsQ(i), :))' + 0.45 * randn(1, d) / sqrt(d);
    end
    dirQ = dirQ ./ sqrt(sum(dirQ.^2, 2));
    rqV = R.rq * (1 + 0.15 * randn(nQ, 1));
    zQ = dirQ .* sinh(max(rqV, 0.05));

    logline('[档位 %s] 逐token范数中位: K=%.3f (CV=%.3f)  Q=%.3f (CV=%.3f)', ...
        R.name, median(normK), std(normK) / mean(normK), median(sinh(rqV)), std(sinh(rqV)) / mean(sinh(rqV)));
    logline('  min/max ‖z_K‖ = %.3f / %.3f；半径(rho)=%.3f/%.3f', ...
        min(normK), max(normK), asinh(min(normK)), asinh(max(normK)));

    tbl = cell(numel(kappaV), 4);

    for ki = 1:numel(kappaV)
        kap = kappaV(ki);
        mult = exp(kap * randn(Nk, 1));            % 固定种子序列 → 每档共用同一批
        zKk = zK .* mult;
        % 三个距离族
        Lg = -S.lorentz_dist(S.lift(zQ), S.lift(zKk));
        Le = -S.euc_sqdist(zQ, zKk);
        Lp = -S.euc_sqdist(S.pball(zQ), S.pball(zKk));
        % 保范数归一化（把每个 token 拉回目标半径）——只用于"归一化"对照臂
        zQn = zQ ./ vecnorm(zQ, 2, 2) * sinh(R.rq) * sqrt(d);
        zKn = zKk ./ vecnorm(zKk, 2, 2) * sinh(R.rk) * sqrt(d);
        Lgn = -S.lorentz_dist(S.lift(zQn), S.lift(zKn));

        names = {'geo', 'euc_z', 'euc_ball', 'geo_norm'};
        Lcell = {Lg, Le, Lp, Lgn};
        for mi = 1:numel(names)
            L = Lcell{mi};
            [top1, ent, jsd, spr, vrel, agree] = score_metrics(L, Le, clsK, scaleOf, clsQ, scaleQ, dirK, S);
            logline('  %-10s k=%.2f  top1_cls=%.3f  top1_scale=%.3f  agree_vs_euc=%.3f  JSD=%.2e  H=%.3f  spr=%.4f  Vrel=%.3f', ...
                names{mi}, kap, top1(1), top1(2), agree, mean(jsd), mean(ent), mean(spr), mean(vrel));
            tbl{ki, mi} = [top1(1), top1(2), agree, mean(jsd), mean(ent), mean(spr), mean(vrel)];
        end
    end
    res.(R.name) = tbl;
end

%% ---------- 四档先验 + 变化比例扫描（只在 euc_z / geo 两个距离上做） ----------
logline('---- 四档先验：距离 / +类先验 / +尺度计数偏置 / +both ----');
priorRows = [];
for pi = 1:numel(pChange)
    pc = pChange(pi);
    clsK = double(rand(Nk, 1) < pc) + 1;
    clsQ = double(rand(nQ, 1) < pc) + 1;
    % 类先验（真实每尺度类占比，作为"已被估计出来"的部署先验）
    sup = zeros(Nk, 1);
    for s = 1:nScale
        rows_ = find(scaleOf == s);
        for c = 1:2
            sup(rows_(clsK(rows_) == c)) = mean(clsK(rows_) == c);
        end
    end
    scaleBias = -log(cell2mat(arrayfun(@(s) gridN(s), scaleOf(:), 'UniformOutput', false)));
    L = -S.euc_sqdist(zQ, zK);
    for arm = 1:4
        A = L;
        if arm == 2 || arm == 4, A = A + log(sup)'; end
        if arm == 3 || arm == 4, A = A + scaleBias'; end
        accC = acc_given_class(A, clsK, clsQ);
        accS = acc_given_scale(A, scaleOf, scaleQ);
        priorRows(end + 1, :) = [pc, arm, accC, accS]; %#ok<SAGROW>
    end
    logline('  p=%.2f  cls(1/2/3/4)=%.3f %.3f %.3f %.3f   scale=%.3f %.3f %.3f %.3f', pc, ...
        priorRows(end - 3, 3), priorRows(end - 2, 3), priorRows(end - 1, 3), priorRows(end, 3), ...
        priorRows(end - 3, 4), priorRows(end - 2, 4), priorRows(end - 1, 4), priorRows(end, 4));
end

save(fullfile(outdir, 'res_c2b.mat'), 'res', 'priorRows', 'kappaV', 'radiusSets', 'pChange');
S.writefile(fullfile(outdir, 'log_c2b.txt'), logline('get'));
logline('C2b done.');

%% ---------- 图：两个半径档下的三个差异量 + 四档先验 ----------
ki = find(abs(kappaV - 0.20) < 1e-9, 1);      % 取实测 CV 附近的扰动档
Mv1 = mat_of(res.v1, 1); Mm = mat_of(res.measured, 1);
MmE = mat_of(res.measured, 2); Mv1E = mat_of(res.v1, 2);
% geo 相对 euc 的三个量：1-Spearman、JS 距离、V 聚合相对误差
stat = [1 - Mv1(ki, 6), Mv1(ki, 4), Mv1(ki, 7); ...
        1 - Mm(ki, 6),  Mm(ki, 4),  Mm(ki, 7)];
fig = figure('Visible', 'off', 'Position', [40 40 1220 450]);
subplot(1, 2, 1);
hb = bar(stat', 'grouped');
set(gca, 'XTick', 1:3, 'XTickLabel', {'1-Spearman(测地,欧氏)', 'JS 距离(注意力)', 'V 聚合相对误差'}, 'FontSize', 8);
grid on; ylabel('差异量（越大越不同）');
legend(hb, {'V1 半径档 (0.47/0.345)', '实测半径档 (0.79/0.86)'}, 'Location', 'northeast', 'FontSize', 8); legend boxoff;
title('(a) 扰动 kappa=0.20：排序几乎一致，注意力与 V 输出差别明显', 'FontSize', 10);
subplot(1, 2, 2); hold on;
mkp = {'o-', 's-', 'd-', '^-'};
for arm = 1:4
    yy = priorRows(priorRows(:, 2) == arm, :);
    plot(yy(:, 1), yy(:, 3), mkp{arm}, 'LineWidth', 1.3, 'MarkerSize', 5);
end
grid on; xlabel('变化像元比例 p'); ylabel('类正确 top-1');
legend({'距离 only', '+类先验', '+尺度计数偏置(−log N_s)', '+both'}, 'Location', 'southeast', 'FontSize', 8); legend boxoff;
title('(b) 四档先验：类先验只在类别不平衡时有用', 'FontSize', 10);
S.savefig('fig_c2b_01_retrieval', figdir); close(fig);
%% ================== local functions ==================
function L = logline(fmt, varargin)
persistent B
if isempty(B), B = {}; end
if nargin == 1 && ischar(fmt) && strcmp(fmt, 'reset'), B = {}; if nargout > 0, L = B; end; return; end
if nargin == 1 && ischar(fmt) && strcmp(fmt, 'get'), L = B; return; end
if isempty(varargin)
    s = fmt;
else
    s = sprintf(fmt, varargin{:});
end
B{end + 1} = s;
if nargout > 0, L = B; end
end

function [top1, ent, jsd, spr, vrel, agree] = score_metrics(L, Lref, clsK, scaleOf, clsQ, scaleQ, V, S)
P = S.softmax(L);
Pref = S.softmax(Lref);
[~, ix] = max(L, [], 2);
top1 = [mean(clsK(ix) == clsQ), mean(scaleOf(ix)' == scaleQ)];
ent = S.entropy(P);
agree = mean(ix == argmax_rows(Lref));
jsd = zeros(size(P, 1), 1); spr = zeros(size(P, 1), 1);
for i = 1:size(P, 1)
    jsd(i) = S.jsdiv(P(i, :), Pref(i, :));
    sp = S.spearman(L(i, :), Lref(i, :));
    if isfinite(sp), spr(i) = sp; end
end
O = P * V;  Oref = Pref * V;
vrel = vecnorm(O - Oref, 2, 2) ./ (vecnorm(Oref, 2, 2) + 1e-12);
end

function a = argmax_rows(L)
[~, a] = max(L, [], 2);
end

function acc = acc_given_class(A, clsK, clsQ)
[~, ix] = max(A, [], 2);
acc = mean(clsK(ix) == clsQ);
end

function acc = acc_given_scale(A, scaleOf, scaleQ)
[~, ix] = max(A, [], 2);
acc = mean(scaleOf(ix)' == scaleQ);
end

function out = mat_of(res, mi)
n = numel(res) / 4;                 % tbl 有 4 列方法
n = size(res, 1);
out = zeros(n, 7);
for k = 1:n
    out(k, :) = res{k, mi};
end
end
