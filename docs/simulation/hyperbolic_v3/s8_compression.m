% s8_compression.m — S8 原型压缩保真：固定网格 vs 自适应分配（等预算公平对照）
%
%   执行方案 §4 要求：同预算、同样本、同 guide p 权重、同每区域类原型数，
%   比 FIX-E / FIX-H / ADA-E / ADA-H 四臂；四臂都要在同一组度量下评价
%   （欧氏重构、双曲平方测地、Lorentz 弦代价、方向余弦），不能只用各自"自家"的代价。
%
%   本脚本在合成数据上做 S8a sanity（六工况），因为真实特征四臂需要 S7 的张量落盘
%   （见 README 的"未做项"）。合成数据仍用真实口径：
%     - token 是空间向量 z（已含 /sqrt(D)），半径 r=asinh(||z||)
%     - 类条件原型：E 臂用切空间加权均值，H 臂用 Lorentz 加权重心（OEM alpha=0）
%     - 划分代价：E 臂用欧氏 SSE，H 臂用 2(a-M) 弦代价（对弦代价精确）
%     - 预算 B 固定为区域数，两臂都恰好用满（gain=0 也继续分裂到预算）
%
%   输出：results/log_s8.txt、results/res_s8.mat、figures/fig_s8_01_regimes.png
%         figures/fig_s8_02_deltas.png

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

d = 8; nTok = 1024; budgets = [32 224];
regNames = {'单模态一致变化', '等半径反向', '多模态（2 个方向）', ...
            '稀疏小目标（3%）', '噪声/伪变化', '边界混合'};
logline('===== S8 原型压缩保真：FIX vs ADA（等预算公平对照） =====');
logline('设置：d=%d，token=%d，预算 B=%s；四臂同预算、同样本、同类原型数=每区域 2 个', ...
    d, nTok, mat2str(budgets));
logline('度量：四臂全部同时报 欧氏SSE / 平方测地 / 弦代价 2(cosh d-1) / 方向余弦');
logline('原型：四臂一律 Lorentz 加权重心（与模块一致）→ 四臂差异只来自划分规则');
logline('');

allRes = struct([]);
for ri = 1:numel(regNames)
    [Z, lab, pos] = gen_regime(ri, d, nTok);
    w = ones(nTok, 1);          % guide p 权重（本合成里先用均匀权重，p 只影响代价加权）
    line = sprintf('[工况 %d] %s', ri, regNames{ri});
    logline(line);
    for bi = 1:numel(budgets)
        B = budgets(bi);
        met = zeros(4, 4);      % 臂 × 度量：1=EuclidSSE 2=平方测地 3=弦代价 4=1-余弦
        for arm = 1:4
            isH = arm > 2;
            isAda = mod(arm, 2) == 0;    % 2=ADA-E, 4=ADA-H
            regions = compress(Z, lab, w, pos, B, isH, isAda);
            met(arm, :) = eval_regions(Z, lab, regions, isH);
        end
        names = {'FIX-E', 'ADA-E', 'FIX-H', 'ADA-H'};
        logline('  B=%d：EuclidSSE  %s', B, mat2str(met(:, 1)', 4));
        logline('       平方测地    %s', mat2str(met(:, 2)', 4));
        logline('       弦代价      %s', mat2str(met(:, 3)', 4));
        logline('       1-余弦      %s', mat2str(met(:, 4)', 4));
        logline('       ADA−FIX：E 臂 %.4f、H 臂 %.4f（负=自适应更好，按 EuclidSSE）', ...
            met(2, 1) - met(1, 1), met(4, 1) - met(3, 1));
        logline('       H−E：FIX %.4f、ADA %.4f（按 EuclidSSE）；按平方测地 FIX %.4f、ADA %.4f', ...
            met(3, 1) - met(1, 1), met(4, 1) - met(2, 1), met(3, 2) - met(1, 2), met(4, 2) - met(2, 2));
        allRes = [allRes; struct('regime', ri, 'B', B, 'met', met, 'names', {names})]; %#ok<SAGROW>
    end
    logline('');
end

logline('---- 口径提醒 ----');
logline('合成数据的结论只用于检查"实现是否自洽 + 方向是否可能"，不能当真实特征的证据；');
logline('真实四臂要等 S7 张量落盘后在同 checkpoint 上跑（见 README 未做项）。');

save(fullfile(outdir, 'res_s8.mat'), 'allRes');
S.writefile(fullfile(outdir, 'log_s8.txt'), logline('get'));

%% ---- 图 1：六工况的 EuclidSSE 柱状（FIX-E/ADA-E/FIX-H/ADA-H，B=32） ----
M1 = zeros(numel(regNames), 4);
for i = 1:numel(allRes)
    if allRes(i).B == budgets(1)
        M1(allRes(i).regime, :) = allRes(i).met(:, 1)';
    end
end
fig = figure('Visible', 'off', 'Position', [40 40 1300 430]);
subplot(1, 2, 1);
hb = bar(M1, 'grouped');
set(gca, 'XTick', 1:numel(regNames), 'XTickLabel', {'R1 单模态', 'R2 反向', 'R3 多模态', 'R4 稀疏', 'R5 噪声', 'R6 边界'}, 'FontSize', 8);
grid on; ylabel('欧氏 SSE（越低越好）');
legend(hb, {'FIX-E', 'ADA-E', 'FIX-H', 'ADA-H'}, 'Location', 'north', 'FontSize', 8); legend boxoff;
title(sprintf('(a) 六工况保真（B=%d，同一组度量）', budgets(1)), 'FontSize', 10);
subplot(1, 2, 2);
D = [M1(:, 2) - M1(:, 1), M1(:, 4) - M1(:, 3), M1(:, 3) - M1(:, 1), M1(:, 4) - M1(:, 2)];
hb2 = bar(D, 'grouped');
set(gca, 'XTick', 1:numel(regNames), 'XTickLabel', {'R1', 'R2', 'R3', 'R4', 'R5', 'R6'}, 'FontSize', 8);
grid on; ylabel('\Delta 欧氏 SSE（负=更好）');
legend(hb2, {'ADA−FIX @E', 'ADA−FIX @H', 'H−E @FIX（恒为 0）', 'H−E @ADA'}, ...
       'Location', 'southoutside', 'Orientation', 'horizontal', 'NumColumns', 2, 'FontSize', 7.2);
legend boxoff;
title('(b) 四个对照差', 'FontSize', 10);
S.savefig('fig_s8_01_regimes', figdir); close(fig);

logline('S8 done.');

%% ================== 生成器 ==================
function [Z, lab, pos] = gen_regime(ri, d, n)
% lab: 0/1 变化标签；pos: n×2 归一化空间坐标（用于 dyadic 划分）
rng(100 + ri);
pos = rand(n, 2);
switch ri
    case 1      % 单模态一致变化
        m = randn(1, d) / sqrt(d);
        base = 0.8 * randn(n, d) / sqrt(d);
        lab = rand(n, 1) < 0.2;
        Z = base + lab .* (1.6 * repmat(m, n, 1));
    case 2      % 等半径反向：变化方向与背景恰好相反
        lab = rand(n, 1) < 0.2;
        dirs = randn(n, d); dirs = dirs ./ sqrt(sum(dirs .^ 2, 2));
        r = 0.5 + 0.02 * randn(n, 1);
        Z = (r .* (sinh(1.0) ./ max(1, 1))) .* dirs;
        Z(lab, :) = -Z(lab, :);      % 变化 = 反方向、同半径
    case 3      % 多模态：两个变化方向
        m1 = randn(1, d) / sqrt(d); m2 = randn(1, d) / sqrt(d); m2 = m2 - (m2 * m1') * m1; m2 = m2 / norm(m2);
        lab = rand(n, 1) < 0.3;
        sel = rand(n, 1) < 0.5;
        base = 0.7 * randn(n, d) / sqrt(d);
        Z = base + 1.4 * (lab .* (sel .* repmat(m1, n, 1) + (~sel) .* repmat(m2, n, 1)));
    case 4      % 稀疏小目标：3% 正类，集中在一小块
        lab = false(n, 1);
        c = [0.35 0.65];
        dist = sqrt(sum((pos - c) .^ 2, 2));
        idx = find(dist < 0.08);
        if numel(idx) > 3, lab(idx) = true; end
        m = randn(1, d) / sqrt(d);
        Z = 0.7 * randn(n, d) / sqrt(d) + lab .* (2.2 * repmat(m, n, 1));
    case 5      % 噪声/伪变化：正类方向随机，幅度与噪声同量级
        lab = rand(n, 1) < 0.25;
        Z = 0.9 * randn(n, d) / sqrt(d) + lab .* (0.5 * randn(n, d) / sqrt(d));
    otherwise   % 边界混合：沿一条折线混合
        lab = pos(:, 1) > 0.5;
        m = randn(1, d) / sqrt(d);
        edge = exp(-((pos(:, 1) - 0.5) .^ 2) / 0.002);
        Z = 0.7 * randn(n, d) / sqrt(d) + (lab .* (1.2 + 0.8 * edge)) .* repmat(m, n, 1);
end
lab = logical(lab);
end

%% ================== 压缩：FIX / ADA × E/H ==================
function regions = compress(Z, lab, w, pos, B, isH, isAda)
% 返回 regions：结构体数组，字段 idx（成员行）、boxes（不用于评价，仅供调试）
regions = struct('idx', {1:size(Z, 1)});
if isAda
    while numel(regions) < B
        % 贪心：对每个区域算"分裂收益"，取最大者分裂；收益相同用确定性 tie-break
        best = -inf; bi = 1; bs = 0;
        for i = 1:numel(regions)
            idx = regions(i).idx;
            if numel(idx) < 4, continue; end
            [~, g] = split_gain(Z, lab, w, idx, isH);
            if g > best + 1e-12, best = g; bi = i; bs = 1; end
        end
        if bs == 0, break; end
        idx = regions(bi).idx;
        [two, ~] = split_gain(Z, lab, w, idx, isH);
        regions = [regions(1:bi - 1), struct('idx', {two{1}}), struct('idx', {two{2}}), regions(bi + 1:end)];
    end
else
    % FIX：dyadic 空间四叉树，按 pos 的几何中心递归二分到 B
    regions = quadtree(pos, 1:size(Z, 1), B);
end
end

function [two, gain] = split_gain(Z, lab, w, idx, isH)
% 沿空间第一主方向的中位点二分（确定性），返回两部分与代价下降量
X = Z(idx, :);
if isH
    mu = X ./ max(sqrt(sum(X .^ 2, 2)), 1e-12);
    [~, ~, V] = svd(mu - mean(mu, 1), 'econ');
    pr = mu * V(:, 1);
else
    pr = X * (X' * X / max(numel(idx), 1)) * ones(size(X, 2), 1);
end
thr = median(pr);
a = idx(pr <= thr); b = idx(pr > thr);
if isempty(a) || isempty(b), a = idx(1:floor(end/2)); b = idx(floor(end/2)+1:end); end
J0 = region_cost(Z, lab, w, idx, isH);
J1 = region_cost(Z, lab, w, a, isH) + region_cost(Z, lab, w, b, isH);
gain = J0 - J1;
two = {a, b};
end

function J = region_cost(Z, lab, w, idx, isH)
% 类条件代价：E 臂 SSE，H 臂 2(a-M) 弦代价（按类归一化后相加）
J = 0;
for c = 0:1
    sel = idx(lab(idx) == c);
    if numel(sel) < 1, continue; end
    X = Z(sel, :); ww = w(sel); ww = ww / sum(ww);
    if isH
        p = S_h_pool(X, ww);                       % Lorentz 加权重心（空间部分）
        D = S_h_dist(X, p);                         % 弦代价 2(cosh d - 1)
        J = J + sum(ww .* D) / max(numel(sel), 1);
    else
        mu = sum(X .* ww, 1);
        J = J + sum(ww .* sum((X - mu) .^ 2, 2)) / max(numel(sel), 1);
    end
end
end

function regions = quadtree(pos, idx, B)
regions = struct('idx', {idx});
while numel(regions) < B
    % 分裂成员最多的区域
    [~, mi] = max(arrayfun(@(r) numel(r.idx), regions));
    id = regions(mi).idx;
    if numel(id) < 2, break; end
    P = pos(id, :);
    [~, ax] = max(max(P, [], 1) - min(P, [], 1));
    thr = median(P(:, ax));
    a = id(P(:, ax) <= thr); b = id(P(:, ax) > thr);
    if isempty(a) || isempty(b), break; end
    regions = [regions(1:mi - 1), struct('idx', {a}), struct('idx', {b}), regions(mi + 1:end)];
end
end

%% ================== 评价：四臂同一组度量 ==================
function met = eval_regions(Z, lab, regions, isH)
% met = [欧氏SSE, 平方测地, 弦代价, 1-余弦]，按类条件、按成员数加权
sse = 0; gsq = 0; chord = 0; cosm = 0; ntot = 0;
for i = 1:numel(regions)
    idx = regions(i).idx;
    for c = 0:1
        sel = idx(lab(idx) == c);
        if isempty(sel), continue; end
        X = Z(sel, :); n = numel(sel); ntot = ntot + n;
        % 原型一律用 Lorentz 加权重心（与真实模块 prototype_pooling=lorentz 一致），
        % 这样四臂的差异只来自"划分规则"，不再混入原型类型。
        p = S_h_pool(X, ones(n, 1) / n);
        sse = sse + sum(sum((X - p) .^ 2, 2));
        gsq = gsq + sum(S_g_dist2(X, p));
        chord = chord + sum(S_h_dist(X, p));
        if norm(p) > 1e-12
            cs = (X * p') ./ max(sqrt(sum(X .^ 2, 2)) * norm(p), 1e-12);
            cosm = cosm + sum(1 - cs);
        end
    end
end
met = [sse, gsq, chord, cosm] / max(ntot, 1);
end

function D2 = S_g_dist2(X, p)
% 平方测地（曲率 -1）：cosh d = sqrt(1+|x|^2)sqrt(1+|p|^2) - x·p
tx = sqrt(1 + sum(X .^ 2, 2)); tp = sqrt(1 + sum(p .^ 2));
C = tx * tp - X * p';
D2 = acosh(max(C, 1)) .^ 2;
end

function D = S_h_dist(X, p)
C = sqrt(1 + sum(X .^ 2, 2)) * sqrt(1 + sum(p .^ 2)) - X * p';
D = 2 * (max(C, 1) - 1);
end

function p = S_h_pool(X, w)
% Lorentz 加权重心（确定性 Lift → 加权和 → 归一化 → 空间部分）
w = w(:) / sum(w);
t = sqrt(1 + sum(X .^ 2, 2));
M = [t, X];
mm = (w' * M)';
n2 = mm(1) ^ 2 - sum(mm(2:end) .^ 2);
c = mm / sqrt(max(n2, 1e-12));
p = c(2:end)';
end

%% ================== 日志 ==================
function L = logline(fmt, varargin)
persistent B
if isempty(B), B = {}; end
if nargin == 1 && ischar(fmt) && strcmp(fmt, 'reset'), B = {}; if nargout > 0, L = B; end; return; end
if nargin == 1 && ischar(fmt) && strcmp(fmt, 'get'), L = B; return; end
if isempty(varargin), s = fmt; else, s = sprintf(fmt, varargin{:}); end
B{end + 1} = s;
if nargout > 0, L = B; end
end
