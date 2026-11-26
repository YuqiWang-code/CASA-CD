% c2c_retrieval_v3.m — 修正版 C2（对应执行方案 §3 A1 / A2）
%
%   A1 要修的错（对照 c2b）：
%     1. 距离评分必须与模型一致：负"平方"距离 + 真实温度 tau_s，不是负距离、不是无温度。
%     2. geo_norm 曾把目标范数再乘 sqrt(d)，但输入已是缩放后的 z —— 这会把目标半径推大。
%        → 本脚本只做"同目标半径"的保范重标，且显式用 sinh(rho_target) 作用在 z 上。
%     3. 半径档循环内不许重抽 M/方向/噪声 —— 场景、方向、基础噪声先生成，四族几何与两档半径共用。
%     4. kappa 用一个固定标准正态数组乘 kappa（同一批扰动跨方法共享），不每档重抽。
%     5. 半径档取真实 S7 的 rho_Q / rho_K 分尺度统计（见 RHO 常量与出处），不用 Q/K 均值比反解。
%     6. V 用独立于 K 的值向量（不是单位键方向冒充），两几何共享同一 V。
%     7. 指标：逐查询 Spearman、top1 一致/翻转、JS *距离*（sqrt）、熵、V 绝对与相对 RMS、余弦；
%        另报"单一温度+行偏置能吸收多少 logit 差异"（相对 logit 跨度的残差）。
%
%   A2 要修的错：
%     1. 每个先验比例 p 重新生成"成对特征+标签"（不能沿用上一档的 z 再重贴标签）。
%     2. 四档先验（distance only / +类先验 / +尺度计数偏置 / +both）在 geo 与 euc 上都跑。
%     3. lambda=1 为模型匹配臂；需要调 lambda 时只用 dev 场景选，再冻结到 eval 场景。
%     4. 报 AP、变化召回/精确率、balanced accuracy；背景占比高时总体正确率不算证据。
%
% 输出：results/log_c2c.txt、results/res_c2c.mat、figures/fig_c2c_01_distance.png
%       figures/fig_c2c_02_prior.png
%
% 口径提醒：本脚本是 synthetic sanity；真实结论由 S9a 决定（执行方案 §3 验收）。

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

d = 32;                     % 与 Run10 qkv_dim 一致
tau = 0.4696;               % FROZEN_TAU（hyperbolic_context.FROZEN_TAU）
nScene = 5;                 % 独立场景种子（A1 要求 >=5）
nQ = 200;                   % 每场景查询数
nPerSlot = 28;              % 每 (尺度,类) 成员数 -> 4 尺度 x 2 类 x 28 = 224 键
nScale = 4;
% 真实 S7 半径（R_FULL-2334 SYSU / LEVIR 的 Q 均值与 K 成员加权均值；见 S7_S9_REPORT.md）
% 真实 S7 的 Q 半径（R_FULL-2334，val 120 图均值）；K 半径没有独立实测，
% 用 S7 报的 K/Q 保留比换算（SYSU 0.923/0.859/0.860/0.871、LEVIR 0.875/0.777/0.792/0.903），
% 名字里带 ratio 以免被当成独立测量。
RHO = struct( ...
    'name', {'S7-SYSU2334', 'S7-LEVIR2334', 'EQUAL'}, ...
    'q',    {[0.971 0.871 0.891 0.940], [0.933 1.131 1.167 1.222], [0.49 0.49 0.49 0.49]}, ...
    'k',    {[0.971 0.871 0.891 0.940] .* [0.923 0.859 0.860 0.871], ...
             [0.933 1.131 1.167 1.222] .* [0.875 0.777 0.792 0.903], ...
             [0.49 0.49 0.49 0.49]});
kappaV = [0 0.05 0.1 0.2 0.4 0.8];

logline('===== C2c：与模型一致的平方距离 + 温度（A1） =====');
logline('设置：d=%d，tau=%.4f，场景=%d，查询/场景=%d，成员/槽=%d（224 键）', ...
    d, tau, nScene, nQ, nPerSlot);
logline('半径档：S7-SYSU2334 / S7-LEVIR2334 / EQUAL（真实 S7 统计，不是 Q/K 比反解）');
logline('几何族：geo(平方测地) / euc_z(同 z 的平方欧氏) / euc_p(仅坐标诊断) / geo_norm(保范重标后测地)');
logline('');

%% ================== 数据生成（一次，跨半径档/几何/κ 共用） ==================
nK = nScale * 2 * nPerSlot;
Rc = cell(nScene, 1);
for si = 1:nScene
    rng(2000 + si);
    D = struct();
    D.uQ = randn(nQ, d);          % 方向（后面按行归一化）
    D.uK = randn(nK, d);
    D.V = randn(nK, d);           % 独立值向量（不是单位键方向）
    D.slot = repelem((1:nScale)', 2 * nPerSlot);
    D.cls = repmat(repelem((0:1)', nPerSlot), nScale, 1);
    D.noise = randn(nQ, 1);       % 供形状扰动复用
    Rc{si} = D;
end

tbl = cell(0, 1);
for ri = 1:numel(RHO)
    rq = RHO(ri).q; rk = RHO(ri).k;
    for ki = 1:numel(kappaV)
        kap = kappaV(ki);
        % 每场景对每个查询算一次指标，再按场景平均
        acc = zeros(nScene, 9);
        for si = 1:nScene
            D = Rc{si};
            rng(3000 + 11 * si);      % 固定该场景的扰动 realization
            uQ = D.uQ ./ sqrt(sum(D.uQ .^ 2, 2));
            uK = D.uK ./ sqrt(sum(D.uK .^ 2, 2));
            V = D.V;
            % 逐 token 独立 radial 乘子（同一批跨几何/κ 档共享）
            mQ = exp(kap * randn(nQ, 1) - 0.5 * kap ^ 2);
            mK = exp(kap * randn(nK, 1) - 0.5 * kap ^ 2);
            % 查询按尺度轮转（均匀），保证每尺度都有查询
            sq = mod((1:nQ)' - 1, nScale) + 1;
            rQ = rq(sq)' .* mQ;
            rK = rk(D.slot)' .* mK;
            zQ = uQ .* sinh(rQ);
            zK = uK .* sinh(rK);
            % 四族距离（全部用平方距离）
            DH2 = dist_geo_sq(S, zQ, zK);
            DE2 = dist_euc_sq(zQ, zK);
            DP2 = dist_euc_sq(S.pball(zQ), S.pball(zK));
            % 保范重标：把每个 token 的范数换成目标半径（不再乘 sqrt(d)）
            rzQ = norm_rows(zQ); rzK = norm_rows(zK);
            zQn = zQ ./ max(rzQ, 1e-12) .* sinh(rQ);
            zKn = zK ./ max(rzK, 1e-12) .* sinh(rK);
            DGN2 = dist_geo_sq(S, zQn, zKn);
            % 温度化 logits（先验先不加，A2 单独扫）
            L = struct();
            L.H = -DH2 / (2 * tau ^ 2);
            L.E = -DE2 / (2 * tau ^ 2);
            L.P = -DP2 / (2 * tau ^ 2);
            L.N = -DGN2 / (2 * tau ^ 2);
            % 指标
            PH = S.softmax(L.H); PE = S.softmax(L.E);
            acc(si, 1) = mean(argmax_rows(PH) == argmax_rows(PE));
            acc(si, 2) = mean(js_dist(PH, PE));
            acc(si, 3) = mean(spearman_rows(L.H, L.E));
            acc(si, 4) = mean(rowent(PH));
            OH = PH * V; OE = PE * V;
            acc(si, 5) = rms(OH - OE);
            acc(si, 6) = rms(OH - OE) / max(rms(OH), 1e-12);
            acc(si, 7) = mean(cos_rows(OH, OE));
            % 单一温度+行偏置的残差（相对 logit 跨度）
            acc(si, 8) = absorb_resid(L.H, L.E);
            % JS 距离 geo vs 保范重标（更大的干预）
            acc(si, 9) = mean(js_dist(PH, S.softmax(L.N)));
        end
        m = mean(acc, 1); sd = std(acc, 0, 1) / sqrt(nScene);
        tbl{end + 1, 1} = struct('rho', RHO(ri).name, 'kappa', kap, 'm', m, 'se', sd);
        logline('[%s κ=%.2f] top1一致=%.3f JSD=%.4f Spearman=%.4f H=%.3f |V|差=%.3f Vrel=%.3f cos=%.4f | 温度吸收残差=%.3f | 保范JSD=%.4f', ...
            RHO(ri).name, kap, m(1), m(2), m(3), m(4), m(5), m(6), m(7), m(8), m(9));
    end
    logline('');
end
logline('读法：top1一致 是两类几何 argmax 相同的比例（1=完全一致）；JSD 是 JS *距离*（0–1，不是百分比）；');
logline('      Vrel 是 V 聚合的相对 RMS 差；温度吸收残差 = 用单一标量温度+行偏置拟合后，');
logline('      剩余 logit 差相对该行 logit 跨度的比例（越小说明差异越接近"温度+偏置"）。');
logline('');

%% ================== A2：先验扫描（每 p 重新生成成对特征+标签） ==================
logline('===== C2c：先验扫描（A2：每 p 重新生成特征与标签，四档先验 × 两几何） =====');
logline('评价用平衡查询集（一半变化、一半背景），报 变化召回 / 精确率 / 平衡准确率 / AP。');
pList = [1 3 5 10 20 50] / 100;
priorNames = {'distance only', '+类先验', '+尺度计数偏置', '+both'};
priorTbl = cell(0, 1);
for pi = 1:numel(pList)
    p = pList(pi);
    for gi = 1:2
        geoName = 'geo'; if gi == 2, geoName = 'euc'; end
        accP = zeros(nScene, 6);   % 每场景：4 档先验的平衡准确率 + 固定臂 AP 占位
        for si = 1:nScene
            rng(5000 + 137 * pi + 11 * si);   % 每个 (p, 场景) 独立生成
            % ---- 键：每尺度 2 类，成员数按 p 分配（变化槽 p 比例、背景槽 1-p） ----
            nPer = 24;
            nKp = nScale * 2 * nPer;                 % 本段键数（与第一段不同，别混用 nK）
            slotS = repelem((1:nScale)', 2 * nPer);
            clsK = repmat(repelem((0:1)', nPer), nScale, 1);   % 0=背景 1=变化
            % 类质量：每槽按 p 控制"变化质量占比"，用于类先验；计数用于尺度偏置
            mChg = p * ones(nKp, 1);
            mBg  = (1 - p) * ones(nKp, 1);
            % 按类给不同半径与方向（变化类整体更远，模拟真实统计）
            rK = 0.85 + 0.10 * clsK;
            uK2 = randn(nKp, d); uK2 = uK2 ./ sqrt(sum(uK2 .^ 2, 2));
            zK2 = uK2 .* sinh(rK);
            % ---- 查询：平衡一半变化、一半背景 ----
            nQc = 25; nQb = 25;
            % 变化查询：与"变化类原型方向"相关；背景查询：与背景方向相关
            dirC = mean(uK2(clsK == 1, :), 1); dirC = dirC / norm(dirC);
            dirB = mean(uK2(clsK == 0, :), 1); dirB = dirB / norm(dirB);
            zQc = (repmat(dirC, nQc, 1) + 0.55 * randn(nQc, d) / sqrt(d));
            zQc = zQc ./ sqrt(sum(zQc .^ 2, 2)) * sinh(0.95);
            zQb = (repmat(dirB, nQb, 1) + 0.55 * randn(nQb, d) / sqrt(d));
            zQb = zQb ./ sqrt(sum(zQb .^ 2, 2)) * sinh(0.95);
            zQ3 = [zQc; zQb];
            yTrue = [ones(nQc, 1); zeros(nQb, 1)];
            if gi == 1
                Lq = -dist_geo_sq(S, zQ3, zK2) / (2 * tau ^ 2);
            else
                Lq = -dist_euc_sq(zQ3, zK2) / (2 * tau ^ 2);
            end
            Lq = bsxfun(@minus, Lq, max(Lq, [], 2));
            % ---- 四档先验 ----
            logMassC = log(max(sum(mChg(clsK == 1)), 1e-9));
            logMassB = log(max(sum(mBg(clsK == 0)), 1e-9));
            priorCls = zeros(nKp, 1);
            priorCls(clsK == 1) = logMassC - logMassB;   % 相对背景的类先验
            priorCnt = -log(nPer) * ones(nKp, 1);         % 尺度计数偏置（同尺度内是常数）
            variants = {zeros(nKp, 1), priorCls, priorCnt, priorCls + priorCnt};
            for vi = 1:4
                Lv = bsxfun(@plus, Lq, variants{vi}');
                Pv = S.softmax(Lv);
                massC = sum(Pv(:, clsK == 1), 2);
                massB = sum(Pv(:, clsK == 0), 2);
                yhat = massC > massB;
                % 平衡准确率 = (变化召回 + 背景召回)/2
                recC = mean(yhat(yTrue == 1));
                recB = mean(~yhat(yTrue == 0));
                accP(si, vi) = 0.5 * (recC + recB);
                if vi == 2
                    accP(si, 5) = recC;                          % 变化召回（+类先验）
                    P = massC ./ max(massC + massB, 1e-12);
                    accP(si, 6) = ap_score(P, yTrue);            % AP（+类先验）
                end
            end
        end
        m = mean(accP, 1);
        priorTbl{end + 1, 1} = struct('p', p, 'geo', geoName, 'm', m);
        logline('[p=%2.0f%% %s] 平衡准确率：距离only=%.3f  +类先验=%.3f  +计数偏置=%.3f  +both=%.3f | +类先验 recall=%.3f AP=%.3f', ...
            100 * p, geoName, m(1), m(2), m(3), m(4), m(5), m(6));
    end
    logline('');
end
logline('读法：平衡准确率=0.5 等于随机；"距离only"没有类先验，因此当两类质量差异大时会偏向多数类。');
logline('      计数偏置在同尺度内是常数，理论上不改同一尺度内的类判决，只改跨尺度竞争。');

save(fullfile(outdir, 'res_c2c.mat'), 'tbl', 'priorTbl', 'RHO', 'kappaV', 'd', 'tau');
S.writefile(fullfile(outdir, 'log_c2c.txt'), logline('get'));

%% ================== 图 1：距离族 × κ（JSD / 温度吸收残差） ==================
names = {RHO.name};
cl = lines(numel(names));
fig = figure('Visible', 'off', 'Position', [40 40 1240 460]);
subplot(1, 3, 1); hold on;
for ri = 1:numel(names)
    y = zeros(1, numel(kappaV));
    for ki = 1:numel(kappaV)
        y(ki) = findrow(tbl, names{ri}, kappaV(ki), 2);
    end
    plot(kappaV, y, 'o-', 'Color', cl(ri, :), 'LineWidth', 1.4, 'MarkerSize', 5);
end
grid on; xlabel('逐 token 半径扰动 \kappa'); ylabel('JS 距离（geo vs euc）');
legend(names, 'Location', 'northwest', 'FontSize', 8); legend boxoff;
title('(a) JS 距离：排序接近但注意力不同', 'FontSize', 10);
subplot(1, 3, 2); hold on;
for ri = 1:numel(names)
    y = zeros(1, numel(kappaV));
    for ki = 1:numel(kappaV), y(ki) = findrow(tbl, names{ri}, kappaV(ki), 8); end
    plot(kappaV, y, 's-', 'Color', cl(ri, :), 'LineWidth', 1.4, 'MarkerSize', 5);
end
grid on; xlabel('\kappa'); ylabel('温度+行偏置后的残差 / logit 跨度');
title('(b) 几何差异能否被温度吸收', 'FontSize', 10);
subplot(1, 3, 3); hold on;
for ri = 1:numel(names)
    y = zeros(1, numel(kappaV));
    for ki = 1:numel(kappaV), y(ki) = findrow(tbl, names{ri}, kappaV(ki), 6); end
    plot(kappaV, y, 'd-', 'Color', cl(ri, :), 'LineWidth', 1.4, 'MarkerSize', 5);
end
grid on; xlabel('\kappa'); ylabel('V 聚合相对 RMS 差');
title('(c) 差异能否传到 V', 'FontSize', 10);
S.savefig('fig_c2c_01_distance', figdir); close(fig);

%% ================== 图 2：先验扫描 ==================
fig = figure('Visible', 'off', 'Position', [40 40 980 400]);
for gi = 1:2
    subplot(1, 2, gi); hold on;
    hb = zeros(1, 4);
    for vi = 1:4
        y = zeros(1, numel(pList));
        for pi = 1:numel(pList)
            y(pi) = findprior(priorTbl, pList(pi), gi, vi);
        end
        hb(vi) = plot(100 * pList, y, 'o-', 'LineWidth', 1.3, 'MarkerSize', 5);
    end
    grid on; xlabel('变化像元比例 p（%）'); ylabel('变化召回');
    legend(hb, priorNames, 'Location', 'northwest', 'FontSize', 7.5); legend boxoff;
    if gi == 1, t = '(a) 测地距离'; else, t = '(b) 欧氏距离'; end
    title(t, 'FontSize', 10);
end
S.savefig('fig_c2c_02_prior', figdir); close(fig);

logline('C2c done.');

%% ================== 局部函数 ==================
function v = findrow(tbl, name, kap, col)
v = NaN;
for i = 1:numel(tbl)
    if strcmp(tbl{i}.rho, name) && abs(tbl{i}.kappa - kap) < 1e-12
        v = tbl{i}.m(col); return;
    end
end
end

function v = findprior(tbl, p, gi, col)
v = NaN;
for i = 1:numel(tbl)
    if abs(tbl{i}.p - p) < 1e-12
        if (gi == 1 && strcmp(tbl{i}.geo, 'geo')) || (gi == 2 && strcmp(tbl{i}.geo, 'euc'))
            v = tbl{i}.m(col); return;
        end
    end
end
end

function D2 = dist_geo_sq(S, X, Y)
% 平方测地距离（曲率 -1）。直接用工具箱的 lorentz_dist，避免自己写内积符号错。
% 注意：lift 返回 [t, z]，普通点积 +t_i t_j + z_i·z_j 不是 Minkowski 内积，
% 余弦形式为 cosh d = t_i t_j - z_i·z_j（c1 已核对 2.7e-15）。
D = S.lorentz_dist(S.lift(X), S.lift(Y));
D2 = D .^ 2;
end

function D2 = dist_euc_sq(X, Y)
D2 = max(sum(X .^ 2, 2) + sum(Y .^ 2, 2)' - 2 * (X * Y'), 0);
end

function n = norm_rows(X)
n = sqrt(sum(X .^ 2, 2));
end

function P = argmax_rows(X)
[~, P] = max(X, [], 2);
end

function r = spearman_rows(A, B)
n = size(A, 1); r = zeros(n, 1);
for i = 1:n
    r(i) = corr(rank_avg(A(i, :)'), rank_avg(B(i, :)'));
end
r = r(~isnan(r));
end

function r = rank_avg(x)
x = x(:);
[s, ord] = sort(x);
r = zeros(numel(x), 1);
i = 1;
while i <= numel(x)
    j = i;
    while j < numel(x) && abs(s(j + 1) - s(i)) < 1e-12, j = j + 1; end
    r(ord(i:j)) = (i + j) / 2;
    i = j + 1;
end
end

function d = js_dist(P, Q)
% JS 距离 = sqrt(JS 散度)，行向量
P = max(P, 1e-12); Q = max(Q, 1e-12);
P = bsxfun(@rdivide, P, sum(P, 2));
Q = bsxfun(@rdivide, Q, sum(Q, 2));
M = 0.5 * (P + Q);
d = sqrt(max(0.5 * (sum(P .* log(P ./ M), 2) + sum(Q .* log(Q ./ M), 2)), 0));
end

function h = rowent(P)
P = max(P, 1e-12);
P = bsxfun(@rdivide, P, sum(P, 2));
h = -sum(P .* log(P), 2);
end

function c = cos_rows(A, B)
na = sqrt(sum(A .^ 2, 2)); nb = sqrt(sum(B .^ 2, 2));
c = sum(A .* B, 2) ./ max(na .* nb, 1e-12);
end

function res = absorb_resid(LH, LE)
% 用单一标量 a 与逐行偏置拟合 Lg ≈ a*Le + b，报残差 / 该行 logit 跨度
x = LE(:); y = LH(:);
n = size(LE, 1);
xc = x - mean(x); yc = y - mean(y);
a = (xc' * yc) / max(xc' * xc, 1e-12);
r = yc - a * xc;
span = mean(max(LH, [], 2) - min(LH, [], 2));
res = rms(r) / max(span, 1e-12);
end

function ap = ap_score(score, y)
% 平均精度（AP）：按 score 降序，逐步算 precision 的加权平均
[~, ord] = sort(score, 'descend');
y = y(ord);
tp = cumsum(y); fp = cumsum(1 - y);
prec = tp ./ max(tp + fp, 1e-12);
nPos = sum(y);
if nPos == 0, ap = NaN; return; end
rec = tp / nPos;
ap = sum(prec(y == 1) .* (diff([0; rec(y == 1)]))) ;
if isempty(ap) || isnan(ap), ap = mean(prec(y == 1)); end
end

function v = rms(x)
v = sqrt(mean(x(:) .^ 2));
end

function r = corr(a, b)
a = a(:); b = b(:);
if numel(a) < 3 || std(a) < 1e-12 || std(b) < 1e-12, r = NaN; return; end
r = mean((a - mean(a)) .* (b - mean(b))) / (std(a) * std(b));
end

function L = logline(fmt, varargin)
persistent B
if isempty(B), B = {}; end
if nargin == 1 && ischar(fmt) && strcmp(fmt, 'reset'), B = {}; if nargout > 0, L = B; end; return; end
if nargin == 1 && ischar(fmt) && strcmp(fmt, 'get'), L = B; return; end
if isempty(varargin), s = fmt; else, s = sprintf(fmt, varargin{:}); end
B{end + 1} = s;
if nargout > 0, L = B; end
end
