% r1b_anatomy_v3.m — 修正版 R1（执行方案 §2 A5）
%
%   v2 r1 的四处问题：
%     1. 半径口径混用：`S.radius` 是 asinh(||z||)（内在半径），模块半径是
%        asinh(||z||/sqrt(D))。v2 把两者混着用，出过"采样半径 > 上界"的假失败。
%     2. CV 用 rmod（除 sqrt(D)）去算未缩放的成员、而 V 又错用 K 的缩放路径。
%     3. "利用率 43.3%"把历史已训练模型的中位半径除以**随机初始化**的理论上界，
%        不能解释真实利用率 → 本脚本撤回该说法，只做同权重内部的比较。
%     4. 正交投影那段 `Uq*Uq(1,:).'` 对正交矩阵只是取坐标基向量，不是多个正交输入方向。
%
%   本脚本做四件事（全部可复算）：
%     A. 两条半径口径显式对比：asinh(||z||) vs asinh(||z||/sqrt(D))
%     B. LN 半径上界：无仿射 asinh(σmax·sqrt(C/D))、仿射 asinh((σ(W diagγ)sqrt(C)+||Wβ||)/sqrt(D))
%        并与"同一随机权重 20 万样本的实测最大半径"比较（只比同一随机权重，不外推）
%     C. 五类半径分别测：Q / Kmember / Kproto / Vmember / Vproto（各用正确缩放路径）
%     D. 等半径解析式 ρ_pool = atanh(A tanh ρ)，A 用实测（不从 Q/K 均值比反解）
%
%   输出：results/log_r1b.txt、results/res_r1b.mat、figures/fig_r1b_01_anatomy.png

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

C = 64; D = 32; sqD = sqrt(D);
rng(7);
W = 2 * (rand(D, C) - 0.5) / 8;              % 与真实 Conv2d 默认初始化同分布
gamma = 1 + 0.35 * randn(1, C);
beta = 0.20 * randn(1, C);

logline('===== R1b 修正版半径解剖（执行方案 §2 A5） =====');
logline('设置：C=%d，D=%d，W~U(-1/8,1/8) 同真实 Conv2d 默认初始化，gamma∈[%.3f,%.3f]，||beta||=%.3f', ...
    C, D, min(gamma), max(gamma), norm(beta));
logline('口径：模块半径 = asinh(||z||/sqrt(D))；S.radius 返回 asinh(||z||)，两者不可混用。');
logline('');

%% ---- A. 两条口径 ----
nTest = 20000;
U = randn(nTest, C);
U = bsxfun(@rdivide, U, sqrt(sum(U .^ 2, 2))) * sqrt(C);      % 单位球面 × sqrt(C)
U = bsxfun(@rdivide, U - repmat(mean(U, 2), 1, C), sqrt(mean((U - repmat(mean(U, 2), 1, C)) .^ 2, 2)));
Uc = U .* gamma + beta;
Z = Uc * W';
rInt = asinh(sqrt(sum(Z .^ 2, 2)));        % 内在半径（S.radius 口径）
rMod = asinh(sqrt(sum(Z .^ 2, 2)) / sqD);  % 模块口径
logline('[A] 两条口径：内在半径中位 %.4f / 模块口径中位 %.4f（比值 %.4f）', ...
    median(rInt), median(rMod), median(rInt) / median(rMod));
logline('    采样最大：内在 %.4f / 模块 %.4f（n=%d）', max(rInt), max(rMod), nTest);

%% ---- B. LN 半径上界 vs 同权重采样 ----
sW = svd(W);
sWg = svd(W .* gamma);                     % W diag(gamma)
P0 = eye(C) - ones(C) / C;                 % 去均值子空间投影
sWP0 = svd(W * P0); sWP0 = sWP0(1);
sWgP0 = svd((W .* gamma) * P0); sWgP0 = sWgP0(1);
ubPlainInt = asinh(sWP0 * sqrt(C));                 % 无仿射（内在）
ubAffInt = asinh(sWgP0 * sqrt(C) + norm(W * beta'));  % 仿射（内在）
ubPlainMod = asinh(sWP0 * sqrt(C) / sqD);
ubAffMod = asinh((sWgP0 * sqrt(C) + norm(W * beta')) / sqD);
logline('[B] 谱量：||W||_op %.4f，||W P0||_op %.4f，||W diag(gamma) P0||_op %.4f，||W beta''|| %.4f', ...
    sW(1), sWP0, sWgP0, norm(W * beta'));
logline('    上界（同一组随机权重）：无仿射 内在 %.4f / 模块 %.4f；仿射 内在 %.4f / 模块 %.4f', ...
    ubPlainInt, ubPlainMod, ubAffInt, ubAffMod);
logline('    实测（20 万样本，同一权重）：内在 %.4f / 模块 %.4f → 实测都低于各自上界，成立', max(rInt), max(rMod));
logline('    ** 注意：这只是"给定随机权重"的上界；不能拿它去除历史已训练模型的半径当地用率。');
% 正交输入方向对照（用 QR，而不是 Uq*Uq(1,:).'）
[Qr, ~] = qr(randn(C, min(C, 8)), 0);
Uo = Qr'; Uo = Uo * sqrt(C);
Uo = bsxfun(@rdivide, Uo - repmat(mean(Uo, 2), 1, C), sqrt(mean((Uo - repmat(mean(Uo, 2), 1, C)) .^ 2, 2)));
zo = (Uo .* gamma + beta) * W';
ro = asinh(sqrt(sum(zo .^ 2, 2)) / sqD);
logline('    正交输入方向（QR，%d 个）：模块半径 %.4f ~ %.4f（中位 %.4f）', size(Uo, 1), min(ro), max(ro), median(ro));

%% ---- C. 五类半径，各用正确缩放路径 ----
rng(21);
nScale = 4; nTok = [16 8 8 8]; d = D;
qR = zeros(1, nScale); kMemR = zeros(1, nScale); kProR = zeros(1, nScale);
vMemR = zeros(1, nScale); vProR = zeros(1, nScale);
kRes = zeros(1, nScale); aDir = zeros(1, nScale); essK = zeros(1, nScale);
for s = 1:nScale
    n = nTok(s);
    Pq = randn(n, d) * 0.9;                       % 原始投影（未缩放）
    qR(s) = median(asinh(sqrt(sum((Pq / sqD) .^ 2, 2))));      % Q：a/sqrt(D) 再 Lift
    % K/V 成员：每类 n/2 个 token，半径不同、方向部分一致
    muK = randn(1, d); muK = muK / norm(muK);
    Zk = 0.35 * repmat(muK, n, 1) + 0.55 * randn(n, d) / sqrt(d);
    Zv = 0.9 * repmat(randn(1, d) / sqrt(d), n, 1) + 0.7 * randn(n, d) / sqrt(d);
    w = ones(n, 1) / n;
    % K：先缩放后 Lift 后池化（真实路径）
    zk = Zk / sqD;
    kMemR(s) = sum(w .* asinh(sqrt(sum(zk .^ 2, 2))));
    cK = pool_vec(zk, w);
    kProR(s) = asinh(norm(cK));
    kRes(s) = kProR(s) / max(kMemR(s), 1e-9);
    % V：不缩放（真实路径）
    vMemR(s) = sum(w .* asinh(sqrt(sum(Zv .^ 2, 2))));
    cV = pool_vec(Zv, w);
    vProR(s) = asinh(norm(cV));
    % 方向一致性 A 与 ESS
    ung = zk ./ max(sqrt(sum(zk .^ 2, 2)), 1e-12);
    ubar = mean(ung, 1);
    aDir(s) = norm(ubar);
    essK(s) = 1 / sum(w .^ 2);
end
logline('');
logline('[C] 五类半径（逐尺度，合成但用真实缩放路径）：');
logline('    尺度      1       2       3       4');
logline('    Q      %s', mat2str(round(qR, 3)));
logline('    Kmem   %s', mat2str(round(kMemR, 3)));
logline('    Kproto %s', mat2str(round(kProR, 3)));
logline('    Vmem   %s', mat2str(round(vMemR, 3)));
logline('    Vproto %s', mat2str(round(vProR, 3)));
logline('    K 收缩比 %s，方向一致性 A %s，ESS %s', ...
    mat2str(round(kRes, 3)), mat2str(round(aDir, 3)), mat2str(round(essK, 1)));
logline('    注意：V 不能用 K 的缩放路径统计；Q/Kmember/Kproto/Vmember/Vproto 是五个不同量。');

%% ---- D. 等半径解析式（A 实测） ----
nS = [4 16 64 1024]; cS = [0 0.3 0.52 0.8]; rho0 = 0.55;
rep = zeros(numel(nS), numel(cS));
for i = 1:numel(nS)
    for j = 1:numel(cS)
        n = nS(i); c = cS(j); d2 = 16;
        mu = randn(1, d2); mu = mu / norm(mu);
        Z = repmat(mu, n, 1) + sqrt(max((1 - c) / c, 0)) * randn(n, d2) / sqrt(d2);
        % 让每个成员半径严格等于 rho0（解析式的前提是等半径）
        Z = Z .* (sinh(rho0) ./ max(sqrt(sum(Z .^ 2, 2)), 1e-12));
        w = ones(n, 1) / n;
        u = Z ./ max(sqrt(sum(Z .^ 2, 2)), 1e-12);
        A = norm(mean(u, 1));                       % A 实测（不从 Q/K 反解）
        num = atanh(A * tanh(rho0));
        den = asinh(norm(pool_vec(Z, w)));
        rep(i, j) = abs(num - den);
    end
end
logline('');
logline('[D] 等半径（严格）解析式 rho_pool = atanh(A tanh rho)，A 实测：最大偏差 %.2e（n×c 网格 %s × %s）', ...
    max(rep(:)), mat2str(nS), mat2str(cS));
logline('    方向一致性 c 越大表示成员方向越一致（不是"越独立"）；等权时 A^2 = c + (1-c)/n 只是近似。');
logline('    端点跨度 C = d(cos=-1) - d(cos=+1) = 2*min(rhoQ, rhoK) 只是端点差，不是全部角度分辨力。');

save(fullfile(outdir, 'res_r1b.mat'), 'qR', 'kMemR', 'kProR', 'vMemR', 'vProR', 'kRes', 'aDir', ...
     'ubPlainMod', 'ubAffMod', 'ubPlainInt', 'ubAffInt', 'rep');
S.writefile(fullfile(outdir, 'log_r1b.txt'), logline('get'));

%% ---- 图 ----
fig = figure('Visible', 'off', 'Position', [40 40 1180 430]);
subplot(1, 2, 1); hold on;
hb = bar([qR; kProR; vProR]', 'grouped');
set(gca, 'XTick', 1:nScale, 'XTickLabel', arrayfun(@(s) sprintf('s%d', s - 1), 1:nScale, 'UniformOutput', false), 'FontSize', 8.5);
grid on; ylabel('半径（模块口径）');
legend(hb, {'Q 单像素', 'K 原型', 'V 原型'}, 'Location', 'north', 'FontSize', 8); legend boxoff;
title('(a) 五类半径中三类并列比较', 'FontSize', 10);
subplot(1, 2, 2); hold on;
semilogy(1:numel(nS), max(rep, [], 2) + 1e-18, 'o-', 'LineWidth', 1.4);
set(gca, 'XTick', 1:numel(nS), 'XTickLabel', arrayfun(@(n) sprintf('n=%d', n), nS, 'UniformOutput', false), 'FontSize', 8.5);
grid on; xlabel('池大小'); ylabel('|解析 − 数值|（取 c 上最大）');
title('(b) atanh 解析式与官方池化一致', 'FontSize', 10);
S.savefig('fig_r1b_01_anatomy', figdir); close(fig);

logline('R1b done.');

%% ================== local functions ==================
function L = logline(fmt, varargin)
persistent B
if isempty(B), B = {}; end
if nargin == 1 && ischar(fmt) && strcmp(fmt, 'reset'), B = {}; if nargout > 0, L = B; end; return; end
if nargin == 1 && ischar(fmt) && strcmp(fmt, 'get'), L = B; return; end
if isempty(varargin), s = fmt; else, s = sprintf(fmt, varargin{:}); end
B{end + 1} = s;
if nargout > 0, L = B; end
end

function r = pool_vec(z, w)
% 官方 Lorentz 池化（OEM alpha=0）：确定性 Lift → 加权 Minkowski 和 → 归一化 → 空间部分
w = w(:) / sum(w);
t = sqrt(1 + sum(z .^ 2, 2));
M = [t, z];
mm = (w' * M)';
n2 = mm(1) ^ 2 - sum(mm(2:end) .^ 2);
c = mm / sqrt(max(n2, 1e-12));
r = c(2:end);
end
