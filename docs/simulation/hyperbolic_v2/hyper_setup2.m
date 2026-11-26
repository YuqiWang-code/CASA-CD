function S = hyper_setup2()
%HYPER_SETUP2  修正版双曲/欧氏几何工具集（阶段 0，依据执行方案 §1.2–§1.4）
%
%   与 v1 (../hyperbolic/hyper_setup.m) 的差别——每一条都对应一份被修正的误用：
%   1. v1 里 H.cdist 是 Poincare 测地距离，但多处被当作"欧氏对照"直接喂原始
%      特征（这些点根本不在单位球内）。v2 把三种距离显式分开：
%        S.euc_dist     欧氏距离          |p-q|
%        S.geo_dist     Poincare 测地    输入必须严格在单位球内，否则报错
%        S.lorentz_dist Lorentz 测地     输入是双叶双曲面上的点 (t,z)
%      并且分别提供平方版本 euc_sqdist；测地距离不再偷偷给出平方量。
%   2. v1 的 expmap0 注释把"直接 Lorentz 提升在 Poincare 盘里的像"与"标准
%      Poincare 指数映射"混为一谈。两者相差 |z| = sinh|v|，不是同一个映射：
%        S.pball(z)      直接提升的像        π(z) = z/(1+sqrt(1+|z|^2))
%        S.expmap0(v)    标准指数映射        tanh(|v|/2)·v/|v|
%      且 d(0,π(z)) = asinh|z| 精确成立，d(0,exp_0(v)) = |v|。
%   3. v1 的 pool_einstein 多乘了一次 t、pool_oem(p) 多乘了 p+1 次方，
%      与 HypRAG 官方实现（先转 Klein 聚合）的指数口径错开。v2 统一为官方口径
%        S.pool_oem(z,w,alpha) = NormL( sum_i w_i t_i^alpha x_i ),  t_i = x_i0
%      alpha=0 才是标准 Einstein 中点，也等于 Lorentz 加权重心（本文件里由
%      S.pool_lorentz 给出，两者数值必须一致，c1_math_checks 会验证）。
%      另外补上 E-Coord、真切空间均值（Tangent）、Fréchet 均值三个独立对照。
%   4. 归一化不再混称：S.l2norm 是按行 L2 归一化，S.layernorm 是 PyTorch
%      口径的逐通道 LayerNorm（总体方差、eps、可选 gamma/beta 仿射）。
%
%   约定：所有向量按行存放（N×d）；曲率取 -1（K=-1）；不依赖任何工具箱。
%
%   用法： S = hyper_setup2();

S.ver = 'hyper_setup2 v2 2026-11-26';
S.K   = -1;

% ---- 坐标与半径 -------------------------------------------------------
S.lift        = @lift_direct;     % z -> (t,z)，t=sqrt(1+|z|^2)
S.radius      = @radius_of_z;     % z -> asinh|z|（= Lorentz 时间坐标的反函数）
S.t_of        = @(r) sinh(r);
S.pball       = @pball_direct;    % 直接提升的 Poincare 像
S.expmap0     = @expmap0_std;     % 标准 Poincare 指数映射（球内）
S.logmap0     = @logmap0_std;     % 标准对数映射，返回"双曲半径范数"的切向量
S.mobius      = @mobius_add;
S.lorentz_norm= @lorentz_normalize;

% ---- 距离 ------------------------------------------------------------
S.euc_dist    = @euc_dist;
S.euc_sqdist  = @euc_sqdist;
S.geo_dist    = @geo_dist;
S.geo_dist_c  = @geo_dist_c;      % 任意曲率 c>0 的 Poincare 测地
S.lorentz_dist= @lorentz_dist;
S.check_ball  = @check_ball;      % 显式校验输入是否严格在球内

% ---- 归一化 ----------------------------------------------------------
S.l2norm      = @l2_normalize;    % 按行 L2 归一化（乘 sqrt(C) 保持 LN 的量级）
S.layernorm   = @layer_norm_pop;  % PyTorch 口径：去均值 + 总体方差 + eps + 仿射

% ---- 池化算子（全部吃切向坐标 z 与权重 w）-----------------------------
S.pool_coord   = @pool_coord;     % E-Coord：欧氏坐标均值，再 Lift
S.pool_tangent = @pool_tangent;   % Tangent：Log_0 均值，再 Exp_0
S.pool_lorentz = @(z,w) pool_oem(z,w,0);   % Lorentz 加权重心 = OEM_0
S.pool_oem     = @pool_oem;       % 官方口径 OEM_alpha
S.pool_euclid  = @pool_euclid;    % 旧名兼容：纯坐标均值（不 Lift）
S.frechet      = @frechet_mean;   % Fréchet 均值（迭代，仅参考）

% ---- 统计 ------------------------------------------------------------
S.auc      = @auc_rank;
S.rank_avg = @rank_avg;
S.spearman = @spearman_r;
S.bootci   = @boot_ci;
S.softmax  = @softmax_rows;
S.entropy  = @entropy_rows;
S.effkeys  = @ess_weights;        % 有效样本数 ESS = 1/sum(w^2)
S.jsdiv    = @js_divergence;

% ---- 输出 ------------------------------------------------------------
S.zh       = @set_zh;
S.savefig  = @save_fig_v2;
S.writefile= @write_lines;
end

% =======================================================================
%  坐标、半径、映射
% =======================================================================
function T = lift_direct(Z)
% 支持 N×d 与更高维（按最后一维取范数）
sd = ndims(Z);
T = [sqrt(1+sum(Z.^2,sd)), Z];
end

function r = radius_of_z(Z)
sd = ndims(Z);
t = sqrt(sum(Z.^2,sd));
r = asinh(t);
end

function P = pball_direct(Z)
sd = ndims(Z);
n2 = sum(Z.^2,sd);
P  = Z ./ (1 + sqrt(1+n2));
end

function P = expmap0_std(V)
% 标准 Poincare 指数映射：exp_0(v) = tanh(|v|/2) · v/|v|，零向量取极限 0。
n = sqrt(sum(V.^2,2));
u = zeros(size(V));
nz = n > 0;
u(nz,:) = V(nz,:) ./ n(nz);
r = zeros(size(n));
r(nz)  = tanh(n(nz)/2);
P = r .* u;
end

function V = logmap0_std(P)
% 标准对数映射：log_0(p) = 2 atanh(|p|) · p/|p|。
% 返回向量的范数就是双曲半径（与 asinh|z| 一致，c1 会验证）。
n = sqrt(sum(P.^2,2));
if any(n >= 1)
    error('hyper_setup2:logmap0', ...
        '输入不在单位球内（max|p|=%.6g），不能用 Poincare 对数映射', max(n));
end
u = zeros(size(P));
nz = n > 0;
u(nz,:) = P(nz,:) ./ n(nz);
r = zeros(size(n));
r(nz) = 2*atanh(n(nz));
V = r .* u;
end

function C = mobius_add(X, Y)
% 逐行 Mobius 加法（曲率 -1）
x2 = sum(X.^2,2); y2 = sum(Y.^2,2); xy = sum(X.*Y,2);
den = 1 + 2*xy + x2.*y2;
num = (1 + 2*xy + y2) .* X + (1 - x2) .* Y;
C = num ./ den;
end

function V = lorentz_normalize(V)
n2 = V(:,1).^2 - sum(V(:,2:end).^2,2);
if any(n2 <= 0)
    error('hyper_setup2:lorentz', '向量不在双曲面上的类时区（min <x,x>_L=%.3g）', min(n2));
end
V = V ./ sqrt(n2);
end

% =======================================================================
%  距离
% =======================================================================
function D = euc_dist(X, Y)
D2 = euc_sqdist(X, Y);
D  = sqrt(max(D2, 0));
end

function D2 = euc_sqdist(X, Y)
% 欧氏平方距离矩阵，显式 bsxfun 展开（不依赖 pdist2）
D2 = bsxfun(@plus, sum(X.^2,2), sum(Y.^2,2)') - 2*(X*Y');
D2 = max(D2, 0);
end

function check_ball(P, name)
if nargin < 2, name = 'P'; end
n = sqrt(sum(P.^2,2));
if any(~isfinite(n))
    error('hyper_setup2:ball', '%s 含非有限值', name);
end
if any(n >= 1)
    [mx, k] = max(n);
    error('hyper_setup2:ball', ...
        '%s 有 %d 个点不在单位球内（第 %d 行 |p|=%.6f ≥ 1）；Poincare 距离无定义', ...
        name, sum(n >= 1), k, mx);
end
end

function D = geo_dist(P, Q)
% Poincare 测地距离（曲率 -1）。输入必须严格在单位球内——不做任何 clamp。
check_ball(P, 'P'); check_ball(Q, 'Q');
S2 = euc_sqdist(P, Q);
a  = (1 - sum(P.^2,2)) * (1 - sum(Q.^2,2))';
K  = 1 + 2*S2 ./ a;
D  = acosh(max(K, 1));
end

function D = geo_dist_c(P, Q, c)
% 曲率 c>0 的 Poincare 测地距离：d_c = acosh(1+2cS/((1-c|p|^2)(1-c|q|^2)))/sqrt(c)
nP = sum(P.^2,2); nQ = sum(Q.^2,2);
if any(c*nP >= 1) || any(c*nQ >= 1)
    error('hyper_setup2:ball', '曲率 c=%g 下输入越界', c);
end
S2 = euc_sqdist(P, Q);
K  = 1 + 2*c*S2 ./ ((1-c*nP)*(1-c*nQ)');
D  = acosh(max(K,1)) / sqrt(c);
end

function D = lorentz_dist(X, Y)
% 双曲面上的 Lorentz 测地距离：cosh d = -<x,y>_L = x0*y0 - x·y
ip = bsxfun(@times, X(:,1), Y(:,1)') - X(:,2:end)*Y(:,2:end)';
D  = acosh(max(ip, 1));
end

% =======================================================================
%  归一化
% =======================================================================
function U = l2_normalize(F, targetNorm)
% 按行 L2 归一化，再乘 targetNorm（默认 sqrt(C)，与 LayerNorm 输出量级一致）
C = size(F,3);
X = reshape(F, [], C);
n = sqrt(sum(X.^2,2));
n(n < 1e-12) = 1;
X = X ./ n;
if nargin >= 2 && ~isempty(targetNorm)
    X = X * targetNorm;
else
    X = X * sqrt(C);
end
U = reshape(X, size(F));
end

function U = layer_norm_pop(F, gamma, beta, epsv)
% PyTorch 口径 LayerNorm：mu 是通道均值，var 用总体方差（归一化分母 N，不是 N-1）
if nargin < 2 || isempty(gamma), gamma = 1; end
if nargin < 3 || isempty(beta),  beta  = 0; end
if nargin < 4 || isempty(epsv),  epsv  = 1e-5; end
C = size(F,3);
X = reshape(F, [], C);
mu = mean(X,2);
vr = var(X, 1, 2);           % 总体方差（v1 的 var(X,0,2) 是样本方差）
U  = (X - mu) ./ sqrt(vr + epsv);
U  = bsxfun(@times, U, gamma(:)') + beta(:)';
U  = reshape(U, size(F));
end

% =======================================================================
%  池化算子
% =======================================================================
function z = pool_euclid(Z, w)
% 纯坐标（切空间）加权均值，不回到流形——仅作参考项
w = w(:) / sum(w);
z = w' * Z;
end

function z = pool_coord(Z, w)
% E-Coord：欧氏坐标的加权均值，再 Lift（注意：不是严格的切空间均值）
z = pool_euclid(Z, w);
end

function z = pool_tangent(Z, w)
% Tangent：在原点切空间做加权均值，再 Exp_0。
%   v_i = (asinh|z_i| / |z_i|) z_i ，|z_i|→0 时系数取连续极限 1
w = w(:) / sum(w);
t = sqrt(sum(Z.^2,2));
f = ones(size(t)); nz = t > 0; f(nz) = asinh(t(nz)) ./ t(nz);
V = bsxfun(@times, f, Z);
vbar = w' * V;
nb = norm(vbar);
if nb < 1e-14
    z = zeros(1, size(Z,2));
else
    z = (sinh(nb)/nb) * vbar;
end
end

function z = pool_oem(Z, w, alpha)
% OEM_alpha（HypRAG 官方实现口径）： NormL( sum_i w_i t_i^alpha x_i )
%   alpha = 0  -> 标准 Einstein 中点 == Lorentz 加权重心
%   alpha = 1  -> 官方 OEM(p=1) 的"外偏"形式
if nargin < 3 || isempty(alpha), alpha = 0; end
w = w(:);
if sum(w) <= 0
    error('hyper_setup2:oem', '总权重非正（sum(w)=%.3g）', sum(w));
end
X  = lift_direct(Z);
co = w .* (X(:,1).^alpha);
u  = (co' * X);
z  = u(2:end) / sqrt(max(u(1)^2 - sum(u(2:end).^2), 1e-300));
end

function z = frechet_mean(Z, w, nIter)
% Frechet 均值（原点切空间梯度下降，仅作参考/上限对照）
if nargin < 3 || isempty(nIter), nIter = 30; end
w = w(:) / sum(w);
z = pool_tangent(Z, w);
for it = 1:nIter
    tq = sqrt(sum(z.^2,2));
    P  = pball_direct(z);
    % 到各成员的距离，以及单位切向（用球面方向近似梯度方向）
    Q  = pball_direct(Z);
    D  = geo_dist(P, Q);
    g  = zeros(1, size(Z,2));
    for i = 1:size(Z,1)
        qi = Q(i,:);
        % Poincare 中从 P 指向 qi 的初始方向
        d  = qi - P;
        if norm(d) > 0
            g = g + w(i) * D(i) * d/norm(d);
        end
    end
    if norm(g) < 1e-12, break; end
    z = z - 0.1 * g;
    if sqrt(sum(z.^2,2)) > 12, z = z * 12/sqrt(sum(z.^2,2)); end
end
end

% =======================================================================
%  统计
% =======================================================================
function r = rank_avg(x)
% 平均秩（通用版：输入无需预排序）。并列取平均，NaN 保持 NaN。
x = x(:);
r = nan(size(x));
ok = ~isnan(x);
if ~any(ok), return; end
[~, ord] = sort(x(ok));
xs = x(ok);
rr = zeros(numel(xs),1);
i = 1;
while i <= numel(xs)
    j = i;
    while j < numel(xs) && xs(ord(j+1)) == xs(ord(i))
        j = j + 1;
    end
    rr(ord(i:j)) = (i+j)/2;
    i = j + 1;
end
tmp = nan(size(x)); tmp(ok) = rr; r = tmp;
end

function a = auc_rank(s, y)
% 秩和 AUC；y 为逻辑/0-1；忽略 NaN
s = s(:); y = y(:);
ok = ~isnan(s) & ~isnan(y);
s = s(ok); y = y(ok);
n1 = sum(y == 1); n0 = numel(y) - n1;
if n1 == 0 || n0 == 0, a = NaN; return; end
r = rank_avg(s);
a = (sum(r(y == 1)) - n1*(n1+1)/2) / (n1*n0);
end

function c = spearman_r(a, b)
a = a(:); b = b(:);
ok = ~isnan(a) & ~isnan(b);
if sum(ok) < 3, c = NaN; return; end
ra = rank_avg(a(ok)); rb = rank_avg(b(ok));
if std(ra) == 0 || std(rb) == 0, c = NaN; return; end
cc = corrcoef(ra, rb);
c = cc(1,2);
end

function ci = boot_ci(statfun, data, nBoot, seed)
% 按行重采样的 bootstrap 95% 置信区间
if nargin < 3 || isempty(nBoot), nBoot = 1000; end
if nargin < 4 || isempty(seed), seed = 20261126; end
rs = RandStream('mt19937ar','Seed',seed);
n = size(data,1);
vals = nan(nBoot,1);
for b = 1:nBoot
    idx = randi(rs, n, n);
    vals(b) = statfun(data(idx));
end
vals = vals(~isnan(vals));
if isempty(vals), ci = [NaN NaN]; return; end
ci = quantile(vals, [0.025 0.975]);
end

function P = softmax_rows(L, tau)
if nargin < 2 || isempty(tau), tau = 1; end
L = L/tau;
L = bsxfun(@minus, L, max(L,[],2));
E = exp(L);
P = bsxfun(@rdivide, E, sum(E,2));
end

function h = entropy_rows(P)
P = max(P, 1e-300);
h = -sum(P .* log(P), 2);
end

function ess = ess_weights(w)
w = w(:); s = sum(w);
if s <= 0, ess = 0; return; end
p = w/s;
ess = 1/sum(p.^2);
end

function d = js_divergence(P, Q)
P = max(P, 1e-300); Q = max(Q, 1e-300);
M = 0.5*(P+Q);
kl = @(A,B) sum(A .* (log(A) - log(B)), 2);
d = sqrt(0.5*kl(P,M) + 0.5*kl(Q,M));
end

% =======================================================================
%  输出
% =======================================================================
function set_zh()
set(0,'DefaultAxesFontName','Microsoft YaHei');
set(0,'DefaultTextFontName','Microsoft YaHei');
set(0,'DefaultAxesFontSize',9);
end

function save_fig_v2(name, outdir)
if nargin < 2 || isempty(outdir)
    outdir = fullfile(fileparts(mfilename('fullpath')), 'figures');
end
if ~exist(outdir,'dir'), mkdir(outdir); end
exportgraphics(gcf, fullfile(outdir,[name '.png']), 'Resolution', 150);
end

function write_lines(path, lines)
fid = fopen(path, 'w', 'n', 'UTF-8');
if fid < 0, error('hyper_setup2:io', '无法写入 %s', path); end
for i = 1:numel(lines)
    fprintf(fid, '%s\n', lines{i});
end
fclose(fid);
end
