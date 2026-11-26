function R = rs_setup()
%RS_SETUP  “离双曲原点近/远”研究工具集（曲率 -1，行向量约定，无需任何工具箱）
%
%   记号：z 为点在原点切空间的坐标 (N x d)，t = ||z||（范数），
%         内在半径 rho = asinh(t)，Lorentz 点 x = (sqrt(1+t^2), z)。
%         精确交叉测地距离（无任何近似）：
%             cosh d = sqrt(1+tq^2)*sqrt(1+tk^2) - zq*zk'
%
%   池化算子（输入 z (Nxd)，权重 w (Nx1)，sum(w)=1；返回 1xd）：
%     pool_euc  切空间加权均值后抬升                 （= 我们 aggregate 的默认值通路）
%     pool_amb  Minkowski 环境加权均值 + 归一化到双曲面
%               = lorentz_ops.minkowski_barycenter / HyperbolicContext.manifold_pool
%               = HypRAG 命题 4.3 的 m_Euc（可证明向原点收缩）
%     pool_ein  Einstein 中点（系数 w_i*t_i）
%     pool_oem  Outward Einstein 中点（系数 w_i*t_i^(p+1)），p>=0
%
%   统计：auc / spearman / bootci（自助 95% CI）
%   其它：boxblur（无工具箱）、layernorm、kaiming_proj（复刻 Conv2d 默认初始化）

here = fileparts(mfilename('fullpath'));
R.root   = here;
R.outdir = fullfile(here,'results');
R.figdir = fullfile(R.outdir,'figures');
if ~exist(R.outdir,'dir'), mkdir(R.outdir); end
if ~exist(R.figdir,'dir'), mkdir(R.figdir); end

R.radius    = @rho_of;
R.t_of      = @t_of;
R.lift      = @lorentz_lift;
R.normlz    = @lorentz_normalize;
R.cdist     = @geo_cdist;        % (zq,zk) 精确测地距离矩阵
R.cdist_t   = @geo_cdist_trig;   % (rq,rk,cosT) 由半径与夹角直接算
R.angcontrast = @angular_contrast;
R.pool_euc  = @pool_euclid;
R.pool_amb  = @pool_ambient;
R.pool_ein  = @pool_einstein;
R.pool_oem  = @pool_oem;
R.wmean_r   = @wmean_rho;
R.auc       = @auc_rank;
R.spearman  = @spearman_r;
R.bootci    = @boot_ci;
R.boxblur   = @box_blur;
R.softmax   = @softmax_rows;
R.entropy   = @row_entropy;
R.effkeys   = @row_effkeys;
R.layernorm = @layer_norm;
R.proj      = @kaiming_proj;
R.zh        = @zh_fonts;
R.savefig   = @(name) save_fig_to(R.figdir, name);
R.writefile = @write_lines;
R.anchors   = @measured_anchors;
end

% ============================================================== geometry ====
function t = t_of(rho)
t = sinh(rho);
end

function r = rho_of(z)
r = asinh(sqrt(sum(z.^2,2)));
end

function X = lorentz_lift(z)
X = [sqrt(1 + sum(z.^2,2)), z];
end

function v = lorentz_normalize(v)
% 把 Minkowski 向量按 <v,v>_L = 1 归一化（我们 minkowski_barycenter 做的事）
n2 = v(:,1).^2 - sum(v(:,2:end).^2,2);
s  = sqrt(max(n2, 1e-12));
v  = v ./ s;
end

function D = geo_cdist(zq, zk)
tq = sqrt(1 + sum(zq.^2,2));
tk = sqrt(1 + sum(zk.^2,2));
C  = tq*tk' - zq*zk';
D  = acosh(max(C,1));
end

function D = geo_cdist_trig(rq, rk, cosT)
% 半径 rho_q、rho_k、夹角余弦 cosT 的精确测地距离
C = cosh(rq).*cosh(rk) - sinh(rq).*sinh(rk).*cosT;
D = acosh(max(C,1));
end

function C = angular_contrast(rq, rk)
% 同一半径对下"角度能否被分辨"的幅度：d(cos=-1) - d(cos=+1)
C = geo_cdist_trig(rq, rk, -1) - geo_cdist_trig(rq, rk, 1);
end

% ================================================================= pools ====
function z = pool_euclid(z, w)
z = (w' * z);
if size(z,1) > 1, z = z'; end
end

function z = pool_ambient(z, w)
X = lorentz_lift(z);
v = (w' * X);
v = lorentz_normalize(v);
z = v(2:end);
end

function z = pool_einstein(z, w)
X = lorentz_lift(z);
c = w .* X(:,1);              % w_i * t_i
v = (c' * X);
v = lorentz_normalize(v);
z = v(2:end);
end

function z = pool_oem(z, w, p)
if nargin < 3, p = 1; end
X = lorentz_lift(z);
c = w .* X(:,1).^(p+1);       % w_i * t_i^(p+1)，p=0 即 Einstein
v = (c' * X);
v = lorentz_normalize(v);
z = v(2:end);
end

function m = wmean_rho(z, w)
m = sum(w .* rho_of(z));
end

% ================================================================= stats ====
function a = auc_rank(s, y)
y = y(:); s = s(:);
ok = ~isnan(s) & ~isnan(y);
s = s(ok); y = y(ok);
n1 = sum(y==1); n0 = numel(y)-n1;
if n1==0 || n0==0, a = NaN; return; end
r = rank_avg(s);
a = (sum(r(y==1)) - n1*(n1+1)/2)/(n1*n0);
end

function r = rank_avg(x)
% 平均秩（并列取平均），输入无需有序
x = x(:); n = numel(x);
if n == 0, r = zeros(0,1); return; end
[s, ord] = sort(x);
r = zeros(n,1); i = 1;
while i <= n
    j = i;
    while j < n && s(j+1) == s(i), j = j+1; end
    r(ord(i:j)) = (i+j)/2;
    i = j+1;
end
end

function rho = spearman_r(x, y)
x = x(:); y = y(:);
ok = ~isnan(x) & ~isnan(y);
x = x(ok); y = y(ok);
if numel(x) < 3, rho = NaN; return; end
rx = rank_avg(x); ry = rank_avg(y);
c = corrcoef(rx, ry);
rho = c(1,2);
end

function [lo, hi, bs] = boot_ci(statfun, n, B)
if nargin < 3, B = 1000; end
bs = zeros(B,1);
for b = 1:B
    idx = randi(n, n, 1);
    bs(b) = statfun(idx);
end
bs = bs(~isnan(bs));
lo = prctile(bs, 2.5);
hi = prctile(bs, 97.5);
end

% ============================================================== numerics ====
function B = box_blur(X, h)
% 可分离盒滤波（边缘复制），纯 cumsum 实现，等价于 5x5/… 均值滤波
if h < 1, B = X; return; end
X = double(X);
B = blur1(blur1(X, h).', h).';
end

function Y = blur1(X, h)
[m,n] = size(X);
w = 2*h+1;
P = [repmat(X(:,1),1,h), X, repmat(X(:,n),1,h)];
cs = [zeros(m,1), cumsum(P,2)];
Y = (cs(:,w+1:w+n) - cs(:,1:n)) / w;
end

function Y = layer_norm(X)
mu = mean(X,2);
sd = sqrt(var(X,0,2) + 1e-5);
Y  = (X - mu) ./ sd;
end

function W = kaiming_proj(d_in, d_out, seed)
% 复刻 Conv2d(d_in->d_out,1,bias=False) 的默认初始化：U(-b,b), b=1/sqrt(d_in)
if nargin < 3, seed = 0; end
rng(seed,'twister');
b = 1/sqrt(d_in);
W = (2*b)*rand(d_out, d_in) - b;
end

function S = softmax_rows(L)
L = L - max(L,[],2);
S = exp(L);
S = S ./ sum(S,2);
end

function H = row_entropy(P)
P = max(P, 1e-12);
H = -sum(P .* log(P), 2);
end

function e = row_effkeys(P)
e = exp(row_entropy(P));
end

% ================================================================ figures ===
function zh_fonts()
set(groot,'defaultAxesFontName','Microsoft YaHei');
set(groot,'defaultTextFontName','Microsoft YaHei');
set(groot,'defaultAxesFontSize',10);
end

function save_fig_to(figdir, name)
if ~exist(figdir,'dir'), mkdir(figdir); end
f = gcf; set(f,'Color','w');
exportgraphics(f, fullfile(figdir,[name '.png']), 'Resolution',150);
end

function write_lines(path, lines)
fid = fopen(path,'w','n','UTF-8');
if fid < 0, error('cannot open %s', path); end
for i = 1:numel(lines)
    fprintf(fid,'%s\n', lines{i});
end
fclose(fid);
end

% ====================================================== measured anchors ====
function A = measured_anchors()
% 真实检查点里 record 到的半径统计（radius / key_radius / radius_cv）。
% 来源：outputs/CASA-Pyramid/Run*/probes|selection/*/*/history.json 的
%       first_validation_batch_diagnostics.stage{1..4}
% 说明：key_radius_mean 在同一权重的四个 stage 上相同（键集合只有一份）。
%       Run7 的 selection/* 是 48 个 epoch 的平均。
A = struct( ...
 'run',   {'Run9 H_REFINE','Run9 H_REFINE','Run9 E_REFINE','Run9 E_BASE','Run8 H_NONE', ...
           'Run8 H_CLASS','Run10 H_CENTER_RAW','Run10 H_CENTER_RAW','Run10 H_CENTER_SUP', ...
           'Run10 E_CENTER_SUP','Run7 geometry_euclidean','Run7 direct_geometry'}, ...
 'ds',    {'WHU','SYSU','WHU','SYSU','WHU','WHU','LEVIR','SYSU','LEVIR','LEVIR','LEVIR','SYSU'}, ...
 'q_rho', {0.470, 0.524, 0.467, 0.507, 0.474, 0.777, 0.745, 0.837, 0.545, 0.542, 0.902, 1.011}, ...
 'q_cv',  {0.153, 0.136, 0.146, 0.130, 0.147, 0.150, 0.143, 0.188, 0.174, 0.163, 0.063, 0.118}, ...
 'k_rho', {0.345, 0.378, 0.344, 0.357, 0.336, 0.587, 0.532, 0.697, 0.360, 0.348, 0.920, 0.891});
end
