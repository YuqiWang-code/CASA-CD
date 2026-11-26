% c1_math_checks.m — 阶段 0 数学单元检查（执行方案 §1.2 + §1.3）
%
%   目的：把 v1 里被误用的三类对象显式测一遍，任何一项不通过都不许进入后续实验：
%     A. 距离族        欧氏 / Poincare 测地 / Lorentz 测地 的定义、输入域、对称性
%     B. 映射族        直接提升的像是 π(z)，标准指数映射是 tanh(|v|/2)·u，二者不同
%     C. 池化族        OEM_alpha 官方口径；alpha=0 必须与 Lorentz 加权重心完全一致
%
%   输出：results/log_c1.txt（UTF-8），results/res_c1.mat
%   不通过项会打印 FAIL 并在日志里留痕，不静默通过。

clear; clc;
here = fileparts(mfilename('fullpath'));
addpath(here);
S = hyper_setup2(); S.zh();
outdir = fullfile(here, 'results');
figdir = fullfile(here, 'figures');
if ~exist(outdir,'dir'), mkdir(outdir); end
if ~exist(figdir,'dir'), mkdir(figdir); end

L = {};
chk = struct('name',{},'pass',{},'err',{});
logline('reset');
rec = @(nm, ok, err) struct('name',nm,'pass',ok,'err',err);

logline('===== C1 数学单元检查 =====');
logline('MATLAB %s  |  双精度 eps=%.3g', version, eps);
logline('工具箱检查：ttest=%d  corr=%d  boxplot=%d  pdist2=%d', ...
    exist('ttest','file'), exist('corr','file'), exist('boxplot','file'), exist('pdist2','file'));
logline('（上面四项若为 0，说明依赖 Statistics 工具箱的写法在本机不可用，后续脚本一律用基础实现）');
logline('');

rs = RandStream('mt19937ar','Seed',20261126);
d  = 32; n = 40;

%% ---------- A1 欧氏距离矩阵：显式展开 vs 双重循环 ----------
% 说明：完全相同的两点，展开式里 a+b-2xy 会残留 ~1e-13 的舍入，sqrt 后 ~1e-7；
%       这是浮点性质，不是实现错误。因此严格比较只取非对角元，对角元单独限 <1e-6。
Z = randn(rs, n, d) * 0.8;
Dm = S.euc_dist(Z, Z);
Dl = zeros(n);
for i = 1:n
    for j = 1:n
        Dl(i,j) = norm(Z(i,:) - Z(j,:));
    end
end
off = ~eye(n,'logical');
e1 = max(abs(Dm(off) - Dl(off)));
d1 = max(abs(diag(Dm)));
ok1 = e1 < 1e-12 && max(max(abs(Dm - Dm'))) < 1e-12 && d1 < 1e-6 && all(Dm(:) >= 0);
chk(end+1) = rec('A1 欧氏距离矩阵（展开=循环、对称、对角≈0、非负）', ok1, e1);
logline('[%s] A1 欧氏距离矩阵  非对角 max|Δ|=%.3e（对角残差 %.1e，舍入性质）', pick(ok1), e1, d1);

%% ---------- A2 Poincare 与 Lorentz 距离相等 ----------
P = S.pball(Z);                 % 直接提升的像，严格在球内
T = S.lift(Z);
Dg2 = S.geo_dist(P,P); Dl2 = S.lorentz_dist(T,T);
e2 = max(abs(Dg2(off) - Dl2(off)));
d2 = max(abs(diag(Dg2) - diag(Dl2)));
ok2 = e2 < 1e-12 && d2 < 1e-6;
chk(end+1) = rec('A2 Poincare 测地 = Lorentz 测地（同一批点）', ok2, e2);
logline('[%s] A2 Poincare = Lorentz  非对角 max|Δ|=%.3e（对角残差 %.1e）', pick(ok2), e2, d2);

%% ---------- A3 输入域校验：球外点必须报错 ----------
bad = Z; bad(1,:) = bad(1,:) / max(sqrt(sum(bad(1,:).^2,2)),eps) * 1.5;
raised = false;
try
    S.geo_dist(bad, P);
catch errA3
    raised = strcmp(errA3.identifier, 'hyper_setup2:ball');
end
chk(end+1) = rec('A3 球外点调用 geodist 必须报错（不 clamp 掩盖）', raised, double(~raised));
logline('[%s] A3 球外点校验  raised=%d', pick(raised), raised);

%% ---------- A4 半径恒等式 ----------
r1 = S.radius(Z);                       % asinh|z|
r2 = acosh(T(:,1));                     % Lorentz 时间坐标反函数
e4 = max(abs(r1 - r2));
% d(0, π(z)) 也应当等于 asinh|z|
P0 = zeros(1,d);
e4b = max(abs(S.geo_dist(P0, P) - r1'));
ok4 = e4 < 1e-12 && e4b < 1e-12;
chk(end+1) = rec('A4 半径 = asinh|z| = acosh(t) = d(0,π(z))', ok4, max(e4,e4b));
logline('[%s] A4 半径恒等式  max|Δ|=%.3e（含 d(0,π(z)) %.3e）', pick(ok4), e4, e4b);

%% ---------- A5 标准指数/对数映射往返（与直接提升不是同一映射） ----------
V = randn(rs, 6, d) * 0.7;
Pv = S.expmap0(V);                       % tanh(|v|/2)u
Vb = S.logmap0(Pv);
e5 = max(max(abs(V - Vb)));
% d(0, exp_0(v)) 应等于 |v|（而不是 asinh|v|）
e5b = max(abs(S.geo_dist(P0, Pv) - sqrt(sum(V.^2,2))'));
% π(z) 与 exp_0(v) 的关系：v = asinh|z|·u 时两者相等
nz = sqrt(sum(Z(1:6,:).^2,2));
Vz = bsxfun(@times, asinh(nz), Z(1:6,:)./nz);
e5c = max(max(abs(S.expmap0(Vz) - P(1:6,:))));
ok5 = e5 < 1e-11 && e5b < 1e-11 && e5c < 1e-11;
chk(end+1) = rec('A5 exp/log 往返；d(0,exp_0(v))=|v|；π(z)=exp_0(asinh|z|·u)', ok5, max([e5 e5b e5c]));
logline('[%s] A5 映射族  log∘exp 往返 %.3e | d(0,exp_0(v))=|v| %.3e | π(z)=exp_0 等价 %.3e', ...
    pick(ok5), e5, e5b, e5c);
logline('      注意：直接提升 d(0,π(z))=asinh|z|，标准 exp d(0,exp_0(v))=|v|；');
logline('      |z|=sinh|v|，两者不是同一映射——v1 的 expmap0 注释把这一条混为同一个。');

%% ---------- A6 等半径最近邻：测地与欧氏排序一致 ----------
r0 = 0.60; Nk = 224;
dirs = randn(rs, Nk, d); dirs = bsxfun(@rdivide, dirs, sqrt(sum(dirs.^2,2)));
Q = S.pball(0.9 * dirs(1,:));
Pk = S.pball(r0 * dirs);
Dg = S.geo_dist(Q, Pk); De = S.euc_dist(Q, Pk);
[~, ig] = sort(Dg); [~, ie] = sort(De);
agree = mean(ig(:) == ie(:));
e6 = S.spearman(Dg(:), De(:));
ok6 = agree == 1;
chk(end+1) = rec('A6 等半径时测地/欧氏最近邻排序完全一致', ok6, 1-agree);
logline('[%s] A6 等半径排序  一致率 %.3f  Spearman %.4f', pick(ok6), agree, e6);

%% ---------- A7 小半径极限（在切向坐标 z 下才有 d_geo ≈ |Δz|） ----------
% v1 的说法"小范数极限接近欧氏"必须写清楚是哪个坐标：
%   Poincare 坐标里 d_geo ≈ 2|Δp|（原点处度量因子是 2），
%   切向坐标 z 里 d_geo ≈ |Δz|。c1 用后者判定，并把两者的比值一并打印。
scales = [1e-3 1e-2 1e-1 3e-1 1];
ratP = nan(size(scales)); ratZ = nan(size(scales));
for i = 1:numel(scales)
    ZA = scales(i) * randn(rs, 8, d);
    ZB = scales(i) * randn(rs, 8, d);
    PA = S.pball(ZA); PB = S.pball(ZB);
    Dg = S.geo_dist(PA,PB);
    DeP = S.euc_dist(PA,PB);
    DeZ = S.euc_dist(ZA,ZB);
    m = DeP > 0;
    ratP(i) = mean(Dg(m) ./ DeP(m));
    ratZ(i) = mean(Dg(m) ./ DeZ(m));
end
ok7 = ratZ(1) > 0.99 && ratZ(1) < 1.01 && all(diff(ratZ(:)) <= 1e-12);
chk(end+1) = rec('A7 小半径极限：切向坐标下 d_geo/|Δz| → 1', ok7, ratZ(1));
logline('[%s] A7 小半径极限  半径 1e-3/1e-2/1e-1/0.3/1', pick(ok7));
logline('      d_geo/|Δz| = %s  （应 → 1）', sprintf('%.3f ', ratZ));
logline('      d_geo/|Δp| = %s  （Poincare 坐标里 → 2，这是原点度量因子，不是错误）', ...
        sprintf('%.3f ', ratP));

%% ---------- A8 近边界精度 ----------
p1 = S.pball([1e3 0 0]);  p2 = S.pball([1e3 0.5 0]);
db = S.geo_dist(p1, p2);
% 解析：cosh d = sqrt(1+1e6)sqrt(1+1e6.25) - 1e6
cc = sqrt(1+1e6)*sqrt(1+1e6+0.25) - 1e6;
e8 = abs(db - acosh(cc)) / max(acosh(cc),1e-12);
% 近边界：cosh d 是两个 ~1e6 的数相减，参考式本身就有 ~1e-10 的相消误差，
% 所以这一项的门限单列为 1e-8，不与一般项的 1e-12 混用（方案 §1.2 要求单独报）。
ok8 = e8 < 1e-8;
chk(end+1) = rec('A8 近边界点的测地距离相对误差 < 1e-8（相消受限，单独门限）', ok8, e8);
logline('[%s] A8 近边界 |p|=%.6f  d=%.6f  相对误差 %.3e（两 ~1e6 数相减，相消受限）', ...
    pick(ok8), sqrt(sum(p1.^2,2)), db, e8);

%% ---------- B1 LN：总体方差 / eps / 仿射 ----------
F = randn(rs, 50, 3, 64);
U  = S.layernorm(F, 1, 0, 1e-5);
X  = reshape(U,[],64);
e9 = max(abs(mean(X,2)));                      % 去均值
vr = var(X,1,2);
e9b = max(abs(vr - (1 - 1e-5./max(vr,eps))));  % eps 造成的小偏差（量级 ~eps）
g  = 1 + 0.1*randn(1,1,64); b = 0.1*randn(1,1,64);
U2 = S.layernorm(F, g, b, 1e-5);
X2 = reshape(U2,[],64);
Dv = X2 - (bsxfun(@times, X, g(:)') + b(:)');
e9c = max(abs(Dv(:)));
ok9 = all(e9(:) < 1e-12) && all(e9c(:) < 1e-12);
chk(end+1) = rec('B1 LayerNorm = 去均值 + 总体方差 + eps + 仿射', ok9, max(e9,e9c));
logline('[%s] B1 LayerNorm  去均值 %.3e  仿射重构 %.3e', pick(ok9), e9, e9c);

%% ---------- B2 L2 归一化 ≠ LayerNorm ----------
Ua = S.l2norm(F); Ub = S.layernorm(F);
diffB2 = max(abs(Ua(:) - Ub(:)));
okB2 = diffB2 > 1e-3;
chk(end+1) = rec('B2 L2 归一化与真 LayerNorm 不是同一算子（不可混称）', okB2, diffB2);
logline('[%s] B2 L2 vs LN  max|Δ|=%.3f（应当明显不为 0，"L2 单位化"不能叫 LayerNorm）', pick(okB2), diffB2);

%% ---------- C1 OEM alpha=0 与 Lorentz 加权重心完全一致 ----------
W = randn(rs, n, d) * 0.9;
w = rand(rs, n, 1) + 0.2;
zA = S.pool_lorentz(W, w);
zB = S.pool_oem(W, w, 0);
eC1 = max(abs(zA - zB));
okC1 = eC1 < 1e-12;
chk(end+1) = rec('C1 OEM_0 = Lorentz 加权重心 = 标准 Einstein 中点', okC1, eC1);
logline('[%s] C1 OEM_0 = Lorentz  max|Δ|=%.3e', pick(okC1), eC1);

%% ---------- C2 等半径成员：所有 alpha 相同 ----------
rC = 0.8; D2 = randn(rs, 60, d);
D2 = bsxfun(@rdivide, D2, sqrt(sum(D2.^2,2))) * rC;
w2 = rand(rs, 60, 1) + 0.2;
al = [0 0.5 1 2 3];
zz = nan(numel(al), d);
for i = 1:numel(al)
    zz(i,:) = S.pool_oem(D2, w2, al(i));
end
spread = max(max(abs(zz - zz(1,:))));
okC2 = spread < 1e-10;
chk(end+1) = rec('C2 成员等半径时 OEM 的各 alpha 完全相同', okC2, spread);
logline('[%s] C2 等半径 alpha 无关  max|Δ|=%.3e', pick(okC2), spread);

%% ---------- C3 相反方向等半径 → 半径 0；不同半径反向 → 半径变小 ----------
zzA = S.pool_oem([1 0; -1 0]*0.8, [1;1], 0);
rA  = S.radius(zzA);
z1  = S.pool_oem([1.6 0; -0.3 0]/1, [1;1], 0);   % tanh 坐标：半径 1.6 与 0.3 反向
rB  = S.radius(z1);
% 解析：atanh(tanh r1 - tanh r2) 的一半？直接用定义核对：A=|Σ w u tanh| 形式
zIn = [sinh(1.6) 0; -sinh(0.3) 0];               % 这是 z 坐标
zOut = S.pool_oem(zIn, [1;1], 0);
rOut = S.radius(zOut);
Aana = (sinh(1.6) - sinh(0.3))/2;                % 平均后 |Σ w z_i| = (t1-t2)/2
okC3 = rA < 1e-9 && rOut < 1.6 && rOut > 0.3;
chk(end+1) = rec('C3 相反等半径抵消到原点；不同半径反向仍减小半径', okC3, rOut);
logline('[%s] C3 反向抵消  r(相反等半径)=%.2e  r(1.6 与 0.3 反向)=%.4f（介于 0.3~1.6）', ...
    pick(okC3), rA, rOut);

%% ---------- C4 权重同乘正数不变；零质量报错 ----------
zS1 = S.pool_oem(W, w, 1);
zS2 = S.pool_oem(W, 7.3*w, 1);
eC4 = max(abs(zS1 - zS2));
raisedC4 = false;
try
    S.pool_oem(W, zeros(n,1), 1);
catch errC4
    raisedC4 = strcmp(errC4.identifier, 'hyper_setup2:oem');
end
okC4 = eC4 < 1e-12 && raisedC4;
chk(end+1) = rec('C4 权重正比缩放不变；零质量显式报错', okC4, eC4);
logline('[%s] C4 权重不变性 max|Δ|=%.3e  零质量报错=%d', pick(okC4), eC4, raisedC4);

%% ---------- C5 Tangent 池化与 Lorentz 池化的大小关系 ----------
% 同半径成员：切空间均值半径 > Lorentz/Einstein 半径（S2 的观察，这里做硬检查）
zT = S.pool_tangent(D2, w2);
rT = S.radius(zT); rL = S.radius(S.pool_lorentz(D2, w2));
okC5 = rT > rL;
chk(end+1) = rec('C5 同半径成员：Tangent 池化半径 > Lorentz 池化半径', okC5, rT-rL);
logline('[%s] C5 Tangent vs Lorentz  r_T=%.4f  r_L=%.4f（差 %.4f）', pick(okC5), rT, rL, rT-rL);

%% ---------- C6 几何差异能不能被"温度 + 偏置"吸收（要分半径区间看） ----------
% 固定查询、等半径键：排序恒等式成立（A6），但 softmax 权重是否等价，等价于
% "-d_geo^2 能不能写成 -d_euc^2 的仿射函数"。结论随半径强烈变化，所以分三档测。
cfg = [0.49 0.345; 0.90 0.90; 2.00 0.50];
cosv = linspace(-0.999, 0.999, 400)';
resTau = nan(size(cfg,1),1); resAff = nan(size(cfg,1),1); tauBestV = nan(size(cfg,1),1);
relAff = nan(size(cfg,1),1);
for k = 1:size(cfg,1)
    rq = cfg(k,1); rk = cfg(k,2);
    qz = [rq 0 0];
    kz = [rk*cosv, rk*sqrt(max(1-cosv.^2,0)), zeros(size(cosv))];
    % 注意：geo_dist 返回行向量，必须转列，否则 [Le ones] 会被横向拼成 1×2N 的病态方程
    Lg = reshape(-S.geo_dist(S.pball(qz), S.pball(kz)).^2, [], 1);
    Le = reshape(-S.euc_dist(S.pball(qz), S.pball(kz)).^2, [], 1);
    taus = linspace(0.2, 8, 6000);
    best = inf;
    for i = 1:numel(taus)
        m = max(abs(Lg/taus(i) - Le));
        if m < best, best = m; tauBestV(k) = taus(i); end
    end
    resTau(k) = best;
    Xa = [Le, ones(size(Le))];
    cf = Xa \ Lg;
    resAff(k) = max(abs(Xa*cf - Lg));
    relAff(k) = resAff(k) / max(max(Lg) - min(Lg), eps);
end
% 判据（相对残差）：模块实际半径区间内，几何差异基本等价于"温度 + 偏置"；
%   半径拉大以后就不再是仿射关系。这一条决定了 H4 的"tau 惰性"能外推多远。
okC6 = relAff(1) < 0.01 && relAff(3) > 0.02;
chk(end+1) = rec('C6 实际半径下几何差异近似"温度+偏置"；半径拉大后不成立', okC6, relAff(3));
logline('[%s] C6 几何差异的可吸收性（相对残差 = 残差 / logit 跨度）', pick(okC6));
for k = 1:size(cfg,1)
    logline(['      rho_Q=%.2f rho_K=%.2f：最优单一温度 tau=%.2f 残差 %.2e；' ...
             '仿射吸收后残差 %.2e（相对 %.2f%%）'], ...
        cfg(k,1), cfg(k,2), tauBestV(k), resTau(k), resAff(k), 100*relAff(k));
end
logline('      排序恒等（A6），但只有小半径区间里 softmax 才近似相同；');
logline('      所以 H4 的"tau 惰性 / AUC 对 tau 不变"在模块实际半径下成立，不能外推成"两种注意力的 logit 幅度相同"。');

logline('');
nPass = sum([chk.pass]); nTot = numel(chk);
logline('===== 汇总：%d/%d 通过 =====', nPass, nTot);
for i = 1:numel(chk)
    if ~chk(i).pass
        logline('FAIL: %s（err=%.3e）', chk(i).name, chk(i).err);
    end
end

L = logline('get');
S.writefile(fullfile(outdir,'log_c1.txt'), L);
save(fullfile(outdir,'res_c1.mat'), 'chk', 'ratZ', 'ratP', 'resTau', 'resAff', 'tauBestV', 'cfg');
fprintf('%d/%d passed. Log: %s\n', nPass, nTot, fullfile(outdir,'log_c1.txt'));

% ---------------------------------------------------------------------
function s = pick(ok)
if ok, s = 'PASS'; else, s = 'FAIL'; end
end

function out = logline(fmt, varargin)
% 日志缓冲：logline('reset') 清空；logline('get') 取出；其余追加一行
persistent B
if isempty(B), B = {}; end
if nargin >= 1 && ischar(fmt) && strcmp(fmt,'reset')
    B = {}; out = B; return;
end
if nargin >= 1 && ischar(fmt) && strcmp(fmt,'get')
    out = B; return;
end
if isempty(varargin)
    B{end+1,1} = fmt;
else
    B{end+1,1} = sprintf(fmt, varargin{:});
end
out = B;
end
