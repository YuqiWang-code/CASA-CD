% =========================================================================
% r1_radius_anatomy.m  ——  修正版 S1/S2：半径的解剖
%
% 相对 v1（docs/simulation/radius_semantics/s1_geometry.m, s2_our_radius.m）
% 的修正（执行方案 §1.4）：
%   1. 带仿射 LayerNorm 的半径上界改成
%        rho <= asinh( ( ||W*diag(gamma)||_op * sqrt(C) + ||W*beta|| ) / sqrt(D) )
%      并与无仿射版本 asinh(||W||_op*sqrt(C/D)) 分别报告；"随机均匀初始化"是
%      初始化口径，不叫"正交投影"。
%   2. 四种 CV 分开定义：CV_Q / CV_Kmember / CV_Kproto / CV_V；总体方差；
%      近零均值记 NaN；加权 CV 记 ESS=1/sum(w^2)。
%   3. 池化解析式规范写法 rho_pool = atanh(A*tanh(rho))，A=||sum w_i u_i||；
%      等权 A^2 = c + (1-c)/n（c 越大表示方向一致性越高）。
%   4. "键增益 x1.2" 必须作用在**已池化的空间坐标**上：rho' = asinh(1.2*sinh rho)，
%      不能等价成把 A 乘 1.2。
%
% 输出：results/log_r1.txt、results/res_r1.mat、figures/fig_r1_01_anatomy.png
% =========================================================================
clear; clc;
here = fileparts(mfilename('fullpath'));
addpath(here);
addpath(fullfile(here,'..','hyperbolic_v2'));
S = hyper_setup2(); S.zh;
outdir = fullfile(here,'results'); figdir = fullfile(here,'figures');
if ~exist(outdir,'dir'), mkdir(outdir); end
if ~exist(figdir,'dir'), mkdir(figdir); end
logline('reset');
logline('===== R1 修正版 S1/S2：半径的解剖 =====');

C = 64; D = 32;                       % 模块：channels=64, head_dim=32
% 模块口径的半径：rho = asinh(||z||/sqrt(D))。注意 hyper_setup2 的 S.radius
% 返回 asinh(||z||)（内在半径，不除 sqrt(D)），两者相差 sqrt(D) 倍的哑元缩放，
% 本脚本统一用 rmod，避免混用（v1 的 S2 就是因为混用得出"采样超上界"的假失败）。
rmod = @(Z) asinh(sqrt(sum(Z.^2,2))/sqrt(D));
rs = RandStream('mt19937ar','Seed',7);
W = (rand(rs,D,C)*2-1)/sqrt(C);        % Conv2d 默认 U(-1/sqrt(C), 1/sqrt(C))
P0 = eye(C) - ones(C)/C;              % LN 的零均值子空间
smax = normest_local(W*P0);
logline('W: %dx%d，随机均匀初始化；sigma_max(W*P0)=%.6f', D, C, smax);
logline('无仿射上界 rho_max = asinh(sigma_max*sqrt(C/D)) = %.6f', asinh(smax*sqrt(C/D)));

%% ---------- 1. 仿射 LN 的上界 ----------
gamma = 1 + 0.35*randn(rs,1,C);       % 真实训练后的 gamma/beta 量级
beta  = 0.20*randn(rs,1,C);
Wg = W.*gamma;                        % W*diag(gamma)
num1 = normest_local(Wg*P0)*sqrt(C) + norm(W*beta(:));
bA = asinh(num1/sqrt(D));
b0 = asinh(smax*sqrt(C/D));
logline('');
logline('[1] 仿射 LN 半径上界');
logline('    gamma 范围 [%.3f, %.3f]，beta 范数 %.3f', min(gamma), max(gamma), norm(beta));
logline('    仿射上界 = %.6f（无仿射 %.6f，比值 %.3f）', bA, b0, bA/b0);
% 数值验证：在 LN 输出范数 = sqrt(C) 的球面上取方向，看最大半径
nTest = 200000;
U = randn(rs,nTest,C); U = bsxfun(@rdivide, U, sqrt(sum(U.^2,2)))*sqrt(C);
Uc = U - repmat(mean(U,2),1,C); Uc = bsxfun(@rdivide, Uc, sqrt(mean(Uc.^2,2)));
Z = (bsxfun(@times, Uc, gamma) + beta) * W.';
rNum = max(rmod(Z));
logline('    球面采样最大半径 = %.6f（仿射上界 %.6f，gap %.2e）', rNum, bA, bA-rNum);
logline('    实测 Q 中位 0.5435 / 天花板 %.3f → 利用率 %.1f%%', bA, 100*0.5435/bA);
logline('    口径提醒：随机均匀初始化是初始化口径，不是已训练模型的谱上界；');
logline('    正交对照需另做 QR 并独立命名（这里只在下面各算一次中位，不混用）。');
rRand = median(rmod(Z));
Uq = orth_local(randn(rs,C,C)); Uq = Uq(1:C,1:C).';
Zq0 = (bsxfun(@times, Uq*Uq(1,:).', gamma) + beta) * W.';
rOrth = median(rmod(Zq0));
logline('    随机方向中位半径 %.4f / 正交方向中位半径 %.4f', rRand, rOrth);

%% ---------- 2. 池化收缩与 atanh 解析式 ----------
logline('');
logline('[2] 池化：atanh 解析式 vs 数值（等半径成员 n 个，方向一致性 c）');
logline('%-6s %-8s %10s %10s %10s', 'n','c','A(实测)','解析','数值(alpha=0)');
rows = [];
for n = [4 16 64 1024]
    for cc = [0 0.3 0.52 0.8]
        rho0 = 1.000;   % 内在半径 asinh(||z||)；atanh 解析式只对内在半径成立
        % 造 n 个单位方向；A 必须用实测的 ||mean u||——c 是"单方向与主轴的
        % 平均余弦"，与"平均两两内积"在有限 n 下不是同一个量。
        u = gen_dirs(rs, n, D, cc);
        A = norm(mean(u,1));
        rAna = atanh(min(1,A)*tanh(rho0));
        z = sinh(rho0)*u;
        zz = S.pool_oem(z, ones(n,1)/n, 0);
        rNum = S.radius(zz);   % 内在半径，与解析式同一口径
        rows = [rows; n cc A rAna rNum]; %#ok<AGROW>
        logline('%-6d %-8.2f %10.4f %10.4f %10.4f', n, cc, A, rAna, rNum);
    end
end
logline('    结论：atanh 解析式与官方 alpha=0 池化一致（最大偏差 %.1e）', ...
        max(abs(rows(:,4)-rows(:,5))));

%% ---------- 3. 四种 CV 的口径差别 ----------
logline('');
logline('[3] 四种 CV 不能混用：在 4 尺度合成特征上分别计算');
nPix = [128 32 32 32];  nScale = 4;
rQuery = []; rKmem = []; rKproto = []; rVmem = [];
for s = 1:nScale
    n = nPix(s);
    U = randn(rs,n,D);
    amp = 0.35 + 0.15*s;
    Zq = amp*bsxfun(@times, U, 1./sqrt(sum(U.^2,2)));
    rQuery = [rQuery; rmod(Zq)]; %#ok<AGROW>
    % K 成员来自变化证据通道（幅度不同、方向独立），不能拿查询点冒充
    Uk = randn(rs,n,D);
    Zk = (0.25 + 0.20*s)*bsxfun(@times, Uk, 1./sqrt(sum(Uk.^2,2)));
    g = 8; m = n/g;
    for t = 1:g
        idx = (t-1)*m + (1:m);
        mem = Zk(idx,:);
        rKmem = [rKmem; rmod(mem)]; %#ok<AGROW>
        P = S.pool_oem(mem, ones(m,1)/m, 0);
        rKproto = [rKproto; rmod(P)]; %#ok<AGROW>
        V = 0.7*Zk(idx,:) + 0.3*randn(rs,m,D)/sqrt(D);
        rVmem = [rVmem; rmod(V)]; %#ok<AGROW>
    end
end
cvQ = std(rQuery,1)/mean(rQuery);
cvKm = std(rKmem,1)/mean(rKmem);
cvKp = std(rKproto,1)/mean(rKproto);
cvV = std(rVmem,1)/mean(rVmem);
logline('    CV_Q       = %.4f  (像素查询半径)', cvQ);
logline('    CV_Kmember = %.4f  (池化前成员)', cvKm);
logline('    CV_Kproto  = %.4f  (原型之间；模块里这个量才对应 key_radius 的离散度)', cvKp);
logline('    CV_Vmember = %.4f  (值空间成员)', cvV);
logline('    注：v1 记录的 q_cv≈0.145 只能当 CV_Q，不是 K 成员/K 原型锚点。');

%% ---------- 4. 键增益的正确作用点 ----------
logline('');
logline('[4] "键半径 x1.2" 只能作用在已池化坐标上：rho'' = asinh(1.2*sinh rho)');
rhoK = 0.345; A = 0.757;
rGain = asinh(1.2*sinh(rhoK));
rWrong = atanh(min(1, 1.2*A)*tanh(rhoK));
logline('    正确：rho'' = %.4f；错误做法（把 A 乘 1.2）= %.4f；差 %.4f', rGain, rWrong, rGain-rWrong);
logline('    角分辨保留：正确 %.1f%%，错误 %.1f%%', ...
        100*2*min(0.470,rGain)/(2*0.470), 100*2*min(0.470,rWrong)/(2*0.470));

%% ---------- 5. 端点跨度 C = 2min 与翻倍代价 ----------
logline('');
logline('[5] 端点跨度与"翻倍代价"');
rq = 0.470; rk = 0.345;
Csp = geo_dist_trig_safe(rq,rk,-1) - geo_dist_trig_safe(rq,rk,1);
logline('    C = d(cos=-1) - d(cos=+1) = %.6f，2*min(rq,rk) = %.6f（差 %.1e）', Csp, 2*min(rq,rk), abs(Csp-2*min(rq,rk)));
rr = [0.345 0.470 1.0 3.0 10.0];
logline('    半径翻倍所需范数倍数 2cosh(rho)：%s', mat2str(2*cosh(rr),4));

%% ---------- 图 ----------
fh = figure('Visible','off','Position',[30 40 1220 460]);
subplot(1,2,1); hold on;
kk = unique(rows(:,2)); cl = lines(numel(kk));
for i = 1:numel(kk)
    m = (rows(:,2) == kk(i));
    plot(rows(m,1), rows(m,4), 'o-','Color',cl(i,:),'LineWidth',1.4,'MarkerSize',5);
end
set(gca,'XScale','log'); grid on; xlabel('池大小 n'); ylabel('池化后半径 \rho');
legend(arrayfun(@(c) sprintf('目标一致性 c=%.2f',c), kk,'UniformOutput',false), ...
       'Location','southwest','FontSize',7.5); legend boxoff;
title('(a) rho_{pool} = atanh(A tanh rho)','FontSize',10.5);
subplot(1,2,2); hold on;
hb = bar([cvQ cvKm cvKp cvV], 0.6, 'FaceColor',[.82 .86 .94],'EdgeColor','k');
set(gca,'XTick',1:4,'XTickLabel',{'CV\_Q','CV\_Kmember','CV\_Kproto','CV\_Vmember'},'FontSize',8.5);
set(gca,'TickLabelInterpreter','none'); grid on; ylabel('CV（总体方差）');
text(2.45, 0.075, 'v1 记录的 0.145 只是 CV_Q 那一列','FontSize',7.5,'Color',[.5 .2 .2]);
title('(b) 四种 CV 不是同一个量（4 尺度合成特征）','FontSize',10.5);
S.savefig('fig_r1_01_anatomy', figdir); close(fh);

save(fullfile(outdir,'res_r1.mat'), 'smax','bA','b0','rows','cvQ','cvKm','cvKp','cvV');
logline('R1 done.');

%% ======================= local functions =======================
function u = gen_dirs(rs, n, d, cc)
% 生成 n 个单位方向，平均两两内积 ≈ cc
u = randn(rs,n,d);
u = bsxfun(@rdivide, u, sqrt(sum(u.^2,2)));
m = mean(u,1); m = m/norm(m);
u = bsxfun(@plus, cc*m, sqrt(max(1-cc^2,0))*u);
u = bsxfun(@rdivide, u, sqrt(sum(u.^2,2)));
end

function s = normest_local(A)
s = norm(A,2);
end

function Q = orth_local(A)
[Q,~] = qr(A,0);
end

function d = geo_dist_trig_safe(rq, rk, cosT)
C = cosh(rq).*cosh(rk) - sinh(rq).*sinh(rk).*cosT;
d = acosh(max(C,1));
end

function L = logline(fmt, varargin)
persistent B
if isempty(B) || strcmp(fmt,'reset'), B = {}; end
if strcmp(fmt,'get'), L = B; return; end
if isempty(varargin), s = fmt; else, s = sprintf(fmt, varargin{:}); end
B{end+1} = s;
here = fileparts(mfilename('fullpath'));
if numel(B) == 1
    fid = fopen(fullfile(here,'results','log_r1.txt'),'w','n','UTF-8');
else
    fid = fopen(fullfile(here,'results','log_r1.txt'),'a','n','UTF-8');
end
fprintf(fid,'%s\n', s); fclose(fid);
L = s;
end
