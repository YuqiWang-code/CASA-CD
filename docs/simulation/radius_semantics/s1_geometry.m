function s1_geometry()
%S1_GEOMETRY  纯几何：半径 = 范数；"近/远"的压缩、分辨率与跨半径耦合
%
%   这一节不涉及任何数据或模型，只回答：在曲率 -1 的双曲空间里，
%   "离原点近"与"离原点远"在几何上到底意味着什么。
%
%   产出：results/figures/fig_s1_01_at_a_glance.png  （一眼看懂的四联图）
%         results/figures/fig_s1_02_cross_radius.png（跨半径耦合：非单调与"径向差压制角度"）
%         results/res_s1.mat, results/log_s1.txt

R = rs_setup(); R.zh();
rng(11,'twister');
out = struct(); L = {};
fprintf('\n===== S1 纯几何 =====\n');
L = say(L,'===== S1 纯几何 =====');

%% ---------- 0. 恒等式与旧仿真交叉验证 ----------
addpath(fullfile(R.root,'..','hyperbolic'));
H  = hyper_setup();
zq = randn(64,7); zk = randn(53,7);
Daa = R.cdist(zq,zk); Dbb = H.lcdist(zq,zk);
out.chk_cdist = max(abs(Daa(:) - Dbb(:)));
z  = randn(300,5);
X  = R.lift(z);
out.chk_rho_lift = max(abs(R.radius(z) - acosh(X(:,1))));
% 同半径恒等：d_geo = acosh(1 + d_euc^2/2)
u  = randn(40,9); u = u./sqrt(sum(u.^2,2)); t0 = 1.37;
xu = t0*u; xv = t0*(u + 0.3*randn(size(u))); xv = xv./sqrt(sum(xv.^2,2))*t0;
de = sqrt(sum((xu-xv).^2,2));
Duv = R.cdist(xu,xv);
out.chk_same_radius = max(abs(diag(Duv) - acosh(1+de.^2/2)));
% 角度对比度恒等：C = 2*min(rQ,rK)
rq_test = linspace(0.05,1.2,40); chk_C = zeros(numel(rq_test),1);
for i = 1:numel(rq_test)
    chk_C(i) = abs(R.angcontrast(rq_test(i), 0.37) - 2*min(rq_test(i),0.37));
end
out.chk_contrast = max(chk_C);
L = say(L,'恒等式误差: cdist %.1e | rho=acosh(t) %.1e | 同半径 %.1e | C=2min %.1e', ...
        out.chk_cdist, out.chk_rho_lift, out.chk_same_radius, out.chk_contrast);

%% ---------- 1. 半径 = asinh(范数)：压缩与"翻倍代价" ----------
t_list = [0.05 0.1 0.2 sinh(0.345) sinh(0.470) 0.7 1 2 3 5 10 30]';
rho    = R.radius(t_list);
slope  = 1./sqrt(1+t_list.^2);        % d rho / d t
cost2  = 2*cosh(rho);                 % 半径翻倍所需范数倍数 = sinh(2rho)/sinh(rho)
L = say(L,'');
L = say(L,'  t      rho   d rho/dt  范数需乘几倍才能让半径翻倍');
for i = 1:numel(t_list)
    L = say(L,'%7.3f  %6.3f   %6.3f   %10.2f', t_list(i), rho(i), slope(i), cost2(i));
end
out.t = t_list; out.rho = rho; out.slope = slope; out.cost2 = cost2;

%% ---------- 2. 角度对比度恰好等于 2*min(rho_Q,rho_K) ----------
% 由 cosh d = cosh(rQ)cosh(rK) - sinh(rQ)sinh(rK)cos(theta) 可得精确式：
%   d(cos=+1) = |rQ-rK|,   d(cos=-1) = rQ+rK   =>   C = 2*min(rQ,rK)
% 含义：方向上还能分辨多少，上限由"两者中离原点更近的那个"决定。
pair = [0.470 0.345; 0.545 0.360; 0.902 0.920; 1.011 0.891; 0.490 0.070];
pairC = zeros(size(pair,1),1);
for i = 1:size(pair,1)
    pairC(i) = R.angcontrast(pair(i,1), pair(i,2));
end
out.pair = pair; out.pairC = pairC;
out.pair_ratio = pairC ./ R.angcontrast(pair(:,1), pair(:,1));
L = say(L,'');
L = say(L,'(Q半径, K半径)  角度对比度 C   占同半径对照的比例');
for i = 1:size(pair,1)
    L = say(L,'  (%.3f, %.3f)      %.3f          %3.0f%%', pair(i,1), pair(i,2), ...
            out.pair_ratio(i)*R.angcontrast(pair(i,1),pair(i,1)), 100*out.pair_ratio(i));
end

%% ---------- 3. 容量：224 个键放在半径 rho 的球壳上有多挤 ----------
d_dim = 32; n_key = 224;
rho_grid = [0.05 0.1 0.2 0.345 0.49 0.7 1.0 1.5 2.0 3.0];
nn_med = zeros(size(rho_grid));
for i = 1:numel(rho_grid)
    U = randn(n_key, d_dim); U = U./sqrt(sum(U.^2,2));
    Z = sinh(rho_grid(i))*U;
    D = R.cdist(Z,Z); D(1:n_key+1:end) = inf;
    nn_med(i) = median(min(D,[],2));
end
out.rho_grid = rho_grid; out.nn_med = nn_med;
L = say(L,'');
L = say(L,'224 键在半径 rho 处的典型最近邻距离:');
L = say(L,'  rho  = %s', sprintf('%.2f ', rho_grid));
L = say(L,'  d_nn = %s', sprintf('%.3f ', nn_med));

%% ---------- 图 1：一眼看懂 ----------
f1 = figure('Visible','off','Position',[60 60 1340 1000]);

% (a) Poincare 圆盘：我们的点落在哪里
subplot(2,2,1); hold on; axis equal; axis([-1.18 1.18 -1.18 1.18]); box on;
rectangle('Position',[-1 -1 2 2],'Curvature',[1 1],'EdgeColor',[.2 .2 .2],'LineWidth',1.3);
ring = [0.25 0.5 1 2 3];
for r = ring, rr = tanh(r/2);
    rectangle('Position',[-rr -rr 2*rr 2*rr],'Curvature',[1 1], ...
              'EdgeColor',[.82 .82 .82],'LineStyle',':','LineWidth',.8);
end
pk = tanh(0.345/2); pq = tanh(0.490/2); pf = tanh(2.3/2);
ang = [205 20 112]*pi/180;
h1 = plot(pk*cos(ang(1)), pk*sin(ang(1)), 'o','MarkerSize',10, ...
          'MarkerFaceColor',[.10 .55 .75],'MarkerEdgeColor','k');
h2 = plot(pq*cos(ang(2)), pq*sin(ang(2)), 's','MarkerSize',10, ...
          'MarkerFaceColor',[.90 .35 .15],'MarkerEdgeColor','k');
h3 = plot(pf*cos(ang(3)), pf*sin(ang(3)), '^','MarkerSize',10, ...
          'MarkerFaceColor',[.30 .65 .20],'MarkerEdgeColor','k');
for k = 1:3, plot([0 pf*cos(ang(k))],[0 pf*sin(ang(k))],'-','Color',[.75 .75 .75],'LineWidth',.8); end
text(pk*cos(ang(1))-0.09, pk*sin(ang(1))-0.15, 'K 原型 \rho=0.35', 'FontSize',8.5, ...
     'Color',[.05 .35 .55],'FontWeight','bold');
text(pq*cos(ang(2))+0.03, pq*sin(ang(2))-0.13, 'Q 单像素 \rho=0.49', 'FontSize',8.5, ...
     'Color',[.70 .25 .05],'FontWeight','bold');
text(-0.10, 0.86, '示意：很远 \rho=2.3', 'FontSize',8,'Color',[.20 .50 .15]);
text(-1.13, 1.08, '近（原点）：平均 / 通用 / 背景（池化原型）', 'FontSize',8.5,'Color',[.2 .2 .2]);
text(-1.13,-1.14, '远（外圈）：具体 / 稀有 / 单像素 / 变化', 'FontSize',8.5,'Color',[.2 .2 .2]);
text(-1.13,-0.98, '虚线同心圆 \rho = 0.25 / 0.5 / 1 / 2 / 3', 'FontSize',7.5,'Color',[.5 .5 .5]);
title('(a) 曲率 -1 的 Poincare 圆盘','FontSize',11);

% (b) rho = asinh(t)：近处近似线性，远处对数压缩
subplot(2,2,2); hold on;
tt = logspace(-1.7,1.6,400); plot(tt, R.radius(tt), 'k-','LineWidth',1.9);
m = tt <= 2.5; plot(tt(m), tt(m), '--','Color',[.55 .55 .55],'LineWidth',1.1);
plot(t_list, rho, 'o','MarkerSize',4.5,'MarkerFaceColor','k','MarkerEdgeColor','k');
yline(0.345,'Color',[.10 .55 .75],'LineWidth',1.1,'LineStyle',':');
yline(0.470,'Color',[.90 .35 .15],'LineWidth',1.1,'LineStyle',':');
set(gca,'XScale','log'); ylim([0 4.6]); xlim([0.02 40]); grid on;
text(0.028, 4.25, '远：\rho \approx ln(2t)（对数压缩）', 'FontSize',9,'Color',[.25 .25 .25]);
text(0.028, 3.78, '同样 +1 的范数，换来的半径越来越小', 'FontSize',8.5,'Color',[.35 .35 .35]);
text(0.028, 1.95, '近：\rho \approx t，与欧氏几乎一致', 'FontSize',9,'Color',[.25 .25 .25]);
text(0.028, 0.72, '水平点线 = 实测 K(蓝) / Q(橙) 的半径', 'FontSize',8.5,'Color',[.45 .45 .45]);
xlabel('欧氏范数 t（向量长度）'); ylabel('双曲半径 \rho = asinh(t)');
title('(b) 半径对范数：先近似线性，后对数压缩','FontSize',11);

% (c) 角度对比度 vs 键半径：C = 2 min(rQ,rK)
subplot(2,2,3); hold on;
rkg = linspace(0.01,1.6,300);
cols = [.10 .55 .75; .90 .35 .15; .30 .65 .20];
hh = gobjects(1,3);
for i = 1:3
    C = arrayfun(@(x) R.angcontrast(0.345*(i==1) + 0.470*(i==2) + 0.902*(i==3), x), rkg);
    hh(i) = plot(rkg, C, '-', 'Color', cols(i,:), 'LineWidth',1.8);
    plot(0.345*(i==1)+0.470*(i==2)+0.902*(i==3), ...
         R.angcontrast(0.345*(i==1)+0.470*(i==2)+0.902*(i==3), 0.345*(i==1)+0.470*(i==2)+0.902*(i==3)), ...
         'o','MarkerFaceColor',cols(i,:),'MarkerEdgeColor','k','MarkerSize',6);
end
plot(pair(1,2), pairC(1), 'p','MarkerSize',12,'MarkerFaceColor',[1 1 0],'MarkerEdgeColor','k');
plot(pair(5,2), pairC(5), 'p','MarkerSize',12,'MarkerFaceColor',[1 0 0],'MarkerEdgeColor','k');
grid on; ylim([0 2.0]);
legend(hh, {'\rho_Q=0.345','\rho_Q=0.470','\rho_Q=0.902'},'Location','northwest','FontSize',8); legend boxoff;
text(0.30, 0.17, '实心点 = 同半径时的上限 2\rho_Q', 'FontSize',8.5);
text(0.72, 0.42, '黄五角：实测 (0.470,0.345)，只剩 73%', 'FontSize',8,'Color',[.45 .35 .0]);
text(0.72, 0.30, '红五角：键贴原点，只剩 14%', 'FontSize',8,'Color',[.55 .1 .1]);
xlabel('键半径 \rho_K'); ylabel('角度对比度 C = 2 min(\rho_Q,\rho_K)');
title('(c) 角度可分辨量只由较小的半径决定','FontSize',11);

% (d) 224 个键的最近邻距离 vs 半径
subplot(2,2,4); hold on;
hd = plot(rho_grid, nn_med, 'o-','LineWidth',1.8,'Color',[.2 .2 .2],'MarkerFaceColor',[.2 .2 .2]);
hK = plot(0.345, interp1(rho_grid,nn_med,0.345), 'o','MarkerSize',10, ...
          'MarkerFaceColor',[.10 .55 .75],'MarkerEdgeColor','k');
hQ = plot(0.490, interp1(rho_grid,nn_med,0.490), 's','MarkerSize',10, ...
          'MarkerFaceColor',[.90 .35 .15],'MarkerEdgeColor','k');
grid on; set(gca,'XScale','log','YScale','log');
legend([hd hK hQ], {'224 键最近邻中位数','K 所在半径','Q 所在半径'},'Location','northwest','FontSize',8);
legend boxoff;
xlabel('半径 \rho'); ylabel('224 个键的典型最近邻测地距离');
title('(d) 越远键之间越分得开（代价指数增长）','FontSize',11);
sgtitle('双曲原点附近的“近”与“远”：几何含义','FontSize',13,'FontWeight','bold');
R.savefig('fig_s1_01_at_a_glance'); close(f1);

%% ---------- 图 2：跨半径耦合 ----------
f2 = figure('Visible','off','Position',[60 60 1180 470]);
subplot(1,2,1); hold on;
rkg = linspace(0.01,2.2,400);
cs  = [0.95 0.7 0.3 0 -0.9];
cols = lines(numel(cs));
hl = gobjects(1,numel(cs));
for i = 1:numel(cs)
    hl(i) = plot(rkg, R.cdist_t(0.49, rkg, cs(i)), '-','Color',cols(i,:),'LineWidth',1.7);
end
plot([0 2.2],[0.49 0.49],'k:','LineWidth',1);
rmin = R.radius(sinh(0.49)*0.95/sqrt(1+sinh(0.49)^2*(1-0.95^2)));
plot(rmin, R.cdist_t(0.49,rmin,0.95), 'o','MarkerSize',8, ...
     'MarkerFaceColor',[1 1 0],'MarkerEdgeColor','k');
grid on;
legend(hl, arrayfun(@(c) sprintf('cos\\theta=%.2f',c), cs,'UniformOutput',false), ...
       'Location','northwest','FontSize',8); legend boxoff;
xlabel('键半径 \rho_K'); ylabel('测地距离 d');
title('(a) 固定查询 \rho_Q=0.49：把键推远不一定更远','FontSize',11);
text(1.50, 0.42, '方向一致时 d(\rho_K) 先降后升', 'FontSize',8.5);
text(1.50, 0.28, '黄点 = 最小距离 \rho_K\approx0.47', 'FontSize',8.5,'Color',[.45 .40 .0]);
text(1.50, 0.14, '黑点线 = 键与查询同半径', 'FontSize',8.5,'Color',[.35 .35 .35]);

subplot(1,2,2); hold on;
rqg = linspace(0.02,2.0,220); rkg2 = linspace(0.02,2.0,220);
[RQ,RK] = meshgrid(rqg,rkg2);
CC = arrayfun(@(a,b) R.angcontrast(a,b), RQ, RK);
imagesc(rqg, rkg2, CC); axis xy; colorbar; colormap(parula);
plot([0.02 2],[0.02 2],'w--','LineWidth',1.2);
plot(0.49,0.345,'wp','MarkerSize',13,'MarkerFaceColor','w');
xlabel('查询半径 \rho_Q'); ylabel('键半径 \rho_K');
title('(b) 角度对比度热图：沿对角线最高，偏离即衰减','FontSize',11);
sgtitle('跨半径耦合：半径差不只是“偏移”，它改变角度信息的可用性','FontSize',12,'FontWeight','bold');
R.savefig('fig_s1_02_cross_radius'); close(f2);

%% ---------- 落盘 ----------
save(fullfile(R.outdir,'res_s1.mat'),'out');
R.writefile(fullfile(R.outdir,'log_s1.txt'), L);
fprintf('S1 done.\n');
end

% ---------------------------------------------------------------- helpers --
function L = say(L, fmt, varargin)
s = sprintf(fmt, varargin{:});
fprintf('%s\n', s);
L{end+1,1} = s;
end
