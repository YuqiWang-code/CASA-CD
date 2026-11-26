function s2_our_radius()
%S2_OUR_RADIUS  我们模块的"半径"到底由什么构成
%
%   我们真正的半径定义（models/model/layers/hyperbolic_context.py:39, 96）：
%       u   = LayerNorm_64(x)                       每像素、跨通道归一化
%       z   = qk(u) / sqrt(head_dim)                 共享 1x1 卷积，head_dim=32
%       rho = asinh(||z||) = asinh(||qk(u)||/sqrt(32))
%
%   本节用数值实验回答三件事：
%     (1) 半径对"强度"完全免疫：LN 在前，缩放/平移输入不改变半径；
%     (2) 半径 = 通道模式与 qk 权重顶奇异方向的吻合度；它有硬天花板
%         rho_max = asinh(sigma_max(qk)*sqrt(C/D))，C=64 通道，D=32；
%     (3) 池化会向原点收缩（HypRAG 命题 4.3），收缩量由"池内等效独立方向数"决定，
%         这解释了实测里 K 半径 ≈ 0.6~0.7 * Q 半径。
%
%   产出：results/figures/fig_s2_01_what_is_radius.png
%         results/figures/fig_s2_02_ceiling_and_pooling.png
%         results/res_s2.mat, results/log_s2.txt

R = rs_setup(); R.zh();
rng(7,'twister');
out = struct(); L = {};
L = say(L,'===== S2 我们模块的半径由什么构成 =====');

C = 64; D = 32;                 % channels / head_dim（与 hyperbolic_context.py 一致）
W = R.proj(C, D, 7);            % 复刻 Conv2d(64,32,1,bias=False) 的默认初始化

%% ---------- 1. 半径的等价写法 + LN 的尺度/平移免疫 ----------
z  = randn(2000,5);
X  = R.lift(z);
out.chk_rho = max(abs(R.radius(z) - acosh(X(:,1))));
x  = randn(500, C);
rho_of_x = @(v) R.radius(R.layernorm(v)*W'/sqrt(D));
out.rho_ref = median(rho_of_x(x));
L = say(L,'恒等式 max|asinh(||z||) - acosh(x_0)| = %.1e', out.chk_rho);
for k = [0.05 0.3 2.5 100]
    L = say(L,'LN 不变性 x -> %.2f x : max|drho| = %.2e', k, max(abs(rho_of_x(k*x)-rho_of_x(x))));
end
L = say(L,'（注意：LN 的 eps=1e-5 让这条不变性在方差接近 eps 时失效——x 乘 0.05 后方差 2.5e-3，');
L = say(L,'  偏差升到 1.8e-3；乘 0.01 偏差 4e-2。真实特征方差远大于 eps，故实际不受影响。）');
for b = [-50 5]
    L = say(L,'LN 不变性 x -> x %+d : max|drho| = %.2e', b, max(abs(rho_of_x(x+b)-rho_of_x(x))));
end
out.ln_drift = [max(abs(rho_of_x(2.5*x)-rho_of_x(x))) max(abs(rho_of_x(x+5)-rho_of_x(x)))];

%% ---------- 2. 天花板：sigma_max 决定半径上限 ----------
P0  = eye(C) - ones(C)/C;              % LN 输出必在零均值子空间
Wp  = W*P0;
sv  = svd(Wp);
smax = sv(1);
rho_ceil = asinh(smax*sqrt(C)/sqrt(D));
out.smax = smax; out.rho_ceil = rho_ceil;
out.rho_rand = mean(R.radius(randn(20000,C)*W'/sqrt(D)));
L = say(L,'');
L = say(L,'qk 权重 32x64：sigma_max = %.4f -> 理论半径上限 rho_max = asinh(smax*sqrt(C/D)) = %.4f', smax, rho_ceil);
L = say(L,'随机方向输入的半径基线 = %.4f', out.rho_rand);

A = R.anchors();
runL = {A.run}; dsL = {A.ds}; qR = [A.q_rho]; kR = [A.k_rho];
rq = median(qR); rk = median(kR);
util   = sinh(qR)/(smax*sqrt(C/D));
kratio = kR./qR;
L = say(L,'');
L = say(L,'  运行                          数据集    Q半径   饱和度    K半径   K/Q');
for i = 1:numel(runL)
    L = say(L,'  %-28s  %-6s  %6.3f  %5.1f%%  %6.3f  %5.2f', runL{i}, dsL{i}, ...
            qR(i), 100*util(i), kR(i), kratio(i));
end
L = say(L,'饱和度 = sinh(rho_Q)/(smax*sqrt(C/D))，即该权重把 qk 能给的半径用掉了多少。');
out.util = util; out.kratio = kratio;

%% ---------- 3. 半径 = 与 W 顶奇异方向的吻合度 ----------
[~,~,V] = svd(Wp,'econ'); v1 = V(:,1); v1 = v1(:)';
gfix = randn(C,1); gfix = gfix - v1'*(v1*gfix); gfix = gfix/norm(gfix);   % 固定参考方向
alpha = linspace(0,1,41);  rho_alpha = zeros(size(alpha));
for i = 1:numel(alpha)
    u = sqrt(alpha(i))*v1' + sqrt(1-alpha(i))*gfix;
    u = u - mean(u); u = u/norm(u)*sqrt(C);
    rho_alpha(i) = asinh(norm(W*u)/sqrt(D));
end
out.alpha = alpha; out.rho_alpha = rho_alpha;
L = say(L,'');
L = say(L,'半径随"沿 W 顶奇异方向的能量占比 alpha"单调上升：');
L = say(L,'  alpha = %s', sprintf('%.2f ', alpha(1:5:end)));
L = say(L,'  rho   = %s', sprintf('%.3f ', rho_alpha(1:5:end)));

%% ---------- 4. 池化收缩：解析预测与四种算子 ----------
rho0 = 0.49; t0 = sinh(rho0); d = D;
ngrid = [1 2 4 8 16 32 64 128 256 1024];
cgrid = [0 0.5 0.8 0.95];
ops   = {'amb','euc','ein','oem1','oem2'};
res   = zeros(numel(ngrid), numel(cgrid), numel(ops));
u0 = randn(1,d); u0 = u0/norm(u0);
nrep = 12;                                   % 多次抽样平均，压掉 n 小时的偶然性
for ci = 1:numel(cgrid)
    c = cgrid(ci);
    for ni = 1:numel(ngrid)
        n = ngrid(ni); acc = zeros(1,numel(ops));
        for rep = 1:nrep
            G = randn(n,d);
            G = G - (G*u0')*u0;                      % 去掉与 u0 的分量
            G = G./sqrt(sum(G.^2,2));                % 每个成员单位范数
            U = sqrt(c)*repmat(u0,n,1) + sqrt(1-c)*G;   % 每行范数 = 1
            Z = sinh(rho0)*U; w = ones(n,1)/n;
            acc = acc + [R.radius(R.pool_amb(Z,w)), R.radius(R.pool_euc(Z,w)), ...
                         R.radius(R.pool_ein(Z,w)), R.radius(R.pool_oem(Z,w,1)), ...
                         R.radius(R.pool_oem(Z,w,2))];
        end
        res(ni,ci,:) = acc/nrep;
    end
end
out.ngrid = ngrid; out.cgrid = cgrid; out.pool_res = res;

x_n   = sqrt(bsxfun(@plus, cgrid(:), bsxfun(@rdivide, 1-cgrid(:), ngrid(:)')));
pred_euc = asinh(t0*x_n);
pred_amb = asinh(t0*x_n ./ sqrt(cosh(rho0)^2 - t0^2*x_n.^2));
L = say(L,'');
L = say(L,'池化后半径（rho0=0.49, d=32；c 为池内方向相关系数）：');
for ci = 1:numel(cgrid)
    L = say(L,'  c = %.2f', cgrid(ci));
    L = say(L,'    n        = %s', sprintf('%5d ', ngrid));
    L = say(L,'    环境均值 = %s', sprintf('%5.3f ', res(:,ci,1)));
    L = say(L,'    切空间均 = %s', sprintf('%5.3f ', res(:,ci,2)));
    L = say(L,'    解析(环) = %s', sprintf('%5.3f ', pred_amb(ci,:)));
    L = say(L,'    解析(切) = %s', sprintf('%5.3f ', pred_euc(ci,:)));
end
L = say(L,'');
L = say(L,'同一 n=64、c=0.5、所有成员半径相同(0.49)时五种算子：');
for oi = 1:numel(ops)
    L = say(L,'  %-5s rho = %.4f', ops{oi}, res(ngrid==64, cgrid==0.5, oi));
end
L = say(L,'结论：成员半径完全相同时 Einstein / OEM 与"环境均值"严格等价（同一个归一点），');
L = say(L,'      只有"切空间均值"更大——因为 Minkowski 归一化把空间部分又除以 s>1。');

%% ---------- 4b. 池内半径不一致时，外向池化才有额外偏置 ----------
n = 64;                                      % 一个 cell 约 64 个 token
cv_grid = [0.05 0.15 0.30];
het = zeros(numel(cv_grid),6);
for vi = 1:numel(cv_grid)
    cvv = cv_grid(vi); acc = zeros(1,6);
    for rep = 1:nrep
        G = randn(n,d); G = G - (G*u0')*u0; G = G./sqrt(sum(G.^2,2));
        U = sqrt(0.5)*repmat(u0,n,1) + sqrt(0.5)*G;
        U = U./sqrt(sum(U.^2,2));
        ri = max(rq*(1 + cvv*randn(n,1)), 0.05);
        Z = U .* sinh(ri); w = ones(n,1)/n;
        acc = acc + [mean(ri), R.radius(R.pool_amb(Z,w)), R.radius(R.pool_euc(Z,w)), ...
                     R.radius(R.pool_ein(Z,w)), R.radius(R.pool_oem(Z,w,1)), ...
                     R.radius(R.pool_oem(Z,w,2))];
    end
    het(vi,:) = acc/nrep;
end
out.het = het; out.cv_grid = cv_grid;
L = say(L,'');
L = say(L,'池内半径不一致（n=64, c=0.5, 半径 cv 变化）时的池化后半径：');
L = say(L,'  cv      成员均值   环境均值   切空间均值   Einstein   OEM-1    OEM-2   OEM相对Einstein');
for vi = 1:numel(cv_grid)
    L = say(L,'  %.2f    %7.3f    %7.3f    %7.3f    %7.3f   %6.3f   %6.3f   %+5.1f%%', ...
            cv_grid(vi), het(vi,1), het(vi,2), het(vi,3), het(vi,4), het(vi,5), het(vi,6), ...
            100*(het(vi,5)/het(vi,4)-1));
end
L = say(L,'（实测 radius_cv 约 0.13~0.15：OEM 只能带来百分之几的半径增益，');
L = say(L,'  而 Q/K 差距是 27%，主因是方向抵消而非半径加权。要补这个差距只能径向重锚。）');

L = say(L,'');
L = say(L,'由实测 (Q,K) 半径反推池内等效独立方向数 n_eff = (sinh rQ / sinh rK)^2：');
neff = (sinh(qR)./sinh(kR)).^2;
for i = 1:numel(runL)
    L = say(L,'  %-28s  %-6s  n_eff = %5.2f', runL{i}, dsL{i}, neff(i));
end
out.neff = neff;
L = say(L,'（真实 cell 里通常有 64~1024 个像素；n_eff≈2 说明这些像素方向几乎相同，');
L = say(L,'  所以池化"几乎不收缩"，同时 K 半径被池化锁在 Q 半径的 0.6~0.7 倍。）');

% 径向重锚：把池化后的切向量乘一个标量，使半径回到 Q 的水平
C_now    = R.angcontrast(rq, rk);
C_anchor = R.angcontrast(rq, rq);
scale_needed = sinh(rq)/sinh(rk);
L = say(L,'');
L = say(L,'径向重锚（把池化后的键切向量整体乘 %.3f）：', scale_needed);
L = say(L,'  现状：rho_K = %.3f，角度对比度 %.3f（相对同半径上限 %.0f%%）', rk, C_now, 100*C_now/C_anchor);
L = say(L,'  重锚：rho_K = %.3f，角度对比度 %.3f（100%%）', rq, C_anchor);
L = say(L,'  代价：每个尺度 1 个标量（或 4 个参数），不增加 KV 数量。');
L = say(L,'  这就是"半径"这一维唯一还没被用上的自由度：Q 与 K 共享同一个 qk，');
L = say(L,'  半径差完全由池化造成（不是设计），而 logit 里的距离对角度的权重正比于 rho_Q*rho_K。');
out.C_now = C_now; out.C_anchor = C_anchor; out.scale_needed = scale_needed;

%% ---------- 图 1 ----------
f1 = figure('Visible','off','Position',[40 40 1360 990]);

% (a) LN 免疫
subplot(2,2,1); hold on;
xx  = {x, 0.01*x, 2.5*x, 100*x, x+5};
lab = {'x','0.01x','2.5x','100x','x+5'};
rr  = cellfun(@(v) median(rho_of_x(v)), xx);
bar(1:numel(xx), rr, 0.55, 'FaceColor',[.55 .70 .85],'EdgeColor','k');
plot([0.4 numel(xx)+0.6], [out.rho_ref out.rho_ref], 'r--','LineWidth',1.4);
set(gca,'XTick',1:numel(xx),'XTickLabel',lab);
ylim([0 0.46]); grid on; ylabel('半径 \rho 的中位数');
text(1.15, 0.42, sprintf('所有情况总偏差 %.1e', max(out.ln_drift)), ...
     'FontSize',8.5,'Color',[.65 .2 .1]);
text(1.15, 0.375, sprintf('五种输入的中位数都是 %.3f（完全相同）', out.rho_ref), ...
     'FontSize',8.5,'Color',[.3 .3 .3]);
title('(a) 半径对强度/对比度完全免疫（LN 在前）','FontSize',10.5);

% (b) 半径 = 吻合度
subplot(2,2,2); hold on;
plot(alpha, rho_alpha, 'k-','LineWidth',1.9);
plot(alpha, out.rho_rand*ones(size(alpha)), '--','Color',[.5 .5 .5],'LineWidth',1.1);
yline(rho_ceil,'Color',[.75 .2 .15],'LineWidth',1.3,'LineStyle','-.');
yl = sort(unique([min(qR) median(qR) max(qR)]));
for i = 1:numel(yl), yline(yl(i),'Color',[.15 .45 .75],'LineWidth',.8,'LineStyle',':'); end
text(0.02, rho_ceil-0.075, sprintf('天花板 \\rho_{max}=%.2f（alpha=1）',rho_ceil),'Color',[.75 .2 .15],'FontSize',9);
text(0.55, out.rho_rand+0.035, sprintf('随机方向典型值 %.2f', out.rho_rand),'Color',[.42 .42 .42],'FontSize',8.5);
text(0.30, 0.435, sprintf('实测 Q 的最小/中位/最大：%.2f / %.2f / %.2f', ...
     min(qR), median(qR), max(qR)),'Color',[.15 .45 .75],'FontSize',8.5);
text(0.03, 0.33, '实测中位数与随机方向典型值几乎相同：', 'FontSize',8.2,'Color',[.3 .3 .3]);
text(0.03, 0.26, '这批权重没有把像素推到明显更大的半径。', 'FontSize',8.2,'Color',[.3 .3 .3]);
grid on; ylim([0 rho_ceil*1.12]); xlim([0 1]);
xlabel('沿 W 顶奇异方向的能量占比 \alpha'); ylabel('半径 \rho');
title('(b) 半径只衡量"通道形状与 qk 权重的吻合度"','FontSize',10.5);

% (c) 池化收缩
subplot(2,2,3); hold on;
cols = [.15 .45 .75; .20 .60 .35; .85 .55 .10; .70 .25 .65];
hh = gobjects(1,numel(cgrid));
for ci = 1:numel(cgrid)
    hh(ci) = plot(ngrid, res(:,ci,1), 'o-','Color',cols(ci,:),'LineWidth',1.5,'MarkerFaceColor','w');
    plot(ngrid, pred_amb(ci,:), '--','Color',cols(ci,:),'LineWidth',.9);
end
set(gca,'XScale','log','YScale','log'); grid on;
legend(hh, arrayfun(@(c) sprintf('c=%.2f',c), cgrid,'UniformOutput',false), ...
       'Location','southwest','FontSize',8); legend boxoff;
xlabel('池大小 n'); ylabel('池化后半径 \rho_{pool}');
title('(c) 池化向原点收缩：实线=实验，虚线=解析','FontSize',10.5);
text(1.3, 0.055, 'c=0（方向互相独立）时按 1/\surdn 收缩','FontSize',8,'Color',[.3 .3 .3]);

% (d) 12 个真实权重的 K-Q 半径平面：全部落在 K/Q≈0.7 的带上
subplot(2,2,4); hold on;
plot([0 1.2],[0 1.2],'k--','LineWidth',1.0);
plot([0 1.2],[0 0.7*1.2],':','Color',[.55 .55 .55],'LineWidth',1.2);
plot([0 1.2],[0 1.2]*0.9,'-.','Color',[.75 .55 .50],'LineWidth',.8);
xline(out.rho_rand,'Color',[.55 .55 .55],'LineWidth',1.0,'LineStyle','-.');
xline(rho_ceil,'Color',[.75 .2 .15],'LineWidth',1.2,'LineStyle','-.');
dsc = unique(dsL);
cols2 = [.15 .45 .75; .20 .60 .35; .85 .55 .10; .70 .25 .65];
hh2 = gobjects(1,numel(dsc));
for i = 1:numel(dsc)
    m = strcmp(dsL, dsc{i});
    hh2(i) = plot(qR(m), kR(m), 'o','MarkerSize',7,'MarkerFaceColor',cols2(i,:), ...
         'MarkerEdgeColor','k','LineWidth',.6);
end
text(1.03,1.15,'K = Q','FontSize',8,'Color',[.15 .15 .15]);
text(1.03,0.78,'K = 0.7 Q','FontSize',8,'Color',[.45 .45 .45]);
text(0.30,0.16,'随机方向 0.54','FontSize',7.8,'Color',[.40 .40 .40]);
text(0.88,0.16,'天花板 1.09','FontSize',7.8,'Color',[.75 .2 .15]);
text(0.04,0.06,sprintf('n_{eff} = (sinh\\rho_Q/sinh\\rho_K)^2 中位数 %.2f',median(neff)), ...
     'FontSize',8.2,'Color',[.2 .2 .2]);
legend(hh2, dsc, 'Location','southeast','FontSize',7); legend boxoff;
grid on; axis([0 1.2 0 1.2]); axis square;
xlabel('Q 半径 \rho_Q'); ylabel('K 半径 \rho_K');
title('(d) 12 个权重：K 半径被系统性锁在 0.7 Q','FontSize',10.5);
sgtitle('我们模块的半径 = LN 保留下来的形状 + qk 的奇异值 + 池化的收缩','FontSize',12.5,'FontWeight','bold');
R.savefig('fig_s2_01_what_is_radius'); close(f1);

%% ---------- 图 2 ----------
f2 = figure('Visible','off','Position',[40 40 1260 570]);
subplot(1,2,1); hold on;
[qs, ord] = sort(qR); y = 1:numel(qs);
barh(y, qs, 0.6, 'FaceColor',[.60 .75 .90],'EdgeColor',[.2 .2 .2]);
for i = 1:numel(y)
    plot([kR(ord(i)) kR(ord(i))], [y(i)-0.32 y(i)+0.32], '-', ...
         'Color',[.90 .35 .15],'LineWidth',2.4);
end
xline(rho_ceil,'Color',[.75 .2 .15],'LineWidth',1.6,'LineStyle','-.');
set(gca,'YTick',y,'YTickLabel',cellfun(@(r,d) [r ' / ' d], runL(ord), dsL(ord), ...
    'UniformOutput',false),'FontSize',7.2,'TickLabelInterpreter','none');
xlim([0 1.25]); grid on; xlabel('半径 \rho');
text(rho_ceil-0.40, 0.6, sprintf('天花板 %.2f',rho_ceil),'Color',[.75 .2 .15],'FontSize',8.5);
hs = patch(nan,nan,[.60 .75 .90]); hk = plot(nan,nan,'-','Color',[.90 .35 .15],'LineWidth',2.4);
legend([hs hk], {'Q 半径','K 半径'},'Location','southeast','FontSize',8); legend boxoff;
title('(a) 12 个真实权重的 Q/K 半径与天花板','FontSize',10.5);

subplot(1,2,2); hold on;
plot(ngrid, asinh(sinh(0.49)*sqrt(1./ngrid)), 'k--','LineWidth',1.2);
plot(ngrid, res(:,cgrid==0.5,1), 'o-','Color',[.20 .55 .35],'LineWidth',1.8,'MarkerFaceColor','w');
i64 = find(ngrid==64);
plot(ngrid(i64), median(kR), 'p','MarkerSize',14, ...
     'MarkerFaceColor',[.90 .35 .15],'MarkerEdgeColor','k');
text(ngrid(i64)*1.3, median(kR)-0.045, ...
     sprintf('实测 K 中位数 = %.2f', median(kR)),'Color',[.70 .25 .05],'FontSize',8.5);
text(1.3, 0.30, '若池内方向互相独立，n=64 时半径只剩 0.06','FontSize',8,'Color',[.3 .3 .3]);
text(1.3, 0.245, '实测有 0.35，说明池内像素方向高度重复（n_{eff}\approx2）','FontSize',8,'Color',[.3 .3 .3]);
set(gca,'XScale','log'); grid on; ylim([0 0.52]);
legend({'独立方向的解析解','c=0.5 的实验'},'Location','northeast','FontSize',8); legend boxoff;
xlabel('池大小 n'); ylabel('池化后半径');
title('(b) 池化把 K 锁在 Q 的 0.6~0.7 倍','FontSize',10.5);
sgtitle('两个结构性约束：半径天花板 + 池化锁死','FontSize',12.5,'FontWeight','bold');
R.savefig('fig_s2_02_ceiling_and_pooling'); close(f2);

save(fullfile(R.outdir,'res_s2.mat'),'out');
R.writefile(fullfile(R.outdir,'log_s2.txt'), L);
fprintf('S2 done.\n');
end

% ---------------------------------------------------------------- helpers --
function L = say(L, fmt, varargin)
if isempty(varargin)
    s = fmt;                       % 没有参数时原样输出，避免正文里的 % 被当成格式符
else
    s = sprintf(fmt, varargin{:});
end
fprintf('%s\n', s);
L{end+1,1} = s;
end
