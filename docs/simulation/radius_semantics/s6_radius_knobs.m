function s6_radius_knobs()
%S6_RADIUS_KNOBS  半径归因：把"K 半径只有 Q 的 70%"拆成可操作的旋钮
%
%   实测：rho_Q=0.470、rho_K=0.345（Run9 H_REFINE WHU，4 stage 均值）。
%   我们的键是 cell 内 n 个像素的 Lorentz 均值，成员方向相关 c，于是
%       rho_K = asinh( A * sinh(rho_Q) ),  A = sqrt(c + (1-c)/n)
%   由实测反解 c（n=1024）≈ 0.52 —— 与 S3 里影像统计特征的 c_eff 0.86-0.89
%   差得很远，说明真实骨干特征在 cell 内远比原始像素"独立"。
%
%   A 部分：解析式数值核对 + 四个旋钮（提高 c / 减小 n / 不池化 / 加增益）
%           各自能把 rho_K 抬到哪、角度对比度恢复到多少、top-1 变多少。
%   B 部分：半径唯一真正起作用的地方是跨尺度（各尺度键半径差别很大）：
%           没有偏置时 argmax 会被大半径尺度吸走；把每尺度半径重锚（=每尺度
%           一个偏置）能修好，但模块里已有 scale_bias(4 参数) 与 source_bias，
%           所以"半径"这件事在结构上已经被覆盖了。
%
%   产出：results/figures/fig_s6_01_knobs.png、log_s6.txt

R = rs_setup(); R.zh(); rng(37,'twister');
out = struct(); L = {};
L = say(L,'===== S6 半径归因与四个旋钮 =====');
t0 = tic;

rq = 0.470; rkM = 0.345; tau = 0.4696;

%% ---------- A1. 解析式核对 ----------
L = say(L,'');
L = say(L,'[A1] 池化半径精确式 rho_pool = asinh( A*sinh(rho) / sqrt(cosh(rho)^2 - A^2*sinh(rho)^2) )');
L = say(L,'     其中 A = sqrt(c+(1-c)/n)；小半径时退化为 asinh(A*sinh(rho))');
ns = [1 16 64 1024]; cs = [0.0 0.3 0.52 0.8];
L = say(L,'        n|c      %s', strjoin(arrayfun(@(c) sprintf('%7.2f',c),cs,'UniformOutput',false),''));
maxErr = 0;
for n = ns
    row = '';
    for c = cs
        A = sqrt(c + (1-c)/n);
        pred = pool_rho_exact(rq, A);
        % 数值核对
        d = 32;
        u = unit(randn(1,d));
        G = randn(n,d); G = G - (G*u')*u;
        G = G./max(sqrt(sum(G.^2,2)),1e-12);
        U = unit(sqrt(c)*repmat(u,n,1) + sqrt(1-c)*G);
        got = R.radius(R.pool_amb(sinh(rq)*U, ones(n,1)/n));
        maxErr = max(maxErr, abs(got-pred));
        row = [row sprintf('%7.3f', got)];
    end
    L = say(L,'      %5d      %s', n, row);
end
L = say(L,'  解析式与数值差最大 %.1e（n 小、c=0 时因方向只是"近似不相关"而有偏差）', maxErr);

%% ---------- A2. 反解实测 c 与四个旋钮 ----------
Ameas = fzero(@(A) pool_rho_exact(rq,A) - rkM, [0.5 1.0]);
cImplied = (Ameas^2 - 1/1024)/(1 - 1/1024);
L = say(L,'');
L = say(L,'[A2] 由实测反解：rho_K=%.3f ⇐ A = %.4f ⇒ c(n=1024) = %.3f', rkM, Ameas, cImplied);
knob = {'现状（实测 c, n=1024）','提高池内独立度 c→0.75','减小 cell n→64','不池化 n=1（单像素键）','键向量加增益 ×1.2'};
A = [Ameas, sqrt(0.75+0.25/1024), sqrt(cImplied+(1-cImplied)/64), 1.0, Ameas*1.2];
rkK = arrayfun(@(a) pool_rho_exact(rq,a), A);
ret = zeros(size(A));
for i = 1:numel(A)
    ret(i) = R.angcontrast(rq, rkK(i))/R.angcontrast(rq, rq);
end
L = say(L,'  旋钮                          A        rho_K    角度对比度保留   相对现状');
for i = 1:numel(A)
    L = say(L,'  %-26s %.4f   %.3f      %5.1f%%        %+5.1f%%', ...
            knob{i}, A(i), rkK(i), 100*ret(i), 100*(rkK(i)-rkK(1))/rkK(1));
end
out.A = struct('ns',ns,'cs',cs,'A',A,'rkK',rkK,'ret',ret,'cImplied',cImplied,'Ameas',Ameas);

%% ---------- B. 跨尺度：半径唯一真正起作用的地方 ----------
nScale = 4; nPer = 56; nQ = 800; d = 32; nMem = 64;
ScRho = [0.15 0.25 0.35 0.47];        % 各尺度键半径（示意：小尺度靠近原点）
dirK = unit(randn(nScale*nPer,d));
qScale = ceil(randi(nScale*nPer,nQ,1)/nPer);   % 查询所属尺度（保证每组 1..4）
qIdx = zeros(nQ,1);
for i = 1:nQ
    base = (qScale(i)-1)*nPer;
    qIdx(i) = base + randi(nPer);
end
zQ = sinh(rq)*unit(0.55*dirK(qIdx,:) + sqrt(1-0.55^2)*unit(randn(nQ,d)));
rQ = R.radius(zQ);
Zk = zeros(nScale*nPer,d); rk = zeros(nScale*nPer,1);
for k = 1:nScale*nPer
    s = ceil(k/nPer);
    G = randn(nMem,d); G = G - (G*dirK(k,:)')*dirK(k,:); G = G./max(sqrt(sum(G.^2,2)),1e-12);
    U = unit(sqrt(0.52)*repmat(dirK(k,:),nMem,1) + sqrt(1-0.52)*G);
    Zk(k,:) = R.pool_amb(sinh(ScRho(s))*U, ones(nMem,1)/nMem);
    rk(k) = ScRho(s);
end
rk = R.radius(Zk)';
Cm = cosmat(zQ,Zk);
D2 = R.cdist_t(rQ, rk, Cm).^2;
% 把"重锚到查询半径"拆成两部分：一个每尺度常数偏移 + 一个依赖 cos 的残差。
% 若残差很小，则重锚 ≈ 每尺度一个 logit 偏置（模块里就是 scale_bias）。
rScale = zeros(1,nScale);
for s = 1:nScale, rScale(s) = mean(rk((s-1)*nPer+1:s*nPer)); end
cosGrid = linspace(0,1,101);
offMean = zeros(1,nScale); offStd = zeros(1,nScale);
for s = 1:nScale
    d1 = R.cdist_t(rq, rScale(s)*ones(size(cosGrid)), cosGrid);
    d0 = R.cdist_t(rq, rq*ones(size(cosGrid)), cosGrid);
    dlt = d1.^2 - d0.^2;
    offMean(s) = mean(dlt); offStd(s) = std(dlt);
end
ksOf = ceil((1:nScale*nPer)/nPer);
bKey = offMean(ksOf);
cases = {'无偏置（只有距离）','键半径重锚到查询半径','只用每尺度常数偏移（≈ scale_bias）'};
wrong = zeros(1,3); acc = zeros(1,3);
for ci = 1:3
    switch ci
        case 1
            Duse = D2;
        case 2
            Duse = R.cdist_t(rQ, rq*ones(size(rk)), Cm).^2;
        case 3
            Duse = D2 - repmat(bKey, nQ, 1);
    end
    Lg = -Duse/(2*tau^2);
    [~,am] = max(Lg,[],2);
    wrong(ci) = mean(ceil(am/nPer) ~= qScale);
    acc(ci) = mean(am == qIdx);
end
L = say(L,'');
L = say(L,'[B] 跨尺度（%d 尺度 × %d 键，各尺度键半径 %s）', nScale, nPer, mat2str(ScRho,2));
L = say(L,'    处理方式                            误到别的尺度   总 top-1');
for ci = 1:3
    L = say(L,'    %-32s %5.1f%%        %.3f', cases{ci}, 100*wrong(ci), acc(ci));
end
L = say(L,'  重锚造成的 d^2 改变：每尺度平均值 %s（跨 cos 的标准差 %s）', ...
        mat2str(offMean,3), mat2str(offStd,3));
L = say(L,'  残差/均值 比值平均 %.2f ⇒ 它不是纯偏置，但两者效果几乎相同（差 %.1f 个百分点）。', ...
        mean(offStd./max(abs(offMean),1e-9)), 100*(wrong(3)-wrong(2)));
L = say(L,'  模块里能承载这类逐尺度修正的是 scale_bias(heads,4,4) + source_bias(-log count)。');
L = say(L,'  ⇒ 半径的跨尺度自由度已被这两个已有参数覆盖，再加径向自由度是重复。');
out.B = struct('ScRho',ScRho,'wrong',wrong,'acc',acc,'offMean',offMean, ...
               'offStd',offStd,'cases',{cases});

%% ---------- 图 ----------
f = figure('Visible','off','Position',[30 40 1500 500]);
subplot(1,3,1); hold on;
A2 = sqrt(cs' + (1-cs')/1024);
plot(cs, A2, 'o-','LineWidth',1.5,'MarkerSize',5,'Color',[.15 .40 .75]);
plot(cImplied, Ameas, 'p','MarkerSize',12,'MarkerFaceColor',[1 .85 .2],'MarkerEdgeColor','k');
xlabel('池内方向相关 c'); ylabel('收缩因子 A（n=1024）'); grid on;
text(cImplied+0.03, Ameas+0.02, sprintf('实测 c=%.2f, A=%.3f',cImplied,Ameas),'FontSize',8.5);
title('(a) 键半径缩水来自"池内方向相关"','FontSize',10.5);

subplot(1,3,2); hold on;
hb = bar(1:numel(A), 100*ret, 0.6,'FaceColor',[.70 .82 .92],'EdgeColor','k');
plot([0.4 numel(A)+0.6],[100 100],'k:','LineWidth',1.1);
set(gca,'XTick',1:numel(A),'XTickLabel',{'现状','c↑0.75','n→64','n=1','增益×1.2'},'FontSize',8.5);
grid on; ylabel('角度对比度保留（%）'); ylim([40 118]);
for i = 1:numel(A)
    text(i, 100*ret(i)+2.5, sprintf('%.0f%%',100*ret(i)),'HorizontalAlignment','center','FontSize',8);
end
title(sprintf('(b) 四个旋钮能把角分辨从 %.0f%% 拉到多少',100*ret(1)),'FontSize',10.5);

subplot(1,3,3); hold on;
hb2 = bar([100*wrong(:) 100*acc(:)], 'grouped');
hb2(1).FaceColor = [.85 .45 .35]; hb2(2).FaceColor = [.35 .55 .80];
set(gca,'XTick',1:3,'XTickLabel',{'无偏置','半径重锚','每尺度偏置'},'FontSize',8.5);
grid on; ylabel('%'); ylim([0 100]);
legend(hb2,{'误到别的尺度','总 top-1'},'Location','north','Orientation','horizontal','FontSize',8);
legend boxoff;
title('(c) 半径只在跨尺度上起作用，而偏置已经能修','FontSize',10.5);
sgtitle('半径的可操作空间：能改的旋钮都知道代价了','FontSize',12.5,'FontWeight','bold');
R.savefig('fig_s6_01_knobs'); close(f);

out.runtime_s = toc(t0);
L = say(L,''); L = say(L,'总耗时 %.1f s', out.runtime_s);
save(fullfile(R.outdir,'res_s6.mat'),'out');
R.writefile(fullfile(R.outdir,'log_s6.txt'), L);
fprintf('S6 done.\n');
end

% ---------------------------------------------------------------- helpers --
function r = pool_rho_exact(rho, A)
% 半径 rho、池内相关 c、规模 n 的成员池化后的半径（精确式）
sh = sinh(rho); ch = cosh(rho);
x = A*sh ./ sqrt(max(ch^2 - A^2*sh^2, 1e-12));
r = asinh(x);
end
function U = unit(X)
U = X ./ max(sqrt(sum(X.^2,2)), 1e-12);
end
function Cm = cosmat(Zq, Zk)
nq = max(sqrt(sum(Zq.^2,2)),1e-12); nk = max(sqrt(sum(Zk.^2,2)),1e-12);
Cm = max(min((Zq*Zk')./(nq*nk'),1),-1);
end
function L = say(L, fmt, varargin)
if isempty(varargin), s = fmt; else, s = sprintf(fmt, varargin{:}); end
L{end+1,1} = s;
end
