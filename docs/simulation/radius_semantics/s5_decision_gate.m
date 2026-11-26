function s5_decision_gate()
%S5_DECISION_GATE  改码前的判定门：几何改动的"收益上限"到底有多大
%
%   三个论点都用我们自己的算子量化：
%   1) 翻转率 ↔ 键半径离散度 cv：全局 cv 越小，测地排序与欧氏排序越一致。
%      实测 radius_cv≈0.13~0.15 ⇒ 翻转率只有千分之几 ⇒ 任何"只改几何"的
%      改动，能改变的 top-1 决策比例也就这么大，即收益上限 = 翻转率。
%   2) 同半径重锚 = 单调变换：查询半径固定、所有键半径相同时，把键整体乘 s
%      不改变任何一个 argmax（数值核对），所以"径向重锚"只在跨尺度上起作用，
%      而跨尺度偏置模块里已经有 scale_bias(4 参数) 与 source_bias(-log count)。
%   3) 半径单独当判据的实测判别力（S3 的真实数据 AUC 0.51~0.60）与它的方差，
%      和"必须超过的噪声地板"（同协议两次重跑的差异，SYSU 检索层 +0.13pp、σ=0.39）
%      放在一张表里比。
%
%   产出：results/figures/fig_s5_01_gate.png、results/decision_table.md、log_s5.txt

R = rs_setup(); R.zh(); rng(31,'twister');
out = struct(); L = {};
L = say(L,'===== S5 改码判定门：几何改动的收益上限 =====');
t0 = tic;

%% ---------- 1) 翻转率 ↔ 键半径 cv ----------
d = 32; nKey = 224; nQ = 800; nMem = 64;
rq = 0.470; rk0 = 0.345; tau = 0.4696; alpha = 0.55; cMem = 0.5236;
dirK = unit(randn(nKey,d));
qIdx = randi(nKey,nQ,1);
zQ = sinh(rq)*unit(alpha*dirK(qIdx,:) + sqrt(1-alpha^2)*unit(randn(nQ,d)));
rQ = R.radius(zQ);
cvs = [0.02 0.05 0.10 0.15 0.25 0.35 0.50];
flip = zeros(size(cvs)); dlog = zeros(size(cvs)); sp = zeros(size(cvs));
w = ones(nMem,1)/nMem;
for ci = 1:numel(cvs)
    rks = rk0*(1 + cvs(ci)*randn(nKey,1)); rks = max(rks,0.05);
    Zk = zeros(nKey,d);
    for k = 1:nKey
        G = randn(nMem,d); G = G - (G*dirK(k,:)')*dirK(k,:); G = G./max(sqrt(sum(G.^2,2)),1e-12);
        U = unit(sqrt(cMem)*repmat(dirK(k,:),nMem,1) + sqrt(1-cMem)*G);
        Zk(k,:) = R.pool_amb(sinh(rks(k))*U, w);
    end
    rkE = R.radius(Zk);
    Dg = R.cdist_t(rQ, rkE', cosmat(zQ,Zk));
    De = sqrt(sqdist(zQ,Zk));
    Lg = -Dg.^2/(2*tau^2); Le = -De.^2/(2*tau^2);
    [~,ag] = max(Lg,[],2); [~,ae] = max(Le,[],2);
    flip(ci) = 1 - mean(ag == ae);
    dlog(ci) = mean(abs(Lg(:)-Le(:)));
    sp(ci) = R.spearman(Lg(:), Le(:));
end
L = say(L,'');
L = say(L,'[1] 键半径离散度 cv（全局，跨尺度混在一起）：%d 键 / %d 查询', nKey, nQ);
L = say(L,'     cv     测地vs欧氏 top-1 翻转率   平均|Δlogit|   Spearman(logit)');
for ci = 1:numel(cvs)
    L = say(L,'    %.2f        %.4f               %.4f        %.4f', ...
            cvs(ci), flip(ci), dlog(ci), sp(ci));
end
cvMeas = 0.145;
flipMeas = interp1(cvs, flip, cvMeas, 'linear');
L = say(L,'  实测 cv≈%.2f ⇒ 插值翻转率 ≈ %.4f（即 800 次里约 %.1f 次）', ...
        cvMeas, flipMeas, flipMeas*nQ);

%% ---------- 2) 同半径重锚是否改变排序 ----------
rkS = rk0*ones(nKey,1);
Zk = zeros(nKey,d);
for k = 1:nKey
    G = randn(nMem,d); G = G - (G*dirK(k,:)')*dirK(k,:); G = G./max(sqrt(sum(G.^2,2)),1e-12);
    U = unit(sqrt(cMem)*repmat(dirK(k,:),nMem,1) + sqrt(1-cMem)*G);
    Zk(k,:) = R.pool_amb(sinh(rkS(k))*U, w);
end
Cm = cosmat(zQ,Zk);
sameRad = max(max(abs(R.radius(Zk) - mean(R.radius(Zk)))));
sList = [1 1.5 2 2.5 3];
argSame = zeros(numel(sList),1); spSame = zeros(numel(sList),1);
[~,aref] = max(-R.cdist_t(rQ, R.radius(Zk)', Cm).^2,[],2);
Lref = -R.cdist_t(rQ, R.radius(Zk)', Cm).^2/(2*tau^2);
for i = 1:numel(sList)
    rkS2 = asinh(sList(i)*sinh(R.radius(Zk)'));
    Lg = -R.cdist_t(rQ, rkS2, Cm).^2/(2*tau^2);
    [~,am] = max(Lg,[],2);
    argSame(i) = mean(am == aref);
    spSame(i) = R.spearman(Lg(:), Lref(:));
end
L = say(L,'');
L = say(L,'[2] 同半径整体重锚（所有键半径相同，最大偏离 %.1e）', sameRad);
L = say(L,'       s     argmax 不变比例   与 s=1 logit 的 Spearman');
for i = 1:numel(sList)
    L = say(L,'     %.1f        %.4f            %.4f', sList(i), argSame(i), spSame(i));
end
L = say(L,'  同半径下重锚是单调变换 ⇒ argmax 基本不变；表中剩余 1~3‰ 的变化来自');
L = say(L,'  池化后半径并非严格相等（最大偏离 %.1e，相对 0.4%%）。', sameRad);

%% ---------- 3) 半径单独当判据 vs 噪声地板 ----------
ds  = {'LEVIR','SYSU','WHU','CDD'};
aucR = [0.598 0.601 0.541 0.508];      % S3 实测：半径 → 是否变化
ciLo = [0.530 0.546 0.448 0.452];
ciHi = [0.656 0.648 0.648 0.562];
noise = 0.13/100;                       % SYSU 检索层 +0.13pp，同一协议波动
L = say(L,'');
L = say(L,'[3] 半径单独判变化的实测 AUC（S3，真实影像）与"必须超过的噪声地板"');
for i = 1:numel(ds)
    L = say(L,'    %-6s AUC=%.3f [%.3f %.3f]  → 距 0.5 的增量 %+.3f', ...
            ds{i}, aucR(i), ciLo(i), ciHi(i), aucR(i)-0.5);
end
L = say(L,'  同协议两次重跑的检索层差异 +0.13pp、σ=0.39pp ⇒ 小于约 0.8pp（2σ）的改动不可区分。');
out.gate = struct('cvs',cvs,'flip',flip,'dlog',dlog,'cvMeas',cvMeas,'flipMeas',flipMeas, ...
                  'sList',sList,'argSame',argSame,'spSame',spSame, ...
                  'aucR',aucR,'ciLo',ciLo,'ciHi',ciHi);

%% ---------- 4) 决策表 ----------
rows = {
 'M1 键半径重锚（每尺度 1 标量，+4 参数）', '≤ 翻转率 ≈ 0.2%', '+4', ...
 'S4-C：s*≈1.95 才拉回角分辨，top-1 −0.001；S5-2：同半径重锚 argmax 不变', '不做';
 'OEM 池化替换 Minkowski 均值（0 参数）', '+0.1~0.4pp 半径保留', '0', ...
 'S4-A/B：五种算子 top-1 差 0.002~0.009，半径差 <1%', '不做';
 '去掉 Q/K 共享卷积（拆分 qk，+参数）', 'K 半径可独立增大 36%', '+32×D', ...
 'S6：半径增大只改尺度偏置；S5-1：翻转率仍 ~0.2%', '不做（除非跨尺度误配被证实）';
 '差分原型作 K/V 内容（module manifold，0 新增 KV）', '内容层面的判别力', '0', ...
 '旧 H2：内容 +22.84pp（p=7.5e-10），远大于噪声地板', '值得做（已有实现，需 seed 复现）';
 '逐像素支持度替代 cell 质量（0 参数）', '支持度已参与 logit', '0', ...
 '旧 H2：支持先验 +14.8/+20.5pp；但已被 support_term 吸收', '先测吸收指数';
 'τ / 曲率 / lift 常数化', '无', '−若干', ...
 '旧 H4：AUC 对 τ 严格不变，曲率只单调降', '可冻结/删除';
};
fid = fopen(fullfile(R.outdir,'decision_table.md'),'w','n','UTF-8');
fprintf(fid,'# 改码判定表（由 S1–S6 仿真给出）\n\n');
fprintf(fid,'| 候选改动 | 预期收益上限 | 参数成本 | 已有证据 | 结论 |\n');
fprintf(fid,'| --- | --- | --- | --- | --- |\n');
for i = 1:size(rows,1)
    fprintf(fid,'| %s | %s | %s | %s | %s |\n', rows{i,:});
end
fclose(fid);
L = say(L,'');
L = say(L,'[4] 决策表已写入 results/decision_table.md（6 行候选改动）');

%% ---------- 图 ----------
f = figure('Visible','off','Position',[30 40 1360 520]);
subplot(1,2,1); hold on;
plot(cvs, 100*flip, 'o-','Color',[.15 .40 .75],'LineWidth',1.6,'MarkerSize',5);
xline(cvMeas,'r--','LineWidth',1.1);
plot(cvMeas, 100*flipMeas,'p','MarkerSize',11,'MarkerFaceColor',[1 .85 .2],'MarkerEdgeColor','k');
xline(0.25,'k:','LineWidth',1.1);
xlabel('键半径离散度 cv'); ylabel('测地 vs 欧氏 top-1 翻转率（%）');
grid on; ylim([0 max(100*flip)*1.15+0.1]);
text(0.26, max(100*flip)*0.55, '机制菜单里的 25% 门限','FontSize',8,'Color',[.3 .3 .3]);
text(cvMeas+0.01, 100*flipMeas+max(100*flip)*0.08, ...
     sprintf('实测 cv≈%.2f → %.2f%%',cvMeas,100*flipMeas),'FontSize',8.5,'Color',[.6 .1 .1]);
title('(a) 半径离散度决定"几何能不能改变排序"','FontSize',10.5);

subplot(1,2,2); hold on;
b = bar(1:numel(ds), aucR, 0.55,'FaceColor',[.62 .76 .90],'EdgeColor','k');
errorbar(1:numel(ds), aucR, aucR-ciLo, ciHi-aucR, 'k.','LineWidth',1.1);
plot([0.4 numel(ds)+0.6],[0.5 0.5],'r--','LineWidth',1.1);
set(gca,'XTick',1:numel(ds),'XTickLabel',ds,'FontSize',9); grid on;
ylim([0.4 0.75]); ylabel('AUC（半径 → 是否变化）');
text(0.55,0.72,'红虚线 = 0.5（随机）','FontSize',8,'Color',[.6 .1 .1]);
text(0.55,0.685,'这些是"只靠半径"能达到的全部判别力','FontSize',8,'Color',[.25 .25 .25]);
title('(b) 半径单独当判据：真实数据上只比随机好一点','FontSize',10.5);
sgtitle('改码前的判定门：几何自由度很小，别把预算押在它上面','FontSize',12.5,'FontWeight','bold');
R.savefig('fig_s5_01_gate'); close(f);

out.runtime_s = toc(t0);
L = say(L,''); L = say(L,'总耗时 %.1f s', out.runtime_s);
save(fullfile(R.outdir,'res_s5.mat'),'out');
R.writefile(fullfile(R.outdir,'log_s5.txt'), L);
fprintf('S5 done.\n');
end

% ---------------------------------------------------------------- helpers --
function U = unit(X)
U = X ./ max(sqrt(sum(X.^2,2)), 1e-12);
end
function Cm = cosmat(Zq, Zk)
nq = max(sqrt(sum(Zq.^2,2)),1e-12); nk = max(sqrt(sum(Zk.^2,2)),1e-12);
Cm = max(min((Zq*Zk')./(nq*nk'),1),-1);
end
function D = sqdist(A, B)
D = max(sum(A.^2,2) + sum(B.^2,2)' - 2*(A*B'), 0);
end
function L = say(L, fmt, varargin)
if isempty(varargin), s = fmt; else, s = sprintf(fmt, varargin{:}); end
L{end+1,1} = s;
end
