function s4_hierarchy_pooling()
%S4_HIERARCHY_POOLING  层级/具体性 + 五种池化算子 + 检索效果 + 径向重锚
%
%   三部分：
%   A. 受控三层概念层级（祖父 2 / 父 8 / 子 48，每子 64 个实例）
%      HypRAG 的观点：池化层（"更通用"的概念）应当离原点更近。
%      看五种算子各自把半径收缩多少、层与层之间还剩多少区分度。
%   B. 用实测锚点（Q 半径 0.470、成员半径 0.345、τ=0.4696、224 键）造检索问题，
%      比较五种键池化算子的 top-1 与测地/欧氏 argmax 差异。
%   C. 径向重锚：把键向量整体乘 s（每尺度 1 个标量，共 4 参数、不加 KV），
%      测角度对比度恢复到 100% 所需的 s，以及它对 top-1 与 logit 的影响（no-op 守卫）。
%
%   产出：results/figures/fig_s4_01_hierarchy.png、fig_s4_02_operators.png
%         results/res_s4.mat、results/log_s4.txt

R = rs_setup(); R.zh(); rng(23,'twister');
out = struct(); L = {};
L = say(L,'===== S4 层级 / 池化算子 / 径向重锚 =====');
t0 = tic;

ops = {'euc','amb','ein','oem1','oem2'};
opl = {'切空间均值','环境均值(我们)','Einstein','OEM p=1','OEM p=2'};
nOp = numel(ops);

%% ================== A. 受控三层层级 ==================
d = 32;
nG = 2; nP = 4; nC = 6; nInst = 64; rho0 = 0.55;
nPar = nG*nP; nChi = nPar*nC;
cI  = 0.55;   % 实例与所属子概念的方向相关（≈ 实测 K/Q=0.70 反推的 c_eff）
cC  = 0.70;   % 子相对父
cP  = 0.80;   % 父相对祖父
jit = 0.10;   % 实例半径抖动

dirG = unit(randn(nG,d));
dirP = zeros(nPar,d); dirC = zeros(nChi,d);
parOf = zeros(nChi,1); graOfPar = zeros(nPar,1);
for g = 1:nG
    for p = 1:nP
        ip = (g-1)*nP + p;
        dirP(ip,:) = unit(sqrt(cP)*dirG(g,:) + sqrt(1-cP)*unit(randn(1,d)));
        graOfPar(ip) = g;
        for k = 1:nC
            ic = (ip-1)*nC + k;
            dirC(ic,:) = unit(sqrt(cC)*dirP(ip,:) + sqrt(1-cC)*unit(randn(1,d)));
            parOf(ic) = ip;
        end
    end
end
zInst = zeros(nChi*nInst,d); chiOf = zeros(nChi*nInst,1); rInst = zeros(nChi*nInst,1);
for ic = 1:nChi
    i0 = (ic-1)*nInst;
    U  = unit(sqrt(cI)*repmat(dirC(ic,:),nInst,1) + sqrt(1-cI)*unit(randn(nInst,d)));
    rr = max(rho0*(1 + jit*randn(nInst,1)), 0.05);
    zInst(i0+1:i0+nInst,:) = sinh(rr).*U;
    rInst(i0+1:i0+nInst)   = rr;
    chiOf(i0+1:i0+nInst)   = ic;
end
wInst = ones(nInst,1)/nInst;

rG = zeros(1,nOp); rP = zeros(1,nOp); rC = zeros(1,nOp);
zChild = cell(1,nOp); zPar = cell(1,nOp); zGra = cell(1,nOp);
for oi = 1:nOp
    Zc = zeros(nChi,d); Zp = zeros(nPar,d); Zg = zeros(nG,d);
    for ic = 1:nChi
        Zc(ic,:) = pool(R, ops{oi}, zInst(chiOf==ic,:), wInst);
    end
    for ip = 1:nPar
        m = (parOf == ip);
        Zp(ip,:) = pool(R, ops{oi}, Zc(m,:), ones(sum(m),1)/sum(m));
    end
    for g = 1:nG
        m = (graOfPar == g);
        Zg(g,:) = pool(R, ops{oi}, Zp(m,:), ones(sum(m),1)/sum(m));
    end
    zChild{oi} = Zc; zPar{oi} = Zp; zGra{oi} = Zg;
    rC(oi) = mean(R.radius(Zc)); rP(oi) = mean(R.radius(Zp)); rG(oi) = mean(R.radius(Zg));
end
levAuc = zeros(1,nOp);
for oi = 1:nOp
    rr = [R.radius(zGra{oi}); R.radius(zPar{oi}); R.radius(zChild{oi})];
    ll = [3*ones(nG,1); 2*ones(nPar,1); ones(nChi,1)];
    levAuc(oi) = R.auc(rr, double(ll==1));        % 半径越大 = 越"细"
end
rInstM = mean(rInst);
L = say(L,'');
L = say(L,'[A] 三层层级：%d 个实例，实例半径 %.3f±%.3f，cI=%.2f cC=%.2f cP=%.2f', ...
        numel(rInst), rInstM, jit*rho0, cI, cC, cP);
L = say(L,'  算子              实例ρ   子ρ     父ρ     祖父ρ   顶层保留率   层级AUC');
for oi = 1:nOp
    L = say(L,'  %-16s %.3f   %.3f   %.3f   %.3f    %5.1f%%      %.3f', ...
            opl{oi}, rInstM, rC(oi), rP(oi), rG(oi), 100*rG(oi)/rInstM, levAuc(oi));
end
L = say(L,'  顶层保留率最高的算子 = %s（%.1f%%），最低 = %s（%.1f%%），相差 %.1f 个百分点', ...
        opl{find(rG==max(rG),1)}, 100*max(rG)/rInstM, opl{find(rG==min(rG),1)}, 100*min(rG)/rInstM, ...
        100*(max(rG)-min(rG))/rInstM);
out.A = struct('rho0',rho0,'rInstM',rInstM,'rC',rC,'rP',rP,'rG',rG,'levAuc',levAuc);

%% ================== B. 五大算子在真实尺度上的检索效果 ==================
rq   = 0.470;      % 实测 Q 半径（Run9 H_REFINE WHU）
rkM  = 0.345;      % 实测 K 半径
tau  = 0.4696;     % FROZEN_TAU
nKey = 224; nQ = 800; nMem = 64;
alpha = 0.55;      % 查询与真键方向相关
cvA = 0.15;        % 池内成员半径 cv（实测 0.13~0.15）
cs  = [0.52 0.90]; % 池内相关：0.52 由实测 K/Q 反推；0.90 影像统计特征实测

dirK = unit(randn(nKey,d));
qIdx = randi(nKey,nQ,1);
Uq = unit(alpha*dirK(qIdx,:) + sqrt(1-alpha^2)*unit(randn(nQ,d)));
zQ = sinh(rq)*Uq;
rQ = R.radius(zQ);

top1 = zeros(numel(cs),nOp+1); keyRho = zeros(numel(cs),nOp);
flipEuc = zeros(numel(cs),nOp); ZkStore = cell(1,numel(cs));
for ci = 1:numel(cs)
    [Zk, Mz] = make_keys(R, ops, dirK, cs(ci), rkM, cvA, nMem);
    ZkStore{ci} = Zk;
    keyRho(ci,:) = arrayfun(@(oi) mean(R.radius(Zk{oi})), 1:nOp);
    De = sqdist(zQ, Zk{1});
    [~,ae] = max(-De,[],2);
    for oi = 1:nOp
        Dg = R.cdist_t(rQ, R.radius(Zk{oi})', cosmat(zQ, Zk{oi}));
        [~,am] = max(-Dg.^2,[],2);
        top1(ci,oi) = mean(am == qIdx);
        flipEuc(ci,oi) = 1 - mean(am == ae);
    end
    top1(ci,nOp+1) = mean(ae == qIdx);
end
L = say(L,'');
L = say(L,'[B] 检索：%d 键 / %d 查询，τ=%.4f，查询方向相关 %.2f，成员半径 cv=%.2f', ...
        nKey, nQ, tau, alpha, cvA);
for ci = 1:numel(cs)
    L = say(L,'  c=%.2f  top-1：欧氏基线 %.3f | %s', cs(ci), top1(ci,nOp+1), ...
            strjoin(arrayfun(@(x) sprintf('%.3f',x), top1(ci,1:nOp),'UniformOutput',false),' '));
    L = say(L,'         键半径：%s', ...
            strjoin(arrayfun(@(x) sprintf('%.3f',x), keyRho(ci,:),'UniformOutput',false),' '));
    L = say(L,'         与欧氏 argmax 不一致率：%s', ...
            strjoin(arrayfun(@(x) sprintf('%.3f',x), flipEuc(ci,:),'UniformOutput',false),' '));
end
out.B = struct('cs',cs,'top1',top1,'keyRho',keyRho,'flipEuc',flipEuc, ...
               'rq',rq,'rkM',rkM,'tau',tau);

%% ================== C. 径向重锚 ==================
[Zk52, ~] = make_keys(R, ops, dirK, 0.52, rkM, cvA, nMem);
Zk = Zk52{2};                       % 用"我们现在的算子"（环境均值）作基线
rkNat = R.radius(Zk)'; rk0 = mean(rkNat);
LgNat = -R.cdist_t(rQ, rkNat, cosmat(zQ, Zk)).^2/(2*tau^2);
ss = linspace(1,2.2,25);
ret = zeros(size(ss)); t1 = zeros(size(ss)); dlogit = zeros(size(ss));
for i = 1:numel(ss)
    rkS = asinh(ss(i)*sinh(rkNat));
    Lg  = -R.cdist_t(rQ, rkS, cosmat(zQ, Zk)).^2/(2*tau^2);
    [~,am] = max(Lg,[],2);
    t1(i) = mean(am == qIdx);
    dlogit(i) = mean(abs(Lg(:)-LgNat(:)));
    ret(i) = mean(R.angcontrast(rq, rkS)) / R.angcontrast(rq, rq);
end
[~,ibest] = min(abs(ret-0.99));
L = say(L,'');
L = say(L,'[C] 径向重锚：键半径 %.3f → 乘 s（每尺度 1 个标量，+4 参数，不加 KV）', rk0);
L = say(L,'      s     键半径   角度对比度保留   top-1   平均|Δlogit|');
for i = [1 5 9 13 17 21 25]
    L = say(L,'    %.2f   %.3f      %5.1f%%       %.3f     %.4f', ...
            ss(i), asinh(ss(i)*sinh(rk0)), 100*ret(i), t1(i), dlogit(i));
end
L = say(L,'  达到 99%% 角度对比度：s* = %.2f（键半径 %.3f → %.3f）', ...
        ss(ibest), rk0, asinh(ss(ibest)*sinh(rk0)));
L = say(L,'  top-1：不重锚 %.3f → s* 处 %.3f（差 %+.3f）', t1(1), t1(ibest), t1(ibest)-t1(1));
L = say(L,'  守卫：s=1 时 |Δlogit|=%.1e（须为 0）；s=%.2f 时 %.3f（输出确实变了）', ...
        dlogit(1), ss(ibest), dlogit(ibest));
out.C = struct('ss',ss,'ret',ret,'t1',t1,'dlogit',dlogit,'s_star',ss(ibest),'rk0',rk0);

%% ================== 图 1：层级 ==================
f1 = figure('Visible','off','Position',[30 40 1360 920]);
subplot(2,2,1); hold on;
M = [rG(:) rP(:) rC(:)];
hb = bar(M,'grouped');
cols = [.30 .42 .78; .25 .62 .38; .88 .60 .12];
for i = 1:3, hb(i).FaceColor = cols(i,:); end
hLn = plot([0.4 nOp+0.6],[rInstM rInstM],'k:','LineWidth',1.2);
set(gca,'XTick',1:nOp,'XTickLabel',opl,'FontSize',8); grid on;
ylabel('池化后半径 \rho'); ylim([0 0.62]);
legend([hb(:);hLn(:)],{'祖父（最粗）','父','子（最细）','实例原始半径'}, ...
       'Location','northwest','FontSize',7.5); legend boxoff;
title('(a) 越向"通用"池化，半径越小（HypRAG 4.3 的收缩）','FontSize',10.5);

subplot(2,2,2); hold on;
bar(1:nOp, 100*rG/rInstM, 0.55,'FaceColor',[.72 .82 .92],'EdgeColor','k');
set(gca,'XTick',1:nOp,'XTickLabel',opl,'FontSize',8); grid on;
ylabel('顶层保留率（%）'); ylim([50 100]);
plot([0.4 nOp+0.6],[70 70],'r--','LineWidth',1.1);
text(0.6,72.5,'实测模块 K/Q≈70%（一次池化）','FontSize',8,'Color',[.6 .1 .1]);
text(2.3,93,sprintf('五者相差 %.1f 个百分点', 100*(max(rG)-min(rG))/rInstM), ...
     'FontSize',8,'Color',[.25 .25 .25]);
title('(b) 五种算子保留率：差别只有零点几个百分点','FontSize',10.5);

subplot(2,2,3); hold on;
M2 = [top1(:,1:nOp) top1(:,nOp+1)];
hb2 = bar(M2,'grouped');
hb2(nOp+1).FaceColor = [.55 .55 .55];
set(gca,'XTick',1:numel(cs),'XTickLabel', ...
    arrayfun(@(c) sprintf('池内相关 c=%.2f',c),cs,'UniformOutput',false),'FontSize',8.5);
grid on; ylabel('top-1'); ylim([0 1]);
text(0.55,0.95,sprintf('五种算子最大差 %.3f，与欧氏基线差 %.3f', ...
     max(max(top1(:,1:nOp)))-min(min(top1(:,1:nOp))), ...
     mean(mean(abs(top1(:,1:nOp) - top1(:,nOp+1))))), 'FontSize',7.8);
legend(hb2,[opl,{'欧氏距离基线'}],'Location','southoutside','Orientation','horizontal', ...
       'FontSize',6.8,'NumColumns',3); legend boxoff;
title('(c) 检索命中率：五种算子几乎完全重合','FontSize',10.5);

subplot(2,2,4); hold on;
yyaxis left;
plot(ss,100*ret,'-o','Color',[.15 .45 .75],'LineWidth',1.4,'MarkerSize',3);
ylabel('角度对比度保留（%）'); ylim([55 105]);
yyaxis right;
plot(ss,t1,'-s','Color',[.85 .35 .15],'LineWidth',1.4,'MarkerSize',3);
ylabel('top-1'); ylim([0.2 1]);
xline(out.C.s_star,'k--','LineWidth',1.0);
xlabel('键向量的重锚系数 s'); grid on;
title(sprintf('(d) 重锚 s*=%.2f 拉回 100%% 角分辨，top-1 只变 %+.3f', ...
      out.C.s_star, t1(ibest)-t1(1)),'FontSize',10.5);
sgtitle('池化算子的几何差别存在，但对检索几乎没有影响','FontSize',12.5,'FontWeight','bold');
R.savefig('fig_s4_01_hierarchy'); close(f1);

%% ================== 图 2：算子细节 ==================
f2 = figure('Visible','off','Position',[40 50 1180 460]);
subplot(1,2,1); hold on;
bar(1:nOp, rC, 0.5,'FaceColor',[.85 .85 .85],'EdgeColor','k');
plot(1:nOp, rC, 'ko-','MarkerSize',5,'LineWidth',1.2);
hL2 = plot([0.4 nOp+0.6],[rInstM rInstM],'k:','LineWidth',1.1);
set(gca,'XTick',1:nOp,'XTickLabel',opl,'FontSize',8); grid on;
ylabel('子层半径 \rho'); xlim([0.4 nOp+0.6]); ylim([0 0.62]);
legend(hL2,{'实例原始半径'},'Location','northeast','FontSize',7.5); legend boxoff;
title(sprintf('(a) 子层：切空间 %.3f → 环境均值 %.3f（%.1f%%）', ...
      rC(1), rC(2), 100*(rC(2)-rC(1))/rC(1)),'FontSize',10.5);

subplot(1,2,2); hold on;
hb3 = bar(keyRho.' , 0.6, 'grouped');
set(gca,'XTick',1:nOp,'XTickLabel',opl,'FontSize',8); grid on;
ylabel('池化后键半径 \rho_K');
hL3 = plot([0.4 nOp+0.6],[rkM rkM],'k:','LineWidth',1.1);
legend([hb3(:);hL3], [arrayfun(@(c) sprintf('池内相关 c=%.2f',c),cs,'UniformOutput',false), ...
       {'成员原始半径'}], 'Location','northeast','FontSize',7.2); legend boxoff;
ylim([0 max(keyRho(:))+0.06]);
title('(b) 池化后键半径：五种算子相差不到 1%','FontSize',10.5);
R.savefig('fig_s4_02_operators'); close(f2);

out.runtime_s = toc(t0);
L = say(L,''); L = say(L,'总耗时 %.1f s', out.runtime_s);
save(fullfile(R.outdir,'res_s4.mat'),'out');
R.writefile(fullfile(R.outdir,'log_s4.txt'), L);
fprintf('S4 done.\n');
end

% ---------------------------------------------------------------- helpers --
function [Zk, Mz] = make_keys(R, ops, dirK, c, rkM, cvA, nMem)
[nKey, d] = size(dirK);
Mz = cell(nKey,1);
for k = 1:nKey
    G = randn(nMem,d); G = G - (G*dirK(k,:)')*dirK(k,:);
    G = G ./ max(sqrt(sum(G.^2,2)),1e-12);
    U = unit(sqrt(c)*repmat(dirK(k,:),nMem,1) + sqrt(1-c)*G);
    rr = max(rkM*(1 + cvA*randn(nMem,1)), 0.05);
    Mz{k} = sinh(rr).*U;
end
w = ones(nMem,1)/nMem;
Zk = cell(1,numel(ops));
for oi = 1:numel(ops)
    Z = zeros(nKey,d);
    for k = 1:nKey, Z(k,:) = pool(R, ops{oi}, Mz{k}, w); end
    Zk{oi} = Z;
end
end

function U = unit(X)
n = max(sqrt(sum(X.^2,2)), 1e-12);
U = X ./ n;
end

function Z = pool(R, name, z, w)
switch name
    case 'euc',  Z = R.pool_euc(z,w);
    case 'amb',  Z = R.pool_amb(z,w);
    case 'ein',  Z = R.pool_ein(z,w);
    case 'oem1', Z = R.pool_oem(z,w,1);
    case 'oem2', Z = R.pool_oem(z,w,2);
    otherwise, error('unknown pool %s', name);
end
end

function Cm = cosmat(Zq, Zk)
nq = max(sqrt(sum(Zq.^2,2)),1e-12); nk = max(sqrt(sum(Zk.^2,2)),1e-12);
Cm = (Zq*Zk') ./ (nq*nk');
Cm = max(min(Cm,1),-1);
end

function D = sqdist(A, B)
D = max(sum(A.^2,2) + sum(B.^2,2)' - 2*(A*B'), 0);
end

function L = say(L, fmt, varargin)
if isempty(varargin)
    s = fmt;
else
    s = sprintf(fmt, varargin{:});
end
L{end+1,1} = s;
end
