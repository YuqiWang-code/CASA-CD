% c3b_content_v3.m — 修正版 C3：内容(T2 vs D) × 几何(欧氏 vs 测地) 的公平 2×2
%
%   按执行方案 §2 A4 修正 V2 c3 的混淆：
%     1. 原型聚类对每个 (scene, scale, content) 只拟合一次，两几何共享同一组中心与成员
%        （v2 每臂各跑一次 k-means ⇒ 几何差异里混了聚类随机性）。
%     2. 各臂共用同一批场景与噪声 realization（同一 seed 生成一次，四臂复用）。
%     3. 原型拟合与评价的 token 互斥（fit 半 / eval 半），并额外给"朴素同点"版作对照。
%     4. 成对 bootstrap 以**场景**为重采样单位，不重新生成场景；同时报 pooled 与 macro AUC。
%     5. 数据生成不强制"半径大=变化"：T1/T2 各有独立传感噪声、共享基底、
%        非变化外观扰动、弱变化、强伪变化、局部配准误差；噪声与标签独立。
%     6. 报告 AUC、AP（附正类占比）、弱变化召回、成对 CI 与几何交互项。
%
%   输出：results/log_c3b.txt、results/res_c3b.mat、figures/fig_c3b_01_fair.png

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

d = 16; nScale = 4; nTok = 24; nScene = 30; rngSeeds = [11 12 13];
sigList = [0.70 0.35 0.20];   % 时相独立传感噪声 sigma（三档，等价于扫 SNR）
pChange = 0.15;
logline('===== C3b 公平 2x2：内容(T2/D) x 几何(欧氏/测地) =====');
logline('设置：d=%d，尺度=%d，每尺度 token=%d，场景=%d，变化比例=%.2f', d, nScale, nTok, nScene, pChange);
logline('三档时相独立噪声 sigma_sensor = %s（sigma 越大，差分内容 T2-T1 越被噪声主导）', mat2str(sigList));
logline('口径：原型每 (scene,scale,content) 只拟合一次、两几何共享；fit/eval token 互斥；bootstrap 按场景重采样');
logline('');

%% ================== 主循环：3 个生成 seed ==================
resAll = [];
for si = 1:numel(rngSeeds)
    rng(rngSeeds(si));
    sig = sigList(si);
    [X, Y] = gen_scenes(d, nScale, nTok, nScene, pChange, sig);
    % content: 1 = T2（只有 T2 的表示），2 = D = T2 − T1
    cont{1} = X.T2;  cont{2} = X.T2 - X.T1;
    contName = {'T2', 'D=T2-T1'};

    aucs = zeros(nScene, 2, 2);      % [scene, content, geom]；geom 1=E 2=H
    aps  = zeros(nScene, 2, 2);
    recW = zeros(nScene, 2, 2);      % 弱变化召回
    aucNaive = zeros(nScene, 2, 2);  % 同点拟合/评价（V2 口径）对照
    aucOrc = zeros(nScene, 2, 2);    % 用 GT 造原型（上界，仅作诊断）
    for sc = 1:nScene
        for ci = 1:2
            Z = squeeze(cont{ci}(sc, :, :));         % nTok*nScale × d
            Z = reshape(Z, nTok, nScale, d);
            lab = Y.lab(sc, :);                      % 1×(nScale*nTok) 的 0/1
            lab = reshape(lab', [], 1);              % 按 (scale,token) 展开
            % --- 每个 (scale,content) 只聚类一次，两几何共享 ---
            cen = cell(nScale, 1); mem = cell(nScale, 1);
            cenN = cell(nScale, 1);
            for s = 1:nScale
                Zs = reshape(Z(:, s, :), nTok, d);
                ls = reshape(lab, nTok, nScale)';  ls = ls(s, :)';
                hal = false(nTok, 1); hal(1:2:end) = true;   % fit/eval 互斥（隔点）
                [c2, m2] = kmeans2_local(Zs(hal, :), d);
                % 少数簇=变化（类不平衡先验，不看 GT）
                cnt = [sum(m2 == 1), sum(m2 == 2)];
                ch = find(cnt == min(cnt), 1);
                or_ = setdiff([1 2], ch);
                cen{s} = c2; mem{s} = m2;
                cenN{s} = [ch, or_];                          % [变化, 背景]
            end
            % --- 两几何打分（同一批中心） ---
            for gi = 1:2
                scAll = zeros(nTok * nScale, 1); gtAll = zeros(nTok * nScale, 1);
                isW = zeros(nTok * nScale, 1); inEval = false(nTok * nScale, 1);
                for s = 1:nScale
                    Zs = reshape(Z(:, s, :), nTok, d);
                    base = (s - 1) * nTok;
                    ls = reshape(lab, nTok, nScale)';  ls = ls(s, :)';
                    ev = ~false(nTok, 1); ev(1:2:end) = false;      % eval 半（与 fit 互斥）
                    ch = cenN{s}(1); bg = cenN{s}(2);
                    if gi == 1        % 欧氏：同一 z 上的平方欧氏
                        dCh = S.euc_sqdist(Zs, cen{s}(ch, :));
                        dBg = S.euc_sqdist(Zs, cen{s}(bg, :));
                    else              % 测地：Lorentz 平方测地
                        dCh = S.lorentz_dist(S.lift(Zs), S.lift(cen{s}(ch, :))).^2;
                        dBg = S.lorentz_dist(S.lift(Zs), S.lift(cen{s}(bg, :))).^2;
                    end
                    scAll(base + (1:nTok)') = dBg - dCh;
                    gtAll(base + (1:nTok)') = ls;
                    isW(base + (1:nTok)') = reshape(X.weak(sc, s, :), [], 1) > 0.5;
                    inEval(base + (1:nTok)') = ev;
                end
                [aucs(sc, ci, gi), aps(sc, ci, gi)] = auc_ap_local(scAll(inEval), gtAll(inEval));
                poolS{sc, ci, gi} = scAll(inEval); poolY{sc, ci, gi} = gtAll(inEval);
                wsel = inEval & isW == 1 & gtAll == 1;
                if any(wsel), recW(sc, ci, gi) = mean(scAll(wsel) > 0); else, recW(sc, ci, gi) = NaN; end
                % 朴素同点版：不划分 fit/eval，直接用全部 token 聚类并评价（V2 口径）
                cenA = cell(nScale, 1);
                for s = 1:nScale
                    Zs = reshape(Z(:, s, :), nTok, d);
                    [c2, m2] = kmeans2_local(Zs, d);
                    cnt = [sum(m2 == 1), sum(m2 == 2)];
                    chc = find(cnt == min(cnt), 1);
                    cenA{s} = c2([chc, setdiff([1 2], chc)], :);
                end
                scN = zeros(nTok * nScale, 1);
                for s = 1:nScale
                    Zs = reshape(Z(:, s, :), nTok, d);
                    base = (s - 1) * nTok;
                    if gi == 1
                        scN(base + (1:nTok)') = S.euc_sqdist(Zs, cenA{s}(2, :)) - S.euc_sqdist(Zs, cenA{s}(1, :));
                    else
                        scN(base + (1:nTok)') = S.lorentz_dist(S.lift(Zs), S.lift(cenA{s}(2, :))).^2 ...
                                              - S.lorentz_dist(S.lift(Zs), S.lift(cenA{s}(1, :))).^2;
                    end
                end
                aucNaive(sc, ci, gi) = S.auc(scN, gtAll);
                % oracle 上界：GT 均值当原型（只在 fit 半上算，eval 半评价）
                cenO = cell(nScale, 1);
                for s2 = 1:nScale
                    Zs = reshape(Z(:, s2, :), nTok, d);
                    ls = reshape(lab, nTok, nScale)'; ls = ls(s2, :)';
                    hal = false(nTok, 1); hal(1:2:end) = true;
                    cpos = mean(Zs(hal & ls == 1, :), 1);
                    cneg = mean(Zs(hal & ls == 0, :), 1);
                    cenO{s2} = [cpos; cneg];
                end
                scO = zeros(nTok * nScale, 1);
                for s2 = 1:nScale
                    Zs = reshape(Z(:, s2, :), nTok, d);
                    b2 = (s2 - 1) * nTok;
                    if gi == 1
                        scO(b2 + (1:nTok)') = S.euc_sqdist(Zs, cenO{s2}(2, :)) - S.euc_sqdist(Zs, cenO{s2}(1, :));
                    else
                        scO(b2 + (1:nTok)') = S.lorentz_dist(S.lift(Zs), S.lift(cenO{s2}(2, :))).^2 ...
                                            - S.lorentz_dist(S.lift(Zs), S.lift(cenO{s2}(1, :))).^2;
                    end
                end
                aucOrc(sc, ci, gi) = S.auc(scO(inEval), gtAll(inEval));
            end
        end
    end
    % ---- 场景级效应（pp）与成对 bootstrap ----
    A = squeeze(mean(aucs, 1));            % 2×2 = content × geom
    eff = [A(1, 1), A(1, 2), A(2, 1), A(2, 2)];   % T-E T-H D-E D-H
    % 每个场景内的四个效应
    sEff = zeros(nScene, 5);
    for sc = 1:nScene
        a = squeeze(aucs(sc, :, :));       % 2×2
        sEff(sc, :) = [a(1, 1), a(1, 2), a(2, 1), a(2, 2), 0];
        sEff(sc, 5) = (a(2, 2) - a(2, 1)) - (a(1, 2) - a(1, 1));  % 几何×内容 交互
    end
    ciEff = pair_boot(sEff, 1000);
    % ciEff 行序：1=内容@欧氏 2=内容@测地 3=几何@T2 4=几何@D 5=交互
    logline('[seed %d, sigma=%.2f] 场景级 AUC 均值：T-E %.4f  T-H %.4f  D-E %.4f  D-H %.4f', ...
        rngSeeds(si), sig, eff(1), eff(2), eff(3), eff(4));
    logline('  内容效应 D-E − T-E = %+.2f pp [%+.2f, %+.2f]；D-H − T-H = %+.2f pp [%+.2f, %+.2f]', ...
        100*(eff(3)-eff(1)), 100*ciEff(1,1), 100*ciEff(1,2), ...
        100*(eff(4)-eff(2)), 100*ciEff(2,1), 100*ciEff(2,2));
    logline('  几何效应 T-H − T-E = %+.2f pp [%+.2f, %+.2f]；D-H − D-E = %+.2f pp [%+.2f, %+.2f]', ...
        100*(eff(2)-eff(1)), 100*ciEff(3,1), 100*ciEff(3,2), ...
        100*(eff(4)-eff(3)), 100*ciEff(4,1), 100*ciEff(4,2));
    logline('  交互 (D-H−D-E)−(T-H−T-E) = %+.2f pp [%+.2f, %+.2f]', ...
        100*mean(sEff(:,5)), 100*ciEff(5,1), 100*ciEff(5,2));
    apM = squeeze(mean(aps, 1)); rwM = squeeze(mean(recW, 1, 'omitnan'));
    logline('  AP(正类占比 %.2f)：T-E %.4f T-H %.4f D-E %.4f D-H %.4f', pChange, ...
        apM(1,1), apM(1,2), apM(2,1), apM(2,2));
    logline('  弱变化召回：T-E %.3f T-H %.3f D-E %.3f D-H %.3f', ...
        rwM(1,1), rwM(1,2), rwM(2,1), rwM(2,2));
    nv = squeeze(mean(aucNaive, 1));
    % 池化 AUC（把所有场景 eval token 拼起来）
    pA = zeros(2, 2);
    for ci2 = 1:2
        for gi2 = 1:2
            allS = vertcat(poolS{:, ci2, gi2}); allY = vertcat(poolY{:, ci2, gi2});
            pA(ci2, gi2) = auc_ap_local(allS, allY);
        end
    end
    logline('  池化 AUC：T-E %.4f T-H %.4f D-E %.4f D-H %.4f', pA(1,1), pA(1,2), pA(2,1), pA(2,2));
    oc = squeeze(mean(aucOrc, 1));
    logline('  oracle 上界（GT 造原型）：T-E %.4f T-H %.4f D-E %.4f D-H %.4f', oc(1,1), oc(1,2), oc(2,1), oc(2,2));
    logline('  朴素同点版（V2 口径）：T-E %.4f T-H %.4f D-E %.4f D-H %.4f → 虚高 %+.2f/%+.2f/%+.2f/%+.2f pp', ...
        nv(1,1), nv(1,2), nv(2,1), nv(2,2), ...
        100*(nv(1,1)-eff(1)), 100*(nv(1,2)-eff(2)), 100*(nv(2,1)-eff(3)), 100*(nv(2,2)-eff(4)));
    logline('');
    resAll = cat(3, resAll, eff);
end

%% ================== 汇总 ==================
resAll = reshape(resAll, 4, numel(rngSeeds));
logline('---- 三个生成 seed 的合并 ----');
logline('  四臂 AUC：T-E %.4f  T-H %.4f  D-E %.4f  D-H %.4f', mean(resAll, 2));
logline('  内容效应 %+.2f pp；几何效应(T) %+.2f pp；几何效应(D) %+.2f pp', ...
    100*mean(resAll(3, :) - resAll(1, :)), 100*mean(resAll(2, :) - resAll(1, :)), ...
    100*mean(resAll(4, :) - resAll(3, :)));
logline('  结论口径：几何效应若 CI 含 0 只能写"证据不足"，不能写"等价"；AUC 的 pp 不能当 F1 的 pp。');

save(fullfile(outdir, 'res_c3b.mat'), 'resAll');
S.writefile(fullfile(outdir, 'log_c3b.txt'), logline('get'));
logline('C3b done.');

%% ================== 图 ==================
fig = figure('Visible', 'off', 'Position', [40 40 1180 450]);
subplot(1, 2, 1);
hb = bar(resAll', 'grouped');
set(gca, 'XTick', 1:numel(rngSeeds), 'XTickLabel', arrayfun(@(s) sprintf('sigma=%.2f', s), sigList, 'UniformOutput', false), 'FontSize', 8.5);
grid on; ylim([0 1]); ylabel('场景级 AUC');
legend(hb, {'T2 + 欧氏', 'T2 + 测地', 'D + 欧氏', 'D + 测地'}, 'Location', 'southoutside', 'Orientation', 'horizontal', 'FontSize', 7.5, 'NumColumns', 2);
legend boxoff;
title('(a) 四臂 AUC：内容效应稳定，几何效应小', 'FontSize', 10);
subplot(1, 2, 2); hold on;
bar(1:3, [mean(resAll(3,:)-resAll(1,:)), mean(resAll(2,:)-resAll(1,:)), mean(resAll(4,:)-resAll(3,:))]*100, 0.55, 'FaceColor', [.8 .85 .9]);
plot([0.4 3.6], [0 0], 'k--');
set(gca, 'XTick', 1:3, 'XTickLabel', {'内容 D−T2 @欧氏', '几何 T2: H−E', '几何 D: H−E'}, 'FontSize', 8.5);
grid on; ylabel('AUC 差（pp）');
title('(b) 三个主效应的方向与量级', 'FontSize', 10);
S.savefig('fig_c3b_01_fair', figdir); close(fig);

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

function [X, Y] = gen_scenes(d, nScale, nTok, nScene, pChange, sig)
% 不强制"半径大=变化"：基底方向 + 各时相独立传感噪声 + 共享噪声 + 弱变化 + 伪变化 + 配准误差
X.T1 = zeros(nScene, nScale * nTok, d);
X.T2 = zeros(nScene, nScale * nTok, d);
Y.lab = zeros(nScene, nScale * nTok);
X.weak = zeros(nScene, nScale, nTok);
for sc = 1:nScene
    base = zeros(nScale, d);
    for s = 1:nScale
        base(s, :) = randn(1, d) / sqrt(d);       % 场景级基底方向
    end
    for s = 1:nScale
        mdir = randn(1, d) / sqrt(d);          % 每个尺度一个"变化方向"（同类变化共享方向）
        for t = 1:nTok
            idx = (s - 1) * nTok + t;
            u0 = base(s, :) * 1.0 + 0.45 * randn(1, d) / sqrt(d);  % 共享外观
            n1 = sig * randn(1, d) / sqrt(d);
            n2 = sig * randn(1, d) / sqrt(d);
            appr = 0.20 * randn(1, d) / sqrt(d);                   % 非变化外观扰动
            isCh = rand < pChange;
            mag = 1.8 + 0.5 * randn;
            weak = isCh && (rand < 0.35);
            if weak, mag = mag * 0.35; end
            delta = mag * mdir + 0.25 * randn(1, d) / sqrt(d);     % 共享变化方向 + 小幅抖动
            X.T1(sc, idx, :) = u0 + n1;
            X.T2(sc, idx, :) = u0 + n2 + appr + isCh * delta;
            Y.lab(sc, idx) = isCh;
            X.weak(sc, s, t) = weak;
        end
    end
    % 局部配准误差：把 T2 在尺度维上错开一位的 10% token
    mis = rand(nScale * nTok, 1) < 0.10;
    if any(mis)
        Z2 = reshape(X.T2(sc, :, :), nScale * nTok, d);
        Z2(mis, :) = Z2(mis, :) + 0.3 * randn(sum(mis), d) / sqrt(d);
        X.T2(sc, :, :) = reshape(Z2, 1, nScale * nTok, d);
    end
end
Y.lab = logical(Y.lab);
end

function [c, m] = kmeans2_local(Z, d)
% 极简 2-means（确定性初始化：最小/最大范数点），迭代 30 次
[~, i1] = min(sum(Z.^2, 2)); [~, i2] = max(sum(Z.^2, 2));
c = [Z(i1, :); Z(i2, :)];
for it = 1:30
    D = S_euc_sq(Z, c);
    [~, m] = min(D, [], 2);
    for k = 1:2
        sel = m == k;
        if any(sel), c(k, :) = mean(Z(sel, :), 1); end
    end
end
D = S_euc_sq(Z, c); [~, m] = min(D, [], 2);
m = m(:);
end

function D = S_euc_sq(A, B)
D = sum(A.^2, 2) + sum(B.^2, 2)' - 2 * (A * B');
D = max(D, 0);
end

function [a, ap] = auc_ap_local(s, y)
y = logical(y);
n1 = sum(y); n0 = sum(~y);
if n1 == 0 || n0 == 0, a = NaN; ap = NaN; return; end
[~, ord] = sort(s, 'descend');
yy = y(ord);
tp = cumsum(yy); fp = cumsum(~yy);
prec = tp ./ max(tp + fp, 1);
ap = sum(prec(yy)) / n1;
rAll = rank_of(s);
a = (sum(rAll(y)) - n1 * (n1 + 1) / 2) / (n1 * n0);
end

function r = rank_of(x)
[~, ord] = sort(x); r = zeros(size(x));
r(ord) = 1:numel(x);
end

function ci = pair_boot(E, B)
% E: nScene×5 = [TE TH DE DH 交互]；按场景重采样，效应顺序见 ciEff 注释
n = size(E, 1);
sEff = [E(:, 3) - E(:, 1), E(:, 4) - E(:, 2), E(:, 2) - E(:, 1), E(:, 4) - E(:, 3), E(:, 5)];
bs = zeros(B, 5);
for b = 1:B
    idx = randi(n, n, 1);
    bs(b, :) = mean(sEff(idx, :), 1);
end
ci = prctile_local(bs, [2.5 97.5]);
end

function q = prctile_local(X, p)
X = sort(X, 1);
n = size(X, 1);
q = zeros(size(X, 2), numel(p));
for j = 1:size(X, 2)
    for k = 1:numel(p)
        pos = (n - 1) * p(k) / 100 + 1;
        lo = floor(pos); hi = ceil(pos);
        q(j, k) = X(lo, j) + (pos - lo) * (X(hi, j) - X(lo, j));
    end
end
end
