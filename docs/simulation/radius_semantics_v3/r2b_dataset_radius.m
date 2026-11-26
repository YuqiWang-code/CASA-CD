% r2b_dataset_radius.m — 修正版 R2（执行方案 §2 A6）
%
%   v2 r2 的四个问题，本脚本逐个修：
%     1. 池化顺序错：v2 先 Lift 未缩放 Z 再把聚合结果除 sqrt(D)；真实 K 是
%        z = a/sqrt(d) 先缩放、再 Lift、再池化 → 统一走 pool_radius_k(Z,w,sqD)。
%     2. CV 口径错：v2 把所有成员半径拼起来算 CV。这里分开
%        CV_Kmember_within_cell（每个 cell 内）/ CV_Kmember_global（整图成员）/ CV_Kproto（池化后）。
%     3. 统计口径错：v2 用 `if ~any(Y(:)), continue` 把 AUC 限制在正样本图；
%        这里只对"同时含前景与背景"的图算 AUC，并记录有效/混合/缺失图数。
%     4. 命名错：v2 的 spStr 只用了背景像素 → 改名 Spearman_bg，另报全图与变化区。
%   另按 A6：三类权重（硬分簇 / 软能量 / 外部 p 双类）都过同一 pool_radius_k；
%   只用 val split、抽样清单先固定并落盘、标签只在算指标时读。
%
%   输出：results/log_r2b.txt、results/res_r2b.mat、results/r2b_sample_list.txt
%         figures/fig_r2b_01_dataset.png、figures/fig_r2b_02_maps.png

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

C = 64; D = 32; nmax = 40;
rng(20261126);
W = 2 * (rand(D, C) - 0.5) / 8;      % 复刻 Conv2d 默认 U(-1/8, 1/8)
gamma = 1 + 0.35 * randn(1, C);
beta = 0.20 * randn(1, C);
sqD = sqrt(D);
cellSizes = [8 16 32 64];

rootD = 'F:/一些数据文件/变化检测CD数据集';
DS = struct('name', {'LEVIR', 'SYSU', 'WHU', 'CDD'}, ...
            'dir', {'LEVIR-CD-256', 'SYSU-CD-256', 'WHU-CD-256', 'CDD-CD-256'}, ...
            'ext', {'.png', '.png', '.png', '.jpg'}, ...
            'prefix', {'', '', '', 'test_'}, ...
            'numfmt', {false, false, false, true});

logline('===== R2b 修正版 S3：真实影像上的半径统计（val 口径、池化顺序已修） =====');
logline('设置：C=%d，D=%d，每库 <=%d 对（val split），cell 尺寸 %s', C, D, nmax, mat2str(cellSizes));
logline('口径：只用 list/val.txt；抽样清单先固定并落盘；池化用 pool_radius_k（先缩放后 Lift）；');
logline('      CV 分池内/全局/原型；AUC 只对同时含前景与背景的图有效；标签只在算指标时读。');
logline('');

acc = struct([]);
sampleLines = {};
for di = 1:numel(DS)
    d = DS(di);
    root = fullfile(rootD, d.dir);
    lstF = fullfile(root, 'list', 'val.txt');
    if ~exist(lstF, 'file')
        logline('[%s] 缺少 val.txt，跳过', d.name); continue;
    end
    nm = strsplit(strtrim(fileread(lstF)), sprintf('\n'));
    nm = nm(~cellfun(@isempty, nm));
    files = pick_files(root, nm, d);
    if isempty(files), logline('[%s] 未找到可用文件，跳过', d.name); continue; end
    idx = unique(round(linspace(1, numel(files), min(nmax, numel(files)))));
    files = files(idx);
    for i = 1:numel(files)
        [~, stem] = fileparts(files{i}.A);
        sampleLines{end + 1} = sprintf('%s %s', d.name, stem); %#ok<SAGROW>
    end

    nValid = 0; nMix = 0; nMiss = 0;
    rhoBg = []; rhoFg = []; aucV = []; spBg = []; spAll = []; spCh = [];
    cvCellAll = []; cvGlobAll = []; cvProtoAll = [];
    ret = zeros(3, numel(cellSizes));   % 行：硬分簇/软能量/外部 p
    maps = {};
    for fi = 1:numel(files)
        try
            A = imread(files{fi}.A); B = imread(files{fi}.B); L = imread(files{fi}.L);
        catch
            nMiss = nMiss + 1; continue;
        end
        if size(L, 3) > 1, L = rgb2gray(L); end
        L = L >= 128;
        if ~isequal(size(L), size(A, 1), size(A, 2))
            L = imresize(L, [size(A, 1), size(A, 2)], 'nearest') >= 0.5;
        end
        nValid = nValid + 1;
        [rA, ZA] = proj_z(build_features(double(A) / 255, C), W, gamma, beta, sqD);
        [rB, ZB] = proj_z(build_features(double(B) / 255, C), W, gamma, beta, sqD);
        rhoBg = [rhoBg; rA(~L)]; rhoFg = [rhoFg; rB(L)]; %#ok<AGROW>
        if any(L(:)) && any(~L(:))
            nMix = nMix + 1;
            s = [rB(:); rA(:)]; y = [L(:); L(:)];
            aucV(end + 1) = S.auc(s, y); %#ok<SAGROW>
            Gd = mean(abs(double(A) - double(B)), 3) / 255;
            spBg(end + 1) = S.spearman(rA(~L), Gd(~L)); %#ok<SAGROW>
            spAll(end + 1) = S.spearman(rA(:), Gd(:)); %#ok<SAGROW>
            spCh(end + 1) = S.spearman(rB(L), Gd(L)); %#ok<SAGROW>
        else
            Gd = mean(abs(double(A) - double(B)), 3) / 255; %#ok<NASGU>
        end
        % --- 池化：三种权重，同一 pool_radius_k；CV 分开统计 ---
        for ci = 1:numel(cellSizes)
            [retC, cvC, cvG, cvP] = pool_stats(rA, rB, ZA, ZB, L, cellSizes(ci), S);
            ret(:, ci) = ret(:, ci) + retC;
            cvCellAll = [cvCellAll; cvC]; %#ok<SAGROW>
            cvGlobAll = [cvGlobAll; cvG]; %#ok<SAGROW>
            cvProtoAll = [cvProtoAll; cvP]; %#ok<SAGROW>
        end
        if numel(maps) < 2 && nMix == 1
            maps{end + 1} = struct('A', A, 'B', B, 'rho', rA, 'name', d.name); %#ok<SAGROW>
        end
    end

    a = struct();
    a.name = d.name; a.n_valid = nValid; a.n_mix = nMix; a.n_miss = nMiss;
    a.rho_bg = median(rhoBg); a.rho_fg = median(rhoFg);
    av = aucV(~isnan(aucV));
    a.auc = mean(av); a.auc_ci = boot_ci(av, 1000);
    a.sp_bg = mean(spBg(~isnan(spBg))); a.sp_all = mean(spAll(~isnan(spAll))); a.sp_ch = mean(spCh(~isnan(spCh)));
    a.ret = ret / max(nValid, 1);
    a.cv_cell = median(cvCellAll); a.cv_global = median(cvGlobAll); a.cv_proto = median(cvProtoAll);
    acc = [acc, a]; %#ok<SAGROW>
    if ~isempty(maps), save_maps(maps, d.name, figdir, S); end

    logline('[%s] 有效 %d 图（混合 %d，缺失 %d）', d.name, nValid, nMix, nMiss);
    logline('  背景 rho %.4f / 变化 rho %.4f / Δ %+.4f', a.rho_bg, a.rho_fg, a.rho_fg - a.rho_bg);
    logline('  AUC(半径→变化) %.4f [%.4f, %.4f]', a.auc, a.auc_ci(1), a.auc_ci(2));
    logline('  Spearman_bg %.3f / 全图 %.3f / 变化区 %.3f', a.sp_bg, a.sp_all, a.sp_ch);
    logline('  CV：池内 %.4f / 整图成员 %.4f / 原型 %.4f', a.cv_cell, a.cv_global, a.cv_proto);
    logline('  cell=8/16/32/64 保留率(%%): 硬 %s | 软 %s | 外部p %s', ...
        mat2str(round(100 * a.ret(1, :), 1)), mat2str(round(100 * a.ret(2, :), 1)), ...
        mat2str(round(100 * a.ret(3, :), 1)));
    logline('');
end

S.writefile(fullfile(outdir, 'r2b_sample_list.txt'), sampleLines);
save(fullfile(outdir, 'res_r2b.mat'), 'acc');
S.writefile(fullfile(outdir, 'log_r2b.txt'), logline('get'));

%% ---------- 图 ----------
names = {acc.name};
fig = figure('Visible', 'off', 'Position', [40 40 1320 420]);
subplot(1, 3, 1); hold on;
for k = 1:numel(acc)
    bar(k - 0.18, acc(k).rho_bg, 0.32, 'FaceColor', [.65 .70 .78]);
    bar(k + 0.18, acc(k).rho_fg, 0.32, 'FaceColor', [.90 .55 .30]);
end
set(gca, 'XTick', 1:numel(acc), 'XTickLabel', names, 'FontSize', 8.5); grid on;
ylabel('半径中位数 \rho'); ylim([0.55 0.72]);
legend({'背景', '变化'}, 'Location', 'north', 'NumColumns', 2, 'FontSize', 8); legend boxoff;
title('(a) 变化与背景的半径差异很小', 'FontSize', 10);

subplot(1, 3, 2); hold on;
for k = 1:numel(acc)
    plot(k, acc(k).auc, 'o', 'MarkerSize', 7, 'MarkerFaceColor', [.35 .5 .75], 'MarkerEdgeColor', 'k');
    plot([k k], acc(k).auc_ci, 'k-', 'LineWidth', 1.2);
end
plot([0.4 numel(acc)+0.6], [0.5 0.5], 'r--');
set(gca, 'XTick', 1:numel(acc), 'XTickLabel', names, 'FontSize', 8.5); grid on;
ylim([0 1]); ylabel('AUC（半径 → 变化）');
title('(b) 单靠半径判别力有限', 'FontSize', 10);

subplot(1, 3, 3); hold on;
M = zeros(numel(cellSizes), numel(acc));
for k = 1:numel(acc), M(:, k) = 100 * acc(k).ret(3, :)'; end
hb = bar(M, 'grouped');
set(gca, 'XTick', 1:numel(cellSizes), 'XTickLabel', ...
    arrayfun(@(c) sprintf('%d px', c), cellSizes, 'UniformOutput', false), 'FontSize', 8.5);
grid on; ylim([45 90]); ylabel('池化后半径 / 成员加权半径（%）');
plot([0.4 numel(cellSizes)+0.6], [70 70], 'r--');
legend(hb, names, 'Location', 'southoutside', 'Orientation', 'horizontal', 'FontSize', 7.5, 'NumColumns', 2);
legend boxoff;
title('(c) 真实影像统计特征：cell 越大池化收缩越多', 'FontSize', 10);
S.savefig('fig_r2b_01_dataset', figdir); close(fig);

logline('R2b done.');

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

function F = build_features(img, C)
g = mean(img, 3);
gx = conv2(g, [1 -1], 'same'); gy = conv2(g, [1; -1], 'same');
h = sqrt(gx .^ 2 + gy .^ 2);
chans = {g, std(img, 0, 3), h, mean(img, 3) .^ 2, h .^ 2, img(:, :, 1), img(:, :, 2), img(:, :, 3)};
hs = [0 1 2 3 4 6 8 12];
F = zeros(size(img, 1), size(img, 2), C);
k = 0;
for bi = 1:numel(chans)
    for si = 1:numel(hs)
        k = k + 1;
        if k > C, break; end
        X = blur1(double(chans{bi}), hs(si));
        m = mean(abs(X(:)));
        if m > 1e-9, X = X / m; end
        F(:, :, k) = X;
    end
    if k >= C, break; end
end
end

function B = blur1(X, h)
if h < 1, B = X; return; end
k = 2 * h + 1;
B = boxf(X, k, 1); B = boxf(B, k, 2);
end

function Y = boxf(X, k, dim)
if dim == 1
    P = [repmat(X(1, :), k, 1); X; repmat(X(end, :), k, 1)];
    Y = conv2(P, ones(k, 1) / k, 'same');
    Y = Y(k + 1:end - k, :);
else
    P = [repmat(X(:, 1), 1, k), X, repmat(X(:, end), 1, k)];
    Y = conv2(P, ones(1, k) / k, 'same');
    Y = Y(:, k + 1:end - k);
end
end

function [r, Z3] = proj_z(F, W, gamma, beta, sqD)
X = reshape(F, [], size(F, 3));
mu = mean(X, 2); sd = sqrt(var(X, 1, 2) + 1e-5);
U = (X - mu) ./ sd;
U = U .* gamma + beta;
Z = U * W';
z = Z / sqD;                                  % 真实 K/Q 口径：先缩放再 Lift
r = reshape(asinh(sqrt(sum(z .^ 2, 2))), size(F, 1), size(F, 2));
Z3 = reshape(z, size(F, 1), size(F, 2), size(z, 2));
end

function [retC, cvC, cvG, cvP] = pool_stats(rA, rB, ZA, ZB, L, cw, S)
[n1, n2] = size(rA);
ny = floor(n1 / cw); nx = floor(n2 / cw);
if ny < 1 || nx < 1
    retC = [NaN; NaN; NaN]; cvC = NaN; cvG = NaN; cvP = NaN; return;
end
rA = rA(1:ny*cw, 1:nx*cw); rB = rB(1:ny*cw, 1:nx*cw); L2 = L(1:ny*cw, 1:nx*cw);
ZA = ZA(1:ny*cw, 1:nx*cw, :); ZB = ZB(1:ny*cw, 1:nx*cw, :);
allMembers = [rA(:); rB(:)];
cvG = std(allMembers, 1) / max(mean(allMembers), 1e-9);
perCellCv = []; pooled = []; rH = []; rS = []; rD = [];
for j = 1:nx
    for i = 1:ny
        m = [reshape(rA((i-1)*cw + (1:cw), (j-1)*cw + (1:cw)), [], 1); ...
             reshape(rB((i-1)*cw + (1:cw), (j-1)*cw + (1:cw)), [], 1)];
        p = [reshape(L2((i-1)*cw + (1:cw), (j-1)*cw + (1:cw)), [], 1); ...
             reshape(L2((i-1)*cw + (1:cw), (j-1)*cw + (1:cw)), [], 1)];
        Zcell = [reshape(ZA((i-1)*cw + (1:cw), (j-1)*cw + (1:cw), :), [], size(ZA, 3)); ...
                 reshape(ZB((i-1)*cw + (1:cw), (j-1)*cw + (1:cw), :), [], size(ZB, 3))];
        perCellCv(end + 1) = std(m, 1) / max(mean(m), 1e-9); %#ok<SAGROW>
        % 三种权重（都归一化，且都过同一个 pool_radius_k）
        wHard = double(m > median(m)); if ~any(wHard) || all(wHard), wHard = ones(size(m)); end
        wSoft = sinh(m) .^ 2;
        wDual = double(p); if ~any(wDual), wDual = ones(size(m)); end
        rH(end + 1) = ratio(m, Zcell, wHard); %#ok<SAGROW>
        rS(end + 1) = ratio(m, Zcell, wSoft); %#ok<SAGROW>
        rD(end + 1) = ratio(m, Zcell, wDual); %#ok<SAGROW>
        pooled(end + 1) = pool_radius_vec(Zcell, wDual); %#ok<SAGROW>
    end
end
retC = [mean(rH); mean(rS); mean(rD)];
cvC = median(perCellCv);
cvP = std(pooled, 1) / max(mean(pooled), 1e-9);
end

function r = ratio(m, Zcell, w)
% m: 成员半径（asinh 后）；Zcell: 成员空间向量 z（已含 /sqrt(D)）
w = w(:) / sum(w);
den = sum(w .* m);                  % 成员加权平均半径
r = pool_radius_vec(Zcell, w) / max(den, 1e-9);
end

function r = pool_radius_vec(z, w)
% 真实 Lorentz 向量池化：每个 z 先确定性 Lift → 加权和 → 归一化 → 取空间部分范数
w = w(:) / sum(w);
t = sqrt(1 + sum(z .^ 2, 2));
M = [t, z];
mm = (w' * M)';
n2 = mm(1) ^ 2 - sum(mm(2:end) .^ 2);
c = mm / sqrt(max(n2, 1e-12));
r = norm(c(2:end));
end

function ci = boot_ci(x, B)
x = x(~isnan(x));
if numel(x) < 3, ci = [NaN NaN]; return; end
n = numel(x); bs = zeros(B, 1);
for b = 1:B, bs(b) = mean(x(randi(n, n, 1))); end
bs = sort(bs);
ci = [bs(max(1, round(0.025 * B))), bs(min(B, round(0.975 * B)))];
end

function files = pick_files(root, nm, d)
files = {};
for i = 1:numel(nm)
    if d.numfmt
        stem = sprintf('%s%05d', d.prefix, str2double(nm{i}));
    else
        [~, stem] = fileparts(nm{i});
    end
    f = struct('A', fullfile(root, 'A', [stem d.ext]), ...
               'B', fullfile(root, 'B', [stem d.ext]), ...
               'L', fullfile(root, 'label', [stem d.ext]));
    if exist(f.A, 'file') && exist(f.B, 'file') && exist(f.L, 'file')
        files{end + 1} = f; %#ok<SAGROW>
    end
end
end

function save_maps(maps, dsname, figdir, S)
fig = figure('Visible', 'off', 'Position', [40 40 1400 430]);
for i = 1:numel(maps)
    m = maps{i};
    b = (i - 1) * 3;
    subplot(2, 3, b + 1); imshow(m.A); title(sprintf('%s: A', m.name), 'FontSize', 9);
    subplot(2, 3, b + 2); imshow(m.B); title('B', 'FontSize', 9);
    subplot(2, 3, b + 3); imshow(m.rho); colormap(gca, parula); colorbar;
    title('半径 \rho（亮=远）', 'FontSize', 9);
end
S.savefig(sprintf('fig_r2b_02_maps_%s', lower(dsname)), figdir);
close(fig);
end
