% =========================================================================
% r2_dataset_radius.m  ——  修正版 S3：真实数据集上的半径语义（val 口径）
%
% 相对 v1（docs/simulation/radius_semantics/s3_datasets.m）的修正
% （执行方案 §1.5 "S3"）：
%   1. 只用 **val 列表**，不碰 list/test.txt（v1 用 test 前 60 对）；
%   2. 抽样**固定清单**并写盘（排序后等距抽取，不用"前 N 张"，不依赖 RNG）；
%   3. LayerNorm 用**总体方差**（v1 用样本方差），eps 与 PyTorch 一致；
%   4. 标签统一 `gray >= 128`；文件名按各数据集实际规则（CDD 是 test_%05d.jpg）；
%   5. 池化**对向量做**（Lorentz 重心）再算半径，不是对半径做平均；
%      并补软混合权重（cell 内按局部能量加权）与硬分簇对照；
%   6. 特征基全部**对 A/B 对称**，于是 A/B 交换一致性应恒为 1（实测即校核）；
%   7. 半径诊断不看标签，标签只在算指标时读。
%
% 输出：results/log_r2.txt、res_r2.mat、r2_sample_list.txt、
%       figures/fig_r2_01_dataset_radius.png、fig_r2_02_radius_maps.png
% =========================================================================
clear; clc;
here = fileparts(mfilename('fullpath'));
addpath(here);
outdir = fullfile(here,'results'); figdir = fullfile(here,'figures');
if ~exist(outdir,'dir'), mkdir(outdir); end
if ~exist(figdir,'dir'), mkdir(figdir); end
logline('reset');
logline('===== R2 修正版 S3：真实数据集上的半径语义（val 口径）=====');

ROOT = 'F:/一些数据文件/变化检测CD数据集';
DS = struct('name',{'LEVIR','SYSU','WHU','CDD'}, ...
            'dir', {'LEVIR-CD-256','SYSU-CD-256','WHU-CD-256','CDD-CD-256'}, ...
            'ext',{'.png','.png','.png','.jpg'}, ...
            'prefix',{'','','','test_'}, 'numfmt',{false,false,false,true});
nPer = 40;
D = 32; C = 64; nBase = 8; hs = [0 1 2 3 4 6 8 12];
assert(numel(hs) == nBase);
rs = RandStream('mt19937ar','Seed',11);
W = (rand(rs,D,C)*2-1)/sqrt(C);      % 复刻 Conv2d 默认初始化（不是"正交投影"）
cellSizes = [8 16 32 64];
sqD = sqrt(D);

acc = []; sampleTxt = {}; rmap1 = []; rmap2 = []; ab1 = []; ab2 = [];
for di = 1:numel(DS)
    d = DS(di);
    root = fullfile(ROOT, d.dir);
    listFile = fullfile(root,'list','val.txt');
    if ~exist(listFile,'file'), logline('[%s] 缺 val.txt，跳过', d.name); continue; end
    nm = sort(readlines_simple(listFile));
    if numel(nm) > nPer, nm = nm(round(linspace(1, numel(nm), nPer))); end
    files = {}; stems = {};
    for i = 1:numel(nm)
        if d.numfmt
            stem = sprintf('%s%05d', d.prefix, str2double(nm{i}));
        else
            [~,stem] = fileparts(nm{i});
        end
        f = struct('A',fullfile(root,'A',[stem d.ext]), ...
                   'B',fullfile(root,'B',[stem d.ext]), ...
                   'L',fullfile(root,'label',[stem d.ext]));
        if exist(f.A,'file') && exist(f.B,'file') && exist(f.L,'file')
            files{end+1} = f; stems{end+1} = stem; %#ok<SAGROW>
        end
    end
    for i = 1:numel(stems), sampleTxt{end+1} = sprintf('%s: %s', d.name, stems{i}); end %#ok<SAGROW>
    logline('');
    logline('[%s] val 列表 %d 条 → 固定抽样 %d 对可用（清单见 r2_sample_list.txt）', ...
            d.name, numel(nm), numel(files));

    rhoBg = []; rhoFg = []; aucImg = []; spStr = []; swapCorr = [];
    retHard = cell(1,numel(cellSizes)); retSoft = cell(1,numel(cellSizes));
    memRho = []; protoRho = [];
    t0 = tic; rhoShow = []; show = [];
    for i = 1:numel(files)
        A0 = imread(files{i}.A); B0 = imread(files{i}.B); M0 = imread(files{i}.L);
        if size(A0,3) == 1, A0 = repmat(A0,1,1,3); end
        if size(B0,3) == 1, B0 = repmat(B0,1,1,3); end
        A = double(A0)/255; B = double(B0)/255;
        if size(M0,3) > 1, M0 = M0(:,:,1); end
        Y = double(M0) >= 128;

        Fq = proj_features(A,B,hs,W);            % 对称特征基
        [n1,n2,~] = size(Fq);
        Zq = reshape(Fq, n1*n2, D);
        rho = reshape(asinh(sqrt(sum(Zq.^2,2))/sqD), n1, n2);
        Fb = proj_features(B,A,hs,W);
        Zb = reshape(Fb, n1*n2, D);
        rhoS = reshape(asinh(sqrt(sum(Zb.^2,2))/sqD), n1, n2);
        if std(rho(:)) > 1e-12 && std(rhoS(:)) > 1e-12
            swapCorr(end+1) = corr_safe(rho(:), rhoS(:)); %#ok<SAGROW>
        end
        if i == 1, rhoShow = rho; show = {A,B,abs(A-B)}; end
        if ~isequal(size(Y), size(rho))
            Y = imresize(double(Y), [n1 n2], 'nearest') >= 0.5;
        end
        if ~any(Y(:)), continue; end
        rhoBg = [rhoBg; rho(~Y)]; rhoFg = [rhoFg; rho(Y)]; %#ok<SAGROW>
        aucImg(end+1) = auc_rank(rho(:), double(Y(:))); %#ok<SAGROW>
        str = sqrt(mean((A-B).^2,3));
        if std(rho(~Y)) > 1e-12 && std(str(~Y)) > 1e-12
            spStr(end+1) = spearman_r(rho(~Y), str(~Y)); %#ok<SAGROW>
        end
        for ci = 1:numel(cellSizes)
            [rH, rS, mR, pR] = cell_pool(Zq, n1, n2, cellSizes(ci), D, sqD);
            if ~isempty(rH)
                retHard{ci}(end+1) = rH; retSoft{ci}(end+1) = rS; %#ok<SAGROW>
            end
            if cellSizes(ci) == 32
                memRho = [memRho; mR]; protoRho = [protoRho; pR]; %#ok<SAGROW>
            end
        end
    end
    el = toc(t0);
    a = struct();
    a.name = d.name; a.nFig = numel(files); a.nAuc = numel(aucImg); a.sec = el;
    a.rhoBg = median_safe(rhoBg); a.rhoFg = median_safe(rhoFg);
    a.auc = boot_mean(aucImg);
    a.spStr = mean_safe(spStr); a.swap = mean_safe(swapCorr);
    a.retHard = cellfun(@mean_safe, retHard); a.retSoft = cellfun(@mean_safe, retSoft);
    a.cvMem = std(memRho,1)/mean(memRho);
    if isempty(protoRho), a.cvProto = NaN; else, a.cvProto = std(protoRho,1)/mean(protoRho); end
    logline('    有效图 %d / 读入 %d；耗时 %.1f s', a.nAuc, a.nFig, a.sec);
    logline('    背景 ρ 中位 %.3f / 变化 ρ 中位 %.3f（差 %+.3f）', a.rhoBg, a.rhoFg, a.rhoFg-a.rhoBg);
    logline('    每图 AUC（半径→变化）= %.3f，95%% CI [%.3f, %.3f]', a.auc(1), a.auc(2), a.auc(3));
    logline('    Spearman(ρ, |A−B|) = %+.3f；A/B 交换相关 = %+.4f（对称特征应 = 1）', a.spStr, a.swap);
    logline('    cell 池化保留率 硬/软：8px %.1f/%.1f%%  16px %.1f/%.1f%%  32px %.1f/%.1f%%  64px %.1f/%.1f%%', ...
            100*a.retHard(1),100*a.retSoft(1),100*a.retHard(2),100*a.retSoft(2), ...
            100*a.retHard(3),100*a.retSoft(3),100*a.retHard(4),100*a.retSoft(4));
    logline('    32px cell：成员 CV %.3f → 原型 CV %.3f（模块实测 K/Q≈0.70~0.73）', a.cvMem, a.cvProto);
    if di == 1, rmap1 = rhoShow; ab1 = show; end
    if di == 2, rmap2 = rhoShow; ab2 = show; end
    acc = [acc, a]; %#ok<SAGROW>
end
writefile_utf8(fullfile(outdir,'r2_sample_list.txt'), sampleTxt);

%% ---------------- 图 1 ----------------
if ~isempty(acc)
    ns = numel(acc);
    fh = figure('Visible','off','Position',[30 40 1280 470]);
    subplot(1,3,1); hold on;
    for k = 1:ns
        bar(k-0.18, acc(k).rhoBg, 0.32,'FaceColor',[.72 .76 .84],'EdgeColor','k');
        bar(k+0.18, acc(k).rhoFg, 0.32,'FaceColor',[.92 .62 .35],'EdgeColor','k');
    end
    set(gca,'XTick',1:ns,'XTickLabel',{acc.name},'FontSize',8.5); grid on;
    ylabel('半径中位数 \rho'); ylim([0.45 0.62]);
    legend({'背景','变化'},'Location','north','NumColumns',2,'FontSize',8); legend boxoff;
    title('(a) 变化与背景的半径差异 |\Delta\rho| <= 0.02','FontSize',10.5);

    subplot(1,3,2); hold on;
    for k = 1:ns
        plot([k k],[acc(k).auc(2) acc(k).auc(3)],'k-','LineWidth',1.2);
        plot(k, acc(k).auc(1),'o','MarkerSize',7,'MarkerFaceColor',[.3 .5 .8],'MarkerEdgeColor','k');
    end
    plot([0.4 ns+0.6],[0.5 0.5],'r--','LineWidth',1);
    set(gca,'XTick',1:ns,'XTickLabel',{acc.name},'FontSize',8.5); grid on;
    xlim([0.4 ns+0.6]); ylabel('AUC（半径→变化）');
    title('(b) 单靠半径判别力有限','FontSize',10.5);

    subplot(1,3,3); hold on;
    kk = 1:numel(cellSizes);
    Mh = reshape([acc.retHard], numel(cellSizes), [])';
    Ms = reshape([acc.retSoft], numel(cellSizes), [])';
    bar(kk-0.16, 100*mean(Mh,1), 0.3,'FaceColor',[.45 .62 .82],'EdgeColor','k');
    bar(kk+0.16, 100*mean(Ms,1), 0.3,'FaceColor',[.95 .75 .45],'EdgeColor','k');
    plot([0.5 kk(end)+0.5],[70 70],'r--','LineWidth',1.1);
    set(gca,'XTick',kk,'XTickLabel', ...
        arrayfun(@(c) sprintf('%dpx',c), cellSizes,'UniformOutput',false),'FontSize',8.5);
    grid on; ylabel('池化后半径 / 成员半径（%）'); ylim([68 100]); xlim([0.5 kk(end)+0.5]);
    legend({'硬分簇','软混合'},'Location','southoutside','Orientation','horizontal','FontSize',8);
    legend boxoff;
    text(1.35, 71.5, '红虚线 = 模块实测 70%','FontSize',7.5,'Color',[.6 .1 .1]);
    title('(c) 真实影像统计特征的池化保留率','FontSize',10.5);
    savefig_png('fig_r2_01_dataset_radius', figdir); close(fh);
end

%% ---------------- 图 2 ----------------
if ~isempty(rmap1)
    fh = figure('Visible','off','Position',[30 40 1300 470]);
    R = {ab1, rmap1; ab2, rmap2};
    nmz = {'LEVIR','SYSU'};
    for r = 1:2
        for c = 1:4
            subplot(2,4,(r-1)*4+c);
            if c <= 3
                imshow(R{r,1}{c});
            else
                imagesc(R{r,2}); axis image off; colormap(gca,parula); colorbar;
                lo = quantile(R{r,2}(:),0.02); hi = quantile(R{r,2}(:),0.98);
                caxis([lo hi]);
            end
            if r == 1
                tits = {'A','B','|A-B|','\rho（亮=远）'};
                title(tits{c},'FontSize',9);
            end
            if c == 1, ylabel(nmz{r},'FontSize',9); end
        end
    end
    savefig_png('fig_r2_02_radius_maps', figdir); close(fh);
end

save(fullfile(outdir,'res_r2.mat'),'acc');
logline('');
logline('R2 done.');

%% ======================= local functions =======================
function X = proj_features(A, B, hs, W)
% 8 个基础图**全部对 A/B 对称**，swap 后逐像素特征相同 → 交换一致性应为 1
gA = mean(A,3); gB = mean(B,3);
eA = edge_simple(gA); eB = edge_simple(gB);
cA = var(A,0,3); cB = var(B,0,3);
base = { (gA+gB)/2, abs(gA-gB), (eA+eB)/2, max(gA,gB), min(gA,gB), ...
         sqrt(mean((A-B).^2,3)), (cA+cB)/2, abs(cA-cB) };
F = [];
for b = 1:numel(base)
    for h = hs
        F = cat(3, F, boxblur(base{b}, h));
    end
end
F = layernorm3(F);
[n1,n2,C] = size(F);
F = reshape(reshape(F,n1*n2,C) * W.', n1, n2, []);
X = F;
end

function Y = layernorm3(F)
mu = mean(F,3);
V = mean(bsxfun(@minus, F, mu).^2, 3);    % 总体方差
Y = bsxfun(@rdivide, bsxfun(@minus, F, mu), sqrt(V + 1e-5));
end

function E = edge_simple(G)
E = abs([zeros(size(G,1),1) diff(G,1,2)]) + abs([zeros(1,size(G,2)); diff(G,1,1)]);
end

function Y = boxblur(X, h)
if h == 0, Y = X; return; end
k = 2*h+1;
Xp = padrep(padrep(X, h, 1), h, 2);
I = cumsum(cumsum(Xp,1),2);
I = [zeros(1,size(I,2)); I]; I = [zeros(size(I,1),1) I];
Y = I(k+1:end,k+1:end) - I(1:end-k,k+1:end) - I(k+1:end,1:end-k) + I(1:end-k,1:end-k);
Y = Y/(k*k);
end

function P = padrep(X, h, dim)
if dim == 1
    P = [repmat(X(1,:),h,1); X; repmat(X(end,:),h,1)];
else
    P = [repmat(X(:,1),1,h), X, repmat(X(:,end),1,h)];
end
end

function r = pool_radius(Z, w, sqD)
% Lorentz（Minkowski）重心的空间部分半径：lift → 加权和 → 归一化 → asinh(|z|/sqrt(D))
t = sqrt(1 + sum(Z.^2,2));
v = [t, Z];
s = sum(bsxfun(@times, v, w(:)), 1);
n2 = s(:,1).^2 - sum(s(:,2:end).^2, 2);
if n2 <= 1e-9, r = 0; return; end
s = s/sqrt(n2);
r = asinh(norm(s(:,2:end))/sqD);
end

function [rHard, rSoft, memRho, protoRho] = cell_pool(Zq, n1, n2, cw, D, sqD)
ny = floor(n1/cw); nx = floor(n2/cw);
Zimg = reshape(Zq, n1, n2, D);
memRho = []; protoRho = []; hR = []; sR = [];
for iy = 1:ny
    for ix = 1:nx
        blk = Zimg((iy-1)*cw+(1:cw), (ix-1)*cw+(1:cw), :);
        Zc = reshape(blk, cw*cw, D);
        rr = asinh(sqrt(sum(Zc.^2,2))/sqD);
        memRho = [memRho; rr]; %#ok<AGROW>
        % 硬分簇：按"离原点远近"分两簇（与模块"半径大=变化"的假设一致）
        m = median(rr);
        c1 = rr > m; c0 = ~c1;
        if any(c1) && any(c0)
            p1 = pool_radius(Zc(c1,:), ones(nnz(c1),1)/nnz(c1), sqD);
            p0 = pool_radius(Zc(c0,:), ones(nnz(c0),1)/nnz(c0), sqD);
            protoRho = [protoRho; p1; p0]; %#ok<AGROW>
            hR = [hR; 0.5*(p1+p0)]; %#ok<AGROW>
        end
        % 软混合：cell 内按局部能量加权（不读标签）
        e = sum(Zc.^2, 2); w = e/sum(e);
        sR = [sR; pool_radius(Zc, w, sqD)]; %#ok<AGROW>
    end
end
if isempty(memRho) || isempty(hR)
    rHard = []; rSoft = []; return;
end
rHard = mean(hR)/mean(memRho);
rSoft = mean(sR)/mean(memRho);
end

function m = boot_mean(x)
x = x(~isnan(x));
if isempty(x), m = [NaN NaN NaN]; return; end
rs2 = RandStream('mt19937ar','Seed',3); B = 2000;
bs = zeros(B,1);
for b = 1:B, bs(b) = mean(x(randi(rs2, numel(x), 1))); end
m = [mean(x), quantile(bs,0.025), quantile(bs,0.975)];
end

function a = auc_rank(s, y)
s = s(:); y = y(:) > 0;
r = rank_avg(s);
n1 = sum(y); n0 = numel(y) - n1;
if n1 == 0 || n0 == 0, a = NaN; return; end
a = (sum(r(y)) - n1*(n1+1)/2) / (n1*n0);
end

function r = rank_avg(x)
x = x(:); n = numel(x);
[s, ord] = sort(x);
r = zeros(n,1); i = 1;
while i <= n
    j = i;
    while j < n && s(j+1) == s(i), j = j + 1; end
    r(ord(i:j)) = (i+j)/2;
    i = j + 1;
end
end

function v = spearman_r(x, y)
rx = rank_avg(x); ry = rank_avg(y);
if std(rx) < 1e-12 || std(ry) < 1e-12, v = NaN; return; end
c = corrcoef(rx, ry); v = c(1,2);
end

function v = mean_safe(x)
x = x(:); x = x(~isnan(x));
if isempty(x), v = NaN; else, v = mean(x); end
end

function v = median_safe(x)
x = x(:); x = x(~isnan(x));
if isempty(x), v = NaN; else, v = median(x); end
end

function c = corr_safe(x, y)
x = x(:); y = y(:);
if std(x) < 1e-12 || std(y) < 1e-12, c = NaN; return; end
cc = corrcoef(x, y); c = cc(1,2);
end

function L = readlines_simple(p)
fid = fopen(p,'r','n','UTF-8');
if fid < 0, L = {}; return; end
L = {};
while true
    t = fgetl(fid);
    if ~ischar(t), break; end
    t = strtrim(t);
    if ~isempty(t), L{end+1} = t; end %#ok<AGROW>
end
fclose(fid);
end

function writefile_utf8(p, lines)
fid = fopen(p,'w','n','UTF-8');
for i = 1:numel(lines), fprintf(fid,'%s\n', lines{i}); end
fclose(fid);
end

function savefig_png(name, figdir)
exportgraphics(gcf, fullfile(figdir,[name '.png']), 'Resolution', 150);
end

function L = logline(fmt, varargin)
persistent B
if isempty(B) || strcmp(fmt,'reset'), B = {}; end
if isempty(varargin), s = fmt; else, s = sprintf(fmt, varargin{:}); end
B{end+1} = s;
here2 = fileparts(mfilename('fullpath'));
if numel(B) == 1, mode = 'w'; else, mode = 'a'; end
fid = fopen(fullfile(here2,'results','log_r2.txt'), mode, 'n', 'UTF-8');
fprintf(fid,'%s\n', s); fclose(fid);
L = s;
end
