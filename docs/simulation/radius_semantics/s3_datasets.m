function s3_datasets()
%S3_DATASETS  在真实 LEVIR / SYSU / WHU / CDD 数据上，双曲半径"近/远"分别对应什么
%
%   本地没有训练好的权重，因此这里用一个**结构等价的替代前端**：把 256x256 的
%   A/B 影像做成 64 通道、8 个空间尺度的局部统计特征（见 build_features），
%   再走和我们模块完全相同的三步：LayerNorm(64) -> 32x64 的 1x1 投影 -> 半径
%       rho = asinh(||W * LN(x)|| / sqrt(32))
%   替代前端保留"多尺度局部统计 + LN + 1x1 投影"这三件决定半径的事；
%   不保真的是具体权重（真实 qk 是学出来的），所以本脚本只报告统计结构与排序关系。
%
%   回答四个问题：
%     Q1 半径能不能区分变化/背景？（每图 AUC + 自助 CI）
%     Q2 半径与"变化强度""窗口内变化占比""处在变化区域多深"的排序关系
%     Q3 真实像素按 cell 池化会把半径压掉多少？cell ∈ {8,16,32,64} 像素
%        （模块 FLAT_GRIDS=(8,4,4,4) 作用在 stride-4 特征图上，
%          grid=8 的 cell 恰好等于 256 图里的 32x32 像素 = 1024 个像素）
%     Q4 整幅 A/B 乘 0.5 时半径是否不变（LN 免疫性的真实数据证据，含 eps 归因）
%
%   产出：results/figures/fig_s3_01_dataset_semantics.png
%         results/figures/fig_s3_02_radius_maps.png
%         results/res_s3.mat, results/log_s3.txt

R = rs_setup(); R.zh();
rng(11,'twister');
out = struct(); L = {};
L = say(L,'===== S3 真实数据集上"近/远"的语义 =====');
t0 = tic;

C = 64; D = 32;
W = R.proj(C, D, 11);
cellSizes = [8 16 32 64];

roots = 'F:/一些数据文件/变化检测CD数据集';
DS = struct('name',{'LEVIR','SYSU','WHU','CDD'}, ...
            'dir', {'LEVIR-CD-256','SYSU-CD-256','WHU-CD-256','CDD-CD-256'}, ...
            'ext',{'.png','.png','.png','.jpg'}, ...
            'prefix',{'','','','test_'}, ...      % CDD 文件名带 test_ 前缀，list 里只有编号
            'numfmt',{false,false,false,true}, ...
            'nmax',{60,60,60,60});

acc = struct();
for di = 1:numel(DS)
    d = DS(di); d.root = fullfile(roots, d.dir);
    files = pick_files(d);
    L = say(L,'');
    L = say(L,'[%s] 用 %d 张测试图', d.name, numel(files));
    a = run_dataset(R, d, files, W, C, D, cellSizes);
    acc(di).name = d.name; acc(di).a = a;
    L = say(L,'  背景 ρ 中位 %.3f ｜ 变化 ρ 中位 %.3f ｜ 差 %+.3f', a.med_bg, a.med_fg, a.med_fg-a.med_bg);
    L = say(L,'  Q1 半径判变化：每图 AUC 均值 %.3f（95%% CI %.3f~%.3f，%d 张有效）', ...
            a.auc_mean, a.auc_lo, a.auc_hi, numel(a.auc_img));
    L = say(L,'  Q2 Spearman(ρ, 变化强度)=%+.3f ｜ (ρ, 窗口变化占比)=%+.3f ｜ (ρ, 变化内部深度)=%+.3f', ...
            a.sp_strength, a.sp_core, a.sp_insider);
    L = say(L,'     有效图数：%d / %d / %d（无变化区域的图记为 NaN 并跳过）', a.n_sp(1), a.n_sp(2), a.n_sp(3));
    for ci = 1:numel(cellSizes)
        L = say(L,'  Q3 cell=%2dpx (n=%4d)：成员均值 %.3f → 池化后 %.3f（保留 %.0f%%，c_eff=%.2f）', ...
                cellSizes(ci), cellSizes(ci)^2, a.pool_member(ci), a.pool_after(ci), ...
                100*a.pool_after(ci)/a.pool_member(ci), a.c_eff(ci));
    end
    L = say(L,'     先池化内容再投影（cell=32）：%.3f（与先投影后池化差 %.3f）', ...
            a.pool_content(3), abs(a.pool_content(3)-a.pool_after(3)));
    L = say(L,'  Q4 A/B 整体乘 0.5：半径最大变化 %.2e（eps=1e-5）｜ %.2e（eps=0，见 log 说明）', ...
            a.gain_drift, a.gain_drift0);
    L = say(L,'  用时 %.1f s', a.secs);
end
out.acc = acc;
dsn = {acc.name};

% ============================== 图 1 ==============================
f1 = figure('Visible','off','Position',[30 30 1420 1010]);

% (a) 分位数箱线：每个数据集两行（背景/变化）
subplot(2,2,1); hold on;
y = 0; tkl = {};
for di = 1:numel(acc)
    for which = 1:2
        y = y + 1;
        if which == 1, v = acc(di).a.rho_bg; else, v = acc(di).a.rho_fg; end
        q = quantile(v, [0.05 0.25 0.50 0.75 0.95]);
        c = [.72 .72 .72]; if which == 2, c = [.88 .45 .30]; end
        patch([q(2) q(4) q(4) q(2)],[y-0.30 y-0.30 y+0.30 y+0.30], c, 'EdgeColor',[.25 .25 .25]);
        plot([q(1) q(5)],[y y],'-','Color',[.25 .25 .25],'LineWidth',1.0);
        plot([q(3) q(3)],[y-0.30 y+0.30],'k-','LineWidth',1.6);
        if which == 1, tag = '背景'; else, tag = '变化'; end
        tkl{y} = [acc(di).name ' ' tag]; %#ok<AGROW>
    end
end
set(gca,'YTick',1:y,'YTickLabel',tkl,'FontSize',7.5,'YDir','reverse');
xlim([0 1.15]); grid on; xlabel('半径 \rho');
title('(a) 背景 vs 变化：变化整体更靠外，但高度重叠','FontSize',10.5);

% (b) AUC
subplot(2,2,2); hold on;
mu = zeros(1,numel(acc)); lo = mu; hi = mu;
for di = 1:numel(acc)
    a = acc(di).a; mu(di)=a.auc_mean; lo(di)=a.auc_lo; hi(di)=a.auc_hi;
end
bar(1:numel(acc), mu, 0.55, 'FaceColor',[.62 .76 .90],'EdgeColor','k');
errorbar(1:numel(acc), mu, mu-lo, hi-mu, 'k.','LineWidth',1.1);
plot([0.4 numel(acc)+0.6],[0.5 0.5],'r--','LineWidth',1.2);
set(gca,'XTick',1:numel(acc),'XTickLabel',dsn);
ylim([0.35 0.85]); grid on; ylabel('AUC（半径 → 是否变化）');
title('(b) 只用半径判变化：只比随机好一点','FontSize',10.5);
text(0.55, 0.39, '红虚线 = 0.5（随机猜测）','FontSize',8,'Color',[.7 .2 .1]);

% (c) 四种 cell 尺寸下的"半径保留率"
subplot(2,2,3); hold on;
Ret = zeros(numel(acc), numel(cellSizes));
for di = 1:numel(acc)
    Ret(di,:) = 100*acc(di).a.pool_after./acc(di).a.pool_member;
end
hb2 = bar(Ret, 'grouped');
cc2 = [.15 .45 .75; .20 .60 .35; .85 .55 .10; .70 .25 .65];
for ci = 1:numel(cellSizes), hb2(ci).FaceColor = cc2(ci,:); end
plot([0.4 numel(acc)+0.6],[70 70],'r--','LineWidth',1.4);
set(gca,'XTick',1:numel(acc),'XTickLabel',dsn);
ylim([40 105]); grid on; ylabel('池化后半径 / 成员半径均值（%）');
legend(hb2, arrayfun(@(n) sprintf('cell=%dpx (n=%d)',n,n^2), cellSizes,'UniformOutput',false), ...
       'Location','southoutside','Orientation','horizontal','FontSize',7.5,'NumColumns',2);
legend boxoff;
title('(c) 影像特征池化只掉 2~8%，模块实测却掉到 70%（红虚线）','FontSize',10.5);

% (d) 排序关系
subplot(2,2,4); hold on;
M = zeros(numel(acc),3);
for di = 1:numel(acc)
    M(di,:) = [acc(di).a.sp_strength, acc(di).a.sp_core, acc(di).a.sp_insider];
end
hb = bar(M, 'grouped');
hb(1).FaceColor = [.88 .45 .30]; hb(2).FaceColor = [.30 .62 .45]; hb(3).FaceColor = [.35 .45 .80];
set(gca,'XTick',1:numel(acc),'XTickLabel',dsn,'FontSize',9);
xlim([0.4 numel(acc)+0.6]); ylim([-0.25 0.25]); grid on; ylabel('Spearman 相关');
plot(xlim,[0 0],'k-','LineWidth',1.0);
legend(hb, {'与变化强度','与窗口内变化占比','与变化区域内部深度'}, ...
       'Location','southoutside','Orientation','horizontal','FontSize',7.5);
legend boxoff;
title('(d) 半径与三个"具体性"代理量几乎不相关','FontSize',10.5);

sgtitle('真实数据上双曲半径的语义：靠外 ≈ 变化/边缘/高频，但单靠半径判别力有限', ...
        'FontSize',12.5,'FontWeight','bold');
R.savefig('fig_s3_01_dataset_semantics'); close(f1);

% ============================== 图 2 ==============================
f2 = figure('Visible','off','Position',[30 30 1520 700]);
case_k = [1 7];                                   % LEVIR / SYSU 各挑一张有变化的
for r = 1:2
    d = DS(r); d.root = fullfile(roots, d.dir);
    files = pick_files(d);
    k = pick_change_image(files, case_k(r));
    A = imread(files{k}.A); B = imread(files{k}.B); Lb = imread(files{k}.L);
    A = double(A)/255; B = double(B)/255;
    if size(A,3) == 1, A = repmat(A,1,1,3); end
    if size(B,3) == 1, B = repmat(B,1,1,3); end
    F = build_features(A,B,R); U = ln_image(F);
    Z = (reshape(U,[],C)*W')/sqrt(D);
    rho = reshape(asinh(sqrt(sum(Z.^2,2))), size(A,1), size(A,2));
    dlt = mean(abs(A-B),3);
    cl  = quantile(rho(:), [0.02 0.98]);
    pos = (r-1)*5;
    subplot(2,5,pos+1); imshow(A); title(sprintf('%s #%d：A', d.name, k),'FontSize',9);
    subplot(2,5,pos+2); imshow(B); title('B','FontSize',9);
    subplot(2,5,pos+3); imshow(dlt, []); title('|A-B|','FontSize',9);
    subplot(2,5,pos+4); imshow(double(Lb)>127); title('标注（变化）','FontSize',9);
    subplot(2,5,pos+5); imshow(rho, cl); colormap(gca, parula); colorbar('FontSize',7);
    title('半径 \rho（亮=远）','FontSize',9);
end
sgtitle('同一张图内：暗/蓝 = 近原点（同质背景），亮/黄 = 远（变化/边缘/纹理）；色标按 2~98% 分位拉伸', ...
        'FontSize',12,'FontWeight','bold');
R.savefig('fig_s3_02_radius_maps'); close(f2);

out.runtime_s = toc(t0);
L = say(L,''); L = say(L,'总耗时 %.1f s', out.runtime_s);
save(fullfile(R.outdir,'res_s3.mat'),'out');
R.writefile(fullfile(R.outdir,'log_s3.txt'), L);
fprintf('S3 done.\n');
end

% ======================================================================= io
function files = pick_files(d)
root = d.root;
lf = fullfile(root,'list','test.txt');
if exist(lf,'file')
    fid = fopen(lf,'r'); nm = textscan(fid,'%s'); fclose(fid); nm = nm{1};
else
    dd = dir(fullfile(root,'A',['*' d.ext])); nm = sort({dd.name})';
end
nm = nm(1:min(d.nmax,numel(nm)));
files = cell(numel(nm),1);
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
        files{i} = f;
    end
end
files = files(~cellfun(@isempty,files));
end

% ================================================================= features
function F = build_features(A, B, R)
% 64 通道 = 8 个基础图 x 8 个空间尺度（h = 0,1,2,3,4,6,8,12，都是局部尺度）；A,B 为 0..1 double
gA = mean(A,3);
M = cat(3, A(:,:,1),A(:,:,2),A(:,:,3), B(:,:,1),B(:,:,2),B(:,:,3), ...
           mean(abs(A-B),3), abs(gA - R.boxblur(gA,1)));
hs = [0 1 2 3 4 6 8 12];
F = zeros(size(A,1), size(A,2), 64);
for i = 1:8
    for j = 1:numel(hs)
        F(:,:,(i-1)*numel(hs)+j) = R.boxblur(M(:,:,i), hs(j));
    end
end
end

function U = ln_image(F, epsv)
if nargin < 2, epsv = 1e-5; end
X = reshape(F, [], size(F,3));
mu = mean(X,2); sd = sqrt(var(X,0,2) + epsv);
U = (X - mu) ./ sd;
U = reshape(U, size(F));
end

% ================================================================== dataset
function a = run_dataset(R, d, files, W, C, D, cellSizes)
tstart = tic;
auc_img = []; rho_bg = []; rho_fg = [];
sp_s = []; sp_c = []; sp_i = [];
nc = numel(cellSizes);
pool_member = zeros(1,nc); pool_after = zeros(1,nc); pool_content = zeros(1,nc);
member_all = cell(1,nc); after_all = cell(1,nc);
sel_count = zeros(1,nc);
drift = 0; drift0 = 0;

for k = 1:numel(files)
    A = imread(files{k}.A); B = imread(files{k}.B); Lb = imread(files{k}.L);
    if size(A,3) == 1, A = repmat(A,1,1,3); end
    if size(B,3) == 1, B = repmat(B,1,1,3); end
    A = double(A)/255; B = double(B)/255;
    mask = double(Lb) > 127;

    F = build_features(A,B,R);
    U = ln_image(F);
    Z = (reshape(U,[],C)*W')/sqrt(D);
    rho = asinh(sqrt(sum(Z.^2,2)));

    % ---- Q1
    y = reshape(mask, [], 1);
    if any(y) && ~all(y)
        auc_img(end+1) = R.auc(rho, double(y)); %#ok<AGROW>
    end

    % ---- Q4 整体增益不变性（eps=1e-5 与 eps=0 各一次）
    F2 = build_features(A*0.5, B*0.5, R);
    r2 = asinh(sqrt(sum(((reshape(ln_image(F2),[],C)*W')/sqrt(D)).^2,2)));
    drift = max(drift, max(abs(r2 - rho)));
    r1b = asinh(sqrt(sum(((reshape(ln_image(F,0),[],C)*W')/sqrt(D)).^2,2)));
    r2b = asinh(sqrt(sum(((reshape(ln_image(F2,0),[],C)*W')/sqrt(D)).^2,2)));
    drift0 = max(drift0, max(abs(r2b - r1b)));

    % ---- 分层采样
    ib = find(~y); ib = ib(randperm(numel(ib), min(400,numel(ib))));
    rho_bg = [rho_bg; rho(ib)]; %#ok<AGROW>
    if any(y)
        iy = find(y); if numel(iy) > 400, iy = iy(randperm(numel(iy),400)); end
        rho_fg = [rho_fg; rho(iy)]; %#ok<AGROW>
    end

    % ---- Q2 排序关系
    strength = reshape(mean(abs(A-B),3), [], 1);
    core     = reshape(R.boxblur(double(mask),8), [], 1);
    depth = zeros(numel(core),1);
    for h = [1 2 4 8 16]
        depth = depth + (reshape(R.boxblur(double(mask),h),[],1) > 0.99);
    end
    idx = 1:numel(rho);
    if numel(idx) > 15000, idx = idx(randperm(numel(idx),15000)); end
    sp_s(end+1) = R.spearman(rho(idx), strength(idx)); %#ok<AGROW>
    sp_c(end+1) = R.spearman(rho(idx), core(idx)); %#ok<AGROW>
    if any(y)
        iy = find(y);
        if numel(iy) > 15000, iy = iy(randperm(numel(iy),15000)); end
        sp_i(end+1) = R.spearman(rho(iy), depth(iy)); %#ok<AGROW>
    end

    % ---- Q3 多种 cell 尺寸的池化
    for ci = 1:nc
        cw = cellSizes(ci);
        ny = size(A,1)/cw; nx = size(A,2)/cw;
        Zc = reshape(Z, cw,ny, cw,nx, D);
        Tc = sqrt(1 + sum(Zc.^2,5));
        Zb = squeeze(mean(mean(Zc,1),3));
        Tb = squeeze(mean(mean(Tc,1),3));
        rho_cell = asinh(sqrt(sum(Zb.^2,3) ./ max(Tb.^2 - sum(Zb.^2,3), 1e-9)));
        rho_mem  = squeeze(mean(mean(asinh(sqrt(sum(Zc.^2,5))),1),3));
        mk = reshape(double(mask), cw,ny, cw,nx);
        pf = squeeze(mean(mean(mk,1),3));
        sel = (pf > 0.95) | (pf < 0.05);
        if any(sel(:))
            pool_member(ci) = pool_member(ci) + mean(rho_mem(sel));
            pool_after(ci)  = pool_after(ci)  + mean(rho_cell(sel));
            Uc = reshape(U, cw,ny, cw,nx, C);
            Ub = squeeze(mean(mean(Uc,1),3));
            zb = reshape(Ub, [], C)*W'/sqrt(D);
            pool_content(ci) = pool_content(ci) + mean(asinh(sqrt(sum(zb.^2,2))));
            member_all{ci} = [member_all{ci}; rho_mem(sel)]; %#ok<AGROW>
            after_all{ci}  = [after_all{ci};  rho_cell(sel)]; %#ok<AGROW>
            sel_count(ci) = sel_count(ci) + 1; %#ok<AGROW>
        end
    end
end
sel_count(end+1:nc) = 0;
pool_member = pool_member ./ max(sel_count,1);
pool_after  = pool_after  ./ max(sel_count,1);
pool_content= pool_content./ max(sel_count,1);

a.auc_img = auc_img; a.auc_mean = mean(auc_img);
[lo,hi] = R.bootci(@(ii) mean(auc_img(ii)), numel(auc_img), 1000);
a.auc_lo = lo; a.auc_hi = hi;
a.med_bg = median(rho_bg); a.med_fg = median(rho_fg);
a.rho_bg = rho_bg; a.rho_fg = rho_fg;
a.sp_strength = nanmean_safe(sp_s); a.sp_core = nanmean_safe(sp_c); a.sp_insider = nanmean_safe(sp_i);
a.n_sp = [numel(sp_s) sum(~isnan(sp_c)) sum(~isnan(sp_i))];
a.pool_member = pool_member; a.pool_after = pool_after; a.pool_content = pool_content;
a.pool_member_all = member_all; a.pool_after_all = after_all;
a.c_eff = zeros(1,nc);
for ci = 1:nc
    n = cellSizes(ci)^2;
    x2 = (sinh(pool_after(ci))/sinh(pool_member(ci)))^2;
    a.c_eff(ci) = max(min((x2 - 1/n)/(1-1/n), 1), 0);
end
a.gain_drift = drift; a.gain_drift0 = drift0;
a.secs = toc(tstart);
end

% ---------------------------------------------------------------- helpers --
function k = pick_change_image(files, prefer)
% 从 prefer 开始找一张"变化像素 >= 80"的图，找不到就退回 prefer
k = min(prefer, numel(files));
for i = prefer:numel(files)
    Lb = imread(files{i}.L);
    if sum(double(Lb(:)) > 127) >= 80
        k = i; return;
    end
end
end

function m = nanmean_safe(v)
v = v(~isnan(v));
if isempty(v), m = NaN; else, m = mean(v); end
end

function L = say(L, fmt, varargin)
if isempty(varargin)
    s = fmt;
else
    s = sprintf(fmt, varargin{:});
end
fprintf('%s\n', s);
L{end+1,1} = s;
end
