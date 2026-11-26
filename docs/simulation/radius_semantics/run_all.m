%RUN_ALL  依次运行 S1–S6，并把总日志写到 results/log_all.txt
%
%   用法：在 MATLAB 里 cd 到本目录后直接运行  run_all
%         （不需要任何工具箱；MATLAB R2020b 及以上，R2022a 实测通过）

here = fileparts(mfilename('fullpath'));
if isempty(here), here = pwd; end
addpath(here);
outdir = fullfile(here,'results');
if ~exist(outdir,'dir'), mkdir(outdir); end

names = {'S1 纯几何：近/远意味着什么','S2 我们模块的半径由什么构成', ...
         'S3 四个真实数据集上的半径语义','S4 层级/池化算子/径向重锚', ...
         'S5 改码判定门','S6 半径归因与四个旋钮'};
funs = {@s1_geometry, @s2_our_radius, @s3_datasets, ...
        @s4_hierarchy_pooling, @s5_decision_gate, @s6_radius_knobs};

times = zeros(1,numel(funs));
fprintf('=== run_all: %d 个脚本 ===\n', numel(funs));
for i = 1:numel(funs)
    fprintf('[%d/%d] %s ...\n', i, numel(funs), names{i});
    t = tic;
    funs{i}();
    times(i) = toc(t);
    fprintf('        完成，%.1f s\n', times(i));
end

lines = {'===== run_all 总日志 =====',''};
for i = 1:numel(funs)
    lines{end+1,1} = sprintf('  %-28s %7.1f s', names{i}, times(i));
end
lines{end+1,1} = '';
lines{end+1,1} = sprintf('  合计 %.1f s', sum(times));
lines{end+1,1} = '';
lines{end+1,1} = '  每个脚本的详细数值见同目录 log_s*.txt，结论见 SUMMARY.md。';
fid = fopen(fullfile(outdir,'log_all.txt'),'w','n','UTF-8');
for i = 1:numel(lines), fprintf(fid,'%s\n',lines{i}); end
fclose(fid);
fprintf('run_all 完成，总耗时 %.1f s。日志：results/log_all.txt\n', sum(times));
