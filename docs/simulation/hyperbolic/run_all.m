% run_all.m -- single entry point for the CASA-CD hyperbolic simulation.
% Runs H5 (geometry switched on/off), H1 (hierarchy), H2 (content vs geometry),
% H3 (224-prototype retrieval), H4 (tau and curvature), writes all figures to
% ./figures and a full console log to ./run_all_log.txt.
%
%   >> run docs/simulation/hyperbolic/run_all.m      % or press Run in the editor
%
% Order matters: H5-Q1 is the first figure produced (the captain's requirement:
% the "is the geometry switched on at all?" measurement must come first).

clear; close all; clc;
here = fileparts(mfilename('fullpath'));
if isempty(here), here = pwd; end
addpath(here); cd(here);
figdir = fullfile(here,'figures'); if ~exist(figdir,'dir'), mkdir(figdir); end
logFile = fullfile(here,'run_all_log.txt');
if exist(logFile,'file'), delete(logFile); end
diary(logFile); diary on;

t0 = tic;
fprintf('=====================================================================\n');
fprintf(' CASA-CD hyperbolic simulation -- run_all\n');
fprintf(' MATLAB %s | %s\n', version, datestr(now,'yyyy-mm-dd HH:MM:SS'));
fprintf(' question: what does hyperbolic geometry give that Euclidean cannot?\n');
fprintf('=====================================================================\n');

fprintf('\n\n#################### PART 1/5 : H5 geometry on/off ####################\n');
fprintf('first figure = H5-Q1 (same shell vs cross shell rearrangement)\n');
h5_geometry;

fprintf('\n\n#################### PART 2/5 : H1 hierarchy ####################\n');
h1_hierarchy;

fprintf('\n\n#################### PART 3/5 : H2 content vs geometry ####################\n');
h2_decisive;

fprintf('\n\n#################### PART 4/5 : H3 224-KV retrieval ####################\n');
h3_prototypes;

fprintf('\n\n#################### PART 5/5 : H4 tau and curvature ####################\n');
h4_temperature;

fprintf('\n\n=====================================================================\n');
fprintf(' FIGURES (what to look at)\n');
fprintf('---------------------------------------------------------------------\n');
idx = { ...
 'fig_h5_rearrangement.png','H5-Q1 first: Spearman(hyperbolic, euclid) vs radial spread; flat shell -> geometry off', ...
 'fig_h5_q2_arms.png',    'H5-Q2: top-1 retrieval, LN vs raw, with/without a radius bias', ...
 'fig_h5_shells.png',     'H5-Q3 (M2): weak/strong recall at 1% FPR, raw vs LN vs radius-layered KV', ...
 'fig_h5_tree_shell.png', 'H5-Q4: tree stress and radius-vs-depth under a one-shell constraint', ...
 'fig_h1_disk.png',       'H1: 2-D Poincare embedding of the flat KV tree vs a quad-tree KV', ...
 'fig_h1_dims.png',       'H1: stress vs dimension, Euclidean vs hyperbolic (dimension budget)', ...
 'fig_h2_arms.png',       'H2: linear-probe AUC per arm (content arms vs geometry arms)', ...
 'fig_h2_effect.png',     'H2: paired effect sizes with confidence intervals', ...
 'fig_h3_distmat.png',    'H3: 224x224 distance matrix, hyperbolic vs Euclidean (block structure)', ...
 'fig_h3_retrieval.png',  'H3: retrieval accuracy vs prototype noise, all four arms', ...
 'fig_h4_tau.png',        'H4: AUC flat in tau, threshold follows the prevalence', ...
 'fig_h4_curvature.png',  'H4: curvature only reorders far pairs, AUC barely moves'};
for i = 1:2:size(idx,2)
    have = exist(fullfile(figdir,idx{1,i}),'file') > 0;
    mark = ' '; if have, mark = 'x'; end
    fprintf(' [%s] %-26s %s\n', mark, idx{1,i}, idx{1,i+1});
end
fprintf('---------------------------------------------------------------------\n');
fprintf(' total runtime %.1f s ; log written to %s\n', toc(t0), logFile);
fprintf('=====================================================================\n');
diary off;
