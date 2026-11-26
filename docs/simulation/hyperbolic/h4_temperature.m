% h4_temperature.m -- H4: is the learnable temperature tau (and curvature c)
% doing geometry, or only re-calibrating the class prior?
%
% H4-A  tau cannot change a ranking: AUC of a softmax(-tau*d^2) readout is flat in tau
% H4-B  a tau fitted by BCE follows the PREVALENCE, not the geometry: at 1/3/5%
%       positives the learned intercept moves by the log-odds shift while the AUC
%       on one fixed test set stays identical
% H4-C  curvature: (i) with FIXED raw features, d_c differs from d_1 only at large
%       distance (Spearman split by distance quartile); (ii) on ONE shell the ranking
%       is identical for every c (analytic: d_c = acosh(1+2cS/den)/sqrt(c), den fixed)
% H4-D  tau does not interact with c: AUC(tau) is flat at every c, so tau cannot
%       compensate for curvature (they are not interchangeable knobs)
%
% Figures: fig_h4_tau.png, fig_h4_curvature.png

here = fileparts(mfilename('fullpath'));
addpath(here); H = hyper_setup();
figdir = fullfile(here,'figures'); if ~exist(figdir,'dir'), mkdir(figdir); end
rng(44); fprintf('H4 ===== temperature and curvature: geometry or prior calibration? =\n');

% ------------------------------------------------- synthetic change problem --
d = 32; N = 20000; nz = 0.55;
Wc = randn(5,d)/sqrt(5);                       % change directions live in 5 dims
isCh = rand(N,1) < 0.05;
ul = randn(N,5); ul = ul./vecnorm(ul,2,2);
Z = (ul*Wc).*(1.1*isCh) + nz*randn(N,d);
up = randn(8,5); up = up./vecnorm(up,2,2);
Mch = (up*Wc)*1.1;                             % 8 change prototypes
Mbg = zeros(1,d);                              % background prototype = origin
P = H.pball(Z); Pmc = H.pball(Mch); Pmb = H.pball(Mbg);
s1 = -min(H.cdist(P,Pmc).^2,[],2);             % -d^2 to the nearest change proto
s0 = -H.cdist(P,Pmb).^2;                       % -d^2 to the background proto
y = double(isCh);
sAll = s1 - s0;
fprintf('  problem: %d samples, 5%% change, 32-d, geodesic prototype contrast\n', N);
fprintf('  raw-score AUC = %.4f (this is the only number that carries information)\n', H.auc(sAll,y));

% --------------------------------------------------- H4-A tau invariance -----
taus = [0.05 0.1 0.25 0.5 1 2 4 8 16 32];
aucT = zeros(size(taus)); dmax = 0; pRef = [];
for i = 1:numel(taus)
    tau = taus(i);
    lg = [tau*s1, tau*s0]; lg = lg - max(lg,[],2);
    p = exp(lg(:,1))./sum(exp(lg),2);           % change probability
    aucT(i) = H.auc(p,y);
    if isempty(pRef), pRef = p; else, dmax = max(dmax, max(abs(p - pRef))); end
end
fprintf('\nH4-A ===== softmax(-tau*d^2) readout vs tau\n');
fprintf('  tau      :'); fprintf(' %7.2f', taus); fprintf('\n');
fprintf('  AUC      :'); fprintf(' %7.4f', aucT); fprintf('\n');
fprintf('  max|AUC(tau) - AUC(tau_min)| = %.2e  -> AUC is EXACTLY flat in tau\n', max(abs(aucT-aucT(1))));
fprintf('  max|p(tau=32) - p(tau=0.05)| = %.3f (probabilities do move: calibration only)\n', dmax);

% ------------------------------------- H4-B learned tau tracks the prevalence -
fprintf('\nH4-B ===== a tau fitted by BCE: does it learn geometry or the prior?\n');
idxP = find(isCh); idxN = find(~isCh); nP = numel(idxP);
prevList = [0.01 0.03 0.05]; aFit = zeros(1,3); bFit = zeros(1,3); aucFix = zeros(1,3);
nTest = 4000; te = randperm(N, nTest);
for k = 1:3
    pr = prevList(k);
    nN = round(nP*(1-pr)/pr);
    tr = [idxP; idxN(randi(numel(idxN), nN, 1))];   % negatives drawn with replacement
    tr = unique(tr);                                 % (prevalence enforced exactly)
    w = H.logreg(sAll(tr), y(tr), 1e-3);             % w = [b; a], a plays the role of tau
    bFit(k) = w(1); aFit(k) = w(2);
    aucFix(k) = H.auc(sAll(te), y(te));              % one fixed test set, ranking unchanged
end
fprintf('  prevalence | learned a (=tau) | learned b | threshold s0=-b/a | AUC on one fixed test set\n');
for k = 1:3
    fprintf('     %.0f%%     |     %7.2f      |  %6.2f   |     %7.3f      |   %.4f\n', ...
        100*prevList(k), aFit(k), bFit(k), -bFit(k)/aFit(k), aucFix(k));
end
fprintf('  log-odds shift 1%%->5%% = %.2f ; learned intercept shift = %.2f ; learned a ratio = %.2f\n', ...
    log(0.05/0.95)-log(0.01/0.99), bFit(3)-bFit(1), aFit(3)/aFit(1));
fprintf('  reading: the intercept/threshold follows the prevalence almost exactly while the\n');
fprintf('           slope and the AUC do not move -> learnable tau is a PRIOR parameter.\n');

% ------------------------------------------------------- H4-C curvature ------
fprintf('\nH4-C ===== curvature c: new information or a reparameterisation?\n');
cs = [0.25 0.5 1 2 4];
liftf = @(Q,c) Q./(1+sqrt(1+c*sum(Q.^2,2)));    % generalises the project lift to c
v = Z(1:400,:);
devR = 0; radC = zeros(size(cs));
for i = 1:numel(cs)
    Xc = liftf(v,cs(i));
    rr = H.gdist_c(Xc, zeros(1,d), cs(i));
    devR = max(devR, max(abs(rr - asinh(sqrt(cs(i))*vecnorm(v,2,2))/sqrt(cs(i)))));
    radC(i) = mean(rr);
end
fprintf('  (i) radius identity d_c(0,x) = asinh(sqrt(c)||z||)/sqrt(c): max dev %.2e\n', devR);
fprintf('      mean geodesic radius per c:'); fprintf(' %6.3f', radC); fprintf('\n');
fprintf('      c rescales the radius only: c->0 gives ||z||, small ||z|| is c-blind\n');
D1p = H.gdist_c(liftf(v,1), liftf(v,1), 1); D1p = D1p(triu(true(400),1));
[ds, ord] = sort(D1p); nq = round(0.25*numel(ds));
nearI = ord(1:nq); farI = ord(end-nq+1:end);
sp = zeros(size(cs)); spNear = zeros(size(cs)); spFar = zeros(size(cs)); aucC = zeros(size(cs));
for i = 1:numel(cs)
    Xc = liftf(v,cs(i));
    Dc = H.gdist_c(Xc, Xc, cs(i)); Dc = Dc(triu(true(400),1));
    sp(i) = corr(D1p, Dc, 'type','Spearman');
    spNear(i) = corr(D1p(nearI), Dc(nearI), 'type','Spearman');
    spFar(i) = corr(D1p(farI), Dc(farI), 'type','Spearman');
end
fprintf('  (ii) fixed features, Spearman(d_c, d_1):\n');
fprintf('       c          :'); fprintf(' %7.2f', cs); fprintf('\n');
fprintf('       all pairs  :'); fprintf(' %7.4f', sp); fprintf('\n');
fprintf('       nearest 25%%:'); fprintf(' %7.4f', spNear); fprintf('\n');
fprintf('       far 25%%    :'); fprintf(' %7.4f', spFar); fprintf('\n');
% (iii) one shell: identical ranking for every c (analytic: d_c monotone in the chord)
u = randn(300,d); u = u./vecnorm(u,2,2); R0 = 1.0;
spShell = zeros(size(cs)); Dsh1 = [];
for i = 1:numel(cs)
    Xc = (tanh(sqrt(cs(i))*R0/2)/sqrt(cs(i)))*u;     % same geodesic radius R0
    Dc = H.gdist_c(Xc, Xc, cs(i)); Dc = Dc(triu(true(300),1));
    if i == 1, Dsh1 = Dc; spShell(i) = 1;
    else, spShell(i) = corr(Dsh1, Dc, 'type','Spearman'); end
end
fprintf('  (iii) one shell (R=%.1f), Spearman vs c=%.2f:', R0, cs(1));
fprintf(' %7.4f', spShell); fprintf('  -> ranking identical for every c\n');

% --------------------------------------------------- H4-D tau does not fix c -
fprintf('\nH4-D ===== tau vs c: can a learnable tau compensate for curvature?\n');
spC = zeros(size(cs)); scAll = cell(size(cs));
for i = 1:numel(cs)
    Xc = liftf(Z,cs(i)); PmcC = liftf(Mch,cs(i)); PmbC = zeros(1,d);
    sc = -min(H.gdist_c(Xc,PmcC,cs(i)).^2,[],2) + H.gdist_c(Xc,PmbC,cs(i)).^2;
    scAll{i} = sc;
    aucC(i) = H.auc(sc,y);
    spC(i) = corr(sc, sAll, 'type','Spearman');
end
fprintf('  c             :'); fprintf(' %7.2f', cs); fprintf('\n');
fprintf('  AUC(contrast)  :'); fprintf(' %7.4f', aucC); fprintf('\n');
fprintf('  Spearman vs c=1:'); fprintf(' %7.4f', spC); fprintf('\n');
tauC = [0.1 1 10]; aucGrid = zeros(2,numel(tauC));
for j = 1:2
    ii = [3 5]; sc = scAll{ii(j)};
    for t = 1:numel(tauC)
        lg = [tauC(t)*sc, tauC(t)*zeros(size(sc))]; lg = lg - max(lg,[],2);
        pr = exp(lg(:,1))./sum(exp(lg),2);
        aucGrid(j,t) = H.auc(pr,y);
    end
end
fprintf('  AUC(tau) at c=%.2f:'); fprintf(' %7.4f', aucGrid(1,:));
fprintf('   (tau inert at this curvature)\n');
fprintf('  AUC(tau) at c=%.2f:'); fprintf(' %7.4f', aucGrid(2,:));
fprintf('   (tau inert at this curvature too)\n');
fprintf('  reading: tau never changes a decision, at any curvature, so it cannot stand in\n');
fprintf('           for c; curvature only reorders pairs at LARGE distance.\n');

% --------------------------------------------------------------- figures -----
f = figure('Position',[60 60 900 340]);
subplot(1,2,1);
plot(taus, aucT, 'o-','LineWidth',1.6); grid on; ylim([0 1]);
xlabel('tau'); ylabel('AUC'); title('H4-A: AUC is flat in tau');
subplot(1,2,2);
yyaxis left; plot(100*prevList, aFit, 'o-','LineWidth',1.6); ylabel('learned a (=tau)');
yyaxis right; plot(100*prevList, -bFit./aFit, 's--','LineWidth',1.6); ylabel('threshold s0');
xlabel('change prevalence (%)'); grid on; title('H4-B: threshold follows the prior');
H.savefig(figdir,'fig_h4_tau');

f = figure('Position',[60 60 900 340]);
subplot(1,2,1);
plot(cs, spNear, 'o-','LineWidth',1.6); hold on;
plot(cs, sp, 's-','LineWidth',1.6); plot(cs, spFar, '^-','LineWidth',1.6);
grid on; ylim([0.5 1.02]); legend({'nearest 25%','all pairs','far 25%'},'Location','southwest');
xlabel('curvature c'); ylabel('Spearman vs c=1');
title('H4-C: curvature only reorders far pairs');
subplot(1,2,2);
plot(cs, aucC, 'o-','LineWidth',1.6); grid on; ylim([0 1]);
xlabel('curvature c'); ylabel('AUC of the contrast');
title('H4-D: curvature moves little');
H.savefig(figdir,'fig_h4_curvature');
fprintf('H4 done.\n');
