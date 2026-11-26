function H = hyper_setup()
%HYPER_SETUP Handles for Poincare / Lorentz geometry + small stats helpers.
%   All maps act on ROW vectors (N x d). Curvature is -1 unless stated.
%   Poincare ball of radius 1: {x : ||x|| < 1}.
%
%   H.expmap0  tangent-at-origin -> ball          exp_0(v) = tanh(||v||) v/||v||
%   H.logmap0  ball -> tangent-at-origin          log_0(x) = atanh(||x||) x/||x||
%   H.mobius   Mobius addition x (+) y
%   H.cdist    Poincare cross distances  (N x K)
%   H.pdist    Poincare pairwise distances (N x N)
%   H.lift     direct Lorentz lift  z -> (sqrt(1+||z||^2), z)
%   H.lcdist   Lorentz cross distances from raw features (N x K)
%   H.gdist_c  Poincare cross distances with curvature -c
%   H.logreg   ridge logistic regression (IRLS), returns weights on [1 X]
%   H.auc      rank AUC
%   H.fisher   Fisher ratio of a 1-D score
%   H.embed    stress-minimising embedding, geom = 'hyp' | 'euc'
%   H.savefig  exportgraphics wrapper

H.expmap0 = @expmap0;
H.logmap0 = @logmap0;
H.pball   = @pball_image;
H.mobius  = @mobius_add;
H.cdist   = @geo_cdist;
H.pdist   = @geo_pdist;
H.lift    = @lorentz_lift;
H.lcdist  = @lorentz_cdist;
H.gdist_c = @geo_cdist_c;
H.logreg  = @logreg_irls;
H.auc     = @auc_rank;
H.fisher  = @fisher_ratio;
H.embed   = @stress_embed;
H.savefig = @save_fig;
end

% ---------------------------------------------------------------- geometry --
function X = expmap0(V)
% exp_0 with the 2/(1-r^2) metric: d(0,exp_0(v)) = ||v||.
% Equals the Poincare image of the direct Lorentz lift: v/(1+sqrt(1+||v||^2)).
n = sqrt(sum(V.^2,2));
f = ones(size(n));
m = n > 1e-12;
f(m) = tanh(0.5*n(m))./n(m);
X = V.*f;
end

function V = logmap0(X)
n = sqrt(sum(X.^2,2));
n = min(n, 1-1e-9);
f = ones(size(n));
m = n > 1e-12;
f(m) = 2*atanh(n(m))./n(m);
V = X.*f;
end

function X = pball_image(Z)
% Poincare coordinates of the DIRECT Lorentz lift of Z: (t,s)=(sqrt(1+||Z||^2),Z).
% Then d_poincare(0,pball_image(Z)) = asinh(||Z||), i.e. radius depends on ||Z|| only.
X = Z./(1 + sqrt(1 + sum(Z.^2,2)));
end

function Z = mobius_add(X,Y)
x2 = sum(X.^2,2); y2 = sum(Y.^2,2); xy = sum(X.*Y,2);
num = (1 + 2*xy + y2).*X + (1 - x2).*Y;
den = 1 + 2*xy + x2.*y2;
Z = num./max(den,1e-12);
end

function D = geo_cdist(X,P)
x2 = sum(X.^2,2); p2 = sum(P.^2,2);
S  = max(x2 + p2' - 2*(X*P'), 0);
den = max((1-x2)*(1-p2'), 1e-12);
K = 1 + 2*S./den;
D = acosh(max(K,1));
end

function D = geo_pdist(X)
D = geo_cdist(X,X);
end

function T = lorentz_lift(Z)
T = [sqrt(1+sum(Z.^2,2)), Z];
end

function D = lorentz_cdist(Z,P)
t1 = sqrt(1+sum(Z.^2,2)); t2 = sqrt(1+sum(P.^2,2));
inner = -(t1*t2') + Z*P';
D = acosh(max(-inner,1));
end

function D = geo_cdist_c(X,P,c)
% Poincare ball of curvature -c.
x2 = sum(X.^2,2); p2 = sum(P.^2,2);
S  = max(x2 + p2' - 2*(X*P'), 0);
den = max((1-c*x2)*(1-c*p2'), 1e-12);
K = 1 + 2*c*S./den;
D = acosh(max(K,1))/sqrt(c);
end

% ------------------------------------------------------------------- stats --
function w = logreg_irls(X, y, lambda)
if nargin < 3, lambda = 1e-2; end
[n,d] = size(X);
Xa = [ones(n,1), X];
w  = zeros(d+1,1);
R  = lambda*eye(d+1); R(1,1) = 0;
for it = 1:40
    z = Xa*w;
    p = 1./(1+exp(-z));
    g = (Xa'*(p-y))/n + (R*w)/n;
    Wv = max(p.*(1-p), 1e-8);
    Hm = (Xa'*(Xa.*Wv))/n + R/n + 1e-9*eye(d+1);
    step = Hm\g;
    sn = norm(step);
    if sn > 1, step = step/sn; end
    w = w - step;
    if sn < 1e-9, break; end
end
end

function a = auc_rank(s, y)
y = y(:); s = s(:);
[~, idx] = sort(s);
ys = y(idx);
r  = avg_rank(s(idx));
n1 = sum(ys==1); n0 = numel(ys)-n1;
if n1==0 || n0==0, a = NaN; return; end
a = (sum(r(ys==1)) - n1*(n1+1)/2)/(n1*n0);
end

function r = avg_rank(s)
n = numel(s); r = zeros(n,1); i = 1;
while i <= n
    j = i;
    while j < n && s(j+1) == s(i), j = j+1; end
    r(i:j) = (i+j)/2;
    i = j+1;
end
end

function f = fisher_ratio(s, y)
s = s(:); y = y(:);
m1 = mean(s(y==1)); m0 = mean(s(y==0));
v1 = var(s(y==1));  v0 = var(s(y==0));
f  = (m1-m0)^2/max(v1+v0, eps);
end

% --------------------------------------------------------------- embedding --
function [X, stress1] = stress_embed(Dt, dim, geom, iters, restarts, X0)
% Minimise stress = ||alpha*D_emb - Dt||_F / ||Dt||_F with normalised-gradient
% descent; for 'hyp' the step is a geodesic step (expmap0(logmap0(X) - eta*g)).
n  = size(Dt,1);
if nargin < 4, iters = 800; end
if nargin < 5, restarts = 1; end
if nargin < 6, X0 = []; end
Dt = Dt/mean(Dt(tril(true(n),-1)));
bestS = inf; Xbest = zeros(n,dim);
for r = 1:restarts
    rng(1000+r);
    if ~isempty(X0)
        X = X0;
    elseif strcmp(geom,'hyp')
        X = expmap0(0.05*randn(n,dim));
    else
        X = 0.10*randn(n,dim);
    end
    for it = 1:iters
        G  = grad_stress(X, Dt, geom);
        gn = norm(G,'fro');
        if gn < 1e-12, break; end
        eta = 0.06*(1 - 0.7*it/iters);
        if strcmp(geom,'hyp')
            X = expmap0(logmap0(X) - eta*G/gn);
        else
            X = X - eta*G/gn;
        end
    end
    D = emb_dist(X, geom);
    al = sum(D(:).*Dt(:))/max(sum(D(:).^2),eps);
    S  = norm(al*D - Dt,'fro')/norm(Dt,'fro');
    if S < bestS, bestS = S; Xbest = X; end
end
X = Xbest; stress1 = bestS;
end

function D = emb_dist(X, geom)
if strcmp(geom,'hyp')
    D = geo_pdist(X);
else
    x2 = sum(X.^2,2);
    D = sqrt(max(x2 + x2' - 2*(X*X'),0));
end
end

function G = grad_stress(X, Dt, geom)
n = size(X,1);
x2 = sum(X.^2,2);
S  = max(x2 + x2' - 2*(X*X'),0);
W  = ones(n) - eye(n);
if strcmp(geom,'euc')
    D = sqrt(S);
    C = (D - Dt).*W./max(D,1e-9);
    C(S < 1e-12) = 0;
    G = 2*((sum(C,2)).*X - C*X);
else
    a = max(1 - x2, 1e-6);
    A = a*a';
    K = 1 + 2*S./max(A,1e-12);
    D = acosh(max(K,1));
    C = (D - Dt).*W./sqrt(max(K.^2-1,1e-12));
    C(S < 1e-12) = 0;
    Q = 4*C./max(A,1e-12);
    G = (sum(Q,2)).*X - Q*X;
    R = 4*C.*S./max(A,1e-12);
    G = G + (sum(R,2)./a.^2).*X;
end
end

% ------------------------------------------------------------------ figures --
function save_fig(figdir, name)
if ~exist(figdir,'dir'), mkdir(figdir); end
f = gcf; set(f,'Color','w');
exportgraphics(f, fullfile(figdir,[name '.png']), 'Resolution',150);
end
