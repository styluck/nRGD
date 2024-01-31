function [Xk, outs] = nmRGD(Xk, func, M, opts, varargin)
%[Xnew,iter,objf] = nmRGD(Xk, func, opts, varargin)
% non-monotone Riemannian gradient descent method
%  with BB step size as the initial step size
%   min F(x), s.t., x in M
%
% Input:
%           x --- initial guess
%         fun --- objective function and its gradient:
%                 [F, G] = fun(X,  data1, data2)
%                 F, G are the objective function value and gradient, repectively
%                 data1, data2 are addtional data, and can be more
%                 Calling syntax:
%                   [X, out]= OptStiefelGBB(X0, @fun, opts, data1, data2);
%           M --- manifold
%
%        opts --- option structure with fields:
%                 record = 0, no print out
%                 mxitr       max number of iterations
%                 xtol        stop control for ||X_k - X_{k-1}||
%                 gtol        stop control for the projected gradient
%                 ftol        stop control for |F_k - F_{k-1}|/(1+|F_{k-1}|)
%                             usually, max{xtol, gtol} > ftol
%                 tau         initial step size
%        rhols, eta, nt       parameters in line search
%
% Output:
%           x --- solution
%         out --- output information

% termination rule
if ~isfield(opts, 'gtol');      opts.gtol = 1e-3; end
if ~isfield(opts, 'ftol');      opts.ftol = 1e-6; end
if ~isfield(opts, 'xtol');      opts.xtol = 1e-6; end

% parameters for control the linear approximation in line search,
if ~isfield(opts, 'tau');      opts.tau  = 1.0e-2; end % stepsize
if ~isfield(opts, 'mintau');   opts.mintau  = 1e-12; end % mini-stepsize
if ~isfield(opts, 'maxtau');   opts.maxtau  = opts.tau; end % max-stepsize
if ~isfield(opts, 'rhols');    opts.rhols  = 1e-4; end
if ~isfield(opts, 'eta');      opts.eta  = 0.1; end
if ~isfield(opts, 'gamma');    opts.gamma  = 0.85; end
if ~isfield(opts, 'mxitr');    opts.mxitr  = 100; end
if ~isfield(opts, 'record');   opts.record = 0; end
if ~isfield(opts, 'mu');       opts.mu   = 1; end

%%
%-------------------------------------------------------------------------------
% copy parameters
[n,r] = size(Xk);
sqtn = sqrt(n);

gtol   = opts.gtol*sqtn; %
ftol   = opts.ftol*sqtn; %
xtol   = opts.xtol*sqtn; %
rhols  = opts.rhols;
eta    = opts.eta;
tau    = opts.tau; % initial step size
mintau = opts.mintau;
maxtau = opts.maxtau;
record = opts.record; %
kappa  = 5;


%% ****************************************************************
% initial function value and gradient
[Fold, ge] = feval(func, Xk, varargin{:});
grad = M.egrad2rgrad(Xk, ge);


Fk_list = zeros(opts.mxitr,1);

outs.nfe = 0;

if (record)
    fprintf('\n itr   nls    \t\t  fval  \t   ||grad(x)||    time');
    tstart = tic;
end

Cval = Fold;

for itr = 1:opts.mxitr

    nls = 1;

    while 1   %% to search a desired step-size

        Vk = M.lincomb(Xk, -tau, grad);

        normVk = M.norm(Xk, Vk);
        sqnormVk = normVk^2;

        Xnew = M.retr(Xk,Vk);
        [Fnew, ge_new] = feval(func, Xnew, varargin{:});

        if (Fnew<= Cval-(0.5/tau)*rhols*sqnormVk)||(nls>=5)
            break;
        end

        tau = eta*tau;

        nls = nls + 1;
    end
    outs.nfe = outs.nfe + nls;

    Fk_list(itr) = Fnew;

    % update the direction vector
    grad_new = M.egrad2rgrad(Xnew, ge_new);
    norm_grad = M.norm(Xnew,grad_new);

    if norm_grad <= gtol % && Fnew<=Fold && dist1 < xtol
        break;

    elseif (norm_grad <= 5*gtol && itr>=5 ...
            && abs(Fnew-Fold)/(1+abs(Fnew))<=ftol) 
        break;
    end

    %% *************** to estimate the step-size via BB *****************

    DetaX = Xnew - Xk;  DetaY = grad_new - grad;

    DetaXY = abs(sum(dot(DetaX,DetaY)));

    tau1 = norm(DetaX,'fro')^2/DetaXY;

    tau2 = DetaXY/norm(DetaY,'fro')^2;

    tau = max(min(min(tau1, tau2),maxtau), mintau);

    % recorder
    if (record) && rem(itr,10) == 0
        ttime = toc(tstart);
        fprintf('\n %4d  %3d     %5.4e   %6.2e    %2.1f',...
            itr, nls, Fnew, norm_grad, ttime);
    end

    Xk = Xnew;  grad = grad_new;

    if itr <= kappa % non-monotone line search
        Cval = max(Fk_list(1:itr));
    else
        Cval = max(Fk_list(itr-kappa+1:itr));
    end

    gtol = max(0.99*gtol,1.0e-3);
end

% recorder
if (record)
    ttime = toc(tstart);
    fprintf('\n %4d  %3d     %5.4e   %6.2e    %2.1f',...
        itr, nls, Fnew, norm_grad, ttime);
end
outs.itr = itr;
outs.fval = Fnew;
outs.nrmG = norm_grad;
%-------------------------------------------------------------------------------

end


% [EOF]