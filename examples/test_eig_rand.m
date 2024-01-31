function test_eig_rand

%-------------------------------------------------------------
% A demo of solving
%   min f(X), s.t., X'*X = I, where X is an n-by-p matrix
%
%  This demo solves the eigenvalue problem by letting
%  f(X) = -0.5*Tr(X'*A*X);
%
%  The result is compared to the MATLAB function "eigs",
%  which call ARPACK (FORTRAN) to find leading eigenvalues.
%
%  Our solver can be faster when n is large and p is small
%
%  The advantage of our solver is not obvious in this demo 
%  since our solver is a general MATLAB code while ARPACK implemented
%  many tricks for computing the eigenvalues.
% -------------------------------------
%
% Reference: 
%  Z. Wen and W. Yin
%  A feasible method for optimization with orthogonality constraints
%
% Author: Zaiwen Wen
%   Version 0.1 .... 2010/10
%   Version 0.5 .... 2013/10
%-------------------------------------------------------------

clc

% seed = 2010;
% fprintf('seed: %d\n', seed);
% if exist('RandStream','file')
%    RandStream.setDefaultStream(RandStream('mt19937ar','seed',seed));
% else
%    rand('state',seed); randn('state',seed^2);
% end

% nlist = [500, 1000, 2000, 3000, 4000, 5000];
nlist = [1000];
nlen = length(nlist);

perf = zeros(10,nlen);

for dn = 1:nlen
    n = nlist(dn);
    fprintf('matrix size: %d\n', nlist(dn));
    
    A = randn(n); A = A'*A;
    k = 6;
    opteig.issym = 1;
    nAx = 0;
    
    % --- MATLAB eigs ---
    tic; [V, D] = eigs(@funAX, n, k, 'la',opteig); teig = toc; D = diag(D); feig = sum(D(1:k));
    
    fprintf('\neigs: obj val %7.6e, cpu %f, #func eval %d\n', feig, teig, nAx);
    feasi = norm(V'*V - eye(k), 'fro');
    
    % --- our solver ---
    % X0 = eye(n,k);
    X0 = randn(n,k);    X0 = orth(X0);
    
    opts.record = 0;
    opts.mxitr  = 1000;
    opts.xtol = 1e-5;
    opts.gtol = 1e-5;
    opts.ftol = 1e-8;
    out.tau = 1e-3;
    %opts.nt = 1;
    
    %profile on;
    M = stiefelfactory(n,k);
    
    %nmRGD
    tic;
    [X,  out] = nmRGD(X0, @funeigsym, M, opts, A); 
    tsolve = toc;
    
    %Wen2013
    tic; 
    [X1, out1]= OptStiefelGBB(X0, @funeigsym, opts, A); 
    tsolve1 = toc;
    
    %RGBB
%     tic; 
    [X2, ~,out2]= RGBB(X0, @funeigsym, M, opts, A); 
%     [X2, ~,out2]= arnt(X0, @funeigsym, M, opts, A); 
    tsolve2 = toc;

%     [X, out] = landing_cal(X0, @funeigsym, opts, A);
    
    % profile viewer;
    out.fval = -2*out.fval;
    fprintf('nnRGD: obj val %7.6e, cpu %f, #func eval %d, itr %d, |XT*X-I| %3.2e\n', ...
             out.fval, tsolve, out.nfe, out.itr, norm(X'*X - eye(k), 'fro'));
%     fprintf('relative difference between two obj vals: %3.2e\n',err);
    fprintf('Wen2013: obj val %7.6e, cpu %f, #func eval %d, itr %d, |XT*X-I| %3.2e\n', ...
             -2*out1.fval, tsolve1, out1.nfe, out1.itr, norm(X1'*X1 - eye(k), 'fro'));
    fprintf('RGBB: obj val %7.6e, cpu %f, #func eval %d, itr %d, |XT*X-I| %3.2e\n', ...
             out2.fval, tsolve2, out2.nfe, out2.iter, norm(X2'*X2 - eye(k), 'fro'));
    
end
% save('results/eig_rand_perf', 'perf', 'nlist');


    function AX = funAX(X)
        nAx = nAx + 1;
        AX = A*X;
        %fprintf('iter: %d, size: (%d, %d)\n', nAx, size(X));
    end

    function [F, G] = funeigsym(X,  A)
        
        G = -(A*X);
        %F = 0.5*sum(sum( G.*X ));
        F = 0.5*sum(dot(G,X,1));
        % F = sum(sum( G.*X ));
        % G = 2*G;
        
    end


end
