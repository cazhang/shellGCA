function [cost, grad] = objDualCaculus2mPlus1(lambda_in, p, qs, Topology,Ev,Ef,Eo,boundaryedges,...
    eta,opt,id)
% optimise all lambdas together, using explicit constraint, i.e.
% \sum_{i=0}^{2n} \lambda_i = 1.
% INPUT: qs: must have reflections
% p: shape to recon
% lambda_0: vector of size 2n+1, [1 0 0 ... 0]
% step 1: given lambda_0, get the shape q with nonlinear optimisation
if isfield(opt, 'useRedu')
    useRedu = opt.useRedu;
else
    useRedu = false;
end

if isfield(opt, 'useGlobal')
    useGlobal = opt.useGlobal;
else
    useGlobal = false;
end

global q0;

N = size(p.vertices, 1)*3;
if useRedu % make up lambda_0_0
    lambda_0 = [0 lambda_in];
    lambda_0(1) = 1 - sum(lambda_0(2:end));
else
    lambda_0 = lambda_in;
end
n = length(lambda_0);

option0 = [];
option0.datweights = lambda_0;
option0.eta = eta;
if useGlobal && id > 0
    option0.FVinit = q0;
end
option0.regHess = false;
option0.useLagrange = true;
option0.useMem = true;

%disp(q0.vertices(1,:));
q = MultiResElasticAv( qs,Topology,Ev,Ef,Eo,boundaryedges, option0 );
if useGlobal && id > 0
    q0 = q;
end
[cost, ~] = geomDSD( p,q,Topology,Ev,Ef,eta,1 );
% gradient

% step 2: given q, compute mu with lsqlin using eq. 35. 
% Cx-d = 0, C = partial_q (G)', d = - partial_q (J)', anx x = mu'
% get C
x = q.vertices(:);
option1 = [];
option1.eta = eta;
option1.datweights = lambda_0;
%[~,~,C]=shellGradDeformed( x, qs, Topology, Ev, Ef, Eo, boundaryedges,option1);
[~,~,C]=geomShellGradDef( x, qs, Topology, Ev, Ef, Eo, boundaryedges,option1);
% get d
fv = cell(1,1);
fv{1}=p;
option2 = [];
option2.eta = eta;
%[~,d,~]=shellGradDeformed( x, fv, Topology, Ev, Ef, Eo, boundaryedges,option2);
[~,d]=geomShellGradDef( x, fv, Topology, Ev, Ef, Eo, boundaryedges,option2);
d = -1.*d;
% compute mu
mu = lsqlin(C,d,[],[]);
% step 3: given mu, compute update of lambda_delta using eq. 37. 
% assemble partial_lambda (G)
if useRedu
    G = zeros(N,n-1);
    for i=1:n-1
        fv{1} = qs{i+1};
        %[~,G(:,i),~]=shellGradDeformed( x, fv, Topology, Ev, Ef, Eo, boundaryedges,option2);
        [~,G(:,i)]=geomShellGradDef( x, fv, Topology, Ev, Ef, Eo, boundaryedges,option2);
        
    end
else  
    G = zeros(N, n);
    for i=1:n
        fv{1} = qs{i};
        %[~,G(:,i),~]=shellGradDeformed( x, fv, Topology, Ev, Ef, Eo, boundaryedges,option2);
        [~,G(:,i)]=geomShellGradDef( x, fv, Topology, Ev, Ef, Eo, boundaryedges,option2);
    end
end

grad = mu'*G;
grad = grad';
%disp(grad');

end