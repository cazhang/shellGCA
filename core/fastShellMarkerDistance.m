function [cost,grad,H]=fastShellMarkerDistance( x,FVs,LM,corres_lm,Topo, Ev, Ef, Eo, boundaryedges,options)
% Note: x is deformed Shell, while FVs holds mesh estimate 
% LM is the sparse landmarks of input
% cost has two terms:
% lm term: minimize landmarks Euc. distance (correspondence given in corres_lm)
% shell term: minimize mesh editing cost (either bending or membrane)

%% input
% x: init value for average, only vertices is supplied
% FVs: mesh estimate
% LM: data landmarks
% Evs(nEdge): i and j
% Eos(nEdge): k and l
% Efs(nEdge): adjacent faces list
% allboundaryedges: boundary edges
% Topo: similar to FV.faces, which is nface x 3 mat
%% output
% f: cost (1,1)
% g: gradient (nvertsx3, 1)
% H: hessian (nvertsx3, nvertsx3)
if ~isfield(options, 'mu')
    options.mu = 1;
end
if ~isfield(options, 'lambda')
    options.lambda = 1;
end
if ~isfield(options, 'eta')
    options.eta = 1;
end

% landmarks weights
alpha = options.alpha;
nmesh = length(FVs);
if ~isfield(options, 'datweights')
    datweights(1:nmesh) = 1;
else
    datweights = options.datweights;
end
nverts = length(x)/3;
if isempty(corres_lm)
    corres_lm = [1:nverts];
end

% add a step to remove invalid markers, i.e. all zeros
sum_lm = sum(LM.^2, 2);
valid_lm_id = sum_lm > 0;
corres_lm = corres_lm(valid_lm_id);
LM = LM(valid_lm_id, :);

v = reshape(x, nverts, 3);
FV_def.vertices = v;
FV_def.faces = Topo;
model_LM = FV_def.vertices(corres_lm, :);
% add shell cost, grad, and hess
%[ cost,grad,H ] = fastShellGradAndHessDef( x,FVs,Ev,Eo,Ef,mu,lambda,eta,datweights,boundaryedges );
[cost,grad,H]=geomShellGradDef( x, FVs, Topo, Ev, Ef, Eo, boundaryedges,options);
% add marker cost, grad and hess
% cost
diff_LM = model_LM - LM;
cost_lm = sum(sum(diff_LM.^2, 2)).*datweights(1) .* alpha;
cost = cost + cost_lm;
% grad
grad = reshape(grad, nverts, 3);
tmps = 2*datweights(1)*alpha.*(model_LM - LM);
grad(corres_lm,:) = grad(corres_lm,:) + tmps;
grad = grad(:);
% hess
cont = 2*datweights(1)*alpha;
for i=1:length(corres_lm)
    vid = corres_lm(i);
    hess = cont.*eye(3);
    H = addToH(H, hess, vid, vid, nverts);
end
end

function mat = addToH(mat, tmp, r, c, nverts)
xr = r; xc = c;
yr = r+nverts; yc = c+nverts;
zr = r+2*nverts; zc = c+2*nverts;
mat(xr,xc) = mat(xr,xc) + tmp(1,1);
mat(yr,yc) = mat(yr,yc) + tmp(2,2);
mat(zr,zc) = mat(zr,zc) + tmp(3,3);
mat(xr,yc) = mat(xr,yc) + tmp(1,2);
mat(xr,zc) = mat(xr,zc) + tmp(1,3);
mat(yr,xc) = mat(yr,xc) + tmp(2,1);
mat(yr,zc) = mat(yr,zc) + tmp(2,3);
mat(zr,xc) = mat(zr,xc) + tmp(3,1);
mat(zr,yc) = mat(zr,yc) + tmp(3,2);
end

