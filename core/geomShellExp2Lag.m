function [f, g] = geomShellExp2Lag(x,FV_ref,FV_vari,Topo,Ev,Ef,Eo,boundaryedges,options)
% the function computes the cost and gradient to optimise Exp. shell
% FV_ref: S_0, FV_vari: S_1, and x is S_2
% x is a nverts+6 x 1 vector
% d2 W[S_0, S_1] * w1 = - d1 W[S1, S2] * w2
% new added: Lagrange option
if ~isfield(options, 'mu')
    options.mu = 1;
end
if ~isfield(options, 'lambda')
    options.lambda = 1;
end
if ~isfield(options, 'eta')
    options.eta = 0.0001;
end
if ~isfield(options, 'firstDerivWRTDef')
    options.firstDerivWRTDef = true;
end
if ~isfield(options, 'regHess')
    options.regHess = true;   
end
if ~isfield(options, 'eps')
    options.eps = 1e-4;
end
if ~isfield(options, 'setBrd')
    options.setBrd = false;
end

nverts = size(FV_ref.vertices, 1);
xvec = x(1:nverts*3);
FV_exp = cell(1,1);
FV_tmp = cell(1,1);
FV_exp{1}.vertices = reshape(xvec, nverts, 3);
FV_exp{1}.faces = FV_ref.faces;
FV_tmp{1}.vertices = FV_ref.vertices;
FV_tmp{1}.faces = FV_ref.faces;

xlag = x(nverts*3+1:end);
if ~isfield(FV_exp{1}, 'de')
    FV_exp{1} = precompute(FV_exp{1},Topo,Ev,Ef,boundaryedges);
end
if ~isfield(FV_tmp{1}, 'de')
    FV_tmp{1} = precompute(FV_tmp{1},Topo,Ev,Ef,boundaryedges);
end

% F[S] = d_2 W[S_0, S_1] + d_1 W[S_1, S]
x1 = FV_vari.vertices(:);

% x1 --> S1, FV_tmp --> S0
[~,g1] = geomShellGradDef(x1,FV_tmp,Topo,Ev,Ef,Eo,boundaryedges,options);
[~,g2] = geomShellGradUnd(x1,FV_exp,Topo,Ev,Ef,Eo,boundaryedges,options);
%[~,g2] = shellGradUndeformed(x1,FV_exp,Topo,Ev,Ef,Eo,boundaryedges,options);

f = options.expweights(1).*g1 + options.expweights(2).*g2;
% set zero
if options.setBrd  
    f = reshape(f, nverts, 3);
    if ~isfield(options, 'brdPts')
        bverts = Ev(boundaryedges,:);
        bverts = unique(bverts(:));
    else
        bverts = options.brdPts;
    end
    f(bverts,:)=0;
    f = f(:);
end
f = addLagrangeConstraintCost(xlag, f, FV_vari, FV_exp{1});

% d F[S] = d_2 d_1 W[S1,S]
if nargout > 1
    g = geomMixedHessianShell(FV_vari,FV_exp{1},Topo,Ev,Ef,Eo,boundaryedges,options);
    %g2 = getMixedHessianShell(FV_vari,FV_exp{1},Topo,Ev,Ef,Eo,boundaryedges,options);
    g = options.expweights(2).*g;
    if options.setBrd
        if ~isfield(options, 'brdPts')
            bverts = Ev(boundaryedges,:);
            bverts = unique(bverts(:));
        else
            bverts = options.brdPts;
        end
        for i=1:length(bverts)
            vid = bverts(i);
            hess = eye(3);
            g = setToH(g, hess, vid, vid, nverts);
        end
    end
    g = addLagrangeConstraintGrad(g, FV_vari, FV_exp{1});
end
end

function M = getMassMat(FV)
nverts = size(FV.vertices, 1);
M = zeros(nverts, nverts);
for i=1:nverts  
    ind =  sum((FV.faces(:,:)==i),2);
    ind = logical(ind);   
    M(i,i) = sum(FV.at(ind));   
end
%M = speye(nverts);
end

function Lg = addLagrangeConstraintGrad(g, FV_ref, FV_def)
% f - original cost in 3n
% FV_ref - undeformed shell
% FV_exp - deformed shell
% make lumped matrix M
nverts = size(g,1)/3;
if nverts ~= size(FV_ref.vertices, 1)
    error('cost vector dimension wrong');
end
refV = FV_ref.vertices;
defV = FV_def.vertices;
M = getMassMat(FV_ref);
I = ones(nverts, 1);
Zmat = zeros(nverts, 1);
Lg = g;

% right submatrix, 3n * 6
submat = [M*I,Zmat,Zmat, M*refV(:,2),Zmat,-1.*(M*refV(:,3));
    Zmat,M*I,Zmat, -1.*(M*refV(:,1)), M*refV(:,3), Zmat;
    Zmat,Zmat, M*I, Zmat, -1.*(M*refV(:,2)), M*refV(:,1)];

Lg = [Lg submat; submat' zeros(6)];

end


function Lf = addLagrangeConstraintCost(xlag, f, FV_ref, FV_def)
% f - original cost in 3n
% FV_ref - undeformed shell
% FV_exp - deformed shell
% make lumped matrix M
nverts = length(f) / 3;
if nverts ~= size(FV_ref.vertices, 1)
    error('cost vector dimension wrong');
end
refV = FV_ref.vertices;
defV = FV_def.vertices;
M = getMassMat(FV_ref);
I = ones(nverts, 1);
Lf = f;
% add original functionals
Lf(1:nverts, 1) = Lf(1:nverts,1) + xlag(1).*(M*I) + xlag(4).*(M*refV(:,2)) - ...
    xlag(6).*(M*refV(:,3));
Lf(nverts+1:nverts*2, 1) = Lf(nverts+1:nverts*2,1) + xlag(2).*(M*I) + ...
   xlag(5).*(M*refV(:,3)) - xlag(4).*(M*refV(:,1));
Lf(nverts*2+1:nverts*3, 1) = Lf(nverts*2+1:nverts*3,1) + xlag(3).*(M*I)...
    + xlag(6).*(M*refV(:,1)) - xlag(5).*(M*refV(:,2));

% add lagrange functionals
trans_func(1) = I'*(M*(defV(:,1)-refV(:,1)));
trans_func(2) = I'*(M*(defV(:,2)-refV(:,2)));
trans_func(3) = I'*(M*(defV(:,3)-refV(:,3)));
rot_func(1) = refV(:,2)'*(M*(defV(:,1)-refV(:,1))) - refV(:,1)'*(M*(defV(:,2)-refV(:,2)));
rot_func(2) = refV(:,3)'*(M*(defV(:,2)-refV(:,2))) - refV(:,2)'*(M*(defV(:,3)-refV(:,3)));
rot_func(3) = refV(:,1)'*(M*(defV(:,3)-refV(:,3))) - refV(:,3)'*(M*(defV(:,1)-refV(:,1)));

Lf = [Lf; trans_func'; rot_func'];

end

function mat = setToH(mat, tmp, r, c, nverts)
xr = r; xc = c;
yr = r+nverts; yc = c+nverts;
zr = r+2*nverts; zc = c+2*nverts;
mat(xr,xc) = tmp(1,1);
mat(yr,yc) = tmp(2,2);
mat(zr,zc) = tmp(3,3);
mat(xr,yc) = tmp(1,2);
mat(xr,zc) = tmp(1,3);
mat(yr,xc) = tmp(2,1);
mat(yr,zc) = tmp(2,3);
mat(zr,xc) = tmp(3,1);
mat(zr,yc) = tmp(3,2);
end