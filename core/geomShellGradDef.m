function [ cost, grad, H ] = geomShellGradDef( x, FVs, Topo, Ev, Ef, Eo, boundaryedges,options )
%compute the cost and gradient wrt. deformed shell-FV_def.
%   x: initial value of the deformed shell
%   FVs: a set of undeformed shells
v = reshape(x, length(x)/3, 3);

nverts = size(v, 1);
if nargout > 1
    g = zeros(nverts, 3);
end
if nargout > 2
    rows=[]; cols=[]; vals=[];
end
if isfield(options, 'mu')
    mu = options.mu;
else
    mu = 1;
end
if isfield(options, 'lambda')
    lambda = options.lambda;
else
    lambda = 1;
end
if isfield(options, 'eta')
    eta = options.eta;
else
    error('eta must be set');
end

nmesh = length(FVs);
if ~isfield(options, 'datweights')
    datweights(1:nmesh) = 1;
else
    datweights = options.datweights;
end

% cost term
f = zeros(1, nmesh);
FV_def.vertices = v;
FV_def.faces = Topo;
% compute intermediate quantities
if ~isfield(FV_def, 'de')
    FV_def = precompute(FV_def,Topo,Ev,Ef,boundaryedges);
end
for m = 1:nmesh
    if ~isfield(FVs{m}, 'de')
        FVs{m} = precompute(FVs{m},Topo,Ev,Ef,boundaryedges);
    end
end
% make f
for m=1:nmesh
    [f(m)] = geomDSD(FVs{m}, FV_def, Topo, Ev, Ef,eta,1 );
end
cost = sum(f);


% make gradient
if nargout > 1
    % membrane energy
    for m=1:nmesh
        
        % membrane term
        %[g1, loop_out] = getMemGradLoop(FVs{m}, FV_def, Topo, mu, lambda,datweights(m));
        g1_fast = getMemGradVec(FVs{m}, FV_def, Topo, mu, lambda,datweights(m));

        %res = norm(g1(:) - g1_fast(:));
        %disp(res);
        % dihedral angle term
        g3_fast = getAngleGradVec(FVs{m}, FV_def, Ev, Eo, boundaryedges,eta,datweights(m));
        
        g = g + g1_fast + g3_fast;       
    end 
end

if nargout > 2
    %% TODO: get Hessian for deformed shell
    % mem term
    for m=1:nmesh
               
        [rows,cols,vals] = getMemHessVec(rows,cols,vals,FVs{m}, FV_def, Topo, mu, lambda,datweights(m));
        %H = H + H1_fast;
        
        % bending part: dihedral angle    
        %H3 = getAngleHessLoop(FVs{m}, FV_def, Ev, Eo, eta, boundaryedges, datweights(m));
        %H = H + H3;    
        [rows,cols,vals] = getAngleHessVec(rows,cols,vals,FVs{m},FV_def,Ev,Eo,eta,boundaryedges,datweights(m));
        
    end 
     
end

if nargout > 1 % gradient required
    grad = g(:);
end

if nargout > 2
    H = sparse(rows,cols,vals,3*nverts,3*nverts);
end
end

function mat = getCrossOp(a)

mat = zeros(3,3);
mat(1,2) = -a(3);
mat(1,3) = a(2);
mat(2,1) = a(3);
mat(2,3) = -a(1);
mat(3,1) = -a(2);
mat(3,2) = a(1);
end

function Mat = getCrossOpVec(X)
nverts = size(X, 1);
Mat = zeros(nverts,3,3);
Mat(:,1,2) = -X(:,3);
Mat(:,1,3) = X(:,2);
Mat(:,2,1) = X(:,3);
Mat(:,2,3) = -X(:,1);
Mat(:,3,1) = -X(:,2);
Mat(:,3,2) = X(:,1);
end

function m = getProjection(x)
m = eye(3);
temp = x'*x;
m = m - temp;
end

function M = getProjectionVec(X)
tmp = ones(size(X,1), 1);
M = outerProductVectorised(X,X);
M = -M;
M(:,1,1) = tmp + M(:,1,1);
M(:,2,2) = tmp + M(:,2,2);
M(:,3,3) = tmp + M(:,3,3);
end

function m = getReflection(x)
m = eye(3);
temp = x'*x;
m = m - 2.*temp;
end

function M = getReflectionVec(X)
M = outerProductVectorised(X,X);
M = bsxfun(@times,-2,M);
tmp = ones(size(X,1),1);
M(:,1,1) = tmp + M(:,1,1);
M(:,2,2) = tmp + M(:,2,2);
M(:,3,3) = tmp + M(:,3,3);
end

function N = getNormal(Pi, Pj, Pk)
e1 = Pk - Pi;
e3 = Pk - Pj;
N = cross(e1, e3, 2);
nLeng = sqrt(sum(N.^2, 2));
N = N./ nLeng;
end

function Ns = getNormalVec(Pis, Pjs, Pks)
e1s = Pks-Pis;
e3s = Pks-Pjs;
Ns = cross(e1s, e3s, 2);
nlength = sqrt(sum(Ns.^2, 2));
Ns = Ns ./ repmat(nlength, 1, 3);
end

function area = getArea(Pi, Pj, Pk)
e1 = Pk - Pi;
e3 = Pk - Pj;
N = cross(e1,e3,2);
area = sqrt(sum(N.^2, 2));
area = area ./ 2;
end
    
function areas = getAreaVec(Pis,Pjs,Pks)
e1s = Pks - Pis;
e3s = Pks - Pjs;
Ns = cross(e1s,e3s,2);
areas = sqrt(sum(Ns.^2, 2))./2;
end

function grad = getAreaGradK(Pi,Pj,Pk)
e = Pj-Pi;
normal = getNormal(Pi,Pj,Pk);
grad = cross(0.5*normal, e);
end
function Grad = getAreaGradKVec(Pis,Pjs,Pks)
es = Pjs-Pis;
normals = getNormalVec(Pis,Pjs,Pks);
Grad = cross(0.5.*normals, es, 2);
end

function hess = getHessAreaKK(Pi,Pj,Pk)
e = Pj-Pi;
eNormalized = e/norm(e);
gradAreaK = getAreaGradK(Pi,Pj,Pk);
hess = gradAreaK'*gradAreaK;
proj = getProjection(eNormalized);
hess = hess - 0.25*(e*e').*proj;
hess = hess .* (-1/getArea(Pi,Pj,Pk));
end
function Hess = getHessAreaKKVec(Pis,Pjs,Pks)
es = Pjs-Pis;
areas = getAreaVec(Pis,Pjs,Pks);
elens = sqrt(sum(es.^2, 2));
eNormalized = es./repmat(elens,1,3);
gradAreaKs = getAreaGradKVec(Pis,Pjs,Pks);
Hess = outerProductVectorised(gradAreaKs, gradAreaKs);
Proj = getProjectionVec(eNormalized);
%eLengthSqrs = diag(es*es');
eLengthSqrs = sum(es.*es,2);
Proj = bsxfun(@times,0.25.*eLengthSqrs, Proj);
Hess = Hess - Proj;
Hess = bsxfun(@times, -1./areas, Hess);
end


function hess = getHessAreaIK(Pi,Pj,Pk)
e = Pj-Pi;
d = Pk-Pj;
temp1 = getAreaGradK(Pj,Pk,Pi);
temp2 = getAreaGradK(Pi,Pj,Pk);
hess = temp1'*temp2;
hess = hess + 0.25.*(e'*d);
hess = hess - 0.25*(d*e').*eye(3);
hess = hess .* (-1/getArea(Pi,Pj,Pk));
auxMat = getCrossOp(getNormal(Pi,Pj,Pk));
hess = hess + 0.5.*auxMat;
end

function Hess = getHessAreaIKVec(Pis,Pjs,Pks)
es = Pjs-Pis;
ds = Pks-Pjs;
areas = getAreaVec(Pis,Pjs,Pks);
normals = getNormalVec(Pis,Pjs,Pks);
temp1 = getAreaGradKVec(Pjs,Pks,Pis);
temp2 = getAreaGradKVec(Pis,Pjs,Pks);
Hess = outerProductVectorised(temp1, temp2); % get temp1'*temp2
epd = outerProductVectorised(es, ds); % get e'*d
Hess = Hess + 0.25.*epd; % plus epd
%tmp = 0.25.*diag(ds*es'); % minus 0.25*d*e' on diag
tmp = 0.25.*sum(ds.*es,2); % minus 0.25*d*e' on diag
Hess(:,1,1) = Hess(:,1,1) - tmp;
Hess(:,2,2) = Hess(:,2,2) - tmp;
Hess(:,3,3) = Hess(:,3,3) - tmp;
Hess = bsxfun(@times,-1./areas,Hess);
auxMat = getCrossOpVec(normals);
Hess = Hess + 0.5.*auxMat;
end


function grad = getThetaGradK(Pi,Pj,Pk)
e = Pj - Pi;
normal = getNormal(Pi,Pj,Pk);
grad = (-0.5*norm(e) / getArea(Pi,Pj,Pk)) .* normal;

end

function Grad = getThetaGradKVec(Pis, Pjs, Pks)
es = Pjs-Pis;
normals = getNormalVec(Pis,Pjs,Pks);
areas = getAreaVec(Pis,Pjs,Pks);
elens = sqrt(sum(es.^2, 2));

Grad = normals;
tmps = -0.5.*elens ./ areas;
Grad = bsxfun(@times,tmps,Grad);
end

function grad = getThetaGradILeftPart(Pi,Pj,Pk)
e = Pj-Pi;
d = Pk-Pj;
grad = getThetaGradK(Pi,Pj,Pk);
grad = (e*d')/(e*e').* grad;
end

function grad = getThetaGradI(Pi,Pj,Pk,Pl)
grad = getThetaGradILeftPart(Pi,Pj,Pk);
grad = grad - getThetaGradILeftPart(Pi,Pj,Pl);
end

function grad = getThetaGradJLeftPart(Pi,Pj,Pk)
e = Pj-Pi;
a = Pi-Pk;
grad = getThetaGradK(Pi,Pj,Pk);
grad = ((a*e')/(e*e')).*grad;
end

function grad = getThetaGradJ(Pi,Pj,Pk,Pl)
grad = getThetaGradJLeftPart(Pi,Pj,Pk);
grad = grad - getThetaGradJLeftPart(Pi,Pj,Pl);
end

function Grad = getThetaGradILeftPartVec(Pis,Pjs,Pks)
es = Pjs-Pis;
ds = Pks-Pjs;
Grad = getThetaGradKVec(Pis,Pjs,Pks);
%tmps = diag(es*ds')./diag(es*es');
tmps = sum(es.*ds,2)./sum(es.*es,2);
Grad = bsxfun(@times,tmps,Grad);
end

function Grad = getThetaGradIVec(Pis,Pjs,Pks,Pls)
Grad = getThetaGradILeftPartVec(Pis,Pjs,Pks);
Grad = Grad - getThetaGradILeftPartVec(Pis,Pjs,Pls);
end

function Grad = getThetaGradJLeftPartVec(Pis,Pjs,Pks)
es = Pjs-Pis;
as = Pis-Pks;
Grad = getThetaGradKVec(Pis,Pjs,Pks);
%tmps = diag(as*es')./diag(es*es');
tmps = sum(as.*es,2)./sum(es.*es,2);
Grad = bsxfun(@times,tmps,Grad);
end

function Grad = getThetaGradJVec(Pis,Pjs,Pks,Pls)
Grad = getThetaGradJLeftPartVec(Pis,Pjs,Pks);
Grad = Grad - getThetaGradJLeftPartVec(Pis,Pjs,Pls);
end

% Second order derivative wrt. dihedral angle
function Hkk = getHessThetaKK(Pi, Pj, Pk)
areaSqr = getArea(Pi,Pj,Pk)^2;
e = Pj-Pi;
gradArea = getAreaGradK(Pi,Pj,Pk);
normal = getNormal(Pi,Pj,Pk);
mat1 = getCrossOp(e);
mat2 = gradArea'*normal;
Hkk = norm(e)/(4*areaSqr)*mat1 + norm(e)/areaSqr*mat2;
end

function Hkk = getHessThetaKKVec(Pis,Pjs,Pks)
areas = getAreaVec(Pis,Pjs,Pks);
areaSqrs = areas.^2;
es = Pjs-Pis;
elens = sqrt(sum(es.^2,2));
gradAreas = getAreaGradKVec(Pis,Pjs,Pks);
normals = getNormalVec(Pis,Pjs,Pks);
mat1 = getCrossOpVec(es);
mat2 = outerProductVectorised(gradAreas, normals);
mat1 = bsxfun(@times,elens./(4.*areaSqrs), mat1);
mat2 = bsxfun(@times,elens./areaSqrs, mat2);
Hkk = mat1 + mat2;
end

function Hik = getHessThetaIK(Pi,Pj,Pk)
area = getArea(Pi,Pj,Pk);
areaSqr = area^2;
e = Pj-Pi; 
d = Pk-Pj;
gradArea = getAreaGradK(Pj,Pk,Pi);
normal = getNormal(Pi,Pj,Pk);
mat1 = e'*normal;
mat2 = getCrossOp(d);
mat3 = (1/(2*area*norm(e))).*mat1 + ( norm(e)/(4*areaSqr)).*mat2;
Hik = mat3 + norm(e)/areaSqr .* gradArea'*normal;
end

function Hik = getHessThetaIKVec(Pis,Pjs,Pks)
areas = getAreaVec(Pis,Pjs,Pks);
areaSqrs = areas.^2;
es = Pjs-Pis;
elens = sqrt(sum(es.^2,2));
ds = Pks-Pjs;
gradArea = getAreaGradKVec(Pjs,Pks,Pis);
normals = getNormalVec(Pis,Pjs,Pks);
mat1 = outerProductVectorised(es, normals);
mat2 = getCrossOpVec(ds);
mat1 = bsxfun(@times, 1./(2.*areas.*elens), mat1);
mat2 = bsxfun(@times, elens./(4.*areaSqrs), mat2);
mat3 = mat1 + mat2;
mat1 = outerProductVectorised(gradArea, normals);
mat1 = bsxfun(@times, elens./areaSqrs, mat1);
Hik = mat3 + mat1;
end

function Hjk = getHessThetaJK(Pi,Pj,Pk)
area = getArea(Pi,Pj,Pk);
areaSqr = area^2;
e = Pi-Pj; 
a = Pi-Pk;
gradArea = getAreaGradK(Pk,Pi,Pj);
normal = getNormal(Pi,Pj,Pk);

mat1 = e'*normal;
mat2 = getCrossOp(a);
mat3 = (1/(2*area*norm(e))).*mat1 + ( norm(e)/(4*areaSqr)).*mat2;
Hjk = mat3 + norm(e)/areaSqr .* gradArea'*normal;
end

function Hjk = getHessThetaJKVec(Pis,Pjs,Pks)
areas = getAreaVec(Pis,Pjs,Pks);
areaSqrs = areas.^2;
es = Pis-Pjs;
elens = sqrt(sum(es.^2,2));
as = Pis-Pks;
gradArea = getAreaGradKVec(Pks,Pis,Pjs);
normals = getNormalVec(Pis,Pjs,Pks);
mat1 = outerProductVectorised(es, normals);
mat2 = getCrossOpVec(as);
mat1 = bsxfun(@times, 1./(2.*areas.*elens), mat1);
mat2 = bsxfun(@times, elens./(4.*areaSqrs), mat2);
mat3 = mat1 + mat2;
mat1 = outerProductVectorised(gradArea, normals);
mat1 = bsxfun(@times, elens./areaSqrs, mat1);
Hjk = mat3 + mat1;
end

function HiLeft = getHessThetaILeftPartI(Pi,Pj,Pk)
e = Pj-Pi;
d = Pk-Pj;
eNormalized = (Pj-Pi)/norm(Pj-Pi);
gradThetaK = getThetaGradK(Pi,Pj,Pk);
Refl = getReflection(eNormalized);
temp = d*Refl';
mat1 = temp'*gradThetaK;
mat2 = getHessThetaIK(Pi,Pj,Pk);
HiLeft = -1/(e*e').*mat1 + (d*e')/(e*e').*mat2;

end

function HjLeft = getHessThetaJLeftPartI(Pi,Pj,Pk)
e = Pj-Pi;
d = Pk-Pj;
eNormalized = (Pj-Pi)/norm(Pj-Pi);
gradThetaK = getThetaGradK(Pi,Pj,Pk);
Refl = getReflection(eNormalized);
temp = (d-e)*Refl';
mat1 = temp'*gradThetaK;
mat2 = getHessThetaJK(Pi,Pj,Pk);
HjLeft = 1/(e*e').*mat1 + (d*e')/(e*e').*mat2;

end

function Hii = getHessThetaII(Pi,Pj,Pk,Pl)
Hii = getHessThetaILeftPartI(Pi,Pj,Pk);
Hii = Hii - getHessThetaILeftPartI(Pi,Pj,Pl);
end

function HiLeft = getHessThetaILeftPartIVec(Pis,Pjs,Pks)
es = Pjs-Pis;
ds = Pks-Pjs;
elens = sqrt(sum(es.^2,2));
eNormalized = es./repmat(elens,1,3);
gradThetaK = getThetaGradKVec(Pis,Pjs,Pks);
Refl = getReflectionVec(eNormalized);
Refl = permute(Refl, [1 3 2]);
%temp = ds*Refl';
temp(:,1) = dotProductVectorised(squeeze(Refl(:,:,1)), ds);
temp(:,2) = dotProductVectorised(squeeze(Refl(:,:,2)), ds);
temp(:,3) = dotProductVectorised(squeeze(Refl(:,:,3)), ds);

mat1 = outerProductVectorised(temp, gradThetaK);
%mat1 = bsxfun(@times, -1./diag(es*es'), mat1);
mat1 = bsxfun(@times, -1./sum(es.*es,2), mat1);
mat2 = getHessThetaIKVec(Pis,Pjs,Pks);
%mat2 = bsxfun(@times, diag(ds*es')./diag(es*es'), mat2);
mat2 = bsxfun(@times, sum(ds.*es,2)./sum(es.*es,2), mat2);
HiLeft = mat1 + mat2;
end

function Hii = getHessThetaIIVec(Pis,Pjs,Pks,Pls)
Hii = getHessThetaILeftPartIVec(Pis,Pjs,Pks);
temp = getHessThetaILeftPartIVec(Pis,Pjs,Pls);
Hii = Hii - temp;
end

function Hji = getHessThetaJI(Pi,Pj,Pk,Pl)
% differenr quotient hack
%Hji = getHijDiffQuotient(Pi,Pj,Pk,Pl);
edge = Pj - Pi;
d = Pk - Pj;
c = Pj - Pl;
diff = d - edge;
sum = c + edge;
eLengthSqr = norm(edge)^2;
thetak = getThetaGradK(Pi,Pj,Pk);
thetal = getThetaGradK(Pj,Pi,Pl);
grad = (edge*d').*thetak - (edge*c').*thetal;
Hjk = getHessThetaJK(Pi,Pj,Pk);
Hjl = getHessThetaIK(Pj,Pi,Pl);
Hji = (edge*d'/eLengthSqr).* Hjk' - (edge*c' /eLengthSqr).*Hjl';
tensorProduct = grad'*edge;
Hji = Hji - (2/eLengthSqr^2).*tensorProduct;
tensorProduct = thetak'*diff;
Hji = Hji + (1/eLengthSqr).*tensorProduct;
tensorProduct = thetal'*sum;
Hji = Hji - (1/eLengthSqr).*tensorProduct;
end

function Hji = getHessThetaJIVec(Pis,Pjs,Pks,Pls)
edges = Pjs-Pis;
ds = Pks-Pjs;
cs = Pjs-Pls;
diffs = ds-edges;
sums = cs+edges;
eLengthSqr = sum(edges.^2, 2);
thetaks = getThetaGradKVec(Pis,Pjs,Pks);
thetals = getThetaGradKVec(Pjs,Pis,Pls);
%grad1 = bsxfun(@times, diag(edges*ds'), thetaks);
%grad2 = bsxfun(@times, diag(edges*cs'), thetals);
grad1 = bsxfun(@times, sum(edges.*ds,2), thetaks);
grad2 = bsxfun(@times, sum(edges.*cs,2), thetals);
grad = grad1 - grad2;
Hjk = getHessThetaJKVec(Pis,Pjs,Pks);
Hjl = getHessThetaIKVec(Pjs,Pis,Pls);
Hjk = permute(Hjk, [1 3 2]);
Hjl = permute(Hjl, [1 3 2]);
%Hji1 = bsxfun(@times, diag(edges*ds')./eLengthSqr, Hjk);
%Hji2 = bsxfun(@times, diag(edges*cs')./eLengthSqr, Hjl);
Hji1 = bsxfun(@times, sum(edges.*ds,2)./eLengthSqr, Hjk);
Hji2 = bsxfun(@times, sum(edges.*cs,2)./eLengthSqr, Hjl);
Hji = Hji1 - Hji2;
tp = outerProductVectorised(grad, edges);
tp = bsxfun(@times, 2./(eLengthSqr.^2), tp);
Hji = Hji - tp;
tp = outerProductVectorised(thetaks, diffs);
tp = bsxfun(@times, 1./(eLengthSqr), tp);
Hji = Hji + tp;
tp = outerProductVectorised(thetals, sums);
tp = bsxfun(@times, 1./(eLengthSqr), tp);
Hji = Hji - tp;
end

function mat = addToG(mat, tmp, i)
mat(i,1)=mat(i,1)+tmp(1);
mat(i,2)=mat(i,2)+tmp(2);
mat(i,3)=mat(i,3)+tmp(3);

end

function mat = addToH(mat, tmp, r, c, nverts)
if sum(double(isnan(tmp)))>0
    disp(tmp);
    msg = ([num2str(r),' ',num2str(c)]);
    error(msg);
    
end
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

function C = dotProductVectorised(A,B)
% A and B are N by 3, C is N by 1 containing dot products
C = A(:,1).*B(:,1)+A(:,2).*B(:,2)+A(:,3).*B(:,3);
end

function C = outerProductVectorised(A,B)
% A and B are N by 3, C is N by 3 by 3 containing outer products
C(:,1,1) = A(:,1).*B(:,1);
C(:,1,2) = A(:,1).*B(:,2);
C(:,1,3) = A(:,1).*B(:,3);
C(:,2,1) = A(:,2).*B(:,1);
C(:,2,2) = A(:,2).*B(:,2);
C(:,2,3) = A(:,2).*B(:,3);
C(:,3,1) = A(:,3).*B(:,1);
C(:,3,2) = A(:,3).*B(:,2);
C(:,3,3) = A(:,3).*B(:,3);
end

function [rows,cols,vals] = addToHVectorised(rows,cols,vals,newvals,r,c,nverts)
% newvals is N by 3 by 3
% r and c are N by 1
rows = [rows; r;          r;          r; ...
              r+nverts;   r+nverts;   r+nverts; ...
              r+2*nverts; r+2*nverts; r+2*nverts];
cols = [cols; c;          c+nverts;   c+2*nverts; ...
              c;          c+nverts;   c+2*nverts; ...
              c;          c+nverts;   c+2*nverts;];
vals = [vals; newvals(:,1,1); newvals(:,2,1); newvals(:,3,1); ...
              newvals(:,1,2); newvals(:,2,2); newvals(:,3,2); ...
              newvals(:,1,3); newvals(:,2,3); newvals(:,3,3)];
end

function [g1,loop_out] = getMemGradLoop(FV_ref, FV_def, Topo, mu, lambda, datweight)
nverts = size(FV_ref.vertices, 1);
nFace = size(FV_ref.faces, 1);
g1 = zeros(nverts, 3);
muHalf = mu / 2;
lambdaQuarter = lambda / 4;
muHalfPlusLambdaQuarter = muHalf + lambdaQuarter;
for fid=1:nFace
    %disp(['edge index = ', num2str(eid)]);
    nodesIdx = Topo(fid,:);
    % get undeformed quantities
    nodes = FV_ref.vertices(nodesIdx,:); % each row is one verts
    undefEdges(1,:) = nodes(3,:) - nodes(2,:);
    undefEdges(2,:) = nodes(1,:) - nodes(3,:);
    undefEdges(3,:) = nodes(2,:) - nodes(1,:);
    % compute volume
    temp = cross(undefEdges(1,:), undefEdges(2,:));
    volUndefSqr = norm(temp)^2 / 4;
    volUndef = sqrt(volUndefSqr);
   
    % get deformed quantities
    nodes = FV_def.vertices(nodesIdx,:); % 3 x 3
    % compute volume
    defEdges(1,:) = nodes(3,:) - nodes(2,:);
    defEdges(2,:) = nodes(1,:) - nodes(3,:);
    defEdges(3,:) = nodes(2,:) - nodes(1,:);
    temp = cross(nodes(3,:)-nodes(2,:), nodes(1,:)-nodes(3,:));
    volDefSqr = norm(temp)^2 / 4;
    volDef = sqrt(volDefSqr);
   
        
    factors = zeros(1,3);
    % trace part of gradient
    for i=0:2
        factors(i+1) = -0.25 * mu * (undefEdges(mod(i+2,3)+1,:)*...
            undefEdges(mod(i+1,3)+1,:)') / volUndef;
    end
    
    
    factor = 2*(lambdaQuarter * (volDef / volUndef) - ...
        muHalfPlusLambdaQuarter * (volUndef/volDef) );
    
    
    
    for i=0:2
        temp = getAreaGradK(nodes(mod(i+1,3)+1,:), nodes(mod(i+2,3)+1,:),...
            nodes(i+1,:));
        
        for j=1:3
            
            grad1 = factor * temp(j);
            grad2 = factors(mod(i+1,3)+1) * defEdges(mod(i+1,3)+1,j);
            grad3 = -1*(factors(mod(i+2,3)+1) * defEdges(mod(i+2,3)+1,j));
            grad = grad1 + grad2 + grad3;
            loop_out(fid, :) = grad;
            g1(nodesIdx(i+1),j) = g1(nodesIdx(i+1),j) + grad;
                        
        end
    end
end
g1 = g1.*datweight;

end

function [g1_fast] = getMemGradVec(FV_ref, FV_def, Topo, mu, lambda,datweight)
nverts = size(FV_ref.vertices, 1);
nFace = size(FV_ref.faces, 1);
muHalf = mu / 2;
lambdaQuarter = lambda / 4;
muHalfPlusLambdaQuarter = muHalf + lambdaQuarter;
%Topo = Topo(1:10,:);
g1_fast = zeros(nverts, 3);
is = Topo(:, 1);
js = Topo(:, 2);
ks = Topo(:, 3);
% deformed quantities
% undeformed quantities
Pis = FV_ref.vertices(is, :); % nface x 3
Pjs = FV_ref.vertices(js, :);
Pks = FV_ref.vertices(ks, :);
undEis = Pks - Pjs;
undEjs = Pis - Pks;
undEks = Pjs - Pis;
tempsUnd = cross(undEis, undEjs, 2);

Pis = FV_def.vertices(is, :); % nface x 3
Pjs = FV_def.vertices(js, :);
Pks = FV_def.vertices(ks, :);
defEis = Pks - Pjs;
defEjs = Pis - Pks;
defEks = Pjs - Pis;
tempsDef = cross(defEis, defEjs, 2);
% why are they not the same?
volUndefSqrs = sum(tempsUnd.^2, 2)./4;
volDefSqrs = sum(tempsDef.^2, 2)./4;
% for fid=1:nFace
%     volUndefSqrs(fid,1) = norm(tempsUnd(fid,:))^2 / 4;
%     volDefSqrs(fid,1) = norm(tempsDef(fid,:))^2 / 4;
% end

volUndefs = sqrt(volUndefSqrs);
volDefs = sqrt(volDefSqrs);


% factors(1): 3, 2
factors = zeros(nFace, 3);
tmp = bsxfun(@times, undEks, undEjs);
factors(:,1) = sum(tmp, 2);
% factors(2): 1, 3
tmp = bsxfun(@times, undEis, undEks);
factors(:,2) = sum(tmp, 2);
% factors(3): 2, 1
tmp = bsxfun(@times, undEjs, undEis);
factors(:,3) = sum(tmp, 2);

factors = (-0.25*mu).*(factors./repmat(volUndefs,1,3));
factor = 2*(lambdaQuarter.*(volDefs./volUndefs) - ...
    muHalfPlusLambdaQuarter.* (volUndefs./volDefs));

% fill gradient
% i=1: 2, 3, 1
temps = getAreaGradKVec(Pjs, Pks, Pis);

for j=1:3
    % j=1: factors(2) * defEdges(2) - factors(3) * defEdges(3)
    grad1 = factor.*temps(:,j);
    grad2 = factors(:,2).*defEjs(:,j);
    grad3 = -1.* (factors(:,3).*defEks(:,j));
    
    grad = grad1+grad2+grad3;
    %vec_out = grad;
    g1_fast(1:max(is),j) = g1_fast(1:max(is),j) + accumarray(is,grad);
end

% i=2: 3, 1, 2
temps = getAreaGradKVec(Pks, Pis, Pjs);
for j=1:3
    % j=1: factors(3) * defEdges(3) - factors(1) * defEdges(1)
    grad = factor.*temps(:,j) + factors(:,3).*defEks(:,j) - ...
        factors(:,1).*defEis(:,j);
    g1_fast(1:max(js),j) = g1_fast(1:max(js),j) + accumarray(js,grad);
end

% i=3: 1, 2, 3
temps = getAreaGradKVec(Pis, Pjs, Pks);
for j=1:3
    % j=1: factors(1) * defEdges(1) - factors(2) * defEdges(2)
    grad = factor.*temps(:,j) + factors(:,1).*defEis(:,j) - ...
        factors(:,2).*defEjs(:,j);
    g1_fast(1:max(ks),j) = g1_fast(1:max(ks),j) + accumarray(ks,grad);
end
g1_fast = datweight.*g1_fast;
end


function H = getAngleHessLoop(FV, FV_def, Ev, Eo, eta, boundaryedges, datweight)
nEdge = size(Ev, 1);
nverts = size(FV_def.vertices, 1);
H = zeros(nverts*3, nverts*3);
for edge_idx = 1:nEdge
    if ~boundaryedges(edge_idx)
        i = Ev(edge_idx,1);
        j = Ev(edge_idx,2);
        k = Eo(edge_idx,1);
        l = Eo(edge_idx,2);       
        area = FV.de(edge_idx);
        %area = 1;
        Pi = FV.vertices(i,:);
        Pj = FV.vertices(j,:);
        elengthSqr = norm(Pj-Pi)^2;
        %elengthSqr = 1;
        delThetaDouble = FV.Te(edge_idx)-FV_def.Te(edge_idx);
        delThetaDouble = (-2*elengthSqr*delThetaDouble)/area;
        %delThetaDouble = -1;
        factor = (2*elengthSqr)/area;
        %factor = 0;
        Pi = FV_def.vertices(i,:);
        Pj = FV_def.vertices(j,:);
        Pk = FV_def.vertices(k,:);
        Pl = FV_def.vertices(l,:);                
        thetaK = getThetaGradK(Pi,Pj,Pk);       
        thetaL = getThetaGradK(Pj,Pi,Pl);      
        thetaI = getThetaGradI(Pi,Pj,Pk,Pl);
        thetaJ = getThetaGradJ(Pi,Pj,Pk,Pl);        
        % Start assemble Hessian matrix
        cont = datweight*eta;
        % kk
        HessKK = factor.*(thetaK'*thetaK) + delThetaDouble.*getHessThetaKK(Pi,Pj,Pk);
        HessKK = cont.*HessKK;
        H = addToH(H, HessKK, k, k,nverts);          
        % ik & ki
        HessKI = factor.*(thetaI'*thetaK) + delThetaDouble.*getHessThetaIK(Pi,Pj,Pk);
        HessKI = cont.*HessKI;
        H = addToH(H, HessKI, i, k,nverts);
        H = addToH(H, HessKI', k, i,nverts);           
        %hess = getHessThetaIK(Pi,Pj,Pk);
        %hess_diff = getHikDiffQuotient(Pi,Pj,Pk,Pl);
        
        % jk & kj
        HessKJ = factor.*(thetaJ'*thetaK) + delThetaDouble.*getHessThetaJK(Pi,Pj,Pk);
        HessKJ = cont.*HessKJ;
        H = addToH(H, HessKJ, j, k,nverts);
        H = addToH(H, HessKJ', k, j,nverts); 
        %hess = getHessThetaJK(Pi,Pj,Pk);
        %hess_diff = getHjkDiffQuotient(Pi,Pj,Pk,Pl);
        % ll
        HessLL = factor.*(thetaL'*thetaL) + delThetaDouble.*getHessThetaKK(Pj,Pi,Pl);
        HessLL = cont.*HessLL;
        H = addToH(H, HessLL, l, l,nverts);       
        % il & li
        HessLI = factor.*(thetaI'*thetaL) + delThetaDouble.*getHessThetaJK(Pj,Pi,Pl);
        HessLI = cont.*HessLI;
        H = addToH(H, HessLI, i, l,nverts);
        H = addToH(H, HessLI', l, i,nverts);        
        % jl & lj
        HessLJ = factor.*(thetaJ'*thetaL) + delThetaDouble.*getHessThetaIK(Pj,Pi,Pl);
        HessLJ = cont.*HessLJ;
        H = addToH(H, HessLJ, j, l,nverts);
        H = addToH(H, HessLJ', l, j,nverts);       
        % kl & lk
        HessLK = factor.*(thetaK'*thetaL);
        HessLK = cont.*HessLK;
        H = addToH(H, HessLK, k, l, nverts);
        H = addToH(H, HessLK', l, k, nverts);        
        % ii
        HessII1 = factor.*(thetaI'*thetaI);
        HessII2 = delThetaDouble.*getHessThetaII(Pi,Pj,Pk,Pl);
        HessII =  HessII1 + HessII2;
        HessII = cont.*HessII;
        H = addToH(H, HessII, i, i, nverts);       
        % jj
        HessJJ = factor.*(thetaJ'*thetaJ) + delThetaDouble.*getHessThetaII(Pj,Pi,Pl,Pk);
        HessJJ = cont.*HessJJ;
        H = addToH(H, HessJJ, j, j,nverts);       
        % ij & ji
        HessJI = factor.*(thetaI'*thetaJ) + delThetaDouble.*getHessThetaJI(Pi,Pj,Pk,Pl);
        HessJI = cont.*HessJI;
        H = addToH(H, HessJI, i, j,nverts);
        H = addToH(H, HessJI', j, i,nverts);            
        %hess = getHessThetaJI(Pi,Pj,Pk,Pl);
        %hess_diff = getHijDiffQuotient(Pi,Pj,Pk,Pl);
    end
end
end

function g3_fast = getAngleGradVec(FV_ref,FV_def,Ev,Eo,boundaryedges,eta,datweight)
nverts = size(FV_ref.vertices, 1);

g3_fast = zeros(nverts, 3);
is = Ev(~boundaryedges, 1);
js = Ev(~boundaryedges, 2);
ks = Eo(~boundaryedges, 1);
ls = Eo(~boundaryedges, 2);
areas = 3.*FV_ref.de(~boundaryedges);
% undeformed
Pis = FV_ref.vertices(is, :);
Pjs = FV_ref.vertices(js, :);
edges = Pjs-Pis;
elengthSqr = sum(edges.^2, 2);
% deformed quantity
Pis = FV_def.vertices(is, :);
Pjs = FV_def.vertices(js, :);
Pks = FV_def.vertices(ks, :);
Pls = FV_def.vertices(ls, :);
delThetas = (FV_ref.Te(~boundaryedges) - FV_def.Te(~boundaryedges)).*elengthSqr ./ areas;
delThetas = bsxfun(@times,-2,delThetas);

thetaks = getThetaGradKVec(Pis,Pjs,Pks);
thetals = getThetaGradKVec(Pjs,Pis,Pls);
thetais = getThetaGradIVec(Pis,Pjs,Pks,Pls);
thetajs = getThetaGradJVec(Pis,Pjs,Pks,Pls);

conts = datweight*eta.*delThetas;
thetaks = bsxfun(@times,conts,thetaks);
thetals = bsxfun(@times,conts,thetals);
thetais = bsxfun(@times,conts,thetais);
thetajs = bsxfun(@times,conts,thetajs);
g3_fast(1:max(is),:) = g3_fast(1:max(is),:)+[accumarray(is,thetais(:,1)) accumarray(is,thetais(:,2)) accumarray(is,thetais(:,3))];
g3_fast(1:max(js),:) = g3_fast(1:max(js),:)+[accumarray(js,thetajs(:,1)) accumarray(js,thetajs(:,2)) accumarray(js,thetajs(:,3))];
g3_fast(1:max(ks),:) = g3_fast(1:max(ks),:)+[accumarray(ks,thetaks(:,1)) accumarray(ks,thetaks(:,2)) accumarray(ks,thetaks(:,3))];
g3_fast(1:max(ls),:) = g3_fast(1:max(ls),:)+[accumarray(ls,thetals(:,1)) accumarray(ls,thetals(:,2)) accumarray(ls,thetals(:,3))];
end

function H1 = getMemHessLoop(FV_ref,FV_def,Topo)
H1 = zeros(nverts*3, nverts*3);
for fid = 1:nFace
    
    nodesIdx = Topo(fid,:);
    % get undeformed quantities
    nodes = FVs{m}.vertices(nodesIdx,:);
    for j=0:2
        undefEdges(j+1,:) = nodes(mod(j+2,3)+1,:) - nodes(mod(j+1,3)+1,:);
    end
    % compute volume
    temp = cross(undefEdges(1,:), undefEdges(2,:));
    volUndefSqr = norm(temp)^2 / 4;
    volUndef = sqrt(volUndefSqr);
    
    % get deformed quantities
    nodes = FV_def.vertices(nodesIdx,:);
    for j=0:2
        defEdges(j+1,:) = nodes(mod(j+2,3)+1,:) - nodes(mod(j+1,3)+1,:);
    end
    % compute volume
    temp = cross(defEdges(1,:), defEdges(2,:));
    volDefSqr = norm(temp)^2 / 4;
    volDef = sqrt(volDefSqr);
    % compute trace factors
    for i=0:2
        traceFactors(i+1) = -0.25*mu*undefEdges(mod(i+2,3)+1,:)*...
            undefEdges(mod(i+1,3)+1,:)' / volUndef;
    end
    % mixedFactor and areaFactor
    mixedFactor = 0.5 * lambda / volUndef + 2*muHalfPlusLambdaQuarter...
        * volUndef / volDefSqr;
    areaFactor = 0.5 * lambda * volDef / volUndef - 2*...
        muHalfPlusLambdaQuarter * volUndef / volDef;
    % precompute area gradients
    gradArea = zeros(3,3);
    for i=0:2
        gradArea(i+1,:) = getAreaGradK(nodes(mod(i+1,3)+1,:), nodes(mod(i+2,3)+1,:),...
            nodes(mod(i,3)+1,:));
    end
    % compute local matrices
    % i==j
    for i=0:2
        auxMat = getHessAreaKK(nodes(mod(i+1,3)+1,:),nodes(mod(i+2,3)+1,:),...
            nodes(mod(i,3)+1,:));
        tensorProduct = gradArea(i+1,:)'*gradArea(i+1,:);
        hess = areaFactor.*auxMat + mixedFactor.*tensorProduct;
        diagMat1 = traceFactors(mod(i+1,3)+1).*eye(3);
        diagMat2 = traceFactors(mod(i+2,3)+1).*eye(3);
        
        hess = hess + diagMat1 + diagMat2;
        
        H1 = addToH(H1,hess,nodesIdx(i+1),nodesIdx(i+1),nverts);
    end
    % i~=j
    for i=0:2
        auxMat = getHessAreaIK(nodes(mod(i,3)+1,:),nodes(mod(i+1,3)+1,:),...
            nodes(mod(i+2,3)+1,:));
        tensorProduct = gradArea(i+1,:)'*gradArea(mod(i+2,3)+1,:);
        hess = areaFactor.*auxMat + mixedFactor.*tensorProduct;
        diagMat = traceFactors(mod(i+1,3)+1).*eye(3);
        hess = hess - diagMat;
        
        H1 = addToH(H1,hess,nodesIdx(i+1),nodesIdx(mod(i+2,3)+1),nverts);
        H1 = addToH(H1,hess',nodesIdx(mod(i+2,3)+1),nodesIdx(i+1),nverts);
    end
    
end
end

function [rows,cols,vals] = getMemHessVec(rows,cols,vals,FV_ref, FV_def, Topo, mu, lambda,datweight)
%rows = [];cols = [];vals = [];
nFace = size(FV_ref.faces, 1);
nverts = size(FV_ref.vertices, 1);
muHalf = mu / 2;
lambdaQuarter = lambda / 4;
muHalfPlusLambdaQuarter = muHalf + lambdaQuarter;

is = Topo(:, 1);
js = Topo(:, 2);
ks = Topo(:, 3);
% undeformed quantities
Pis = FV_ref.vertices(is, :); % nface x 3
Pjs = FV_ref.vertices(js, :);
Pks = FV_ref.vertices(ks, :);
undEis = Pks - Pjs;
undEjs = Pis - Pks;
undEks = Pjs - Pis;
tempsUnd = cross(undEis, undEjs);
volUndefSqrs = sum(tempsUnd.^2, 2)./4;
volUndefs = sqrt(volUndefSqrs);
% deformed quantities
Pis = FV_def.vertices(is, :); % nface x 3
Pjs = FV_def.vertices(js, :);
Pks = FV_def.vertices(ks, :);
defEis = Pks - Pjs;
defEjs = Pis - Pks;
defEks = Pjs - Pis;
tempsDef = cross(defEis, defEjs);
volDefSqrs = sum(tempsDef.^2, 2)./4;
volDefs = sqrt(volDefSqrs);
% trace part
traceFactors = zeros(nFace, 3);
% factors(1): 3, 2
traceFactors(:,1) = -0.25.*mu.*sum(undEks.*undEjs, 2)./volUndefs;
% factors(2): 1, 3
traceFactors(:,2) = -0.25.*mu.*sum(undEis.*undEks, 2)./volUndefs;
% factors(3): 2, 1
traceFactors(:,3) = -0.25.*mu.*sum(undEjs.*undEis, 2)./volUndefs;
% mixed factor and area factor
mixedFactors = 0.5*lambda./volUndefs + 2*muHalfPlusLambdaQuarter*...
    (volUndefs./volDefSqrs);
areaFactors = 0.5*lambda.*volDefs./volUndefs - ...
    2*muHalfPlusLambdaQuarter.*volUndefs./volDefs;
gradAreas = zeros(nFace, 3);
gradAreas(:,:,1)=getAreaGradKVec(Pjs,Pks,Pis); % n x 3
gradAreas(:,:,2)=getAreaGradKVec(Pks,Pis,Pjs); % n x 3
gradAreas(:,:,3)=getAreaGradKVec(Pis,Pjs,Pks); % n x 3
% compute local mat
%% i==j
% i=1
auxMats = getHessAreaKKVec(Pjs,Pks,Pis);% n x 3 x 3
tensorProduct = outerProductVectorised(gradAreas(:,:,1),gradAreas(:,:,1));
hess1 = bsxfun(@times, auxMats, areaFactors);
hess2 = bsxfun(@times, tensorProduct, mixedFactors);
hess = hess1 + hess2;
TF1 = repmat(eye(3), [1 1 nFace]);
TF1 = permute(TF1, [3 1 2]);
TF1 = bsxfun(@times, TF1, traceFactors(:,2));
TF2 = repmat(eye(3), [1 1 nFace]);
TF2 = permute(TF2, [3 1 2]);
TF2 = bsxfun(@times, TF2, traceFactors(:,3));
hess = hess + TF1 + TF2;
hess = bsxfun(@times, hess, datweight);

[rows,cols,vals]=addToHVectorised(rows,cols,vals,hess,is,is,nverts);
% i=2
auxMats = getHessAreaKKVec(Pks,Pis,Pjs);% n x 3 x 3
tensorProduct = outerProductVectorised(gradAreas(:,:,2),gradAreas(:,:,2));
hess1 = bsxfun(@times, auxMats, areaFactors);
hess2 = bsxfun(@times, tensorProduct, mixedFactors);
hess = hess1 + hess2;
TF1 = repmat(eye(3), [1 1 nFace]);
TF1 = permute(TF1, [3 1 2]);
TF1 = bsxfun(@times, TF1, traceFactors(:,3));
TF2 = repmat(eye(3), [1 1 nFace]);
TF2 = permute(TF2, [3 1 2]);
TF2 = bsxfun(@times, TF2, traceFactors(:,1));
hess = hess + TF1 + TF2;
hess = bsxfun(@times, hess, datweight);
[rows,cols,vals]=addToHVectorised(rows,cols,vals,hess,js,js,nverts);
% i=3
auxMats = getHessAreaKKVec(Pis,Pjs,Pks);% n x 3 x 3
tensorProduct = outerProductVectorised(gradAreas(:,:,3),gradAreas(:,:,3));
hess1 = bsxfun(@times, auxMats, areaFactors);
hess2 = bsxfun(@times, tensorProduct, mixedFactors);
hess = hess1 + hess2;
TF1 = repmat(eye(3), [1 1 nFace]);
TF1 = permute(TF1, [3 1 2]);
TF1 = bsxfun(@times, TF1, traceFactors(:,1));
TF2 = repmat(eye(3), [1 1 nFace]);
TF2 = permute(TF2, [3 1 2]);
TF2 = bsxfun(@times, TF2, traceFactors(:,2));
hess = hess + TF1 + TF2;
hess = bsxfun(@times, hess, datweight);
[rows,cols,vals]=addToHVectorised(rows,cols,vals,hess,ks,ks,nverts);
%% i~=j
% i=1
auxMats = getHessAreaIKVec(Pis,Pjs,Pks);% n x 3 x 3
tensorProduct = outerProductVectorised(gradAreas(:,:,1),gradAreas(:,:,3));
hess1 = bsxfun(@times, auxMats, areaFactors);
hess2 = bsxfun(@times, tensorProduct, mixedFactors);
hess = hess1 + hess2;
TF1 = repmat(eye(3), [1 1 nFace]);
TF1 = permute(TF1, [3 1 2]);
TF1 = bsxfun(@times, TF1, traceFactors(:,2));

hess = hess - TF1;
hess = bsxfun(@times, hess, datweight);
[rows,cols,vals]=addToHVectorised(rows,cols,vals,permute(hess, [1 3 2]),is,ks,nverts);
[rows,cols,vals]=addToHVectorised(rows,cols,vals,hess,ks,is,nverts);
%H1_fast = sparse(rows,cols,vals,nverts*3,nverts*3);
% i=2
auxMats = getHessAreaIKVec(Pjs,Pks,Pis);% n x 3 x 3
tensorProduct = outerProductVectorised(gradAreas(:,:,2),gradAreas(:,:,1));
hess1 = bsxfun(@times, auxMats, areaFactors);
hess2 = bsxfun(@times, tensorProduct, mixedFactors);
hess = hess1 + hess2;
TF1 = repmat(eye(3), [1 1 nFace]);
TF1 = permute(TF1, [3 1 2]);
TF1 = bsxfun(@times, TF1, traceFactors(:,3));
hess = hess - TF1;
hess = bsxfun(@times, hess, datweight);
[rows,cols,vals]=addToHVectorised(rows,cols,vals,permute(hess, [1 3 2]),js,is,nverts);
[rows,cols,vals]=addToHVectorised(rows,cols,vals,hess,is,js,nverts);
% i=3
auxMats = getHessAreaIKVec(Pks,Pis,Pjs);% n x 3 x 3
tensorProduct = outerProductVectorised(gradAreas(:,:,3),gradAreas(:,:,2));
hess1 = bsxfun(@times, auxMats, areaFactors);
hess2 = bsxfun(@times, tensorProduct, mixedFactors);
hess = hess1 + hess2;
TF1 = repmat(eye(3), [1 1 nFace]);
TF1 = permute(TF1, [3 1 2]);
TF1 = bsxfun(@times, TF1, traceFactors(:,1));
hess = hess - TF1;
hess = bsxfun(@times, hess, datweight);
[rows,cols,vals]=addToHVectorised(rows,cols,vals,permute(hess, [1 3 2]),ks,js,nverts);
[rows,cols,vals]=addToHVectorised(rows,cols,vals,hess,js,ks,nverts);
%H1_fast = sparse(rows,cols,vals,nverts*3,nverts*3);
end

function [rows,cols,vals] = getAngleHessVec(rows,cols,vals,FV_ref,FV_def,Ev,Eo,eta,boundaryedges,datweight)%H3_fast =
nverts = size(FV_ref.vertices, 1);
%rows = [];cols = [];vals = [];
is = Ev(~boundaryedges, 1);
js = Ev(~boundaryedges, 2);
ks = Eo(~boundaryedges, 1);
ls = Eo(~boundaryedges, 2);
areas = 3.*FV_ref.de(~boundaryedges);
Pis = FV_ref.vertices(is,:);
Pjs = FV_ref.vertices(js,:);
edges = Pjs-Pis;
elengthSqr = sum(edges.^2, 2);
delThetaDouble = FV_ref.Te(~boundaryedges)-FV_def.Te(~boundaryedges);
delThetaDouble = (-2.*elengthSqr.*delThetaDouble)./areas;
factors = 2.*elengthSqr./areas;
Pis = FV_def.vertices(is,:);
Pjs = FV_def.vertices(js,:);
Pks = FV_def.vertices(ks,:);
Pls = FV_def.vertices(ls,:);
thetaKs = getThetaGradKVec(Pis,Pjs,Pks);
thetaLs = getThetaGradKVec(Pjs,Pis,Pls);
thetaIs = getThetaGradIVec(Pis,Pjs,Pks,Pls);
thetaJs = getThetaGradJVec(Pis,Pjs,Pks,Pls);
cont = datweight*eta;
% kk
HessKK1 = outerProductVectorised(thetaKs, thetaKs);
HessKK1 = bsxfun(@times, factors, HessKK1);
HessKK2 = getHessThetaKKVec(Pis,Pjs,Pks);
HessKK2 = bsxfun(@times, delThetaDouble, HessKK2);
HessKKs = HessKK1 + HessKK2;
HessKKs = bsxfun(@times, cont, HessKKs);
[rows,cols,vals]=addToHVectorised(rows,cols,vals,HessKKs,ks,ks,nverts);
% ik & ki
% cautions: HessKIs need to be transposed with [1 3 2], this is
% different with loopy version!
HessKI1s = outerProductVectorised(thetaIs, thetaKs);
HessKI1s = bsxfun(@times, factors, HessKI1s);
HessKI2s = getHessThetaIKVec(Pis,Pjs,Pks);
HessKI2s = bsxfun(@times, delThetaDouble, HessKI2s);
HessKIs = HessKI1s + HessKI2s;
HessKIs = bsxfun(@times, cont, HessKIs);
[rows,cols,vals]=addToHVectorised(rows,cols,vals,HessKIs,ks,is,nverts);
[rows,cols,vals]=addToHVectorised(rows,cols,vals,permute(HessKIs,[1 3 2]),is,ks,nverts);
% jk & kj
HessKJ1 = outerProductVectorised(thetaJs, thetaKs);
HessKJ1 = bsxfun(@times, factors, HessKJ1);
HessKJ2 = getHessThetaJKVec(Pis,Pjs,Pks);
HessKJ2 = bsxfun(@times, delThetaDouble, HessKJ2);
HessKJs = HessKJ1 + HessKJ2;
HessKJs = bsxfun(@times, cont, HessKJs);
[rows,cols,vals]=addToHVectorised(rows,cols,vals,HessKJs,ks,js,nverts);
[rows,cols,vals]=addToHVectorised(rows,cols,vals,permute(HessKJs,[1 3 2]),js,ks,nverts);
% ll
HessLL1 = outerProductVectorised(thetaLs, thetaLs);
HessLL1 = bsxfun(@times, factors, HessLL1);
HessLL2 = getHessThetaKKVec(Pjs,Pis,Pls);
HessLL2 = bsxfun(@times, delThetaDouble, HessLL2);
HessLL = HessLL1 + HessLL2;
HessLL = bsxfun(@times, cont, HessLL);
[rows,cols,vals]=addToHVectorised(rows,cols,vals,HessLL,ls,ls,nverts);
% il & li
HessLI1 = outerProductVectorised(thetaIs, thetaLs);
HessLI1 = bsxfun(@times, factors, HessLI1);
HessLI2 = getHessThetaJKVec(Pjs,Pis,Pls);
HessLI2 = bsxfun(@times, delThetaDouble, HessLI2);
HessLI = HessLI1 + HessLI2;
HessLI = bsxfun(@times, cont, HessLI);
[rows,cols,vals]=addToHVectorised(rows,cols,vals,HessLI,ls,is,nverts);
[rows,cols,vals]=addToHVectorised(rows,cols,vals,permute(HessLI,[1 3 2]),is,ls,nverts);
% jl & lj
HessLJ1 = outerProductVectorised(thetaJs, thetaLs);
HessLJ1 = bsxfun(@times, factors, HessLJ1);
HessLJ2 = getHessThetaIKVec(Pjs,Pis,Pls);
HessLJ2 = bsxfun(@times, delThetaDouble, HessLJ2);
HessLJ = HessLJ1 + HessLJ2;
HessLJ = bsxfun(@times, cont, HessLJ);
[rows,cols,vals]=addToHVectorised(rows,cols,vals,HessLJ,ls,js,nverts);
[rows,cols,vals]=addToHVectorised(rows,cols,vals,permute(HessLJ,[1 3 2]),js,ls,nverts);
% kl & lk
HessLK = outerProductVectorised(thetaKs, thetaLs);
HessLK = bsxfun(@times, cont.*factors, HessLK);
[rows,cols,vals]=addToHVectorised(rows,cols,vals,HessLK,ls,ks,nverts);
[rows,cols,vals]=addToHVectorised(rows,cols,vals,permute(HessLK,[1 3 2]),ks,ls,nverts);
% ii
HessII1 = outerProductVectorised(thetaIs, thetaIs);
HessII1 = bsxfun(@times, factors, HessII1);
HessII2 = getHessThetaIIVec(Pis,Pjs,Pks,Pls);
HessII2 = bsxfun(@times, delThetaDouble, HessII2);
HessII = HessII1 + HessII2;
HessII = bsxfun(@times, cont, HessII);
[rows,cols,vals]=addToHVectorised(rows,cols,vals,HessII,is,is,nverts);
% jj
HessJJ1 = outerProductVectorised(thetaJs, thetaJs);
HessJJ1 = bsxfun(@times, factors, HessJJ1);
HessJJ2 = getHessThetaIIVec(Pjs,Pis,Pls,Pks);
HessJJ2 = bsxfun(@times, delThetaDouble, HessJJ2);
HessJJ = HessJJ1 + HessJJ2;
HessJJ = bsxfun(@times, cont, HessJJ);
[rows,cols,vals]=addToHVectorised(rows,cols,vals,HessJJ,js,js,nverts);
% ij & ji
HessJI1 = outerProductVectorised(thetaIs, thetaJs);
HessJI1 = bsxfun(@times, factors, HessJI1);
HessJI2 = getHessThetaJIVec(Pis,Pjs,Pks,Pls);
HessJI2 = bsxfun(@times, delThetaDouble, HessJI2);
HessJI = HessJI1 + HessJI2;
HessJI = bsxfun(@times, cont, HessJI);
[rows,cols,vals]=addToHVectorised(rows,cols,vals,HessJI,js,is,nverts);
[rows,cols,vals]=addToHVectorised(rows,cols,vals,permute(HessJI,[1 3 2]),is,js,nverts);

%H3_fast = sparse(rows,cols,vals,nverts*3,nverts*3);
end

