function [cost,grad,H]=shellGradDeformed( x, FVs, Topo, Ev, Ef, Eo, boundaryedges,options)
% Note: x is deformed Shell, while FVs hold all undeformed shells. 
% compute the cost and gradient of cost function wrt nodes
%% input
% x: init value for average, only vertices is supplied
% FVs: input data (nres x nmeshes)
% Evs(nEdge): i and j
% Eos(nEdge): k and l
% Efs(nEdge): adjacent faces list
% allboundaryedges: boundary edges
% Topo: similar to FV.faces, which is nface x 3 mat

%% output
% f: cost (1,1)
% g: gradient (nvertsx3, 1)
% H: hessian (nvertsx3, nvertsx3)
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
eta = options.eta;

nmesh = length(FVs);
if ~isfield(options, 'datweights')
    datweights(1:nmesh) = 1;
else
    datweights = options.datweights;
end

% recompute length, area, and dihedral angle
v = reshape(x, length(x)/3, 3);
nverts = size(v, 1);
%g = zeros(nverts, 3);
%H = sparse(nverts*3, nverts*3);
% cost term
FV_def.vertices = v;
FV_def.faces = Topo;


% compute intermediate quantities
if ~isfield(FV_def, 'de')
    FV_def = precompute(FV_def,Topo,Ev,Ef,boundaryedges);
end

for i = 1:nmesh
    if ~isfield(FVs{i}, 'de')
        FVs{i} = precompute(FVs{i},Topo,Ev,Ef,boundaryedges);
    end
end

gridSize = max(FV_def.le);

[ cost,grad,H ] = fastShellGradAndHessDef( x,FVs,Ev,Eo,Ef,mu,lambda,eta,datweights,boundaryedges );
% 
% for m=1:nmesh
%     if datweights(m) == 0
%         continue;
%     end
% 
%     % Edge lengths
%     f1 = mu*datweights(m).*FVs{m}.de ./ FVs{m}.le.^2 .*((FVs{m}.le-FV_def.le).^2);
%     f = [f; f1];
%     % Triangle areas
%     f2 = lambda*datweights(m)./FVs{m}.at .* (FV_def.at-FVs{m}.at).^2;
%     f = [f; f2];
%     % Dihedral angles
%     f3 = eta*datweights(m).* (FVs{m}.le(~boundaryedges)).^2 ./ ...
%         (FVs{m}.de(~boundaryedges)) .* (FV_def.Te-FVs{m}.Te).^2;  
%     f = [f; f3];
%     
%     useLoop = false;
%     if nargout > 1 % gradient required
%         % Add gradient
%         if useLoop
%             disp('Cautions: using loop version.');
%             g1 = getEdgeGradLoop(FVs{m},FV_def,Ev,mu,datweights(m));
%             g2 = getFaceGradLoop(FVs{m},FV_def, lambda, datweights(m));
%             g3 = getAngleGradLoop(FVs{m}, FV_def, Ev, Eo, boundaryedges,eta,datweights(m));
%         else
%             g1 = getEdgeGradVec(FVs{m},FV_def,Ev,mu,datweights(m));
%             g2 = getFaceGradVec(FVs{m},FV_def, lambda, datweights(m));
%             g3 = getAngleGradVec(FVs{m}, FV_def, Ev, Eo, boundaryedges,eta,datweights(m));
%         end
%         g = g + g1 + g2 + g3;
%     end
%     
%     if nargout > 2 % hessian required   
%         if useLoop
%             disp('Cautions: using loop version.');
%             H1 = getEdgeHessLoop(FVs{m}, FV_def,Ev,mu,datweights(m));
%             H2 = getFaceHessLoop(FVs{m},FV_def,lambda,datweights(m));
%             H3 = getAngleHessLoop(FVs{m},FV_def,Ev,Eo,eta,boundaryedges,datweights(m));
%         else            
%             H1 = getEdgeHessVec(FVs{m}, FV_def,Ev,mu,datweights(m));
%             H2 = getFaceHessVec(FVs{m},FV_def,lambda,datweights(m));
%             H3 = getAngleHessVec(FVs{m},FV_def,Ev,Eo,eta,boundaryedges,datweights(m));
%         end
%         H = H + H1 + H2 + H3;
%     end
% end
% 
% cost = sum(f);

if nargout > 2 % hessian required
    if ~isfield(options, 'regHess')
        options.regHess = false;
    end
    if options.regHess
        if ~isfield(options, 'eps')
            options.eps = 1e-4;
        end
        H = regularizeH(H, options.eps, gridSize);
    end

end
end

%% SUPPORTING FUNCTIONS BELOW

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

function H = regularizeH(H, eps, gridsize)
nverts = size(H, 1)/3;
for i=1:nverts
    xr = i; xc = i;
    yr = i+nverts; yc = i+nverts;
    zr = i+2*nverts; zc = i+2*nverts;
    H(xr, xc) = H(xr, xc) + eps*gridsize;
    H(yr, yc) = H(yr, yc) + eps*gridsize;
    H(zr, zc) = H(zr, zc) + eps*gridsize;
end

end


function mat = addToG(mat, tmp, i)
mat(i,1)=mat(i,1)+tmp(1);
mat(i,2)=mat(i,2)+tmp(2);
mat(i,3)=mat(i,3)+tmp(3);

end

function mat = addToH(mat, tmp, r, c, nverts)


xr = r; 
xc = c;
yr = r+nverts; 
yc = c+nverts;
zr = r+2*nverts; 
zc = c+2*nverts;

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
% old implementation
e = Pj-Pi;
normal = getNormal(Pi,Pj,Pk);
grad = cross(0.5*normal, e);

% behrend's code
% a = Pi - Pk;
% d = Pk - Pj;
% e = Pj - Pi;
% area = getArea(Pi,Pj,Pk);
% temp1 = -0.25 * e*a' / area;
% temp2 = 0.25 * e*d' / area;
% grad = temp1 .* d + temp2 .* a;
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
eNormalized = es./repmat(elens, 1, 3);
gradAreaKs = getAreaGradKVec(Pis,Pjs,Pks);
Hess = outerProductVectorised(gradAreaKs, gradAreaKs);
Proj = getProjectionVec(eNormalized);
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
grad = ((e*d')/(e*e')).* grad;
end

function Grad = getThetaGradILeftPartVec(Pis,Pjs,Pks)
es = Pjs-Pis;
ds = Pks-Pjs;
Grad = getThetaGradKVec(Pis,Pjs,Pks);
tmps = sum(es.*ds,2)./sum(es.*es,2);
Grad = bsxfun(@times,tmps,Grad);
end

function grad = getThetaGradI(Pi,Pj,Pk,Pl)
grad = getThetaGradILeftPart(Pi,Pj,Pk);
grad = grad - getThetaGradILeftPart(Pi,Pj,Pl);
end

function Grad = getThetaGradIVec(Pis,Pjs,Pks,Pls)
Grad = getThetaGradILeftPartVec(Pis,Pjs,Pks);
Grad = Grad - getThetaGradILeftPartVec(Pis,Pjs,Pls);
end

function grad = getThetaGradJLeftPart(Pi,Pj,Pk)
e = Pj-Pi;
a = Pi-Pk;
grad = getThetaGradK(Pi,Pj,Pk);
grad = ((a*e')/(e*e')).*grad;
end

function Grad = getThetaGradJLeftPartVec(Pis,Pjs,Pks)
es = Pjs-Pis;
as = Pis-Pks;
Grad = getThetaGradKVec(Pis,Pjs,Pks);
tmps = sum(as.*es,2)./sum(es.*es,2);
Grad = bsxfun(@times,tmps,Grad);
end

function grad = getThetaGradJ(Pi,Pj,Pk,Pl)
grad = getThetaGradJLeftPart(Pi,Pj,Pk);
grad = grad - getThetaGradJLeftPart(Pi,Pj,Pl);
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
mat1 = gradArea'*normal;
Hik = mat3 + (norm(e)/areaSqr) .* mat1;
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
mat1 = gradArea' * normal;
Hjk = mat3 + (norm(e)/areaSqr) .* mat1;
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
eNormalized = e/norm(Pj-Pi);
gradThetaK = getThetaGradK(Pi,Pj,Pk);
Refl = getReflection(eNormalized);
temp = d*Refl';
mat1 = temp'*gradThetaK;
mat2 = getHessThetaIK(Pi,Pj,Pk);
HiLeft = (-1/(e*e')).*mat1 + ((d*e')/(e*e')).*mat2;
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
mat1 = bsxfun(@times, -1./sum(es.*es,2), mat1);
mat2 = getHessThetaIKVec(Pis,Pjs,Pks);
mat2 = bsxfun(@times, sum(ds.*es,2)./sum(es.*es,2), mat2);
HiLeft = mat1 + mat2;

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
temp = getHessThetaILeftPartI(Pi,Pj,Pl);
Hii = Hii - temp;
end

function Hii = getHessThetaIIVec(Pis,Pjs,Pks,Pls)
Hii = getHessThetaILeftPartIVec(Pis,Pjs,Pks);
temp = getHessThetaILeftPartIVec(Pis,Pjs,Pls);
Hii = Hii - temp;
end


function Hij = getHijDiffQuotient(Pi,Pj,Pk,Pl)
Hij = zeros(3);
H = 1e-8;
for i=1:3
    PjPlusH = Pj;
    PjPlusH(i) = PjPlusH(i) + H;
    grad1 = getThetaGradI(Pi, PjPlusH, Pk, Pl);
    grad2 = getThetaGradI(Pi, Pj, Pk, Pl);
    grad1 = grad1 - grad2;
    for j=1:3
        Hij(j,i) = grad1(j) / H;
    end
end
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
grad1 = bsxfun(@times, sum(edges.*ds,2), thetaks);
grad2 = bsxfun(@times, sum(edges.*cs,2), thetals);
grad = grad1 - grad2;
Hjk = getHessThetaJKVec(Pis,Pjs,Pks);
Hjl = getHessThetaIKVec(Pjs,Pis,Pls);
Hjk = permute(Hjk, [1 3 2]);
Hjl = permute(Hjl, [1 3 2]);
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

function M = getMassMat(FV)
nverts = size(FV.vertices, 1);
%M = eye(nverts);
M = zeros(nverts, nverts);
for i=1:nverts
    
    ind =  sum((FV.faces(:,:)==i),2);
    ind = logical(ind);
    
    M(i,i) = sum(FV.at(ind));
    
end
end

function f = getLagrangeFunc(x, FV_ref, FV_def)
disp(x');
nverts = size(FV_ref.vertices, 1);
refV = FV_ref.vertices;
defV = FV_def.vertices;
M = getMassMat(FV_ref);
I = ones(nverts, 1);

f = [];
trans_func(1) = x(1).*I'*(M*(defV(:,1)-refV(:,1)));
trans_func(2) = x(2).*I'*(M*(defV(:,2)-refV(:,2)));
trans_func(3) = x(3).*I'*(M*(defV(:,3)-refV(:,3)));
rot_func(1) = x(4).*refV(:,2)'*(M*(defV(:,1)-refV(:,1))) - refV(:,1)'*(M*(defV(:,2)-refV(:,2)));
rot_func(2) = x(5).*refV(:,3)'*(M*(defV(:,2)-refV(:,2))) - refV(:,2)'*(M*(defV(:,3)-refV(:,3)));
rot_func(3) = x(6).*refV(:,1)'*(M*(defV(:,3)-refV(:,3))) - refV(:,3)'*(M*(defV(:,1)-refV(:,1)));

f = [trans_func'; rot_func'];
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

% right submatrix, 3n * 6
submat = [M*I  Zmat Zmat    M*refV(:,2)     Zmat    -1.*(M*refV(:,3));
          Zmat M*I  Zmat  -1.*(M*refV(:,1)) M*refV(:,3) Zmat;
          Zmat Zmat M*I     Zmat            -1.*(M*refV(:,2))   M*refV(:,1)];

Lg = [g submat; submat' zeros(6)];

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


% GRADIENT AND HESSIAN FUNCTIONS LOOP VERSION
% GET EDGE GRAD
function g = getEdgeGradLoop(FV, FV_def, Ev, mu, datweight)
% g = getEdgeGradLoop(FV, FV_def, Ev, mu, datweight)
nverts = size(FV_def.vertices, 1);
g = zeros(nverts, 3);
nEdge = size(Ev, 1);
for edge_idx = 1:nEdge
    i = Ev(edge_idx, 1);
    j = Ev(edge_idx, 2);
    edge = FV_def.vertices(j,:)-FV_def.vertices(i,:);
    cont = datweight*2*mu*FV.de(edge_idx)...
        *(FV_def.le(edge_idx)-FV.le(edge_idx))/(FV.le(edge_idx))^2;
    tmp = -1 * (edge)./FV_def.le(edge_idx);
    tmp = cont.*tmp;
    g = addToG(g,tmp,i);
    tmp = (edge)./FV_def.le(edge_idx);
    tmp=cont.*tmp;
    g = addToG(g,tmp,j);
end
end
% GET FACE GRAD
function g = getFaceGradLoop(FV, FV_def, lambda, datweight)
nFace = size(FV_def.faces, 1);
F = FV_def.faces;
nverts = size(FV_def.vertices, 1);
g = zeros(nverts, 3);
for face_idx = 1:nFace
    i = F(face_idx, 1);
    j = F(face_idx, 2);
    k = F(face_idx, 3);
    Ejk = FV_def.vertices(k,:)-FV_def.vertices(j,:);
    Eki = FV_def.vertices(i,:)-FV_def.vertices(k,:);
    Eij = FV_def.vertices(j,:)-FV_def.vertices(i,:);
    tn = FV_def.Tn(face_idx, :);
    cont = datweight*2*lambda *...
        (FV_def.at(face_idx)-FV.at(face_idx)) / FV.at(face_idx);
    tmp=cross(0.5.*tn, Ejk);
    tmp = cont.*tmp;
    g = addToG(g,tmp,i);
    tmp=cross(0.5.*tn, Eki);
    tmp=cont.*tmp;
    g = addToG(g,tmp,j);
    tmp=cross(0.5.*tn, Eij);
    tmp=cont.*tmp;
    g = addToG(g,tmp,k);
    
end       
end
% GET ANGLE GRAD
function g = getAngleGradLoop(FV, FV_def, Ev, Eo, boundaryedges, eta, datweight)
nverts = size(FV_def.vertices, 1);
nEdge = size(Ev, 1);
g = zeros(nverts, 3);
for eid = 1:nEdge
    if ~boundaryedges(eid)
        i = Ev(eid,1);
        j = Ev(eid,2);
        k = Eo(eid,1);
        l = Eo(eid,2);
        area = FV.de(eid);
        Pi = FV.vertices(i,:);
        Pj = FV.vertices(j,:);
        elengthSqr = norm(Pj-Pi)^2;
        Pi = FV_def.vertices(i,:);
        Pj = FV_def.vertices(j,:);
        Pk = FV_def.vertices(k,:);
        Pl = FV_def.vertices(l,:);
        delTheta = FV.Te(eid)-FV_def.Te(eid);
        delTheta = (-2*elengthSqr/area).*delTheta;
        thetak = getThetaGradK(Pi,Pj,Pk);
        thetal = getThetaGradK(Pj,Pi,Pl);
        thetai = getThetaGradI(Pi,Pj,Pk,Pl);
        thetaj = getThetaGradJ(Pi,Pj,Pk,Pl);  
        cont = datweight*eta*delTheta;
        g = addToG(g,cont.*thetak,k);
        g = addToG(g,cont.*thetal,l);
        g = addToG(g,cont.*thetai,i);
        g = addToG(g,cont.*thetaj,j);
        
    end
end
end

% GET EDGE HESSIAN
function H = getEdgeHessLoop(FV, FV_def, Ev, mu, datweight)
nEdge = size(Ev, 1);
nverts  = size(FV_def.vertices, 1);
H = zeros(nverts*3, nverts*3);
for edge_idx = 1:nEdge
    i = Ev(edge_idx, 1);
    j = Ev(edge_idx, 2);
    edge = FV_def.vertices(j,:)-FV_def.vertices(i,:);
    def_leng = FV_def.le(edge_idx);
    und_leng = FV.le(edge_idx);
    area = FV.de(edge_idx);
    temp = edge'*edge;
    temp = (2*area/(def_leng^3*und_leng)).*temp;
    temp = temp + (2*area*(def_leng-und_leng)/...
        (und_leng^2*def_leng)).*eye(3);
    temp = (datweight*mu).*temp; 
    HessII = temp;
    H = addToH(H, HessII, i, i, nverts);  
    HessJJ = temp;
    H = addToH(H, HessJJ, j, j, nverts);   
    HessIJ = -temp;
    H = addToH(H, HessIJ, j, i, nverts);
    H = addToH(H, HessIJ', i, j, nverts);   
end
end
% GET FACE HESSIAN
function H = getFaceHessLoop(FV,FV_def, lambda, datweight)
F = FV_def.faces;
nFace = size(F, 1);
nverts = size(FV_def.vertices, 1);
H = zeros(nverts*3, nverts*3);
for face_idx = 1:nFace
    i = F(face_idx, 1);
    j = F(face_idx, 2);
    k = F(face_idx, 3);
    Pi = FV_def.vertices(i,:);
    Pj = FV_def.vertices(j,:);
    Pk = FV_def.vertices(k,:);
    und_area = FV.at(face_idx);
    def_area = FV_def.at(face_idx);
    hessFactor = 2*(def_area-und_area)/und_area;
    mixedFactor = 2/und_area;
    gradK = getAreaGradK(Pi,Pj,Pk);
    gradI = getAreaGradK(Pj,Pk,Pi);
    gradJ = getAreaGradK(Pk,Pi,Pj);
    cont = datweight*lambda;
    % kk
    hess = getHessAreaKK(Pi,Pj,Pk);
    Hkk = hessFactor.*hess;
    auxMat = gradK'*gradK;
    Hkk = Hkk + mixedFactor.*auxMat;
    HessKK = cont.*Hkk;
    H = addToH(H, HessKK, k, k, nverts);
    % ik & ki
    auxMat = getHessAreaIK(Pi,Pj,Pk);
    Hik = hessFactor.*auxMat;
    auxMat = gradI'*gradK;
    Hik = Hik + mixedFactor.*auxMat;
    HessIK = cont.*Hik;
    H = addToH(H, HessIK, i, k, nverts);
    H = addToH(H, HessIK', k, i, nverts);
    % jk & kj
    auxMat = getHessAreaIK(Pj,Pi,Pk);
    Hjk = hessFactor.*auxMat;
    auxMat = gradJ'*gradK;
    Hjk = Hjk + mixedFactor.*auxMat;
    HessJK = cont.*Hjk;
    H = addToH(H, HessJK, j, k, nverts);
    H = addToH(H, HessJK', k, j, nverts);
    % jj
    auxMat = getHessAreaKK(Pk,Pi,Pj);
    Hjj = hessFactor.*auxMat;
    auxMat = gradJ'*gradJ;
    Hjj = Hjj + mixedFactor.*auxMat;
    HessJJ = cont.*Hjj;
    H = addToH(H, HessJJ, j, j, nverts);
    % ij & ji
    auxMat = getHessAreaIK(Pi,Pk,Pj);
    Hij = hessFactor.*auxMat;
    auxMat = gradI'*gradJ;
    Hij = Hij + mixedFactor.*auxMat;
    HessIJ = cont.*Hij;
    H = addToH(H, HessIJ, i, j, nverts);
    H = addToH(H, HessIJ', j, i, nverts);
    % ii
    auxMat = getHessAreaKK(Pj,Pk,Pi);
    Hii = hessFactor.*auxMat;
    auxMat = gradI'*gradI;
    Hii = Hii + mixedFactor.*auxMat;
    HessII = cont.*Hii;
    H = addToH(H, HessII, i, i, nverts);
    
end
end
% GET ANGLE HESS
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

function Hji = getHjiDiffQuotient(Pi,Pj,Pk,Pl)
H = 1e-8;
Hji = zeros(3,3);
for i=1:3
    PiPlusH = Pi;
    PiPlusH(i) = PiPlusH(i) + H;
    grad1 = getThetaGradJ(PiPlusH, Pj, Pk,Pl);
    grad2 = getThetaGradJ(Pi,Pj,Pk,Pl);
    grad1 = grad1 - grad2;
    for j=1:3
        Hji(j,i) = grad1(j)/H;
    end
end
end

function Hik = getHikDiffQuotient(Pi,Pj,Pk,Pl)
H = 1e-8;
Hik = zeros(3,3);
for i=1:3
    PkPlusH = Pk;
    PkPlusH(i) = PkPlusH(i) + H;
    grad1 = getThetaGradI(Pi, Pj, PkPlusH,Pl);
    grad2 = getThetaGradI(Pi,Pj,Pk,Pl);
    grad1 = grad1 - grad2;
    for j=1:3
        Hik(j,i) = grad1(j)/H;
    end
end
end

function Hjk = getHjkDiffQuotient(Pi,Pj,Pk,Pl)
H = 1e-8;
Hjk = zeros(3,3);
for i=1:3
    PkPlusH = Pk;
    PkPlusH(i) = PkPlusH(i) + H;
    grad1 = getThetaGradJ(Pi, Pj, PkPlusH,Pl);
    grad2 = getThetaGradJ(Pi,Pj,Pk,Pl);
    grad1 = grad1 - grad2;
    for j=1:3
        Hjk(j,i) = grad1(j)/H;
    end
end
end

% VECTORISED GRAD AND HESS
function g1_fast = getEdgeGradVec(FV_ref,FV_def,Ev,mu,datweight)
nverts = size(FV_def.vertices, 1);
g1_fast = zeros(nverts, 3);
is = Ev(:,1);
js = Ev(:,2);
edges = FV_def.vertices(js,:)-FV_def.vertices(is,:);
conts = datweight.*2.*mu.*FV_ref.de.*(FV_def.le-FV_ref.le)./(FV_ref.le.^2);
tmps = bsxfun(@rdivide,-edges,FV_def.le);
tmps = bsxfun(@times,conts,tmps);
g1_fast(1:max(is),:) = g1_fast(1:max(is),:)+[accumarray(is,tmps(:,1)) accumarray(is,tmps(:,2)) accumarray(is,tmps(:,3))];
g1_fast(1:max(js),:) = g1_fast(1:max(js),:)+[accumarray(js,-tmps(:,1)) accumarray(js,-tmps(:,2)) accumarray(js,-tmps(:,3))];
end
function g2_fast = getFaceGradVec(FV_ref,FV_def,lambda,datweight)
nverts = size(FV_def.vertices, 1);
g2_fast = zeros(nverts, 3);
F = FV_def.faces;
is = F(:,1);
js = F(:,2);
ks = F(:,3);
Ejk = FV_def.vertices(ks,:)-FV_def.vertices(js,:);
Eki = FV_def.vertices(is,:)-FV_def.vertices(ks,:);
Eij = FV_def.vertices(js,:)-FV_def.vertices(is,:);
tns = FV_def.Tn;
conts = datweight*2*lambda .* (FV_def.at - FV_ref.at) ./ FV_ref.at;
tmps = cross(0.5.*tns, Ejk, 2);
tmps = bsxfun(@times,conts,tmps);
g2_fast(1:max(is),:) = g2_fast(1:max(is),:)+[accumarray(is,tmps(:,1)) accumarray(is,tmps(:,2)) accumarray(is,tmps(:,3))];
tmps = cross(0.5.*tns, Eki, 2);
tmps = bsxfun(@times,conts,tmps);
g2_fast(1:max(js),:) = g2_fast(1:max(js),:)+[accumarray(js,tmps(:,1)) accumarray(js,tmps(:,2)) accumarray(js,tmps(:,3))];
tmps = cross(0.5.*tns, Eij, 2);
tmps = bsxfun(@times,conts,tmps);
g2_fast(1:max(ks),:) = g2_fast(1:max(ks),:)+[accumarray(ks,tmps(:,1)) accumarray(ks,tmps(:,2)) accumarray(ks,tmps(:,3))];
end

function g3_fast = getAngleGradVec(FV_ref,FV_def,Ev,Eo,boundaryedges,eta,datweight)
nverts = size(FV_ref.vertices, 1);

g3_fast = zeros(nverts, 3);
is = Ev(~boundaryedges, 1);
js = Ev(~boundaryedges, 2);
ks = Eo(~boundaryedges, 1);
ls = Eo(~boundaryedges, 2);
areas = FV_ref.de(~boundaryedges);
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
delThetas = (FV_ref.Te - FV_def.Te).*elengthSqr ./ areas;
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

% VEC HESS

function H1_fast = getEdgeHessVec(FV_ref,FV_def,Ev,mu,datweight)
nverts = size(FV_ref.vertices, 1);
rows=[]; cols=[]; vals=[];
is = Ev(:,1);
js = Ev(:,2);
edges = FV_def.vertices(js,:)-FV_def.vertices(is,:);
def_lengs = FV_def.le;
und_lengs = FV_ref.le;
areas = FV_ref.de;
temp = outerProductVectorised(edges,edges); % N by 3 by 3
temp = bsxfun(@times,(2.*areas./(def_lengs.^3.*und_lengs)),temp);
temp2 = (2.*areas.*(def_lengs-und_lengs)./(und_lengs.^2.*def_lengs));
% Add identity matrix to temp, scaled by temp2
temp(:,1,1)=temp(:,1,1)+temp2;
temp(:,2,2)=temp(:,2,2)+temp2;
temp(:,3,3)=temp(:,3,3)+temp2;
temp = (datweight.*mu).*temp;
tempT = permute(temp, [1 3 2]);
% Fastest but least readable version:
[rows,cols,vals]=addToHVectorised(rows,cols,vals,[temp; temp; -temp; -tempT],[is; js; js; is],[is; js; is; js],nverts);
H1_fast = sparse(rows,cols,vals,nverts*3,nverts*3);
end

function H2_fast = getFaceHessVec(FV_ref,FV_def,lambda,datweight)
F = FV_ref.faces;
nverts = size(FV_ref.vertices, 1);
rows = []; cols = []; vals = [];
is = F(:,1);
js = F(:,2);
ks = F(:,3);
Pis = FV_def.vertices(is,:);
Pjs = FV_def.vertices(js,:);
Pks = FV_def.vertices(ks,:);
und_areas = FV_ref.at;
def_areas = FV_def.at;

hessFactors = 2.*(def_areas-und_areas)./und_areas;
mixedFactors = 2./und_areas;

gradKs = getAreaGradKVec(Pis,Pjs,Pks);
gradIs = getAreaGradKVec(Pjs,Pks,Pis);
gradJs = getAreaGradKVec(Pks,Pis,Pjs);

cont = datweight*lambda;
% kk
hesss = getHessAreaKKVec(Pis,Pjs,Pks);
Hkks = bsxfun(@times, hessFactors,hesss);
auxMats = outerProductVectorised(gradKs,gradKs);
auxMats = bsxfun(@times, mixedFactors,auxMats);
Hkks = Hkks + auxMats;
Hkks = bsxfun(@times, Hkks, cont);
[rows,cols,vals]=addToHVectorised(rows,cols,vals,Hkks,ks,ks,nverts);
% ik & ki
% note: Hiks should be transposed before assigning to H, indices are different with loopy version
hesss = getHessAreaIKVec(Pis,Pjs,Pks);
Hiks = bsxfun(@times, hessFactors,hesss);
auxMat = outerProductVectorised(gradIs,gradKs);
auxMat = bsxfun(@times, mixedFactors,auxMat);
Hiks = Hiks + auxMat;
Hiks = bsxfun(@times, Hiks, cont);
[rows,cols,vals]=addToHVectorised(rows,cols,vals,Hiks,ks,is,nverts);
[rows,cols,vals]=addToHVectorised(rows,cols,vals,permute(Hiks,[1 3 2]),is,ks,nverts);
% jk & kj
hesss = getHessAreaIKVec(Pjs,Pis,Pks);
Hjks = bsxfun(@times, hessFactors,hesss);
auxMat = outerProductVectorised(gradJs,gradKs);
auxMat = bsxfun(@times, mixedFactors,auxMat);
Hjks = Hjks + auxMat;
Hjks = bsxfun(@times, Hjks, cont);
[rows,cols,vals]=addToHVectorised(rows,cols,vals,Hjks,ks,js,nverts);
[rows,cols,vals]=addToHVectorised(rows,cols,vals,permute(Hjks,[1 3 2]),js,ks,nverts);
% jj
hesss = getHessAreaKKVec(Pks,Pis,Pjs);
Hjjs = bsxfun(@times, hessFactors,hesss);
auxMat = outerProductVectorised(gradJs,gradJs);
auxMat = bsxfun(@times, mixedFactors,auxMat);
Hjjs = Hjjs + auxMat;
Hjjs = bsxfun(@times, Hjjs, cont);
[rows,cols,vals]=addToHVectorised(rows,cols,vals,Hjjs,js,js,nverts);
% ij & ji
hesss = getHessAreaIKVec(Pis,Pks,Pjs);
Hijs = bsxfun(@times, hessFactors,hesss);
auxMat = outerProductVectorised(gradIs,gradJs);
auxMat = bsxfun(@times, mixedFactors,auxMat);
Hijs = Hijs + auxMat;
Hijs = bsxfun(@times, Hijs, cont);
[rows,cols,vals]=addToHVectorised(rows,cols,vals,Hijs,js,is,nverts);
[rows,cols,vals]=addToHVectorised(rows,cols,vals,permute(Hijs,[1 3 2]),is,js,nverts);
% ii
hesss = getHessAreaKKVec(Pjs,Pks,Pis);
Hiis = bsxfun(@times, hessFactors,hesss);
auxMat = outerProductVectorised(gradIs,gradIs);
auxMat = bsxfun(@times, mixedFactors,auxMat);
Hiis = Hiis + auxMat;
Hiis = bsxfun(@times, Hiis, cont);
[rows,cols,vals]=addToHVectorised(rows,cols,vals,Hiis,is,is,nverts);

H2_fast = sparse(rows,cols,vals,nverts*3,nverts*3);
end

function H3_fast = getAngleHessVec(FV_ref,FV_def,Ev,Eo,eta,boundaryedges,datweight)
nverts = size(FV_ref.vertices, 1);
rows = [];cols = [];vals = [];
is = Ev(~boundaryedges, 1);
js = Ev(~boundaryedges, 2);
ks = Eo(~boundaryedges, 1);
ls = Eo(~boundaryedges, 2);
areas = FV_ref.de(~boundaryedges);
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

H3_fast = sparse(rows,cols,vals,nverts*3,nverts*3);
end


