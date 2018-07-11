function H = fastMixedHessianShell(FV_ref, FV_def, Topo, Ev, Ef, Eo, boundaryedges, options)
% fast version of mixed Hessian matrix. 
% FV_ref: undeformed shell, type: FV
% FV_def: deformed shell, type: FV
% firstDerivWRTDef: bool, True if first derivative wrt. Deformed shell
% mu,lambda,eta are weights to be applied to edge length, tri area and
% dihedral angle terms, respectively. 

% for example, E = W[S, \tilde{S}]:
% firstDerivWRTDef = True, d_2 d_1 W[S, \tilde{S}]
% firstDerivWRTDef = False, d_1 d_2 W[S, \tilde{S}]

firstDerivWRTDef = options.firstDerivWRTDef;
if ~isfield(options, 'mu')
    mu = 1;
else
    mu = options.mu;
end
if ~isfield(options, 'lambda')
    lambda = 1;
else
    lambda = options.lambda;
end
eta = options.eta;
nverts = size(FV_ref.vertices,1);

% compute intermediate quantities
if ~isfield(FV_ref, 'de')
    FV_ref = precompute(FV_ref,Topo,Ev,Ef,boundaryedges);
end
if ~isfield(FV_def, 'de')
    FV_def = precompute(FV_def,Topo,Ev,Ef,boundaryedges);
end

rows=[]; cols=[]; vals=[];

[rows,cols,vals] = getMixedEdgeHessVec(rows,cols,vals,FV_ref,FV_def,Ev,Ef,Eo,mu,firstDerivWRTDef);

[rows,cols,vals] = getMixedFaceHessVec(rows,cols,vals,FV_ref,FV_def,lambda,firstDerivWRTDef);

[rows,cols,vals] = getMixedAngleHessVec(rows,cols,vals,FV_ref,FV_def,Ev,Eo,eta,boundaryedges,firstDerivWRTDef);

H = sparse(rows,cols,vals,3*nverts,3*nverts);
end

%% support functions
function Ns = getNormalVec(Pis, Pjs, Pks)
e1s = Pks-Pis;
e3s = Pks-Pjs;
Ns = cross(e1s, e3s, 2);
nlength = sqrt(sum(Ns.^2, 2));
Ns = Ns ./ repmat(nlength, 1, 3);
end

function areas = getAreaVec(Pis,Pjs,Pks)
e1s = Pks - Pis;
e3s = Pks - Pjs;
Ns = cross(e1s,e3s,2);
areas = sqrt(sum(Ns.^2, 2))./2;
end

function M = getProjectionVec(X)
tmp = ones(size(X,1), 1);
M = outerProductVectorised(X,X);
M = -M;
M(:,1,1) = tmp + M(:,1,1);
M(:,2,2) = tmp + M(:,2,2);
M(:,3,3) = tmp + M(:,3,3);
end

function M = getReflectionVec(X)
M = outerProductVectorised(X,X);
M = bsxfun(@times,-2,M);
tmp = ones(size(X,1),1);
M(:,1,1) = tmp + M(:,1,1);
M(:,2,2) = tmp + M(:,2,2);
M(:,3,3) = tmp + M(:,3,3);
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

function N = getNormal(Pi, Pj, Pk)
e1 = Pk - Pi;
e3 = Pk - Pj;
N = cross(e1, e3, 2);
nLeng = sqrt(sum(N.^2, 2));
N = N./ nLeng;
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

function Grad = getThetaGradKVec(Pis, Pjs, Pks)
es = Pjs-Pis;
normals = getNormalVec(Pis,Pjs,Pks);
areas = getAreaVec(Pis,Pjs,Pks);
elens = sqrt(sum(es.^2, 2));

Grad = normals;
tmps = -0.5.*elens ./ areas;
Grad = bsxfun(@times,tmps,Grad);
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

% Second order derivative wrt. dihedral angle
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

function M = getMassMat(FV)
nverts = size(FV.vertices, 1);
%M = eye(nverts);
% M = zeros(nverts, nverts);
% for i=1:nverts
%     
%     ind =  sum((FV.faces(:,:)==i),2);
%     ind = logical(ind);
%     
%     M(i,i) = sum(FV.at(ind));
%     
% end
M = speye(nverts);
end

function [rows,cols,vals] = getMixedEdgeHessVec(rows,cols,vals,FV_ref,FV_def,Ev,Ef,Eo,mu,firstDerivWRTDef) %H1_fast
nverts = size(FV_ref.vertices, 1);
nEdge = size(Ev,1);
f1s = Ef(:,1);
f2s = Ef(:,2);
is = Ev(:,1);
js = Ev(:,2);
ks = Eo(:,1);
ls = Eo(:,2);
nks = (ks~=0);
nls = (ls~=0);
Pis = FV_def.vertices(is,:);
Pjs = FV_def.vertices(js,:);
edges = Pjs-Pis;
defEdgeLength = sqrt(sum(edges.^2,2));
%defGradi = -edges./defEdgeLength;
defGradi = bsxfun(@times, edges, -1./defEdgeLength);
%defGradj = edges./defEdgeLength;
defGradj = bsxfun(@times, edges, 1./defEdgeLength);
% get undeformed quantities
Pis = FV_ref.vertices(is,:);
Pjs = FV_ref.vertices(js,:);
undefEdge = Pjs-Pis;
undefEdgeLengthSqr = sum(undefEdge.^2,2);
undefEdgeLength = sqrt(undefEdgeLengthSqr);
%undefEdge = undefEdge ./ undefEdgeLength;
undefEdge = bsxfun(@times, undefEdge, 1./undefEdgeLength);
Pks = FV_ref.vertices(ks(nks),:);
Pls = FV_ref.vertices(ls(nls),:);
% compute area
areas = zeros(nEdge,1);
areas(nks,:) = areas(nks,:) + FV_ref.at(f1s(nks))./3;
areas(nls,:) = areas(nls,:) + FV_ref.at(f2s(nls))./3;
% area grad
areak = zeros(nEdge,3);
areal = zeros(nEdge,3);
areai = zeros(nEdge,3);
areaj = zeros(nEdge,3);
% if k~=0
tmps = getAreaGradKVec(Pis(nks,:),Pjs(nks,:),Pks);
areak(nks,:) = tmps;
tmps = getAreaGradKVec(Pjs(nks,:),Pks,Pis(nks,:));
areai(nks,:) = areai(nks,:) + tmps;
tmps = getAreaGradKVec(Pks,Pis(nks,:),Pjs(nks,:));
areaj(nks,:) = areaj(nks,:) + tmps;
% if l~=0
tmps = getAreaGradKVec(Pjs(nls,:),Pis(nls,:),Pls);
areal(nls,:) = tmps;
tmps = getAreaGradKVec(Pls,Pjs(nls,:),Pis(nls,:));
areai(nls,:) = areai(nls,:) + tmps;
tmps = getAreaGradKVec(Pis(nls,:),Pls,Pjs(nls,:));
areaj(nls,:) = areaj(nls,:) + tmps;

areaFactor = 2 * (defEdgeLength - undefEdgeLength) ./ (3 * undefEdgeLengthSqr);
edgeFactor = 2*areas .* ( 2*defEdgeLength - undefEdgeLength) ./ (undefEdgeLengthSqr .* undefEdgeLength);

areak = bsxfun(@times, areaFactor, areak);
areal = bsxfun(@times, areaFactor, areal);
areai = bsxfun(@times, areaFactor, areai);
tmps = bsxfun(@times, edgeFactor, undefEdge);
areai = areai + tmps;
areaj = bsxfun(@times, areaFactor, areaj);
tmps = bsxfun(@times, edgeFactor, undefEdge);
areaj = areaj - tmps;

% k*
Hki = outerProductVectorised(areak,defGradi);
Hki = bsxfun(@times, Hki, mu);
if firstDerivWRTDef
    [rows,cols,vals] = addToHVectorised(rows,cols,vals,permute(Hki,[1 3 2]),ks(nks),is(nks),nverts);
else
    [rows,cols,vals] = addToHVectorised(rows,cols,vals,Hki,is(nks),ks(nks),nverts);
end
Hkj = outerProductVectorised(areak,defGradj);
Hkj = bsxfun(@times, Hkj, mu);
if firstDerivWRTDef
    [rows,cols,vals] = addToHVectorised(rows,cols,vals,permute(Hkj,[1 3 2]),ks(nks),js(nks),nverts);
else
    [rows,cols,vals] = addToHVectorised(rows,cols,vals,Hkj,js(nks),ks(nks),nverts);
end
% l*
Hli = outerProductVectorised(areal,defGradi);
Hli = bsxfun(@times, Hli, mu);
if firstDerivWRTDef
    [rows,cols,vals] = addToHVectorised(rows,cols,vals,permute(Hli,[1 3 2]),ls(nls),is(nls),nverts);
else
    [rows,cols,vals] = addToHVectorised(rows,cols,vals,Hli,is(nls),ls(nls),nverts);
end
Hlj = outerProductVectorised(areal,defGradj);
Hlj = bsxfun(@times, Hlj, mu);
if firstDerivWRTDef
    [rows,cols,vals] = addToHVectorised(rows,cols,vals,permute(Hlj,[1 3 2]),ls(nls),js(nls),nverts);
else
    [rows,cols,vals] = addToHVectorised(rows,cols,vals,Hlj,js(nls),ls(nls),nverts);
end
% i*
Hii = outerProductVectorised(areai,defGradi);
Hii = bsxfun(@times, Hii, mu);
if firstDerivWRTDef
    [rows,cols,vals] = addToHVectorised(rows,cols,vals,permute(Hii,[1 3 2]),is,is,nverts);
else
    [rows,cols,vals] = addToHVectorised(rows,cols,vals,Hii,is,is,nverts);
end
Hij = outerProductVectorised(areai,defGradj);
Hij = bsxfun(@times, Hij, mu);
if firstDerivWRTDef
    [rows,cols,vals] = addToHVectorised(rows,cols,vals,permute(Hij,[1 3 2]),is,js,nverts);
else
    [rows,cols,vals] = addToHVectorised(rows,cols,vals,Hij,js,is,nverts);
end
% j*
Hjj = outerProductVectorised(areaj,defGradj);
Hjj = bsxfun(@times, Hjj, mu);
if firstDerivWRTDef
    [rows,cols,vals] = addToHVectorised(rows,cols,vals,permute(Hjj,[1 3 2]),js,js,nverts);
else
    [rows,cols,vals] = addToHVectorised(rows,cols,vals,Hjj,js,js,nverts);
end
Hji = outerProductVectorised(areaj,defGradi);
Hji = bsxfun(@times, Hji, mu);
if firstDerivWRTDef
    [rows,cols,vals] = addToHVectorised(rows,cols,vals,permute(Hji,[1 3 2]),js,is,nverts);
else
    [rows,cols,vals] = addToHVectorised(rows,cols,vals,Hji,is,js,nverts);
end
end

function [rows,cols,vals] = getMixedFaceHessVec(rows,cols,vals,FV_ref,FV_def,lambda,firstDerivWRTDef)

nverts = size(FV_ref.vertices, 1);
F = FV_ref.faces;
nFace = size(F,1);
is = F(:,1);
js = F(:,2);
ks = F(:,3);
% deformed quantity
Pis = FV_def.vertices(is,:);
Pjs = FV_def.vertices(js,:);
Pks = FV_def.vertices(ks,:);
factor = 2*FV_def.at;
defGradk = getAreaGradKVec(Pis,Pjs,Pks);
defGradi = getAreaGradKVec(Pjs,Pks,Pis);
defGradj = getAreaGradKVec(Pks,Pis,Pjs);
% undeformed
Pis = FV_ref.vertices(is,:);
Pjs = FV_ref.vertices(js,:);
Pks = FV_ref.vertices(ks,:);
factor = -factor ./ FV_ref.at.^2;
undefGradk = getAreaGradKVec(Pis,Pjs,Pks);
undefGradk = bsxfun(@times, factor, undefGradk);
undefGradi = getAreaGradKVec(Pjs,Pks,Pis);
undefGradi = bsxfun(@times, factor, undefGradi);
undefGradj = getAreaGradKVec(Pks,Pis,Pjs);
undefGradj = bsxfun(@times, factor, undefGradj);
% k
Hkk = outerProductVectorised(undefGradk,defGradk);
Hkk = bsxfun(@times, Hkk, lambda);
if firstDerivWRTDef
    [rows,cols,vals] = addToHVectorised(rows,cols,vals,permute(Hkk,[1 3 2]),ks,ks,nverts);
else
    [rows,cols,vals] = addToHVectorised(rows,cols,vals,Hkk,ks,ks,nverts);
end

Hki = outerProductVectorised(undefGradk,defGradi);
Hki = bsxfun(@times, Hki, lambda);
% note: transpose Hki first
if firstDerivWRTDef
    [rows,cols,vals] = addToHVectorised(rows,cols,vals,permute(Hki,[1 3 2]),ks,is,nverts);
else
    [rows,cols,vals] = addToHVectorised(rows,cols,vals,Hki,is,ks,nverts);
end 
Hkj = outerProductVectorised(undefGradk,defGradj);
Hkj = bsxfun(@times, Hkj, lambda);
if firstDerivWRTDef
    [rows,cols,vals] = addToHVectorised(rows,cols,vals,permute(Hkj,[1 3 2]),ks,js,nverts);
else
    [rows,cols,vals] = addToHVectorised(rows,cols,vals,Hkj,js,ks,nverts);
end 
% i*
Hii = outerProductVectorised(undefGradi,defGradi);
Hii = bsxfun(@times, Hii, lambda);
if firstDerivWRTDef
    [rows,cols,vals] = addToHVectorised(rows,cols,vals,permute(Hii,[1 3 2]),is,is,nverts);
else
    [rows,cols,vals] = addToHVectorised(rows,cols,vals,Hii,is,is,nverts);
end 
Hij = outerProductVectorised(undefGradi,defGradj);
Hij = bsxfun(@times, Hij, lambda);
if firstDerivWRTDef
    [rows,cols,vals] = addToHVectorised(rows,cols,vals,permute(Hij,[1 3 2]),is,js,nverts);
else
    [rows,cols,vals] = addToHVectorised(rows,cols,vals,Hij,js,is,nverts);
end 
Hik = outerProductVectorised(undefGradi,defGradk);
Hik = bsxfun(@times, Hik, lambda);
if firstDerivWRTDef
    [rows,cols,vals] = addToHVectorised(rows,cols,vals,permute(Hik,[1 3 2]),is,ks,nverts);
else
    [rows,cols,vals] = addToHVectorised(rows,cols,vals,Hik,ks,is,nverts);
end 
% j*
Hjj = outerProductVectorised(undefGradj,defGradj);
Hjj = bsxfun(@times, Hjj, lambda);
if firstDerivWRTDef
    [rows,cols,vals] = addToHVectorised(rows,cols,vals,permute(Hjj,[1 3 2]),js,js,nverts);
else
    [rows,cols,vals] = addToHVectorised(rows,cols,vals,Hjj,js,js,nverts);
end 
Hji = outerProductVectorised(undefGradj,defGradi);
Hji = bsxfun(@times, Hji, lambda);
if firstDerivWRTDef
    [rows,cols,vals] = addToHVectorised(rows,cols,vals,permute(Hji,[1 3 2]),js,is,nverts);
else
    [rows,cols,vals] = addToHVectorised(rows,cols,vals,Hji,is,js,nverts);
end 
Hjk = outerProductVectorised(undefGradj,defGradk);
Hjk = bsxfun(@times, Hjk, lambda);
if firstDerivWRTDef
    [rows,cols,vals] = addToHVectorised(rows,cols,vals,permute(Hjk,[1 3 2]),js,ks,nverts);
else
    [rows,cols,vals] = addToHVectorised(rows,cols,vals,Hjk,ks,js,nverts);
end 

end


function [rows,cols,vals] = getMixedAngleHessVec(rows,cols,vals,FV_ref,FV_def,Ev,Eo,eta,boundaryedges,firstDerivWRTDef)
nverts = size(FV_ref.vertices, 1);
is = Ev(~boundaryedges,1);
js = Ev(~boundaryedges,2);
ks = Eo(~boundaryedges,1);
ls = Eo(~boundaryedges,2);
% deformed quantity
Pis = FV_def.vertices(is,:);
Pjs = FV_def.vertices(js,:);
Pks = FV_def.vertices(ks,:);
Pls = FV_def.vertices(ls,:);
defGradk = getThetaGradKVec(Pis,Pjs,Pks);
defGradl = getThetaGradKVec(Pjs,Pis,Pls);
defGradi = getThetaGradIVec(Pis,Pjs,Pks,Pls);
defGradj = getThetaGradJVec(Pis,Pjs,Pks,Pls);
% undeformed quantity
Pis = FV_ref.vertices(is,:);
Pjs = FV_ref.vertices(js,:);
Pks = FV_ref.vertices(ks,:);
Pls = FV_ref.vertices(ls,:);
area = 3*FV_ref.de(~boundaryedges);
delTheta = FV_def.Te(~boundaryedges)-FV_ref.Te(~boundaryedges);
edges = Pjs-Pis;
elengthSqr = sum(edges.^2,2);
% derivatives
factorGradTheta = -2*elengthSqr./area;
factorGradArea = -2*delTheta.*elengthSqr./(area.^2);
factorGradEdgeLengthSqr = 4*delTheta./area;
% d_k
gradTheta = getThetaGradKVec(Pis,Pjs,Pks);
gradArea = getAreaGradKVec(Pis,Pjs,Pks);
gradTheta = bsxfun(@times, gradTheta, factorGradTheta);
gradArea = bsxfun(@times, gradArea, factorGradArea);
gradk = gradTheta + gradArea;
% d_l
gradTheta = getThetaGradKVec(Pjs,Pis,Pls);
gradArea = getAreaGradKVec(Pjs,Pis,Pls);
gradTheta = bsxfun(@times, gradTheta, factorGradTheta);
gradArea = bsxfun(@times, gradArea, factorGradArea);
gradl = gradTheta + gradArea;
% d_i
gradTheta = getThetaGradIVec(Pis,Pjs,Pks,Pls);
gradArea = getAreaGradKVec(Pjs,Pks,Pis);
gradTheta = bsxfun(@times, gradTheta, factorGradTheta);
gradArea = bsxfun(@times, gradArea, factorGradArea);
gradi = gradTheta + gradArea;
gradArea = getAreaGradKVec(Pls,Pjs,Pis);
gradArea = bsxfun(@times, gradArea, factorGradArea);
gradi = gradi + gradArea;
tmps = bsxfun(@times, factorGradEdgeLengthSqr, Pis-Pjs);
gradi = gradi + tmps;
% d_j
gradTheta = getThetaGradJVec(Pis,Pjs,Pks,Pls);
gradArea = getAreaGradKVec(Pks,Pis,Pjs);
gradTheta = bsxfun(@times, gradTheta, factorGradTheta);
gradArea = bsxfun(@times, gradArea, factorGradArea);
gradj = gradTheta + gradArea;
gradArea = getAreaGradKVec(Pis,Pls,Pjs);
gradArea = bsxfun(@times, gradArea, factorGradArea);
gradj = gradj + gradArea;
tmps = bsxfun(@times, factorGradEdgeLengthSqr, Pjs-Pis);
gradj = gradj + tmps;
% k*
Hkk = outerProductVectorised(gradk,defGradk);
Hkk = bsxfun(@times, Hkk, eta);
if firstDerivWRTDef
    [rows,cols,vals] = addToHVectorised(rows,cols,vals,permute(Hkk,[1 3 2]),ks,ks,nverts);
else
    [rows,cols,vals] = addToHVectorised(rows,cols,vals,Hkk,ks,ks,nverts);
end 
Hkl = outerProductVectorised(gradk,defGradl);
Hkl = bsxfun(@times, Hkl, eta);
if firstDerivWRTDef
    [rows,cols,vals] = addToHVectorised(rows,cols,vals,permute(Hkl,[1 3 2]),ks,ls,nverts);
else
    [rows,cols,vals] = addToHVectorised(rows,cols,vals,Hkl,ls,ks,nverts);
end 
Hki = outerProductVectorised(gradk,defGradi);
Hki = bsxfun(@times, Hki, eta);
if firstDerivWRTDef
    [rows,cols,vals] = addToHVectorised(rows,cols,vals,permute(Hki,[1 3 2]),ks,is,nverts);
else
    [rows,cols,vals] = addToHVectorised(rows,cols,vals,Hki,is,ks,nverts);
end 
Hkj = outerProductVectorised(gradk,defGradj);
Hkj = bsxfun(@times, Hkj, eta);
if firstDerivWRTDef
    [rows,cols,vals] = addToHVectorised(rows,cols,vals,permute(Hkj,[1 3 2]),ks,js,nverts);
else
    [rows,cols,vals] = addToHVectorised(rows,cols,vals,Hkj,js,ks,nverts);
end 
% l*
Hlk = outerProductVectorised(gradl,defGradk);
Hlk = bsxfun(@times, Hlk, eta);
if firstDerivWRTDef
    [rows,cols,vals] = addToHVectorised(rows,cols,vals,permute(Hlk,[1 3 2]),ls,ks,nverts);
else
    [rows,cols,vals] = addToHVectorised(rows,cols,vals,Hlk,ks,ls,nverts);
end 
Hll = outerProductVectorised(gradl,defGradl);
Hll = bsxfun(@times, Hll, eta);
if firstDerivWRTDef
    [rows,cols,vals] = addToHVectorised(rows,cols,vals,permute(Hll,[1 3 2]),ls,ls,nverts);
else
    [rows,cols,vals] = addToHVectorised(rows,cols,vals,Hll,ls,ls,nverts);
end 
Hli = outerProductVectorised(gradl,defGradi);
Hli = bsxfun(@times, Hli, eta);
if firstDerivWRTDef
    [rows,cols,vals] = addToHVectorised(rows,cols,vals,permute(Hli,[1 3 2]),ls,is,nverts);
else
    [rows,cols,vals] = addToHVectorised(rows,cols,vals,Hli,is,ls,nverts);
end 
Hlj = outerProductVectorised(gradl,defGradj);
Hlj = bsxfun(@times, Hlj, eta);
if firstDerivWRTDef
    [rows,cols,vals] = addToHVectorised(rows,cols,vals,permute(Hlj,[1 3 2]),ls,js,nverts);
else
    [rows,cols,vals] = addToHVectorised(rows,cols,vals,Hlj,js,ls,nverts);
end 
% i*
Hik = outerProductVectorised(gradi,defGradk);
Hik = bsxfun(@times, Hik, eta);
if firstDerivWRTDef
    [rows,cols,vals] = addToHVectorised(rows,cols,vals,permute(Hik,[1 3 2]),is,ks,nverts);
else
    [rows,cols,vals] = addToHVectorised(rows,cols,vals,Hik,ks,is,nverts);
end 
Hil = outerProductVectorised(gradi,defGradl);
Hil = bsxfun(@times, Hil, eta);
if firstDerivWRTDef
    [rows,cols,vals] = addToHVectorised(rows,cols,vals,permute(Hil,[1 3 2]),is,ls,nverts);
else
    [rows,cols,vals] = addToHVectorised(rows,cols,vals,Hil,ls,is,nverts);
end 
Hii = outerProductVectorised(gradi,defGradi);
Hii = bsxfun(@times, Hii, eta);
if firstDerivWRTDef
    [rows,cols,vals] = addToHVectorised(rows,cols,vals,permute(Hii,[1 3 2]),is,is,nverts);
else
    [rows,cols,vals] = addToHVectorised(rows,cols,vals,Hii,is,is,nverts);
end 
Hij = outerProductVectorised(gradi,defGradj);
Hij = bsxfun(@times, Hij, eta);
if firstDerivWRTDef
    [rows,cols,vals] = addToHVectorised(rows,cols,vals,permute(Hij,[1 3 2]),is,js,nverts);
else
    [rows,cols,vals] = addToHVectorised(rows,cols,vals,Hij,js,is,nverts);
end 
% j*
Hjk = outerProductVectorised(gradj,defGradk);
Hjk = bsxfun(@times, Hjk, eta);
if firstDerivWRTDef
    [rows,cols,vals] = addToHVectorised(rows,cols,vals,permute(Hjk,[1 3 2]),js,ks,nverts);
else
    [rows,cols,vals] = addToHVectorised(rows,cols,vals,Hjk,ks,js,nverts);
end 
Hjl = outerProductVectorised(gradj,defGradl);
Hjl = bsxfun(@times, Hjl, eta);
if firstDerivWRTDef
    [rows,cols,vals] = addToHVectorised(rows,cols,vals,permute(Hjl,[1 3 2]),js,ls,nverts);
else
    [rows,cols,vals] = addToHVectorised(rows,cols,vals,Hjl,ls,js,nverts);
end 
Hji = outerProductVectorised(gradj,defGradi);
Hji = bsxfun(@times, Hji, eta);
if firstDerivWRTDef
    [rows,cols,vals] = addToHVectorised(rows,cols,vals,permute(Hji,[1 3 2]),js,is,nverts);
else
    [rows,cols,vals] = addToHVectorised(rows,cols,vals,Hji,is,js,nverts);
end 
Hjj = outerProductVectorised(gradj,defGradj);
Hjj = bsxfun(@times, Hjj, eta);
if firstDerivWRTDef
    [rows,cols,vals] = addToHVectorised(rows,cols,vals,permute(Hjj,[1 3 2]),js,js,nverts);
else
    [rows,cols,vals] = addToHVectorised(rows,cols,vals,Hjj,js,js,nverts);
end 

end