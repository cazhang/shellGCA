function [ c,g,H ] = fastShellGradAndHessDef( x,FVs,Ev,Eo,Ef,mu,lambda,eta,datweights,boundaryedges )
%vectorised version of computing cost, grad, hess wrt. deformed shell
%   Detailed explanation goes here

nmesh = length(FVs);
v = reshape(x, length(x)/3, 3);
nverts = size(v, 1);
Topo = FVs{1}.faces;
FV_def.vertices = v;
FV_def.faces = FVs{1}.faces;
% compute intermediate quantities
if ~isfield(FV_def, 'de')
    FV_def = precompute(FV_def,Topo,Ev,Ef,boundaryedges);
end
for i=1:nmesh
    if ~isfield(FVs{i}, 'de')
        FVs{i} = precompute(FVs{i},Topo,Ev,Ef,boundaryedges);
    end
end
if nargout>2
    % Hessian required
    rows=[]; cols=[]; vals=[];
end

f=[];

if nargout>1
    g = zeros(nverts, 3);
end

for m=1:nmesh
    if datweights(m) == 0
        continue;
    end
    % Edge lengths
    f1 = mu*datweights(m).*FVs{m}.de ./ FVs{m}.le.^2 .*((FVs{m}.le-FV_def.le).^2);
    f = [f; f1];
    % Triangle areas
    f2 = lambda*datweights(m)./FVs{m}.at .* (FV_def.at-FVs{m}.at).^2;
    f = [f; f2];
    % Dihedral angles
    f3 = eta*datweights(m).* (FVs{m}.le(~boundaryedges)).^2 ./ ...
        (3.*FVs{m}.de(~boundaryedges)) .* (FV_def.Te(~boundaryedges)-FVs{m}.Te(~boundaryedges)).^2;  
    f = [f; f3];
    
    if nargout > 1 % gradient required
        
        % edge length term       
        g1_fast = getEdgeGradVec(FVs{m},FV_def,Ev,mu,datweights(m));
        
        % triangle area term
        g2_fast = getFaceGradVec(FVs{m},FV_def, lambda, datweights(m));
                       
        % dihedral angle term
        g3_fast = getAngleGradVec(FVs{m}, FV_def, Ev, Eo, boundaryedges,eta,datweights(m));
        
        g = g + (g1_fast + g2_fast + g3_fast);
        
    end
    
    if nargout > 2 % hessian required
        % edge length term
        [rows,cols,vals] = getEdgeHessVec(rows,cols,vals,FVs{m}, FV_def,Ev,mu,datweights(m));
        
        % triangle area term        
        [rows,cols,vals] = getFaceHessVec(rows,cols,vals,FVs{m},FV_def,lambda,datweights(m));
        
        % dihedral angle terms       
        [rows,cols,vals] = getAngleHessVec(rows,cols,vals,FVs{m},FV_def,Ev,Eo,eta,boundaryedges,datweights(m));
        
    end
end

if nargout > 2
    H = sparse(rows,cols,vals,3*nverts,3*nverts);
end

if nargout > 1
    % gradient required
    g = g(:);
end

c = sum(f);

end

%% SUPPORTING FUNCTIONS BELOW

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

function Grad = getAreaGradKVec(Pis,Pjs,Pks)
es = Pjs-Pis;
normals = getNormalVec(Pis,Pjs,Pks);
Grad = cross(0.5.*normals, es, 2);
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

function [rows,cols,vals] = getEdgeHessVec(rows,cols,vals,FV_ref,FV_def,Ev,mu,datweight) %H1_fast
nverts = size(FV_ref.vertices, 1);
%rows=[]; cols=[]; vals=[];
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
%H1_fast = sparse(rows,cols,vals,nverts*3,nverts*3);
end

function [rows,cols,vals] = getFaceHessVec(rows,cols,vals,FV_ref,FV_def,lambda,datweight) %H2_fast =
F = FV_ref.faces;
nverts = size(FV_ref.vertices, 1);
%rows = []; cols = []; vals = [];
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

%H2_fast = sparse(rows,cols,vals,nverts*3,nverts*3);
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