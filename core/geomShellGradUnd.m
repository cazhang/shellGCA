function [ cost, grad, H ] = geomShellGradUnd( x, FVs, Topo, Ev, Ef, Eo, boundaryedges,options )
%compute the cost and gradient wrt. undeformed shell-FV_ref.
%   x: initial value of a deformed shell
%   FVs: a set of deformed shell

v = reshape(x, length(x)/3, 3);

nverts = size(v, 1);
if nargout > 1
    g = zeros(nverts, 3);
end
if nargout > 2
   H = zeros(nverts*3, nverts*3); 
end

mu = options.mu;
lambda = options.lambda;
eta = options.eta;

muHalf = mu / 2;
lambdaQuarter = lambda / 4;
muHalfPlusLambdaQuarter = muHalf + lambdaQuarter;
const = mu + lambdaQuarter;
nmesh = length(FVs);
if ~isfield(options, 'datweights')
    datweights(1:nmesh) = 1;
else
    datweights = options.datweights;
end

% cost term
f = zeros(1, nmesh);
FV_ref.vertices = v;
FV_ref.faces = Topo;
F = Topo;
nEdge = length(Ev);
nFace = size(F, 1);
% compute intermediate quantities
if ~isfield(FV_ref, 'de')
    FV_ref = precompute(FV_ref,Topo,Ev,Ef,boundaryedges);
end

for m = 1:nmesh
    if ~isfield(FVs{m}, 'de')
        FVs{m} = precompute(FVs{m},Topo,Ev,Ef,boundaryedges);
    end
end
% make f
for m=1:nmesh
    f(m) = geomDSD(FV_ref, FVs{m}, Topo, Ev, Ef, eta, 1 );
end
cost = sum(f);
% make gradient
if nargout > 1
    
    for m=1:nmesh
        % membrane energy
        %[g1,loop_out] = getMemGradLoopUnd(FV_ref, FVs{m}, Topo, mu, lambda, datweights(m));
        [g1_fast,vec_out] = getMemGradVecUnd(FV_ref,FVs{m},Topo,mu,lambda,datweights(m));
        %res = norm(g1(:)-g1_fast(:));
        %disp(res);
        % bending energy
        g3 = getAngleGradVecUnd(FV_ref,FVs{m},Ev,Eo,boundaryedges,eta,datweights(m));
        
        g = g + g1_fast + g3;
    end
    
end

if nargout > 2
    %% TODO: get Hessian for deformed shell
    
    for m=1:nmesh
        % mem part
        H1 = zeros(nverts*3, nverts*3);
        for fid = 1:nFace
            nodesIdx = Topo(fid,:);
            % get fixed edges
            % get deformed quantities
            nodes = FVs{m}.vertices(nodesIdx,:);
            % compute volume
            for j=0:2
                fixedEdges(j+1,:) = nodes(mod(j+2,3)+1,:) - nodes(mod(j+1,3)+1,:);
                defLengthSqr(j+1) = norm(fixedEdges(j+1,:))^2;
            end
            temp = cross(nodes(2,:)-nodes(1,:), nodes(3,:)-nodes(2,:));
            volDefSqr = norm(temp)^2 / 4;
            volDef = sqrt(volDefSqr);
            
            % get undeformed quantities
            nodes = FV_ref.vertices(nodesIdx,:); % each row is one verts
            for j=0:2
                undefEdges(j+1,:) = nodes(mod(j+2,3)+1,:) - nodes(mod(j+1,3)+1,:);
            end
            % compute volume
            temp = cross(undefEdges(1,:), undefEdges(2,:));
            volUndefSqr = norm(temp)^2 / 4;
            volUndef = sqrt(volUndefSqr);
            % trace term
            traceTerm = 0;
            for i=0:2
                traceTerm = traceTerm - undefEdges(mod(i+1,3)+1,:)*...
                    undefEdges(mod(i+2,3)+1,:)' * defLengthSqr(i+1);
            end
            
            gradTrace = zeros(3,3);
            for i=0:2
                for j=1:3
                    gradTrace(i+1,j)=defLengthSqr(i+1)*(undefEdges(mod(i+1,3)+1,j)-...
                        undefEdges(mod(i+2,3)+1,j))+ undefEdges(i+1,j)*...
                        (defLengthSqr(mod(i+1,3)+1) - defLengthSqr(mod(i+2,3)+1));
                    
                end
            end
            % precompute area grad
            gradArea = zeros(3,3);
            for i=0:2
                gradArea(i+1,:) = getAreaGradK(nodes(mod(i+1,3)+1,:),...
                    nodes(mod(i+2,3)+1,:), nodes(i+1,:));
                
            end
            areaFactor=0.125*mu*traceTerm+lambdaQuarter*volDefSqr;
            negHessAreaFactor=muHalfPlusLambdaQuarter*(log(volDefSqr/volUndefSqr)-2)...
                + const + areaFactor/volUndefSqr;
            mixedAreaFactor=2*(areaFactor/volUndefSqr+muHalfPlusLambdaQuarter)/volUndef;
            mixedFactor = -0.125*mu/volUndefSqr;
            hessTraceFactor=0.125*mu/volUndef;
            % compute local mat
            tensorProduct = zeros(3,3);
            hess = zeros(3,3);
            auxMat = zeros(3,3);
            % i=j
            for i=0:2
                hess = getHessAreaKK(nodes(mod(i+1,3)+1,:),nodes(mod(i+2,3)+1,:),nodes(i+1,:));
                hess = -1*negHessAreaFactor*hess;
                tensorProduct = gradArea(i+1,:)'*gradTrace(i+1,:);
                hess = hess + mixedFactor.*tensorProduct;
                tensorProduct = gradTrace(i+1,:)'*gradArea(i+1,:);
                hess = hess + mixedFactor.*tensorProduct;
                
                tensorProduct = gradArea(i+1,:)'*gradArea(i+1,:);
                hess = hess + mixedAreaFactor.*tensorProduct;
                
                temp = 2*defLengthSqr(i+1)*hessTraceFactor;
                temp = repmat(temp, [1 3]);
                hess = hess + diag(temp);
                H1 = addToH(H1,hess,nodesIdx(i+1),nodesIdx(i+1),nverts);
            end
            % i~=j
            for i=0:2
                hess = getHessAreaIK(nodes(i+1,:),nodes(mod(i+1,3)+1,:),...
                    nodes(mod(i+2,3)+1,:));
                hess = -1*negHessAreaFactor*hess;
                tensorProduct = gradArea(i+1,:)'*gradTrace(mod(i+2,3)+1,:);
                hess = hess + mixedFactor.*tensorProduct;
                tensorProduct = gradTrace(i+1,:)'*gradArea(mod(i+2,3)+1,:);
                hess = hess + mixedFactor.*tensorProduct;
                
                tensorProduct = gradArea(i+1,:)'*gradArea(mod(i+2,3)+1,:);
                hess = hess + mixedAreaFactor.*tensorProduct;
                
                temp = hessTraceFactor*(defLengthSqr(mod(i+1,3)+1)-...
                    defLengthSqr(i+1) - defLengthSqr(mod(i+2,3)+1));
                temp = repmat(temp, [1 3]);
                hess = hess + diag(temp);
                H1 = addToH(H1,hess,nodesIdx(i+1),nodesIdx(mod(i+2,3)+1),nverts);
                H1 = addToH(H1,hess',nodesIdx(mod(i+2,3)+1),nodesIdx(i+1),nverts);
            end            
        end
        H1 = datweights(m).*H1;
        H = H + H1;
               
        % bending part: dihedral angle       
        H3 = zeros(nverts*3, nverts*3);
        for eid=1:nEdge
            if ~boundaryedges(eid)
                delTheta = FV_ref.Te(eid)-FVs{m}.Te(eid);
                area = 3*FV_ref.de(eid);
                i = Ev(eid,1);
                j = Ev(eid,2);
                k = Eo(eid,1);
                l = Eo(eid,2);
                Pi = FV_ref.vertices(i,:);
                Pj = FV_ref.vertices(j,:);
                Pk = FV_ref.vertices(k,:);
                Pl = FV_ref.vertices(l,:);
                
                e = Pj-Pi;
                elengthSqr = norm(e)^2;
                % compute first derivative of angle
                thetak = getThetaGradK(Pi,Pj,Pk);
                thetal = getThetaGradK(Pj,Pi,Pl);
                thetai = getThetaGradI(Pi,Pj,Pk,Pl);
                thetaj = getThetaGradJ(Pi,Pj,Pk,Pl);
                % compute first derivative of area
                areak = getAreaGradK(Pi,Pj,Pk);
                areal = getAreaGradK(Pj,Pi,Pl);
                areai = getAreaGradK(Pj,Pk,Pi);
                temp = getAreaGradK(Pl,Pj,Pi);
                areai = areai + temp;
                areaj = getAreaGradK(Pk,Pi,Pj);
                temp = getAreaGradK(Pi,Pl,Pj);
                areaj = areaj + temp;
                
                % *k
                auxVec = elengthSqr.*thetak - (delTheta*elengthSqr/area).*areak;
                temp = 2.*thetak - (delTheta/area).*areak;
                % kk
                Hkk = thetak'*auxVec;
                auxMat = getHessThetaKK(Pi,Pj,Pk);
                Hkk = Hkk+ (delTheta*elengthSqr).*auxMat;
                auxMat = areak'*auxVec;
                Hkk = Hkk-(delTheta/area).*auxMat;
                auxMat = getHessAreaKK(Pi,Pj,Pk);
                Hkk = Hkk - (0.5*delTheta^2*elengthSqr/area).*auxMat;
                Hkk = (2/area).*Hkk;
                %Hkk = 3*eta*resweights(res).*Hkk;
                H3 = addToH(H3,Hkk,k,k,nverts);
                % lk & kl
                Hlk = thetal'*auxVec;
                auxMat = areal'*auxVec;
                Hlk = Hlk - (delTheta/area).*auxMat;
                Hlk = (2/area).*Hlk;
                %Hlk = 3*eta*resweights(res).*Hlk;
                H3 = addToH(H3,Hlk,l,k,nverts);
                H3 = addToH(H3,Hlk',k,l,nverts);
                % ik & ki
                Hik = thetai'*auxVec;
                auxMat = getHessThetaIK(Pi,Pj,Pk);
                Hik = Hik + (delTheta*elengthSqr).*auxMat;
                auxMat = areai'*auxVec;
                Hik = Hik - (delTheta/area).*auxMat;
                auxMat = getHessAreaIK(Pi,Pj,Pk);
                Hik = Hik - (0.5*delTheta^2*elengthSqr/area).*auxMat;
                auxMat = e'*temp;
                Hik = Hik - delTheta.*auxMat;
                Hik = (2/area).*Hik;
                %Hik = 3*eta*resweights(res).*Hik;
                H3 = addToH(H3,Hik,i,k,nverts);
                H3 = addToH(H3,Hik',k,i,nverts);
                % jk & kj
                Hjk = thetaj'*auxVec;
                auxMat = getHessThetaJK(Pi,Pj,Pk);
                Hjk = Hjk + (delTheta*elengthSqr).*auxMat;
                auxMat = areaj'*auxVec;
                Hjk = Hjk - (delTheta/area).*auxMat;
                auxMat = getHessAreaIK(Pj,Pi,Pk);
                Hjk = Hjk - (0.5*delTheta^2*elengthSqr/area).*auxMat;
                auxMat = e'*temp;
                Hjk = Hjk + delTheta.*auxMat;
                Hjk = (2/area).*Hjk;
                %Hjk = 3*eta*resweights(res).*Hjk;
                H3 = addToH(H3,Hjk,j,k,nverts);
                H3 = addToH(H3,Hjk',k,j,nverts);
                
                % *l
                auxVec = elengthSqr.*thetal - (delTheta*elengthSqr/area).*areal;
                temp = 2.*thetal - (delTheta/area).*areal;
                % ll
                Hll = thetal'*auxVec;
                auxMat = getHessThetaKK(Pj,Pi,Pl);
                Hll = Hll+ (delTheta*elengthSqr).*auxMat;
                auxMat = areal'*auxVec;
                Hll = Hll-(delTheta/area).*auxMat;
                auxMat = getHessAreaKK(Pj,Pi,Pl);
                Hll = Hll - (0.5*delTheta^2*elengthSqr/area).*auxMat;
                Hll = (2/area).*Hll;
                %Hll = 3*eta*resweights(res).*Hll;
                H3 = addToH(H3,Hll,l,l,nverts);
                % il & li
                Hil = thetai'*auxVec;
                auxMat = getHessThetaJK(Pj,Pi,Pl);
                Hil = Hil + (delTheta*elengthSqr).*auxMat;
                auxMat = areai'*auxVec;
                Hil = Hil - (delTheta/area).*auxMat;
                auxMat = getHessAreaIK(Pi,Pj,Pl);
                Hil = Hil - (0.5*delTheta^2*elengthSqr/area).*auxMat;
                auxMat = e'*temp;
                Hil = Hil - delTheta.*auxMat;
                Hil = (2/area).*Hil;
                %Hil = 3*eta*resweights(res).*Hil;
                H3 = addToH(H3,Hil,i,l,nverts);
                H3 = addToH(H3,Hil',l,i,nverts);
                % jl & lj
                Hjl = thetaj'*auxVec;
                auxMat = getHessThetaIK(Pj,Pi,Pl);
                Hjl = Hjl + (delTheta*elengthSqr).*auxMat;
                auxMat = areaj'*auxVec;
                Hjl = Hjl - (delTheta/area).*auxMat;
                auxMat = getHessAreaIK(Pj,Pi,Pl);
                Hjl = Hjl - (0.5*delTheta^2*elengthSqr/area).*auxMat;
                auxMat = e'*temp;
                Hjl = Hjl + delTheta.*auxMat;
                Hjl = (2/area).*Hjl;
                %Hjl = 3*eta*resweights(res).*Hjl;
                H3 = addToH(H3,Hjl,j,l,nverts);
                H3 = addToH(H3,Hjl',l,j,nverts);
                
                % *j
                auxVec = elengthSqr.*thetaj - (delTheta*elengthSqr/area).*areaj;
                auxVec = auxVec + (2*delTheta).*e;
                temp = 2.*thetaj - (delTheta/area).*areaj;
                % jj
                Hjj = thetaj'*auxVec;
                auxMat = getHessThetaII(Pj,Pi,Pl,Pk);
                Hjj = Hjj + (delTheta*elengthSqr).*auxMat;
                auxVec = auxVec - delTheta.*e;
                auxMat = areaj'*auxVec;
                Hjj = Hjj - (delTheta/area).*auxMat;
                auxMat = getHessAreaKK(Pk,Pi,Pj);
                Hjj = Hjj - (0.5*delTheta^2*elengthSqr/area).*auxMat;
                auxMat = getHessAreaKK(Pi,Pl,Pj);
                Hjj = Hjj - (0.5*delTheta^2*elengthSqr/area).*auxMat;
                auxMat = e'*temp;
                Hjj = Hjj + delTheta.*auxMat;
                Hjj = Hjj + delTheta^2.*eye(3);
                Hjj = (2/area).*Hjj;
                %Hjj = 3*eta*resweights(res).*Hjj;
                H3 = addToH(H3,Hjj,j,j,nverts);
                % ij & ji
                auxVec = auxVec + delTheta.*e;
                Hij = thetai'*auxVec;
                auxMat = getHessThetaJI(Pi,Pj,Pk,Pl);
                Hij = Hij + (delTheta*elengthSqr).*auxMat;
                auxVec = auxVec - delTheta.*e;
                auxMat = areai'*auxVec;
                Hij = Hij - (delTheta/area).*auxMat;
                auxMat = getHessAreaIK(Pi,Pk,Pj);
                Hij = Hij - (0.5*delTheta^2*elengthSqr/area).*auxMat;
                auxMat = getHessAreaIK(Pi,Pl,Pj);
                Hij = Hij - (0.5*delTheta^2*elengthSqr/area).*auxMat;
                auxMat = e'*temp;
                Hij = Hij - delTheta.*auxMat;
                Hij = Hij - (delTheta^2).*eye(3);
                Hij = (2/area).*Hij;
                %Hij = 3*eta*resweights(res).*Hij;
                H3 = addToH(H3,Hij,i,j,nverts);
                H3 = addToH(H3,Hij',j,i,nverts);
                % *i
                auxVec = elengthSqr.*thetai - (delTheta*elengthSqr/area).*areai;
                auxVec = auxVec - (2*delTheta).*e;
                temp = 2.*thetai - (delTheta/area).*areai;
                % ii
                Hii = thetai'*auxVec;
                auxMat = getHessThetaII(Pi,Pj,Pk,Pl);
                Hii = Hii + (delTheta*elengthSqr).*auxMat;
                auxVec = auxVec + delTheta.*e;
                auxMat = areai'*auxVec;
                Hii = Hii - (delTheta/area).*auxMat;
                auxMat = getHessAreaKK(Pl,Pj,Pi);
                Hii = Hii - (0.5*delTheta^2*elengthSqr/area).*auxMat;
                auxMat = getHessAreaKK(Pj,Pk,Pi);
                Hii = Hii - (0.5*delTheta^2*elengthSqr/area).*auxMat;
                auxMat = e'*temp;
                Hii = Hii - delTheta.*auxMat;
                Hii = Hii + delTheta^2.*eye(3);
                Hii = (2/area).*Hii;
                %Hii = 3*eta*resweights(res).*Hii;
                H3 = addToH(H3,Hii,i,i,nverts);
                
            end
        end
        H3 = eta*datweights(m).*H3;
        H = H + H3;
        
    end
        
end

if nargout > 1 % gradient required
    grad = g(:);
end

if nargout > 2
    H = sparse(H);
end
end

%% supporting functions
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

%% submodular functions

function [g1, loop_out] = getMemGradLoopUnd(FV_ref, FV_def, Topo, mu, lambda, datweight)
nverts = size(FV_ref.vertices, 1);
nFace = size(FV_ref.faces, 1);
g1 = zeros(nverts, 3);
muHalf = mu / 2;
lambdaQuarter = lambda / 4;
muHalfPlusLambdaQuarter = muHalf + lambdaQuarter;
const = mu + lambdaQuarter;

for fid=1:nFace
    %disp(['edge index = ', num2str(eid)]);
    nodesIdx = Topo(fid,:);
    % get deformed quantities
    nodes = FV_def.vertices(nodesIdx,:);
    % compute volume
    for j=0:2
        fixedEdges(j+1,:) = nodes(mod(j+2,3)+1,:) - nodes(mod(j+1,3)+1,:);
        defLengthSqr(j+1) = norm(fixedEdges(j+1,:))^2;
    end
    temp = cross(nodes(2,:)-nodes(1,:), nodes(3,:)-nodes(2,:));
    volDefSqr = norm(temp)^2 / 4;
    volDef = sqrt(volDefSqr);
    
    % get undeformed quantities
    nodes = FV_ref.vertices(nodesIdx,:); % each row is one verts
    for j=0:2
        undefEdges(j+1,:) = nodes(mod(j+2,3)+1,:) - nodes(mod(j+1,3)+1,:);
    end
    % compute volume
    temp = cross(nodes(2,:)-nodes(1,:), nodes(3,:)-nodes(2,:));
    volUndefSqr = norm(temp)^2 / 4;
    volUndef = sqrt(volUndefSqr);
    % trace term
    traceTerm = 0;
    for i=0:2
        traceTerm = traceTerm - undefEdges(mod(i+1,3)+1,:)*...
            undefEdges(mod(i+2,3)+1,:)' * defLengthSqr(i+1);
    end
    factor1 = (0.125 * mu * traceTerm + lambdaQuarter * volDefSqr) / volUndefSqr;
    factor2 = muHalfPlusLambdaQuarter * log(volDefSqr/volUndefSqr)+const...
        - 2*muHalfPlusLambdaQuarter;
    factorAreaGrad = factor1 + factor2;
    factorTraceGrad = 0.125 * mu / volUndef;
    gradTrace = zeros(3,3);
    % i: index edges, j index dims
    for i=0:2
        for j=1:3
            gradTrace(i+1,j)=defLengthSqr(i+1)*(undefEdges(mod(i+1,3)+1,j)-...
                undefEdges(mod(i+2,3)+1,j))+ undefEdges(i+1,j)*...
                (defLengthSqr(mod(i+1,3)+1) - defLengthSqr(mod(i+2,3)+1));
            
        end
    end
    
    for i=0:2
        temp = getAreaGradK(nodes(mod(i+1,3)+1,:),nodes(mod(i+2,3)+1,:),...
            nodes(i+1,:));
        
        for j=1:3
            grad = factorTraceGrad * ...
                gradTrace(i+1,j) - factorAreaGrad * temp(j);
            loop_out(fid, :) = grad;
            g1(nodesIdx(i+1),j) = g1(nodesIdx(i+1),j) + grad;
        end
    end  
end
g1 = g1.*datweight;
end

function [g1_fast,vec_out] = getMemGradVecUnd(FV_ref,FV_def,Topo,mu,lambda,datweight)
nverts = size(FV_ref.vertices, 1);
nFace = size(FV_ref.faces, 1);
muHalf = mu / 2;
lambdaQuarter = lambda / 4;
muHalfPlusLambdaQuarter = muHalf + lambdaQuarter;
const = mu + lambdaQuarter;
g1_fast = zeros(nverts, 3);
is = Topo(:, 1);
js = Topo(:, 2);
ks = Topo(:, 3);

% undeformed quantities
Pis = FV_ref.vertices(is, :); % nface x 3
Pjs = FV_ref.vertices(js, :);
Pks = FV_ref.vertices(ks, :);
nodes(:,:,1) = Pis;
nodes(:,:,2) = Pjs;
nodes(:,:,3) = Pks;
undEdges(:,:,1) = Pks - Pjs;
undEdges(:,:,2) = Pis - Pks;
undEdges(:,:,3) = Pjs - Pis;
tempsUnd = cross(undEdges(:,:,1), undEdges(:,:,2));
volUndefSqrs = sum(tempsUnd.^2, 2)./4;
volUndefs = sqrt(volUndefSqrs);
% deformed quantities
Pis = FV_def.vertices(is, :); % nface x 3
Pjs = FV_def.vertices(js, :);
Pks = FV_def.vertices(ks, :);
defEdges(:,:,1) = Pks - Pjs;
defEdges(:,:,2) = Pis - Pks;
defEdges(:,:,3) = Pjs - Pis;
defLenSqrs = zeros(nFace, 3);
defLenSqrs(:,1) = sum(defEdges(:,:,1).^2, 2);
defLenSqrs(:,2) = sum(defEdges(:,:,2).^2, 2);
defLenSqrs(:,3) = sum(defEdges(:,:,3).^2, 2);

tempsDef = cross(defEdges(:,:,1), defEdges(:,:,2));
volDefSqrs = sum(tempsDef.^2, 2)./4;
volDefs = sqrt(volDefSqrs);

% trace trm
traceTerms = zeros(nFace, 1);
for i=0:2
    traceTerms = traceTerms - sum(undEdges(:,:,mod(i+1,3)+1).*...
        undEdges(:,:,mod(i+2,3)+1), 2).*defLenSqrs(:,i+1);
end

factor1 = (0.125*mu.*traceTerms + lambdaQuarter.*volDefSqrs)./volUndefSqrs;
factor2 = muHalfPlusLambdaQuarter * log(volDefSqrs./volUndefSqrs) + ...
    repmat(const, nFace, 1) - 2.*repmat(muHalfPlusLambdaQuarter,nFace,1);
factorAreaGrad = factor1 + factor2;
factorTraceGrad = (0.125*mu)./volUndefs;
% gradTrace
gradTrace = zeros(3,3,nFace);
for i=0:2
    for j=1:3
        gradTrace(i+1,j,:)=...
            defLenSqrs(:,i+1).*...
            (undEdges(:,j,mod(i+1,3)+1)-undEdges(:,j,mod(i+2,3)+1))+...
            undEdges(:,j,i+1).*...
            (defLenSqrs(:,mod(i+1,3)+1)-defLenSqrs(:,mod(i+2,3)+1));
    end
end
% local mat
for i=0:2
    temps = getAreaGradKVec(nodes(:,:,mod(i+1,3)+1),...
        nodes(:,:,mod(i+2,3)+1), nodes(:,:,i+1));
    
    for j=1:3
        grad = factorTraceGrad.*squeeze(gradTrace(i+1,j,:)) - ...
            factorAreaGrad.*temps(:,j);
        vec_out = grad;
        g1_fast(1:max(Topo(:,i+1)),j)=g1_fast(1:max(Topo(:,i+1)),j) + ...
            accumarray(Topo(:,i+1),grad);
    end
end
g1_fast = g1_fast.*datweight;
end

function g3_fast = getAngleGradVecUnd(FV_ref,FV_def,Ev,Eo,boundaryedges,eta,datweight)
nverts = size(FV_ref.vertices, 1);
g3_fast = zeros(nverts,3);
del_theta = FV_def.Te(~boundaryedges)-FV_ref.Te(~boundaryedges);
is = Ev(~boundaryedges, 1);
js = Ev(~boundaryedges, 2);
ks = Eo(~boundaryedges, 1);
ls = Eo(~boundaryedges, 2);
Pis = FV_ref.vertices(is,:);
Pjs = FV_ref.vertices(js,:);
Pks = FV_ref.vertices(ks,:);
Pls = FV_ref.vertices(ls,:);
areas = 3.*FV_ref.de(~boundaryedges);
edges = Pjs-Pis;
elengthSqr = sum(edges.^2, 2);
factorGradTheta = -2 .* del_theta .* elengthSqr ./ areas;
factorGradArea = -1 * del_theta.^2 .* elengthSqr ./ (areas.^2 );
factorGradEdgeLengthSqr = 2 * del_theta.^2 ./ areas;
% d_k
gradTheta = getThetaGradKVec(Pis,Pjs,Pks);
gradArea = getAreaGradKVec(Pis,Pjs,Pks);
gradTheta = bsxfun(@times,gradTheta, factorGradTheta);
gradArea = bsxfun(@times,gradArea, factorGradArea);
gradK = gradTheta + gradArea;
g3_fast(1:max(ks),:) = g3_fast(1:max(ks),:)+[accumarray(ks,gradK(:,1)) accumarray(ks,gradK(:,2)) accumarray(ks,gradK(:,3))];
% d_l
gradTheta = getThetaGradKVec(Pjs,Pis,Pls);
gradArea = getAreaGradKVec(Pjs,Pis,Pls);
gradTheta = bsxfun(@times,gradTheta, factorGradTheta);
gradArea = bsxfun(@times,gradArea, factorGradArea);
gradL = gradTheta + gradArea;
g3_fast(1:max(ls),:) = g3_fast(1:max(ls),:)+[accumarray(ls,gradL(:,1)) accumarray(ls,gradL(:,2)) accumarray(ls,gradL(:,3))];
% d_i
gradTheta = getThetaGradIVec(Pis,Pjs,Pks,Pls);
gradArea = getAreaGradKVec(Pjs,Pks,Pis);
gradTheta = bsxfun(@times,gradTheta, factorGradTheta);
gradArea = bsxfun(@times,gradArea, factorGradArea);
gradI = gradTheta + gradArea;
gradArea = getAreaGradKVec(Pls,Pjs,Pis);
gradArea = bsxfun(@times,gradArea, factorGradArea);
tmps = Pis-Pjs;
tmps = bsxfun(@times, tmps, factorGradEdgeLengthSqr);
gradI = gradI + gradArea + tmps;
g3_fast(1:max(is),:) = g3_fast(1:max(is),:)+[accumarray(is,gradI(:,1)) accumarray(is,gradI(:,2)) accumarray(is,gradI(:,3))];
% d_j
gradTheta = getThetaGradJVec(Pis,Pjs,Pks,Pls);
gradArea = getAreaGradKVec(Pks,Pis,Pjs);
gradTheta = bsxfun(@times,gradTheta, factorGradTheta);
gradArea = bsxfun(@times,gradArea, factorGradArea);
gradJ = gradTheta + gradArea;
gradArea = getAreaGradKVec(Pis,Pls,Pjs);
gradArea = bsxfun(@times,gradArea, factorGradArea);
tmps = Pjs-Pis;
tmps = bsxfun(@times, tmps, factorGradEdgeLengthSqr);
gradJ = gradJ + gradArea + tmps;
g3_fast(1:max(js),:) = g3_fast(1:max(js),:)+[accumarray(js,gradJ(:,1)) accumarray(js,gradJ(:,2)) accumarray(js,gradJ(:,3))];

g3_fast = bsxfun(@times, g3_fast, datweight*eta);
end

