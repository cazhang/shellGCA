function [ c,g,H ] = fastShellGradAndHessUnd( x,FVs,Topo,Ev,Ef,Eo,boundaryedges,options )
%vectorised version of computing cost, grad, hess wrt. deformed shell
%   Detailed explanation goes here

nmesh = length(FVs);
v = reshape(x, length(x)/3, 3);
nverts = size(v, 1);
FV_ref.vertices = v;
FV_ref.faces = FVs{1}.faces;
F = Topo;

nEdge = size(Ev,1);
nFace = size(FV_ref.faces,1);
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
if ~isfield(options, 'datweights')
    datweights(1:nmesh) = 1;
else
    datweights = options.datweights;
end
% compute intermediate quantities
if ~isfield(FV_ref, 'de')
    FV_ref = precompute(FV_ref,Topo,Ev,Ef,boundaryedges);
end
for i=1:nmesh
    if ~isfield(FVs{i}, 'de')
        FVs{i} = precompute(FVs{i},Topo,Ev,Ef,boundaryedges);
    end
end

f=[];

if nargout>1
    g = zeros(nverts, 3);
end
if nargout>2
    % Hessian required
    H = zeros(nverts*3, nverts*3);
end

for m=1:nmesh
    if datweights(m) == 0
        continue;
    end
    % Edge lengths
    f1 = mu * datweights(m).*FV_ref.de ./ FV_ref.le.^2 .*((FVs{m}.le-FV_ref.le).^2);
    f = [f; f1];
    % Triangle areas
    f2 = lambda * datweights(m)./FV_ref.at .* (FV_ref.at-FVs{m}.at).^2;
    f = [f; f2];
    % Dihedral angles
    f3 = eta * datweights(m).* (FV_ref.le(~boundaryedges)).^2 ./ ...
        (3.*FV_ref.de(~boundaryedges)) .* (FV_ref.Te-FVs{m}.Te).^2;  
    f = [f; f3];
    
    if nargout > 1 % gradient required
        
        % edge length term       
        g1_fast = getEdgeGradVecUnd(FV_ref,FVs{m},Ev,Eo,mu,datweights(m));
        
        % triangle area term
        g2_fast = getFaceGradVecUnd(FV_ref,FVs{m}, lambda, datweights(m));
                       
        % dihedral angle term
        g3_fast = getAngleGradVecUnd(FV_ref,FVs{m}, Ev, Eo, boundaryedges,eta,datweights(m));
        
        g = g + (g1_fast + g2_fast + g3_fast);
        
    end
    
    if nargout > 2 % hessian required
        
        H1 = zeros(nverts*3, nverts*3);
        H2 = zeros(nverts*3, nverts*3);
        H3 = zeros(nverts*3, nverts*3);
        
        for eid=1:nEdge
            
            i = Ev(eid,1);
            j = Ev(eid,2);
            k = Eo(eid,1);
            l = Eo(eid,2);
            
            defEdge = FVs{m}.vertices(j,:)-FVs{m}.vertices(i,:);
            defEdgeLength = norm(defEdge);
            
            area = FV_ref.de(eid);
            Pi = FV_ref.vertices(i,:);
            Pj = FV_ref.vertices(j,:);
            Pk = FV_ref.vertices(k,:);
            Pl = FV_ref.vertices(l,:);
            
            if k~=0
                areak = getAreaGradK(Pi,Pj,Pk);
                areai = getAreaGradK(Pj,Pk,Pi);
                areaj = getAreaGradK(Pk,Pi,Pj);
            end
            if l~=0
                areal = getAreaGradK(Pj,Pi,Pl);
                temp = getAreaGradK(Pl,Pj,Pi);
                areai = areai + temp;
                temp = getAreaGradK(Pi,Pl,Pj);
                areaj = areaj + temp;
            end
            
            edge = Pj-Pi;
            undefEdgeLengthSqr = norm(edge)^2;
            undefEdgeLength = norm(edge);
            
            areaFactor = (undefEdgeLength - defEdgeLength)^2 / (3*undefEdgeLengthSqr);
            mixedFactor = 2*(undefEdgeLength - defEdgeLength)*defEdgeLength / (3*undefEdgeLengthSqr^2);
            diagonalFactor = 2*area*(undefEdgeLength - defEdgeLength)*defEdgeLength / (undefEdgeLengthSqr^2);
            eCrossEFactor = 2*area*defEdgeLength*(4*defEdgeLength - 3*undefEdgeLength)/(undefEdgeLengthSqr^2 * undefEdgeLengthSqr);
            
            EcrossE = edge'*edge;
            % *k
            if k~=0
                % kk
                auxMat = getHessAreaKK(Pi,Pj,Pk);
                Hkk = areaFactor .* auxMat;
                %Hkk = mu*resweights(res).*Hkk;
                H1 = addToH(H1, Hkk, k, k,nverts);
                % ik & ki
                auxMat = getHessAreaIK(Pi,Pj,Pk);
                Hik = areaFactor .* auxMat;
                auxMat = edge'*areak;
                Hik = Hik - mixedFactor.*auxMat;
                %Hik = mu*resweights(res).*Hik;
                H1 = addToH(H1, Hik, i, k,nverts);
                H1 = addToH(H1, Hik', k, i,nverts);
                % jk & kj
                auxMat = getHessAreaIK(Pj,Pi,Pk);
                Hjk = areaFactor .* auxMat;
                auxMat = edge'*areak;
                Hjk = Hjk + mixedFactor.*auxMat;
                %Hjk = mu*resweights(res).*Hjk;
                H1 = addToH(H1, Hjk, j, k,nverts);
                H1 = addToH(H1, Hjk', k, j,nverts);
            end
            % *l
            if l~=0
                % ll
                auxMat = getHessAreaKK(Pj,Pi,Pl);
                Hll = areaFactor .* auxMat;
                %Hll = mu*resweights(res).*Hll;
                H1 = addToH(H1, Hll, l, l,nverts);
                % il & li
                auxMat = getHessAreaIK(Pi,Pj,Pl);
                Hil = areaFactor .* auxMat;
                auxMat = edge'*areal;
                Hil = Hil - mixedFactor.*auxMat;
                %Hil = mu*resweights(res).*Hil;
                H1 = addToH(H1, Hil, i, l,nverts);
                H1 = addToH(H1, Hil', l, i,nverts);
                % jl & lj
                auxMat = getHessAreaIK(Pj,Pi,Pl);
                Hjl = areaFactor .* auxMat;
                auxMat = edge'*areal;
                Hjl = Hjl + mixedFactor.*auxMat;
                %Hjl = mu*resweights(res).*Hjl;
                H1 = addToH(H1, Hjl, j, l,nverts);
                H1 = addToH(H1, Hjl', l, j,nverts);
            end
            % jj
            auxMat = areaj'*edge;
            Hjj = mixedFactor .* auxMat;
            auxMat = edge'*areaj;
            Hjj = Hjj + mixedFactor.*auxMat;
            if k~=0
                Hjj = Hjj + areaFactor.*getHessAreaKK(Pk,Pi,Pj);
            end
            if l~=0
                Hjj = Hjj + areaFactor.*getHessAreaKK(Pi,Pl,Pj);
            end
            Hjj = Hjj + eCrossEFactor .* EcrossE;
            Hjj = Hjj + diagonalFactor .* eye(3);
            %Hjj = mu*resweights(res).*Hjj;
            H1 = addToH(H1, Hjj, j, j,nverts);
            
            % ij & ji
            auxMat = areai'*edge;
            Hij = mixedFactor .* auxMat;
            auxMat = edge'*areaj;
            Hij = Hij - mixedFactor.*auxMat;
            if k~=0
                Hij = Hij + areaFactor.*getHessAreaIK(Pi,Pk,Pj);
            end
            if l~=0
                Hij = Hij + areaFactor.*getHessAreaIK(Pi,Pl,Pj);
            end
            Hij = Hij - eCrossEFactor.*EcrossE;
            Hij = Hij - diagonalFactor.*eye(3);
            %Hij = mu*resweights(res).*Hij;
            H1 = addToH(H1, Hij, i, j,nverts);
            H1 = addToH(H1, Hij', j, i,nverts);
            
            % ii
            auxMat = areai'*edge;
            Hii = (-1*mixedFactor).*auxMat;
            auxMat = edge'*areai;
            Hii = Hii - mixedFactor.*auxMat;
            if l~=0
                Hii = Hii + areaFactor.*getHessAreaKK(Pl,Pj,Pi);
            end
            if k~=0
                Hii = Hii + areaFactor.*getHessAreaKK(Pj,Pk,Pi);
            end
            Hii = Hii + eCrossEFactor.*EcrossE;
            Hii = Hii + diagonalFactor.*eye(3);
            %Hii = mu*resweights(res).*Hii;
            H1 = addToH(H1, Hii, i, i,nverts);
        end
        H1 = mu*datweights(m).*H1;
        H = H + H1;
        
        % H2
        for fid=1:nFace
            
            defFaceAreaSqr = FVs{m}.at(fid)^2;
            undFaceAreaSqr = FV_ref.at(fid)^2;
            i = F(fid,1);
            j = F(fid,2);
            k = F(fid,3);
            Pi = FV_ref.vertices(i,:);
            Pj = FV_ref.vertices(j,:);
            Pk = FV_ref.vertices(k,:);
            
            hessFactor = (undFaceAreaSqr-defFaceAreaSqr)/undFaceAreaSqr;
            mixedFactor = 2*defFaceAreaSqr/(undFaceAreaSqr*sqrt(undFaceAreaSqr));
            
            gradk = getAreaGradK(Pi,Pj,Pk);
            gradi = getAreaGradK(Pj,Pk,Pi);
            gradj = getAreaGradK(Pk,Pi,Pj);
            
            % kk
            auxMat = getHessAreaKK(Pi,Pj,Pk);
            Hkk = hessFactor.*auxMat;
            auxMat = gradk'*gradk;
            Hkk = Hkk+mixedFactor.*auxMat;
            %Hkk = lambda*resweights(res).*Hkk;
            H2 = addToH(H2,Hkk,k,k,nverts);
            % ik&ki
            auxMat = getHessAreaIK(Pi,Pj,Pk);
            Hik = hessFactor.*auxMat;
            auxMat = gradi'*gradk;
            Hik = Hik+mixedFactor.*auxMat;
            %Hik = lambda*resweights(res).*Hik;
            H2 = addToH(H2,Hik,i,k,nverts);
            H2 = addToH(H2,Hik',k,i,nverts);
            % jk&kj
            auxMat = getHessAreaIK(Pj,Pi,Pk);
            Hjk = hessFactor.*auxMat;
            auxMat = gradj'*gradk;
            Hjk = Hjk+mixedFactor.*auxMat;
            %Hjk = lambda*resweights(res).*Hjk;
            H2 = addToH(H2,Hjk,j,k,nverts);
            H2 = addToH(H2,Hjk',k,j,nverts);
            % jj
            auxMat = getHessAreaKK(Pk,Pi,Pj);
            Hjj = hessFactor.*auxMat;
            auxMat = gradj'*gradj;
            Hjj = Hjj+mixedFactor.*auxMat;
            %Hjj = lambda*resweights(res).*Hjj;
            H2 = addToH(H2,Hjj,j,j,nverts);
            % ij&ji
            auxMat = getHessAreaIK(Pi,Pk,Pj);
            Hij = hessFactor.*auxMat;
            auxMat = gradi'*gradj;
            Hij = Hij+mixedFactor.*auxMat;
            %Hij = lambda*resweights(res).*Hij;
            H2 = addToH(H2,Hij,i,j,nverts);
            H2 = addToH(H2,Hij',j,i,nverts);
            % ii
            auxMat = getHessAreaKK(Pj,Pk,Pi);
            Hii = hessFactor.*auxMat;
            auxMat = gradi'*gradi;
            Hii = Hii+mixedFactor.*auxMat;
            %Hii = lambda*resweights(res).*Hii;
            H2 = addToH(H2,Hii,i,i,nverts);
        end
        H2 = lambda*datweights(m).*H2;
        H = H + H2;
        % H3
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

if nargout > 2
    %H = sparse(rows,cols,vals,3*nverts,3*nverts);
    H = sparse(H);
end

if nargout > 1
% gradient required
g = g(:);
end

c = sum(f);

end

%% SUPPORTING FUNCTIONS BELOW
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
grad = ((e*d')/(e*e')).* grad;
end

function Grad = getThetaGradILeftPartVec(Pis,Pjs,Pks)
es = Pjs-Pis;
ds = Pks-Pjs;
Grad = getThetaGradKVec(Pis,Pjs,Pks);
%tmps = diag(es*ds')./diag(es*es');
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
%tmps = diag(as*es')./diag(es*es');
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
%mat1 = bsxfun(@times, -1./diag(es*es'), mat1);
mat1 = bsxfun(@times, -1./sum(es.*es,2), mat1);
mat2 = getHessThetaIKVec(Pis,Pjs,Pks);
%mat2 = bsxfun(@times, diag(ds*es')./diag(es*es'), mat2);
mat2 = bsxfun(@times, sum(ds.*es,2)./sum(es.*es,2), mat2);
HiLeft = mat1 + mat2;
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

function g1_fast = getEdgeGradVecUnd(FV_ref,FV_def,Ev,Eo,mu,datweight)
nverts = size(FV_ref.vertices, 1);
g1_fast = zeros(nverts, 3);
is = Ev(:,1);
js = Ev(:,2);
ks = Eo(:,1); nks = (ks~=0);
ls = Eo(:,2); nls = (ls~=0);
def_edges = FV_def.vertices(js,:) - FV_def.vertices(is,:);
defEdgeLength = sqrt(sum(def_edges.^2, 2));
% here, if k=0 (l=0), ignore its value, is and js need to be filtered
Pis = FV_ref.vertices(is,:);
Pjs = FV_ref.vertices(js,:);
Pks = FV_ref.vertices(ks(nks), :); % only non-zero vertices kept
Pls = FV_ref.vertices(ls(nls), :);
areas = FV_ref.de;

und_edges = Pjs - Pis;
undefEdgeLengthSqr = sum(und_edges.^2,2);
undefEdgeLength = sqrt(undefEdgeLengthSqr);

factorGradArea = bsxfun(@times, 1./(3.*undefEdgeLengthSqr), (undefEdgeLength-defEdgeLength).^2);
factorGradEdgeLengthSqr = 2.*areas.*(undefEdgeLength-defEdgeLength).*defEdgeLength ./ (undefEdgeLengthSqr.^2);

% d_k
gradK = zeros(nverts, 3);
tmps = getAreaGradKVec(Pis(nks,:),Pjs(nks,:),Pks);
tmps = bsxfun(@times, tmps, factorGradArea);
gradK(nks,:) = tmps;
g1_fast(1:max(ks),:) = g1_fast(1:max(ks),:)+[accumarray(ks(nks),gradK(:,1)) accumarray(ks(nks),gradK(:,2)) accumarray(ks(nks),gradK(:,3))];
% d_l
gradL = zeros(nverts, 3);
tmps = getAreaGradKVec(Pjs(nls,:),Pis(nls,:),Pls);
tmps = bsxfun(@times, tmps, factorGradArea);
gradL(nls,:) = tmps;
g1_fast(1:max(ls),:) = g1_fast(1:max(ls),:)+[accumarray(ls(nls),gradL(:,1)) accumarray(ls(nls),gradL(:,2)) accumarray(ls(nls),gradL(:,3))];
% d_i
gradIK = zeros(nverts,3);
tmps = getAreaGradKVec(Pjs(nks,:),Pks,Pis(nks,:));
tmps = bsxfun(@times, tmps, factorGradArea);
gradIK(nks,:) = tmps;
gradIL = zeros(nverts,3);
tmps = getAreaGradKVec(Pls,Pjs(nls,:),Pis(nls,:));
tmps = bsxfun(@times, tmps, factorGradArea);
gradIL(nls,:) = tmps;
tmps = bsxfun(@times, factorGradEdgeLengthSqr, und_edges);
gradI = gradIK + gradIL - tmps;
g1_fast(1:max(is),:) = g1_fast(1:max(is),:)+[accumarray(is,gradI(:,1)) accumarray(is,gradI(:,2)) accumarray(is,gradI(:,3))];
% d_j
gradJK = zeros(nverts,3);
tmps = getAreaGradKVec(Pks,Pis(nks,:),Pjs(nks,:));
tmps = bsxfun(@times, tmps, factorGradArea);
gradJK(nks,:) = tmps;
gradJL = zeros(nverts,3);
tmps = getAreaGradKVec(Pis(nls,:),Pls,Pjs(nls,:));
tmps = bsxfun(@times, tmps, factorGradArea);
gradJL(nls,:) = tmps;
tmps = bsxfun(@times, factorGradEdgeLengthSqr, und_edges);
gradJ = gradJK + gradJL + tmps;
g1_fast(1:max(js),:) = g1_fast(1:max(js),:)+[accumarray(js,gradJ(:,1)) accumarray(js,gradJ(:,2)) accumarray(js,gradJ(:,3))];

g1_fast = bsxfun(@times,mu*datweight,g1_fast);
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

function g2_fast = getFaceGradVecUnd(FV_ref,FV_def,lambda,datweight)
nverts = size(FV_ref.vertices,1);
defFaceAreaSqr = FV_def.at.^2;
undFaceAreaSqr = FV_ref.at.^2;
g2_fast = zeros(nverts,3);
F = FV_ref.faces;
is = F(:,1);
js = F(:,2);
ks = F(:,3);
Pis = FV_ref.vertices(is,:);
Pjs = FV_ref.vertices(js,:);
Pks = FV_ref.vertices(ks,:);
% d_k
grad = getAreaGradKVec(Pis,Pjs,Pks);
tmps = (undFaceAreaSqr - defFaceAreaSqr)./undFaceAreaSqr;
gradK = bsxfun(@times, tmps, grad);
gradK = bsxfun(@times, lambda*datweight, gradK);
g2_fast(1:max(ks),:) = g2_fast(1:max(ks),:)+...
    [accumarray(ks,gradK(:,1)) accumarray(ks,gradK(:,2)) accumarray(ks,gradK(:,3))];
% d_i
grad = getAreaGradKVec(Pjs,Pks,Pis);
tmps = (undFaceAreaSqr - defFaceAreaSqr)./undFaceAreaSqr;
gradI = bsxfun(@times, tmps, grad);
gradI = bsxfun(@times, lambda*datweight, gradI);
g2_fast(1:max(is),:) = g2_fast(1:max(is),:)+...
    [accumarray(is,gradI(:,1)) accumarray(is,gradI(:,2)) accumarray(is,gradI(:,3))];
% d_j
grad = getAreaGradKVec(Pks,Pis,Pjs);
tmps = (undFaceAreaSqr - defFaceAreaSqr)./undFaceAreaSqr;
gradJ = bsxfun(@times, tmps, grad);
gradJ = bsxfun(@times, lambda*datweight, gradJ);
g2_fast(1:max(js),:) = g2_fast(1:max(js),:)+...
    [accumarray(js,gradJ(:,1)) accumarray(js,gradJ(:,2)) accumarray(js,gradJ(:,3))];
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

function [rows,cols,vals] = getEdgeHessVecUnd(rows,cols,vals,FV_ref,FV_def,Ev,mu,datweight)
nverts = size(FV_ref.vertices,1);
is = Ev(:,1);
js = Ev(:,2);
ks = Eo(:,1); 
nks = (ks~=0);
ls = Eo(:,2); 
nls = (ls~=0);
defEdges = FV_def.vertices(js,:)-FV_def.vertices(is,:);
defEdgesLen = sqrt(sum(defEdges.^2,2));
areas = FV_ref.de;
Pis = FV_ref.vertices(is,:);
Pjs = FV_ref.vertices(js,:);
Pks = FV_ref.vertices(ks(nks),:);
Pls = FV_ref.vertices(ls(nls),:);
areak = zeros(nverts,3);
areal = zeros(nverts,3);
areai = zeros(nverts,3);
areaj = zeros(nverts,3);
% k~=0
tmps = getAreaGradKVec(Pis(nks,:),Pjs(nks,:),Pks);
areak(nks,:)=tmps;
tmps = getAreaGradKVec(Pjs(nks,:),Pks,Pis(nks,:));
areai(nks,:)=tmps;
tmps = getAreaGradKVec(Pks,Pis(nks,:),Pjs(nks,:));
areaj(nks,:)=tmps;
% l~=0
tmps = getAreaGradKVec(Pjs(nls,:),Pis(nls,:),Pls);
areal(nls,:)=tmps;
tmps = getAreaGradKVec(Pls,Pjs(nls,:),Pis(nls,:));
areai(nls,:) = areai(nls,:) + tmps;
tmps = getAreaGradKVec(Pis(nls,:),Pls,Pjs(nls,:));
areaj(nls,:) = areaj(nls,:) + tmps;

edges = Pjs-Pis;
undEdgesLenSqr = sum(edges.^2,2);
undEdgesLen = sqrt(undEdgesLenSqr);
areaFactor = (undEdgesLen - defEdgesLen).^2 ./ (3*undEdgesLenSqr);
mixedFactor = 2*(undEdgesLen - defEdgesLen).*defEdgesLen ./ (3*undEdgesLenSqr.^2);
diagonalFactor = 2*areas.*(undEdgesLen - defEdgesLen).*defEdgesLen ./ (undEdgesLenSqr.^2);
eCrossEFactor = 2*areas.*defEdgesLen.*(4*defEdgesLen - 3*undEdgesLen)./(undEdgesLenSqr.^2 .* undEdgesLenSqr);

EcrossE = outerProductVectorised(edges,edges);
% *k
auxMat = getHessAreaKKVec(Pis(nks,:),Pjs(nks,:),Pks);
Hkk = bsxfun(@times,areaFactor,auxMat);
[rows,cols,vals]=addToHVectorised(rows,cols,vals,Hkk,ks(nks),ks(nks),nverts);
% ik & ki
auxMat = getHessAreaIKVec(Pis(nks,:),Pjs(nks,:),Pks);
Hik = bsxfun(@times,areaFactor,auxMat);
auxMat = outerProductVectorised(edges,areak);
auxMat = bsxfun(@times,mixedFactor,auxMat);
Hik = Hik - auxMat;
[rows,cols,vals]=addToHVectorised(rows,cols,vals,Hik,is(nks),ks(nks),nverts);
[rows,cols,vals]=addToHVectorised(rows,cols,vals,permute(Hik,[1 3 2]),ks(nks),is(nks),nverts);
% jk & kj
auxMat = getHessAreaIKVec(Pis(nks,:),Pjs(nks,:),Pks);
Hjk = bsxfun(@times,areaFactor,auxMat);
auxMat = outerProductVectorised(edges,areak);
auxMat = bsxfun(@times,mixedFactor,auxMat);
Hjk = Hjk + auxMat;
[rows,cols,vals]=addToHVectorised(rows,cols,vals,Hjk,js(nks),ks(nks),nverts);
[rows,cols,vals]=addToHVectorised(rows,cols,vals,permute(Hjk,[1 3 2]),ks(nks),js(nks),nverts);

% *l
auxMat = getHessAreaKKVec(Pjs(nls,:),Pis(nls,:),Pls);
Hll = bsxfun(@times,areaFactor,auxMat);
[rows,cols,vals]=addToHVectorised(rows,cols,vals,Hll,ls(nls),ls(nls),nverts);
% il & li
auxMat = getHessAreaIKVec(Pis(nls,:),Pjs(nls,:),Pls);
Hil = bsxfun(@times,areaFactor,auxMat);
auxMat = outerProductVectorised(edges,areal);
auxMat = bsxfun(@times,mixedFactor,auxMat);
Hil = Hil - auxMat;
[rows,cols,vals]=addToHVectorised(rows,cols,vals,Hil,is(nls),ls(nls),nverts);
[rows,cols,vals]=addToHVectorised(rows,cols,vals,permute(Hil,[1 3 2]),ls(nls),is(nls),nverts);
% jl & lj
auxMat = getHessAreaIKVec(Pjs(nls,:),Pis(nls,:),Pls);
Hjl = bsxfun(@times,areaFactor,auxMat);
auxMat = outerProductVectorised(edges,areal);
auxMat = bsxfun(@times,mixedFactor,auxMat);
Hjl = Hjl + auxMat;
[rows,cols,vals]=addToHVectorised(rows,cols,vals,Hjl,js(nls),ls(nls),nverts);
[rows,cols,vals]=addToHVectorised(rows,cols,vals,permute(Hjl,[1 3 2]),ls(nls),jl(nls),nverts);
% jj

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

%H3_fast = sparse(rows,cols,vals,nverts*3,nverts*3);
end

