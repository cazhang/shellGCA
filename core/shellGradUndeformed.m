function [ cost, grad, H ] = shellGradUndeformed( x, FVs, Topo, Ev, Ef, Eo, boundaryedges,options )
%compute the cost and gradient wrt. undeformed shell-FV_ref.
%   x: initial value of an undeformed shell
%   FVs: a set of deformed shell

v = reshape(x, length(x)/3, 3);

nverts = size(v, 1);
mu = options.mu;
lambda = options.lambda;
eta = options.eta;

nmesh = length(FVs);
if ~isfield(options, 'datweights')
    datweights(1:nmesh) = 1;
else
    datweights = options.datweights;
end

% cost term
FV_ref.vertices = v;
FV_ref.faces = Topo;
F = Topo;
nEdge = length(Ev);
nFace = size(F, 1);
% compute intermediate quantities
if ~isfield(FV_ref, 'de')
    FV_ref = precompute(FV_ref,Topo,Ev,Ef,boundaryedges);
end

for i = 1:nmesh
    if ~isfield(FVs{i}, 'de')
        FVs{i} = precompute(FVs{i},Topo,Ev,Ef,boundaryedges);
    end
end
% make f
if nargout > 2
    % Hessian required
    rows=[]; cols=[]; vals=[];
    H = zeros(nverts*3,nverts*3);
end
if nargout > 1
    g = zeros(nverts, 3);
end
f = [];

for m=1:nmesh
    if datweights(m) == 0
        continue;
    end
    % Edge lengths
    f1 = mu*datweights(m).*FV_ref.de ./ (FV_ref.le).^2 .*((FV_ref.le-FVs{m}.le).^2);
    f = [f; f1];
    % Triangle areas
    f2 = lambda*datweights(m).* (FV_ref.at-FVs{m}.at).^2 ./ (FV_ref.at);
    f = [f; f2];
    % Dihedral angles
    f3 = eta*datweights(m).* (FV_ref.le(~boundaryedges)).^2 ./ (3.*FV_ref.de(~boundaryedges)) .* (FV_ref.Te-FVs{m}.Te).^2;
    %f3 = (Te(~boundaryedges)-FVs{res,m}.Te(~boundaryedges));
    f = [f; f3];
end

% make gradient
if nargout > 1
    % edge term
    for m=1:nmesh
        g1 = zeros(length(x)/3, 3);
        for eid=1:nEdge
            %disp(['edge index = ', num2str(eid)]);
            i = Ev(eid,1);
            j = Ev(eid,2);
            k = Eo(eid,1);
            l = Eo(eid,2);
            % get deformed quantities
            def_edge = FVs{m}.vertices(j,:)-FVs{m}.vertices(i,:);
            defEdgeLength = norm(def_edge);
          
            % get undeformed values
            Pi = FV_ref.vertices(i,:);
            Pj = FV_ref.vertices(j,:);
            if k~=0
                Pk = FV_ref.vertices(k,:);
            end
            if l~=0
                Pl = FV_ref.vertices(l,:);
            end
            area = FV_ref.de(eid);
            
            und_edge = Pj-Pi;
            undefEdgeLength = norm(und_edge);
            undefEdgeLengthSqr = undefEdgeLength^2;
            
            factorGradArea = (undefEdgeLength-defEdgeLength)^2 / (3*undefEdgeLengthSqr);
            factorGradEdgeLengthSqr = 2*area*(undefEdgeLength-defEdgeLength)*defEdgeLength / (undefEdgeLengthSqr^2);
            
            % d_k
            if k~=0
                gradK = getAreaGradK(Pi,Pj,Pk);
                gradK = factorGradArea .* gradK;
                g1 = addToG(g1,gradK,k);
            end
            % d_l
            if l~=0
                gradL = getAreaGradK(Pj,Pi,Pl);
                gradL = factorGradArea .* gradL;
                g1 = addToG(g1,gradL,l);
            end
            % d_i
            if k~=0
                gradArea = getAreaGradK(Pj,Pk,Pi);
                gradI = factorGradArea .* gradArea;
            end
            if l~=0
                gradArea = getAreaGradK(Pl,Pj,Pi);
                gradI = gradI + factorGradArea .* gradArea;
            end
            gradI = gradI - factorGradEdgeLengthSqr.*und_edge;
            g1 = addToG(g1,gradI,i);
            % d_j
            if k~=0
                gradArea = getAreaGradK(Pk,Pi,Pj);
                gradJ = factorGradArea .* gradArea;
            end
            if l~=0
                gradArea = getAreaGradK(Pi,Pl,Pj);
                gradJ = gradJ + factorGradArea .* gradArea;
            end
            gradJ = gradJ + factorGradEdgeLengthSqr.*und_edge;
            g1 = addToG(g1,gradJ,j);
            
        end
        g = g + mu*datweights(m).*g1;
    end
    
    % area term
    for m=1:nmesh
        g2 = zeros(length(x)/3, 3);
        for faceid = 1:nFace
            defFaceAreaSqr = FVs{m}.at(faceid)^2;
            undFaceAreaSqr = FV_ref.at(faceid)^2;
            i = F(faceid,1);
            j = F(faceid,2);
            k = F(faceid,3);
            Pi = FV_ref.vertices(i,:);
            Pj = FV_ref.vertices(j,:);
            Pk = FV_ref.vertices(k,:);
            % d_k
            grad = getAreaGradK(Pi,Pj,Pk);
            gradK = (undFaceAreaSqr-defFaceAreaSqr)/undFaceAreaSqr .* grad;
            g2 = addToG(g2,gradK,k);
            % d_i
            grad = getAreaGradK(Pj,Pk,Pi);
            gradI = (undFaceAreaSqr-defFaceAreaSqr)/undFaceAreaSqr .* grad;
            g2 = addToG(g2,gradI,i);
            % d_j
            grad = getAreaGradK(Pk,Pi,Pj);
            gradJ = (undFaceAreaSqr-defFaceAreaSqr)/undFaceAreaSqr .* grad;
            g2 = addToG(g2,gradJ,j);
            
        end
        g = g + lambda*datweights(m).*g2;
    end
    
    % dihedral angle 
    for m=1:nmesh
        g3 = zeros(length(x)/3, 3);
        for eid = 1:nEdge
            if ~boundaryedges(eid)
                del_theta = FVs{m}.Te(eid)-FV_ref.Te(eid);
                i = Ev(eid,1);
                j = Ev(eid,2);
                k = Eo(eid,1);
                l = Eo(eid,2);
                Pi = FV_ref.vertices(i,:);
                Pj = FV_ref.vertices(j,:);
                Pk = FV_ref.vertices(k,:);
                Pl = FV_ref.vertices(l,:);
                area = 3*FV_ref.de(eid);
                edge = Pj-Pi;
                elengthSqr = norm(edge)^2;
                % derivative
                factorGradTheta = -2 * del_theta * elengthSqr / area;
                factorGradArea = -1 * del_theta * del_theta * elengthSqr / (area^2 );
                factorGradEdgeLengthSqr = 2 * del_theta * del_theta / area;
                % d_k
                gradTheta = getThetaGradK(Pi,Pj,Pk);
                gradArea = getAreaGradK(Pi,Pj,Pk);
                gradK = factorGradTheta .* gradTheta + factorGradArea .* gradArea;
                g3 = addToG(g3,gradK,k);
                % d_l
                gradTheta = getThetaGradK(Pj,Pi,Pl);
                gradArea = getAreaGradK(Pj,Pi,Pl);
                gradL = factorGradTheta .* gradTheta + factorGradArea .* gradArea;
                g3 = addToG(g3,gradL,l);
                % d_i
                gradTheta = getThetaGradI(Pi,Pj,Pk,Pl);
                gradArea = getAreaGradK(Pj,Pk,Pi);
                gradI = factorGradTheta .* gradTheta + factorGradArea .* gradArea;
                gradArea = getAreaGradK(Pl,Pj,Pi);
                gradI = gradI + factorGradArea .* gradArea;
                gradI = gradI + factorGradEdgeLengthSqr .* (Pi-Pj);
                g3 = addToG(g3,gradI,i);               
                % d_j
                gradTheta = getThetaGradJ(Pi,Pj,Pk,Pl);
                gradArea = getAreaGradK(Pk,Pi,Pj);
                gradJ = factorGradTheta .* gradTheta + factorGradArea .* gradArea;
                gradArea = getAreaGradK(Pi,Pl,Pj);
                gradJ = gradJ + factorGradArea .* gradArea;
                gradJ = gradJ + factorGradEdgeLengthSqr .* (Pj-Pi);
                g3 = addToG(g3,gradJ,j);
            end
        end
        g = g+(eta*datweights(m)).*g3;
    end
    
end

if nargout > 2
    %% TODO: get Hessian for undeformed shell
    % edge length term
    for m=1:nmesh
        H1 = zeros(nverts*3, nverts*3);
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
    end
      
    % area term    
    for m=1:nmesh
        H2 = zeros(nverts*3, nverts*3);
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
    end
    
    % dihedral angle
    for m=1:nmesh
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


cost = sum(f);

if nargout > 1 % gradient required
    grad = g(:);
end

if nargout > 2
    H = sparse(H);
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

function m = getProjection(x)
m = eye(3);
temp = x'*x;
m = m - temp;
end

function m = getReflection(x)
m = eye(3);
temp = x'*x;
m = m - 2.*temp;
end


function N = getNormal(Pi, Pj, Pk)
e1 = Pk - Pi;
e3 = Pk - Pj;
N = cross(e1, e3, 2);
nLeng = sqrt(sum(N.^2, 2));
N = N./ nLeng;
end
function area = getArea(Pi, Pj, Pk)
e1 = Pk - Pi;
e3 = Pk - Pj;
N = cross(e1,e3,2);
area = sqrt(sum(N.^2, 2));
area = area ./ 2;
end
    
function grad = getAreaGradK(Pi,Pj,Pk)
e = Pj-Pi;
normal = getNormal(Pi,Pj,Pk);
grad = cross(0.5*normal, e);
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


function grad = getThetaGradK(Pi,Pj,Pk)
e = Pj - Pi;
normal = getNormal(Pi,Pj,Pk);
grad = (-0.5*norm(e) / getArea(Pi,Pj,Pk)) .* normal;

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
grad = (a*e')/(e*e')*grad;
end

function grad = getThetaGradJ(Pi,Pj,Pk,Pl)
grad = getThetaGradJLeftPart(Pi,Pj,Pk);
grad = grad - getThetaGradJLeftPart(Pi,Pj,Pl);
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

function Hji = getHessThetaJI(Pi,Pj,Pk,Pl)
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


function mat = addToG(mat, tmp, i)
mat(i,1)=mat(i,1)+tmp(1);
mat(i,2)=mat(i,2)+tmp(2);
mat(i,3)=mat(i,3)+tmp(3);

end

function mat = addToH(mat, tmp, r, c, nverts)
if sum(double(isnan(tmp)))>0
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

