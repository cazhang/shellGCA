function [ distSqr,distVec, distVecEm, distVecEd] = geomDSD( FVref,FVdef,Topo,Ev,Ef,eta, factor, vWeight )
%DSD Discrete shell distance: this is the classical one which include mem
%and bending
% accept both FVref is struct, or just vertices, conditioning on Topo 

mu = 1;
lambda = 1;

boundaryedges = Ef(:,2)==0;

if ~isstruct(FVref)
    Vref = FVref;
    clear FVref;
    FVref.vertices = Vref;
    FVref.faces = Topo;
end

if ~isstruct(FVdef)
    Vdef = FVdef;
    clear FVdef;
    FVdef.vertices = Vdef;
    FVdef.faces = Topo;
end

if ~isfield(FVref, 'de')
    FVref = precompute(FVref,Topo,Ev,Ef,boundaryedges);
end
if ~isfield(FVdef, 'de')
    FVdef = precompute(FVdef,Topo,Ev,Ef,boundaryedges);
end

nface = length(FVref.at);
nedge = length(FVref.le);
nverts = size(FVdef.vertices, 1);

if nargin < 8
    eWeight = ones(nedge,1);
    fWeight = ones(nface,1);
else
    eWeight = 0.5*(vWeight(Ev(:,1)) + vWeight(Ev(:,2)));
    fWeight = (vWeight(Topo(:,1)) + vWeight(Topo(:,2)) + vWeight(Topo(:,3)) )/3;
end

% Membrane energy: for each triangle, det, tr, and area

muHalf = mu/2;
lambdaQuarter = lambda/4;
muHalfPlusLambdaQuarter = muHalf + lambdaQuarter;

% vec version
% deformed quantities
is = Topo(:, 1);
js = Topo(:, 2);
ks = Topo(:, 3);

Pis = FVdef.vertices(is, :); % nface x 3
Pjs = FVdef.vertices(js, :);
Pks = FVdef.vertices(ks, :);

Eis = Pks - Pjs;
Ejs = Pis - Pks;
Eks = Pis - Pjs;

liSqrs = sum(Eis.^2, 2);
ljSqrs = sum(Ejs.^2, 2);
lkSqrs = sum(Eks.^2, 2);

temps = cross(Eis, Ejs);
volDefSqrs = sum(temps.^2, 2)./4;

% undeformed quantities
Pis = FVref.vertices(is, :); % nface x 3
Pjs = FVref.vertices(js, :);
Pks = FVref.vertices(ks, :);

Eis = Pks - Pjs;
Ejs = Pis - Pks;
Eks = Pis - Pjs;

temps = cross(Eis, Ejs);
volUndefSqrs = sum(temps.^2, 2)./4;
volUndefs = sqrt(volUndefSqrs);

traceTerms = sum(Ejs.*Eks,2).*liSqrs + sum(Eks.*Eis,2).*ljSqrs - ...
    sum(Eis.*Ejs,2).*lkSqrs;
Em = (mu/8 .* traceTerms + lambdaQuarter.*volDefSqrs) ./ volUndefs - ...
    (muHalfPlusLambdaQuarter .* log(volDefSqrs./volUndefSqrs) + ...
    repmat(mu,nface,1) + repmat(lambdaQuarter,nface,1)) .* volUndefs; 

Em = factor .* Em;

% Dihedral angles
Ed = zeros(nedge,1);
Ed(~boundaryedges) = eta * factor .* (FVref.le(~boundaryedges).^2 ./ (3.*FVref.de(~boundaryedges))...
    .* (FVref.Te(~boundaryedges)-FVdef.Te(~boundaryedges)).^2);

Em = Em.*fWeight;
Ed = Ed.*eWeight;
% sum of total sq. energy
distSqr = sum(Em) + sum(Ed);

distVecEm = zeros(nverts, 1);
distVecEd = zeros(nverts, 1);
if nargout > 1
    % dist of each vertex
    for eid=1:nedge
        v1 = Ev(eid,1);
        v2 = Ev(eid,2);
        distVecEd(v1) = distVecEd(v1) + Ed(eid)*0.5;
        distVecEd(v2) = distVecEd(v2) + Ed(eid)*0.5;
    end
    
    for fid = 1:nface
        v1 = Topo(fid,1);
        v2 = Topo(fid,2);
        v3 = Topo(fid,3);
        distVecEm(v1) = distVecEm(v1) + Em(fid)/3;
        distVecEm(v2) = distVecEm(v2) + Em(fid)/3;
        distVecEm(v3) = distVecEm(v3) + Em(fid)/3;
    end
    distVec = distVecEm + distVecEd;
end
end

function memEnergyLoopy()

for faceid = 1:nface
    % set up deformed vertex and edges
    i = Topo(faceid,1);
    j = Topo(faceid,2);
    k = Topo(faceid,3);
    Pi = FVdef.vertices(i,:);
    Pj = FVdef.vertices(j,:);
    Pk = FVdef.vertices(k,:);
    Ei = Pk - Pj;
    Ej = Pi - Pk;
    Ek = Pi - Pj;
    
    % compute edge lengths
    liSqr = norm(Ei)^2;
    ljSqr = norm(Ej)^2;
    lkSqr = norm(Ek)^2;
    
    % compute volumn
    temp = cross(Ei, Ej);
    volDefSqr = norm(temp)^2 / 4;
    
    % set up undeformed vertices and edges
    Pi = FVref.vertices(i,:);
    Pj = FVref.vertices(j,:);
    Pk = FVref.vertices(k,:);
    Ei = Pk - Pj;
    Ej = Pi - Pk;
    Ek = Pi - Pj;
    
    % compute volumn
    temp = cross(Ei, Ej);
    volUndefSqr = norm(temp)^2 / 4;
    volUndef = sqrt(volUndefSqr);
    
    % compute trace term: caution the signs! Ek is actually -Ek here
    traceTerm = (Ej*Ek')*liSqr + (Ek*Ei')*ljSqr - (Ei*Ej')*lkSqr;
   
    % volume of triangle * evaluation of energy density
    Em(faceid,1) = (mu/8 * traceTerm + lambdaQuarter * volDefSqr) / volUndef -...
        (muHalfPlusLambdaQuarter * log(volDefSqr/volUndefSqr) + mu + ...
        lambdaQuarter) * volUndef; 
end
end
