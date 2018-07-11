function FV = precompute(FV,Topo,Ev,Ef,boundaryedges)
nEdges = size(Ef, 1);
F = Topo; % nface x 3
V = FV.vertices; % nverts x 3
nface = size(F, 1);
% Get the triangle vertices
v1 = F(:, 1);
v2 = F(:, 2);
v3 = F(:, 3);

% Compute the edge vectors
e1s = V(v3, :) - V(v1, :);
%e2s = V2(v3, :) - V2(v1, :);
e3s = V(v3, :) - V(v2, :);

% Compute triangle normals
Tn    = cross(e1s, e3s, 2);
Tnlength = sqrt(sum(Tn.^2,2));
% Compute triangle areas
Ta = Tnlength./2;
% Compute
Tn = Tn./repmat(Tnlength,1,3);

% Compute dihedral angles for non-boundary edges
FV.Te = zeros(nEdges, 1);
FV.Te(~boundaryedges,1) = acos(max(-1,min(1,sum(Tn(Ef(~boundaryedges,1),:).*Tn(Ef(~boundaryedges,2),:),2))));
%Te = acos(sum(Tn(Ef(~boundaryedges,1),:).*Tn(Ef(~boundaryedges,2),:),2));
for edge_idx=1:nEdges  
    if ~boundaryedges(edge_idx)
        e = V(Ev(edge_idx, 2),:)-V(Ev(edge_idx,1),:);
        cross_normal = cross(Tn(Ef(edge_idx,1),:), Tn(Ef(edge_idx,2),:));
        if cross_normal*e' < 0
            FV.Te(edge_idx) = -1*FV.Te(edge_idx);
        end
    end
end
% Compute length scaling terms
FV.de = zeros(nEdges, 1);
FV.de(~boundaryedges) = (1/3).*(Ta(Ef(~boundaryedges,1))+Ta(Ef(~boundaryedges,2)));
% Not sure next one is valid...
FV.de(boundaryedges) = (1/3).*Ta(Ef(boundaryedges,1));
if size(FV.de, 1) < size(FV.de, 2)
    FV.de = FV.de';
end

FV.at = Ta;
FV.le = sqrt(sum((V(Ev(:,1),:)-V(Ev(:,2),:)).^2,2));
FV.Tn = Tn;
end