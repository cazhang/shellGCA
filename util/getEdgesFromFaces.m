function [Ev, Eo, Ef] = getEdgesFromFaces(Topology)
% the function return edges list given faces, keeping the same order as
% faces are. i.e. face (i, j, k) results in edges (i, j), (j, k) and (k, i)
% output: 
%Ev: edge list; 
%Eo: opposite edge list
%Ef: adjacent face list

Ev = [];
Eo = [];
Ef = [];
F = Topology;
nF = length(F);
for f_id = 1:nF
    face = F(f_id, :);
    i = face(1);
    j = face(2);
    k = face(3);
    s_list = [i, j, k];
    e_list = [j, k, i];
    % add edges
    for e_id = 1:3
        edge = [s_list(e_id) e_list(e_id)];
        if isempty(Ev)
            isEdgeAdded = 0;
            isEdgeRevAdded = 0;
        else      
            isEdgeAdded = sum(Ev(:,1)==edge(1) & Ev(:,2)==edge(2));
            isEdgeRevAdded = sum(Ev(:,1)==edge(2) & Ev(:,2)==edge(1));
        end
        
        if isEdgeAdded == 0 && isEdgeRevAdded == 0
            Ev = [Ev; edge];
            Fadj = zeros(1,2);
            Fadj(1) = f_id;
            opp_edge = zeros(1,2);
            opp_edge(1) = setdiff(face, edge);
            Ef = [Ef; Fadj];
            Eo = [Eo; opp_edge];
            %disp('new edge added');
        
        elseif isEdgeAdded == 0 && isEdgeRevAdded == 1
            edge_ind = find(Ev(:,1)==edge(2) & Ev(:,2)==edge(1));
            Ef(edge_ind, 2) = f_id;
            Eo(edge_ind, 2) = setdiff(face, edge);
            %disp('rev edge info added');
            
        
        elseif isEdgeAdded == 1 && isEdgeRevAdded == 0
            disp('do nothing when edges added');
        
        elseif isEdgeAdded == 1 && isEdgeRevAdded == 1
            disp('error when both edges added');
        end
    end
    
end

