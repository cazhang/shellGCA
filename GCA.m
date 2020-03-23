% extrac Principal Components from geodesic paths and geodesic mean
function [FV_pc_ref, C, eVal] = GCA(FV_ga, FVlogs, Topology, gcaopt)
% compute basic geometry information
num_shells = gcaopt.nlength;
cutoff = gcaopt.cutoff;
ninput = gcaopt.ninput;
mu = gcaopt.mu;
lambda = gcaopt.lambda;
eta = gcaopt.eta;
isShowFig = gcaopt.isShowFig;
isNormLog = gcaopt.isNormLog;
if isNormLog
    betas = gcaopt.betas;
end
K = num_shells - 1;
[Ev, Eo, Ef] = getEdgesFromFaces(Topology);
boundaryedges = Ef(:,2)==0;

D2 = zeros(ninput, ninput);
C = zeros(ninput, ninput);

dbg = false;
if dbg
    disp('Using Euclidean distance as metric');
end
% option2:  D computed using logmap
for i=1:ninput
    if dbg
        D2(i,i) = norm(FV_ga.vertices(:) - FVlogs{i}.vertices(:))^2;       
    else
        D2(i,i) = (K^2)*geomDSD(FV_ga, FVlogs{i}, Topology, Ev,Ef,mu,lambda,eta,1);
        if isNormLog
            D2(i,i) = D2(i,i)/(betas(i)^2);
        end
    end
end

for i=1:ninput
    for j=1:ninput
        if i~=j
            if dbg
                D2(i,j) = norm(FVlogs{i}.vertices(:) - FVlogs{j}.vertices(:))^2;               
            else
                D2(i,j) = (K^2)*geomDSD(FVlogs{i},FVlogs{j},...
                    Topology, Ev,Ef,mu,lambda,eta,1);
                if isNormLog
                    D2(i,j)=D2(i,j)/(betas(i)*betas(j));
                end
            end
        end
    end
end

D = 0.5*(D2+D2');

%% gram mat
for i=1:ninput
    C(i,i) = D(i,i);
end
for i=1:ninput
    for j=i+1:ninput
        C(i,j) = 0.5*(D(i,i) + D(j,j) - D(i,j));
        C(j,i) = C(i,j);
    end
end
%% eigendecomposition of C
% [old_eVec, eVal] = eig(C'*C);
% [tmp1,tmp2] = sort(diag(sqrt(eVal)), 'descend');
% [percent, ndim] = caculatePerVar(tmp1, cutoff);
% for i=1:ndim
%     eVec(:,i) = old_eVec(:,tmp2(i));
% end
%% svd
[evec, evalMat] = svd(C);
eVal = diag(evalMat);
[ndim] = caculatePerVar(eVal, cutoff);
for i=1:ndim
    eVec(:,i) = evec(:,i)./sqrt(eVal(i));
end

%% compute PCs, with reflections
options = cell(ndim, 1);
pcW = cell(ndim, 1);

parfor i=1:ndim
    %pcW{i} = cell(1,1);
    fprintf('Computing PC %d of %d...\n', i,ndim);
    pcW{i} = zeros(ninput*2, 1);
    for j=1:ninput
        if eVec(j,i) > 0
            pcW{i}(j) = eVec(j,i);
        else
            pcW{i}(j+ninput) = -1*eVec(j,i);
        end
    end
    % weighted average
    pcW{i} = pcW{i} / sum(pcW{i});
    options{i}.eta = eta;
    options{i}.datweights = pcW{i};
    options{i}.useLagrange = true;
    options{i}.useMem = true;
    [ FV_pc_pos{i}] = MultiResElasticAv( FVlogs,Topology,Ev,Ef,Eo,boundaryedges,options{i} );
    fprintf('PC %d of %d...Done.\n', i,ndim);
    
    if false
        figure
        patch(FV_pc_pos{i}, 'FaceColor', [1 1 0], 'EdgeColor', 'none', 'FaceLighting', 'phong');
        axis equal; axis tight; axis off; cameratoolbar; light; %view(45,0);
    end
end
fprintf('all pc computed.\n');

%%
parfor i=1:ndim
    fprintf('Computing reflection of PC %d of %d...\n', i,ndim);
    % reflections
    pcW{i} = zeros(ninput*2, 1);
    tmp = eVec(:,i);
    for j = 1:ninput
        if tmp(j) > 0
            pcW{i}(j+ninput) = tmp(j);
        else
            pcW{i}(j) = -1*tmp(j);
        end
    end
    pcW{i} = pcW{i} / sum(pcW{i});
    options{i}.eta = eta;
    options{i}.datweights = pcW{i};
    options{i}.useLagrange = true;
    options{i}.useMem = true;
    options{i}.verbose = true;
    [ FV_pc_neg{i}] = MultiResElasticAv(FVlogs,Topology,Ev,Ef,Eo,boundaryedges,options{i} ); 
     fprintf('Reflection of PC %d of %d...Done.\n', i,ndim);
end
fprintf('all pc refs computed.\n');

for i=1:ndim
    FV_pc_ref{i} = FV_pc_pos{i};
    FV_pc_ref{i+ndim} = FV_pc_neg{i};
end

if isShowFig
    figure;
    for i=1:2*ndim
        subplot(2,ndim,i);
        patch(FV_pc_neg{3}, 'FaceColor', [1 1 0], 'EdgeColor', 'none', 'FaceLighting', 'phong');
        axis equal; axis tight; axis off; cameratoolbar; light; %view(45,0);
        hold on
        patch(FV_ga, 'FaceColor', [0 1 1], 'EdgeColor', 'none', 'FaceLighting', 'phong');
        axis equal; axis tight; axis off; cameratoolbar; %view(45,0);
        
    end
end


end