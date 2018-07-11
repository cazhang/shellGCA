function [ FV_gas, FVgeo ] = GeodesicAveragePar( FVs, Topo, opt, ga_init, geo_init )
%The code was originally written by W.Smith to compute geodesic average by
%using a relaxation scheme. 
%This new version is modified to use fminunc optimiser with the support of
%gradient and hessian matrix. 

% FVs: (nres x nmeshes)input data with faces and vertices; FVs{m} is the m-th mesh.
% pmax: max iterations of path subdividing, i.e. pmax=3, 2^3+1=9

% this is to support parfor by modifying data to sliced structure rather
% than 2d grid, i.e. instead of storing geodesics into cell{nmesh, nshell},
% it is put into a 2D cell{nmesh}(nshell)
dbg = opt.dbg;
eta = opt.eta;
cascadic = opt.cascadic;
if cascadic
    max_level = opt.max_level;
    num_shells = 2^max_level+1;
else
    if isfield(opt, 'max_level')
        num_shells = 2^opt.max_level+1;
    else
        num_shells = opt.num_shells;
    end
end

nmeshes = length(FVs);
% compute basic geometry information
[Ev, Eo, Ef] = getEdgesFromFaces(Topo);
boundaryedges = Ef(:,2)==0;

disp('Initialising geodesic average...');
options = [];
options.eta = eta;
options.useLagrange = opt.useLagrange;
options.useMem = opt.useMem;
if nargin > 3
    FV_ea = ga_init;
else
    [ FV_ea] = MultiResElasticAv( FVs,Topo,Ev,Ef,Eo,boundaryedges,options );
end

% Initialise geodesic paths with just two points: shape and average
FV_ga_old = FV_ea;
% Begin time hierarchical alternating relaxation
%FVgeo = cell(nmeshes, num_shells);
FVgeo = cell(nmeshes,1);

% Copy existing shapes: 1 is mean, last one is input
for m=1:nmeshes
    if nargin > 4
        % init geo path
        FVgeo{m} = geo_init{m};  
    else
        % only copy average and input
        FVgeo{m}(1)=FV_ea;
        FVgeo{m}(num_shells) = FVs{m};
    end
end
        
clear geo_init
%% compute geodesic
if cascadic
    % compute geodesic cascadic
    [FV_gas, FVgeo] = computeGeodesicCascadic(FVgeo,FV_ga_old,Topo,Ev,Ef,Eo,...
        boundaryedges, nmeshes, opt);
else
    % compute geodesic temparally . computeGeodesicTemparal
    [FV_gas, FVgeo] = computeGeodesicTemparal(FVgeo,FV_ga_old,Topo,Ev,Ef,Eo,...
        boundaryedges, nmeshes, opt);
end
end


%% compute geodesic cascadicly, from 2^1+1 to 2^level_max+1
function [FV_gas, FVgeo] = computeGeodesicCascadic(FVgeo,FV_ga_old,Topo,Ev,Ef,Eo,boundaryedges,...
    nmeshes, opt)
% start multilevel cascadic: sparse to dense
dbg = opt.dbg;
max_iter = opt.max_iter;
max_level = opt.max_level;
num_shells = 2^max_level+1;
eta = opt.eta;
fixGA = opt.fixGA;
upAdj = opt.upAdj;
startLogmap = opt.startLogmap;
FV_gas{1} = FV_ga_old;
if ~isfield(opt, 'doGlobal')
    opt.doGlobal = true;
end
for level=1:max_level
    aux = 2^(max_level-level);
    maxiter_internal = 2^(level-1);
    free_shape = 2^level-1;
    
    fprintf(['Subdividing discrete geodesic paths at level = ', num2str(level),'\n']);
    if nmeshes>2
        parfor m=1:nmeshes
            % Compute new midpoints
            for i=1:maxiter_internal
                % left shape
                left_idx = aux*(2*(i-1) + 0) + 1;
                right_idx = aux*(2*(i-1) + 2) + 1;
                mid_idx = aux*(2*(i-1) + 1) + 1;
                disp(['Mesh ', num2str(m)]);
                disp(['Computing midpoint of ', num2str(left_idx), ' and ', num2str(right_idx)]);
                options = [];               
                options.eta = eta;
                options.useMem = true;
                options.useLagrange = true;
                % init with the one closer to average
                options.InitNum = 1;
                if dbg
                    FVgeo{m}(mid_idx) = FVgeo{m}(left_idx);
                    FVgeo{m}(mid_idx).vertices = 0.5*(FVgeo{m}(left_idx).vertices...
                        +FVgeo{m}(right_idx).vertices);                   
                else
                    [ FVgeo{m}(mid_idx) ] = MultiResElasticAvOfTwo( FVgeo{m}(left_idx),...
                        FVgeo{m}(right_idx),Topo,Ev,Ef,Eo,boundaryedges,options );
                end
            end
        end
    else
        for m=1:nmeshes
            % Compute new midpoints
            for i=1:maxiter_internal
                % left shape
                left_idx = aux*(2*(i-1) + 0) + 1;
                right_idx = aux*(2*(i-1) + 2) + 1;
                mid_idx = aux*(2*(i-1) + 1) + 1;
                disp(['Mesh ', num2str(m)]);
                disp(['Computing midpoint of ', num2str(left_idx), ' and ', num2str(right_idx)]);
                options = [];                
                options.eta = eta;
                options.useMem = true;
                options.useLagrange = true;
                % init with the one closer to average
                options.InitNum = 1;
                if dbg
                    FVgeo{m}(mid_idx) = FVgeo{m}(left_idx);
                    FVgeo{m}(mid_idx).vertices = 0.5*(FVgeo{m}(left_idx).vertices...
                        +FVgeo{m}(right_idx).vertices);
                else
                    [ FVgeo{m}(mid_idx) ] = MultiResElasticAvOfTwo( FVgeo{m}(left_idx),...
                        FVgeo{m}(right_idx),Topo,Ev,Ef,Eo,boundaryedges,options );
                end
            end
        end
    end
    
    if max_iter > 0
        disp('Alternating relaxation...');
        %log_idx = aux*(2*(max_iter-1) + 1) + 1;
        log_idx = aux+1;
        % 1->5, max_iter=1, aux=4
        % 2->3, max_iter=2, aux=2
        % 4->2, max_iter=4, aux=1
        for iter=1:max_iter
            options = [];
            options.eta = eta;
            options.useMem = true;
            options.useLagrange = true;
            disp('Relaxing geodesic average...');
            FVgeotemp = cell(nmeshes,1);
            for m=1:nmeshes
                FVgeotemp{m}=FVgeo{m}(log_idx);
            end
            options.FVinit = FV_ga_old;
            if dbg || fixGA
                FV_ga = FV_ga_old;
            else
                FV_ga = MultiResElasticAv( FVgeotemp,Topo,Ev,Ef,Eo,boundaryedges,options );
            end
            FV_ga_old = FV_ga;
            FV_gas{(level-1)*max_iter+iter+1} = FV_ga;
            clear FVgeotemp
            disp('Relaxing discrete geodesic paths...');
            
            opts = [];
            opts.aux = aux;
            opts.free_shape = free_shape;
            opts.num_shells = num_shells;
            opts.eta = eta;
            opts.dbg = dbg;
            opts.startLogmap = startLogmap;
            opts.useMem = options.useMem;
            if nmeshes>2
                parfor m=1:nmeshes
                    % update GA
                    if upAdj
                        [FVgeo{m}] = updateFreeShellWithAdj(FV_ga,FVgeo{m},Topo,Ev,Ef,Eo,boundaryedges,...
                            m,opts);
                    else
                        [FVgeo{m}] = updateFreeShellViaWAvg(FV_ga,FVgeo{m},Topo,Ev,Ef,Eo,boundaryedges,...
                            m,opts);
                    end
                end
            else
                for m=1:nmeshes
                    % update GA
                    if upAdj
                        [FVgeo{m}] = updateFreeShellWithAdj(FV_ga,FVgeo{m},Topo,Ev,Ef,Eo,boundaryedges,...
                            m,opts);
                    else
                        [FVgeo{m}] = updateFreeShellViaWAvg(FV_ga,FVgeo{m},Topo,Ev,Ef,Eo,boundaryedges,...
                            m,opts);
                    end
                end
            end
            
            options = [];
            options.eta = eta;
            options.MaxIter = 2000;
            
            cost=0;
            for m=1:nmeshes
                for k=1:2^level
                    start = aux*(k-1)+1;
                    ending = aux*(k)+1;
                    segEner(k) = geomDSD(FVgeo{m}(start),FVgeo{m}(ending),Topo,Ev,Ef,options.eta,1);
                    cost=cost+segEner(k);
                end
                fprintf('Level %d, Iter %d, Mesh %d:',level, iter, m);
                disp(segEner);
            end
            fprintf('Level=%d, Iter=%d, Total Cost=%.5f\n',level,iter,cost );
            Energies(level,iter) = cost;
        end
    end
    % do global after each level
    if opt.doGlobal && (level == max_level)
        
        % global optimisation
        disp('3 - whole geodesic optimisation');
        optglo = [];
        optglo.num_shells = free_shape+2;
        optglo.eta = eta;
        optglo.useMem = true;
        optglo.aux = aux;
        parfor m=1:nmeshes
            tmp = [];
            FVgeo{m} = optimizeGeoPath(FVgeo{m}, Topo, Ev,Ef,Eo,boundaryedges,optglo)
            for k=1:optglo.num_shells-1
                if optglo.useMem
                    tmp(k) = geomDSD(FVgeo{m}(aux*(k-1)+1),FVgeo{m}(aux*k+1),Topo,Ev,Ef,eta,1);
                else
                    tmp(k) = DSD(FVgeo{m}(aux*(k-1)+1),FVgeo{m}(aux*k+1),Topo,Ev,Ef,eta,1);
                end
            end
            fprintf('Level %d after global optimisation, Mesh %d:', level, m);
            disp(num2str(tmp));
        end
    end
end 
end

%% compute geodesics temparaly, starting from 1/K Log, to (K-1)/K Log
function [FV_gas, FVgeo] = computeGeodesicTemparal(FVgeo,FV_ga_old,Topo,Ev,Ef,Eo,boundaryedges,...
    nmeshes, opt)
% start from S_1, to S_K-1
max_iter = opt.max_iter;
if isfield(opt, 'max_level')
    num_shells = 2^opt.max_level+1;
else
    num_shells = opt.num_shells;
end
if ~isfield(opt, 'doGlobal')
    opt.doGlobal = true;
end
eta = opt.eta;
dbg = opt.dbg;
upAdj = opt.upAdj;
fixGA = opt.fixGA;
startLogmap = opt.startLogmap;
free_shape = num_shells-2;   % 3
K = free_shape + 1;         % 4
aux = 1;
FV_gas{1} = FV_ga_old;
useMem = opt.useMem;
useLagrange = opt.useLagrange;

parfor m=1:nmeshes
    % Compute new points
    for k=1:free_shape
        if startLogmap
            updateSet = 1:free_shape;
        else
            updateSet = free_shape:-1:1;
        end
        i = updateSet(k)+1;
        % this index S_k
        
        %i = k+1; % this index FVgeo{m}(i)
        options = [];
        options.eta = eta;
        options.useMem = useMem;
        options.useLagrange = useLagrange;
        options.verbose = false;
        % init with previous shell or given inits
        if isfield(FVgeo{m}(i), 'faces') && ~isempty(FVgeo{m}(i).faces)
            options.FVinit = FVgeo{m}(i);
        else
            if startLogmap
                options.FVinit = FVgeo{m}(i-1);
            else
                options.FVinit = FVgeo{m}(i+1);
            end
        end
        if startLogmap
            options.datweights = [K-k, k];
            options.datweights = options.datweights / sum(options.datweights);
        else
            options.datweights = [k, K-k];
            options.datweights = options.datweights / sum(options.datweights);
        end
        fprintf('Computing mesh %d at point %d\n', m, i);
        %disp(options.datweights);
        
        if dbg
            FVgeo{m}(i) = FVgeo{m}(1);
            FVgeo{m}(i).vertices = (k/K)*(FVgeo{m}(1).vertices...
                +(1-k/K)*FVgeo{m}(num_shells).vertices);
        else
            
            [ FVgeo{m}(i) ] = MultiResElasticAvOfTwo( FVgeo{m}(1),...
                FVgeo{m}(num_shells),Topo,Ev,Ef,Eo,boundaryedges,options );
            
        end
        
    end
end

disp('Alternating relaxation...');
update_options = [];
update_options.free_shape = free_shape;
update_options.num_shells = num_shells;
update_options.aux = aux;
update_options.dbg = dbg;
update_options.eta = eta;
update_options.startLogmap = startLogmap;
update_options.useMem = useMem;

if max_iter > 0
    for iter=1:max_iter
        if mod(iter, 2)==1
            update_options.startLogmap = false;
        else
            update_options.startLogmap = true;
        end
        disp('1 - Relaxing geodesic average...');
        if dbg || fixGA
            FV_ga = FV_ga_old;
            disp('Skip average computation.');
        else
            FV_ga = updateGAFromLog(FV_ga_old, FVgeo,Topo,Ev,Ef,Eo,boundaryedges,eta);
        end
        FV_ga_old = FV_ga;
        FV_gas{iter+1} = FV_ga;
            
        update_options.iter = iter; 
        
        disp('2 - Relaxing discrete geodesic paths...propagating from average');                 
        parfor m=1:nmeshes     
            tmp = [];
            if upAdj
                [FVgeo{m}] = updateFreeShellWithAdj(FV_ga,FVgeo{m},Topo,Ev,Ef,Eo,boundaryedges,...
                    m, update_options);
            else
                [FVgeo{m}] = updateFreeShellViaWAvg(FV_ga,FVgeo{m},Topo,Ev,Ef,Eo,boundaryedges,...
                    m, update_options);
            end
            for k=1:K
                if update_options.useMem
                    tmp(k) = geomDSD(FVgeo{m}(k),FVgeo{m}(k+1),Topo,Ev,Ef,eta,1);
                else
                    tmp(k) = DSD(FVgeo{m}(k),FVgeo{m}(k+1),Topo,Ev,Ef,eta,1);
                end
            end
            fprintf('Iter %d, after averaging, Mesh %d:', iter, m);
            disp(tmp);
            
        end
        if opt.doGlobal
            if (opt.finalGlobal && iter==max_iter) || ~opt.finalGlobal
                % global optimisation
                disp('3 - whole geodesic optimisation');
                optglo = [];
                optglo.num_shells = num_shells;
                optglo.eta = eta;
                optglo.useMem = true;
                optglo.aux = 1;
                parfor m=1:nmeshes
                    tmp = [];
                    FVgeo{m} = optimizeGeoPath(FVgeo{m}, Topo, Ev,Ef,Eo,boundaryedges,optglo)
                    for k=1:K
                        if optglo.useMem
                            tmp(k) = geomDSD(FVgeo{m}(k),FVgeo{m}(k+1),Topo,Ev,Ef,eta,1);
                        else
                            tmp(k) = DSD(FVgeo{m}(k),FVgeo{m}(k+1),Topo,Ev,Ef,eta,1);
                        end
                    end
                    fprintf('Iter %d after global optimisation, Mesh %d:', iter, m);
                    disp(num2str(tmp));
                end
            end
        end
    end
end        
end

%% sub functions
function FV_ga = updateGAFromLog(FV_ga_old, FVgeo,Topo,Ev,Ef,Eo,boundaryedges,eta)
    nmeshes = length(FVgeo);
    FVgeotemp = cell(nmeshes,1);
    for m=1:nmeshes
        FVgeotemp{m}=FVgeo{m}(2);
    end
    options.FVinit = FV_ga_old;
    options.eta = eta;
    options.useLagrange = true;
    [ FV_ga] = MultiResElasticAv( FVgeotemp,Topo,Ev,Ef,Eo,boundaryedges,options );
    
end
%% update free shells using its adjacent shapes
function [FV_path] = updateFreeShellWithAdj(FV_ga,FV_path,Topo,Ev,Ef,Eo,boundaryedges,...
    m, opt)
% given the new average shape, update free shapes using the two adjacent shapes
aux = opt.aux;
free_shape = opt.free_shape;
FV_path(1)=FV_ga;
if opt.startLogmap
    updateSet = 1:free_shape;
else
    updateSet = free_shape:-1:1;
end

for idx=1:length(updateSet)
    k = updateSet(idx);
    left_idx = aux*(k-1)+1;
    right_idx = aux*(k+1)+1;
    mid_idx = aux*k + 1;
    
    %if mod(opt.iter,2) == mod(mid_idx,2)
    %    continue;
    %end
    %fprintf('On mesh %d, refining shell %d using %d and %d\n',m,mid_idx,left_idx,right_idx);
    options = [];
    options.eta = opt.eta;
    options.useMem = opt.useMem;
    options.FVinit = FV_path(mid_idx);
    options.useLagrange = true;
    if opt.dbg
        fprintf('mesh %d: update %d with %d and %d\n', m,mid_idx,left_idx,right_idx);
    else
        [ FV_path(mid_idx) ] = MultiResElasticAvOfTwo( FV_path(left_idx),...
            FV_path(right_idx),Topo,Ev,Ef,Eo,boundaryedges,options );
    end
    
end
end

%%
function [FV_path] = updateFreeShellViaWAvg(FV_ga,FV_path,Topo,Ev,Ef,Eo,boundaryedges,...
    m,opt)
% update free shapes using the geo mean FV_path(1) and input shape
% FV_path(num_shell)
aux = opt.aux;
free_shape = opt.free_shape;
num_shells = opt.num_shells;
FV_path(1)=FV_ga;
K = num_shells - 1;% K=4
eta = opt.eta;
dbg = opt.dbg;

for i=1:free_shape % free_shape = 3
    mid_idx = aux*i + 1; 
    k = mid_idx-1;
    weights = [K-k, k];
    fprintf('On mesh %d, refining shell %d using weighted average.\n',m,mid_idx);
    fprintf('[weight_input, weight_bar] = [%d, %d]\n', K-k, k);
    options = [];
    options.eta = eta;
    options.MaxIter = 2000;
    options.datweights = weights;
    options.FVinit = FV_path(mid_idx);
    options.useLagrange = true;
    if dbg
        FV_path(mid_idx).vertices = ((K-k)/K).*(FV_path(1).vertices...
            +(k/K).*FV_path(num_shells).vertices);
    else
        [ FV_path(mid_idx) ] = MultiResElasticAvOfTwo( FV_path(1),...
            FV_path(num_shells),Topo,Ev,Ef,Eo,boundaryedges,options );
    end
    
end
end

%% update free shells using its adjacent shapes
function [FV_path] = updateFreeShellShoot(FV_ga,FV_path,Topo,Ev,Ef,Eo,boundaryedges,...
    m, opt)
% given the new average shape, update free shapes using the two adjacent shapes
aux = opt.aux;
free_shape = opt.free_shape;
K = free_shape + 1;
FV_path(1)=FV_ga;
if opt.startLogmap
    updateSet = 1:free_shape;
else
    updateSet = free_shape:-1:1;
end

cost = 0;

for k=1:K
    W(k) = geomDSD(FV_path(k),FV_path(k+1),Topo,Ev,Ef,opt.eta,1);
    segLength(k) = sqrt(W(k));
    cost = cost + W(k);
    
end

if ~opt.startLogmap
    segLength = fliplr(segLength);
    FV_ga = FV_path(K+1);
end

for k=1:free_shape
    accLength(k) = 0;
    for i=1:k
        accLength(k) = accLength(k)+segLength(i);
    end
end

aveW = cost / K;
refLength = sqrt(aveW);

for idx=1:length(updateSet)
    k = updateSet(idx);
    mid_idx = aux*k + 1; 
    if opt.dbg
        fprintf('mesh %d: update %d, [real:%f, ref:%f]\n',m,mid_idx,accLength(idx),refLength*idx);
        continue;
    end
    % computer ratio
    beta = (refLength*idx) / accLength(idx);
    [FV_path(mid_idx),~] = rescaleShell(FV_ga, FV_path(mid_idx), Topo, Ev, Ef, Eo, boundaryedges, beta, options);
       
end
end

