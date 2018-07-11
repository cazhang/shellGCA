% Shell GCA main script
% Todo: given a set of input shells, extract geodesic principal components
% as eigenvectors
% Steps: 1. compute geodesic average \tilde{S}, and geodesic path
% connecting \tilde{S} and inputs S_i
% 2. compute geodesic path between inputs S_i, and average-mid paths
% 3. compute distance mat D, and gram matrix C
% 4. do eigen-decomposition and return eigenvectors. 

function [run_time] = demo_GCA(nlevel, percentage, dataid)
% nlevel - max geodesic resolution
% percentage - pca eigenmodes percentage
% dataid - dataset to process
addpath(genpath('../clean_code/'));
tic;
if nargin < 1
    nlevel = 2;
    percentage = 0.99;
    dataid = 1;
end

max_iter = 5; % default max iteration
dbg = false;

isShowFig = false;
isNormLog = false;
isVisPc = false;
expid = 'mem';

if dataid == 1
    dataset = 'faust';% faust holds 10 poses for each subject, use ninput for training, and rest poses for test
    id = 1;
    useTrain = 1;
    useTest = 0;
    ninput = 2;
    FVs = readData(dataset, ninput, id, useTrain);
    FVtest = readData(dataset, ninput, id, useTest);
    ga_init = FVs{1};
    eta = 0.0001;
    
elseif dataid == 2
    dataset = 'cat';
    id = 1;
    useTrain = 1;
    useTest = 0;
    ninput = 71;
    FVs = readData(dataset, ninput, id, useTrain);
    ntest = 71;
    FVtest = readData(dataset, ntest, id, useTest);
    ga_init = FVs{1};
    eta = 0.0001;
end

save_folder = ['results_',expid,'_',dataset, '/'];
if exist(save_folder, 'dir')
    disp('save folder already exists ');
else
    mkdir(save_folder);
    disp('create folder');
end
pc_folder = ['results_',expid,'_',dataset, '/PCs/'];
if exist(pc_folder, 'dir')
    disp('save folder already exists ');
else
    mkdir(pc_folder);
    disp('create folder');
end

fprintf('----- parameters -----------\n');
fprintf(' nlevel : %d\n', nlevel);
fprintf(' cutoff : %.2f\n', percentage);
fprintf(' dataset : %s\n', dataset);
fprintf(' eta : %.e\n', eta);
fprintf(' max_iter : %d\n', max_iter);
fprintf(' savefolder : %s\n', save_folder);
fprintf('----------------------------\n');

Topology = FVs{1}.faces;
[Ev, Eo, Ef] = getEdgesFromFaces(Topology);
boundaryedges = Ef(:,2)==0;

%% compute average and average-input paths
if nlevel > 0
    pathfile = [save_folder,dataset,'Geo_N',num2str(ninput),'L',num2str(nlevel),...
            'E',num2str(eta,'%.e'),'.mat'];
    
    if exist(pathfile, 'file')
        load(pathfile);
    else
        options = [];
        options.eta = eta;
        options.dbg = dbg;
        options.max_iter = max_iter;
        options.max_level = nlevel;
        options.cascadic = false;
        options.fixGA = false;
        options.upAdj = true;        
        options.startLogmap = true;
        options.finalGlobal = true;
        options.useLagrange = true;
        options.useMem = true;
        
        [FV_ga, FV_path] = GeodesicAveragePar( FVs,Topology,options,ga_init);
        
        save(pathfile, 'FV_ga', 'FV_path');
        
    end 
    num_opt = length(FV_ga);
    FV_opt = FV_ga{num_opt};
else
    % use elastic average instead
    options = [];
    options.eta = eta;
    options.useMem = true;
    options.useLagrange = true;   
    FV_opt = MultiResElasticAv( FVs,Topology,Ev,Ef,Eo,boundaryedges,options );   
end

% load pc file or compute it
pcfile = [save_folder,dataset,'PC_N',num2str(ninput),'L',num2str(nlevel),'P',num2str(percentage*100),...
    'E',num2str(eta,'%.e'),'.mat'];

if exist(pcfile, 'file')
    load(pcfile);
    fprintf('PCs already exist. Skip..\n');
  
    FV_pc_norm = FV_pc_ref;
    num_pc = length(FV_pc_norm)/2;
    %[FV_pc_norm, pcLength, pcLength2,refLength,betas] = normalisePC(FV_opt,FV_pc_ref,Topology,Ev,Ef,Eo,boundaryedges,eta,normMode);
    %fprintf('normalisition done.\n');
    if isVisPc
        % save mean shape
        %savename = [pc_folder, 'mean_01.ply'];
        %plywrite(savename, FV_opt.faces, FV_opt.vertices);
        % plot all pc as they are
        for i=1:num_pc
            savename = [pc_folder, 'pos_PC',num2str(i,'%02d'),'.ply'];
            plywrite(savename, FV_pc_norm{i}.faces, FV_pc_norm{i}.vertices);
            savename = [pc_folder, 'neg_PC',num2str(i,'%02d'),'.ply'];
            plywrite(savename, FV_pc_norm{i+num_pc}.faces, FV_pc_norm{i+num_pc}.vertices);
        end
        % check orthogonality of PCs
        % visualize PCs: FV_pc_norm
        nPC = length(FV_pc_norm)/2;
        num_vis = 3;
        num_step = 3;
        parfor i=1:num_vis
            %i = j;
            fprintf('shooting for PC %d\n', i);
            exp2opt = [];
            exp2opt.useLagrange = true;
            exp2opt.useMem = false;
            exp2opt.eta = eta;
            exp2opt.step = num_step;
            exp2opt.id = i;
            exp2opt.saveFolder = pc_folder;
            exp2opt.saveShoot = true;
            FV_pc_vis{i} = TestShellExp2(FV_opt,FV_pc_norm{i},Topology,Ev,Ef,Eo,boundaryedges,exp2opt);
            
        end
        
        parfor i=nPC+1:nPC+num_vis
            fprintf('shooting for PC %d\n', i);
            exp2opt = [];
            exp2opt.useLagrange = true;
            exp2opt.useMem = false;
            exp2opt.eta = eta;
            exp2opt.step = num_step;
            exp2opt.id = i;
            exp2opt.saveFolder = pc_folder;
            exp2opt.saveShoot = true;
            FV_pc_vis{i} = TestShellExp2(FV_opt,FV_pc_norm{i},Topology,Ev,Ef,Eo,boundaryedges,exp2opt);
            
        end
    end
    
else
    %% prepare logmap and reflections
    ntrain = length(FVs);
    FVlogs = cell(ntrain,1);
    Kmax = 2^nlevel;
    logIdx = Kmax / (2^nlevel) + 1;
    on_length = 2^nlevel+1;
    for i=1:ntrain
        if nlevel > 0
            FVlogs{i} = FV_path{i}(logIdx); % logmap
        else
            FVlogs{i} = FVs{i};
        end       
    end
         
    % compute reflections
    exp2opt = [];
    exp2opt.eta = eta;
    exp2opt.InitNum = 2;
    exp2opt.useLagrange = true;
    exp2opt.useMem = true;
    parfor i=1:ntrain
        FVlogsNeg{i} = TestShellExp2( FVlogs{i},FV_opt,Topology,Ev,Ef,Eo,boundaryedges,exp2opt );
    end
    for i=1:ntrain
        FVlogs{end+1} = FVlogsNeg{i};
    end
    clear FVlogsNeg;
    if isShowFig
        figure;
        for i=1:ntrain*2
            subplot(2,ntrain,i);
            patch(FVlogs{i}, 'FaceColor', [1 1 1], 'EdgeColor', 'none', 'FaceLighting', 'phong');
            axis equal; axis tight; axis off; cameratoolbar; light; view(0,90);
        end
    end
    
    gca_opt = [];
    gca_opt.nlength = on_length;
    gca_opt.cutoff = percentage;
    gca_opt.ninput = ntrain;
    gca_opt.eta = eta;
    gca_opt.isShowFig = false;
    gca_opt.isNormLog = isNormLog;
    
    [FV_pc_ref, C, eVal] = GCA(FV_opt, FVlogs, Topology, gca_opt);
    %[FV_pc_norm, pcLen1, pcLen2] = normalisePC(FV_opt,FV_pc_ref,...
    %Topology,Ev,Ef,Eo,boundaryedges,eta,normMode);
    %save(pcfile,'FV_pc_ref','FV_pc_norm','FV_opt','FVlogs','FVlogs_test','pcLen1','pcLen2','eVal');
    save(pcfile,'FV_pc_ref','FV_opt','FVlogs','eVal','C');
end
%% compute geodesics path for test data
     
if nlevel > 0 && length(FVtest) > 0
    pathfile_test = [save_folder,dataset,'Geo_N',num2str(ninput),'L',num2str(nlevel),...
        'E',num2str(eta,'%.e'),'_test.mat'];
    if exist(pathfile_test, 'file')
        load(pathfile_test);
    else
        options = [];
        options.eta = eta;
        options.dbg = dbg;
        options.max_iter = max_iter;
        options.max_level = nlevel;
        options.cascadic = false;
        options.fixGA = true;
        options.upAdj = true;
        options.startLogmap = true;
        options.finalGlobal = true;
        options.useLagrange = true;
        options.useMem = true;
        [ ~,FV_path_test ] = GeodesicAveragePar( FVtest,Topology,options,FV_opt);
        save(pathfile_test, 'FV_path_test');
    end
end
run_time = toc;
end