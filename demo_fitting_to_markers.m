%% demo script to fit model to motion capture file (c3d now) 
function demo_fitting_to_markers(seq_id, max_frame)
if nargin < 1
    seq_id = 1;
    max_frame = 221;
end

addpath(genpath('../clean_code/'));
c3dpath = 'data/c3d_data/';
marker_type = 'MPI'; % CMU or MPI
seq_names = {'stretches', 'jumping_jacks', '05_03', '10_03'};
c3dfile = [seq_names{seq_id}, '.c3d'];
fprintf('Processing motion file: %s\n', c3dfile);
fprintf('Markers type: %s\n', marker_type);
global q0;
isVaxD = C3D_VaxD2PC('file', [c3dpath, c3dfile]);
if isVaxD
    isVaxD = C3D_VaxD2PC('ConvertFile', [c3dpath, c3dfile]);
end

[Markers,VideoFrameRate,AnalogSignals,AnalogFrameRate,Event,ParameterGroup,CameraInfo,ResidualError]=...
    readC3D([c3dpath,c3dfile]);
totalFrame = size(Markers,1);
fprintf('Total frames: %d\n', totalFrame);
      
%% labels store markers names, used to find correspondence from SMPL model
tmp = ParameterGroup(3).Parameter;
len_tmp = size(tmp, 2);
labels = [];
for p=1:len_tmp
    if strcmp(tmp(p).name{1}, 'LABELS')
        labels = ParameterGroup(3).Parameter(p).data;
        break;
    end
end
if isempty(labels)
    error('labels data not found.');
end

c3d_map = mocap_corres_smpl(marker_type);

lm_ind = [];
lm_tf = [];
nFound = 0;
for i=1:length(labels)
    query = labels{1,i};
    auxInd = strfind(query, ':');
    if ~isempty(auxInd)
        query = query(auxInd+1:end);
    end
    if isKey(c3d_map, query)
        nFound = nFound+1;
        fprintf('%d Found %s \n', nFound, query);
        lm_ind = [lm_ind, c3d_map(query)+1];
        lm_tf = [lm_tf; i];
    else
        fprintf('%s Not Found \n', query);
    end
end

a_list = [0.5];
stepFrame = 20;
useTrack = true;
saveMarkerMode = false;
showFig = false;

for alpha_id = 1:length(a_list)
    %% get c3d markers index in low_tr_reg model
    alpha = a_list(alpha_id);
    dataset = 'hc_male3';
    expid = 'mocapSeq';
    trial = ['a',num2str(alpha)];
    seq = [c3dfile(1:end-4), '_', trial, '_S',num2str(stepFrame)];
    % creat save folder if not exist
    savefolder = ['results_',expid,'_',dataset,'/'];
    modelfolder = 'data/precomputed/';
    if ~exist(savefolder, 'dir')
        mkdir(savefolder);
    end
    
    if strcmp(dataset, 'fattySim')||strcmp(dataset, '50021')
        highfile = [modelfolder,'01.ply'];
        lowfile = [modelfolder, 'deci_01.ply'];
    else
        highfile = [modelfolder,'tr_reg_000.ply'];
        lowfile = [modelfolder, 'low_reg_000.ply'];
    end
    if exist(highfile, 'file')
        [highmodel.faces,highmodel.vertices]=plyread(highfile,'tri');
    else
        error('high file not found.');
    end
    if exist(lowfile, 'file')
        [lowmodel.faces, lowmodel.vertices] = plyread(lowfile, 'tri');
    else
        error('low file not found.');
    end
    % get the indice of high-res and low-res mesh
    corres_idx = knnsearch(highmodel.vertices, lowmodel.vertices);
    high_model_lm = highmodel.vertices(lm_ind, :);
    low_idx = knnsearch(lowmodel.vertices, high_model_lm);
    
    % visual markers in human body
    if false
        figure;
        patch(highmodel, 'FaceColor', [1 1 1], 'EdgeColor', 'none', 'FaceLighting', 'phong','FaceAlpha', 1); hold on;
        plot3(high_model_lm(:,1), high_model_lm(:,2),high_model_lm(:,3), 'bo', 'MarkerSize',6,'MarkerFaceColor',[1 0 1]);hold on;
        for k=1:size(high_model_lm,1)
           text(high_model_lm(k,1),high_model_lm(k,2),high_model_lm(k,3),[' ',num2str(k)],'HorizontalAlignment','left','FontSize',20);
        end
        axis equal; axis tight; axis off; cameratoolbar;light;
    end
    fprintf('c3d markers index done.\n');
    %% begin with frame 1, with default model
    % mesh color
    datac = [1 1 1];
    modelc = [1 1 0];
    fitc = [0 1 1];
    % marker color
    mmarkerc = [1 0 0];
    dmarkerc = [1 1 1];
    
    %set up two options
    if strcmp(dataset, 'hc_male3')
        ntrain = 29;
        nlevel = 2;
        pct = 99;
        epoch = 1;
        eta = 0.0001;
        ndim = 1;
        max_iter = 1;
        
    elseif strcmp(dataset, 'fatty') || strcmp(dataset, 'fattySim')
        ntrain = 20;
        nlevel = 2;
        pct = 99;
        epoch = 1;
        eta = 0.0001;
        ndim = 10;
        max_iter = 1;       
    end
    
    seq = [seq,'_D',num2str(ndim),'_I',num2str(max_iter)];
    lmfolder = [savefolder, seq, '/lm/'];
    if ~exist(lmfolder, 'dir')
        mkdir(lmfolder);
    end
    matfolder = [savefolder, seq, '/mat/'];
    if ~exist(matfolder, 'dir')
        mkdir(matfolder);
    end
    plyfolder = [savefolder, seq, '/ply/'];
    if ~exist(plyfolder, 'dir')
        mkdir(plyfolder);
    end
    imgfolder = [savefolder, seq, '/img/'];
    if ~exist(imgfolder, 'dir')
        mkdir(imgfolder);
    end
    
    run_info = [];
    run_info.dataset = dataset;
    run_info.eta = eta;
    run_info.ndim = ndim;
    run_info.max_iter = max_iter;
    
    
    if saveMarkerMode
        for frame_id = 1:totalFrame
            % load mocap data
            mocap_lm = squeeze( Markers(frame_id, lm_tf, :)) ./ 1000;
            % save mocap_lm to txt file
            lm_txt = [lmfolder, 'lm_',num2str(frame_id, '%04d'), '.txt'];
            fid = fopen(lm_txt, 'w');
            fprintf(fid, '%f %f %f\n', mocap_lm');
            fclose(fid);
        end
        fprintf('All markers data save to %s \n', lmfolder);
        fprintf('Marker type: %s \n', marker_type);
        return;
    end
    
    
    % load precomputed model
    pc_file = [modelfolder,dataset,'PC_N',num2str(ntrain),'L',num2str(nlevel),...
        'P',num2str(pct),'Ep',num2str(epoch),'E',num2str(eta,'%.e'),'.mat'];
    pc_data = load(pc_file);
    ref_model = pc_data.FV_opt;
    FV_pc_ref = pc_data.FV_pc_ref;
    ndim_max = length(FV_pc_ref) / 2;
    if ndim > ndim_max
        ndim = ndim_max;
    end
    FV_pc_use = cell(ndim*2,1);
    for i=1:ndim
        FV_pc_use{i} = FV_pc_ref{i};
        FV_pc_use{i+ndim} = FV_pc_ref{i+ndim_max};
    end
    clear pc_data
    % precompute quantities
    Topo = ref_model.faces;
    [Ev, Eo, Ef] = getEdgesFromFaces(Topo);
    boundaryedges = Ef(:,2)==0;
    nverts = size(ref_model.vertices, 1);
    nedge = size(Ev, 1);
    nface = size(Ef, 1);
    vones = ones(nverts,1);
    
    % set options
    opt_fit = [];
    opt_path = [];
    opt_pc = [];
    opt_fit.mu = 1;
    opt_fit.lambda = 1;
    if strcmp(dataset, 'hc_male3')
        opt_fit.eta = 0.001;
        opt_fit.alpha = alpha; % 0.1 for body
        opt_path.eta = 0.0001;
        opt_path.max_level = 2;
        
    elseif strcmp(dataset, 'fatty')||strcmp(dataset, 'fattySim')||strcmp(dataset, '50021')
        opt_fit.eta = 0.001;
        opt_fit.alpha = alpha; % 0.1 for body
        opt_path.eta = 0.0001;
        opt_path.max_level = 2;
    end
    opt_path.dbg = false;
    opt_path.max_iter = 1;
    opt_path.cascadic = false;
    opt_path.fixGA = true;
    opt_path.upAdj = true;
    opt_path.startLogmap = true;
    opt_path.localAdj = false;
    opt_path.usePre = false;
    opt_path.doGlobal = false;
    opt_path.finalGlobal = false;
    opt_path.useLagrange = true;
    opt_path.useMem = true;
    opt_pc.ndim = ndim;
    opt_pc.eta = opt_path.eta;
    opt_pc.nlength = 2^opt_path.max_level + 1;
    opt_pc.dry_run = false;
    opt_pc.ninput = 1;
    opt_pc.weights = vones;
    opt_pc.useLagrange = opt_path.useLagrange;
    opt_pc.useMem = opt_path.useMem;
    opt_pc.alphas = [];
    opt_pc.init = 0;
    opt_pc.useRefLen = true;
    
    nMarkers = length(low_idx);
    vmarkerc = rand(nMarkers, 3);
    
    poolobj = gcp('nocreate');
    if isempty(poolobj)
        parpool(1);
    end   
    
    if totalFrame > max_frame
        totalFrame = max_frame;
    end
    fprintf('Work on %d / %d\n', max_frame, totalFrame);
    
    for frame_id = 1:stepFrame:totalFrame
        %fprintf('reading results of frame %d\n', frame_id);
        opt_pc.id = frame_id;
        fprintf('Fitting to frame %d...\n', frame_id);
        if frame_id == 1
            model = ref_model;
        end
        % align markers (rigid fitting)
        model_lm = model.vertices(low_idx, :); % initial guess
        mocap_lm = squeeze( Markers(frame_id, lm_tf, :)) ./ 1000; % ground truth markers
        % update model markers and then model
        [~, model_lm, TF] = procrustes(mocap_lm, model_lm, 'Scaling', true, 'reflection', false);
        meanC = mean(TF.c);
        C = repmat(meanC, nverts, 1);
        model.vertices =  model.vertices * TF.T + C;        
        
        optoptions = optimoptions('fminunc','Algorithm','trust-region','SpecifyObjectiveGradient',true,...
            'CheckGradients',false,'Display','final', 'HessianFcn','objective');
        
        %% use mesh editing (non-rigid fitting)
        FVs{1} = model;
        fun = @(x) fastShellMarkerDistance(x,FVs,mocap_lm,low_idx,Topo,Ev,Ef,Eo,boundaryedges,opt_fit);
        x0 = model.vertices(:);
        [xopt] = fminunc(fun,x0,optoptions);
        % update model vertices
        model.vertices = reshape(xopt, nverts, 3);      
        
        %% init geo path, recon logmap, shoot back, mesh editing
        FVs{1} = model;
        FV_opt = ref_model;
        [ ~,geo_path ] = GeodesicAveragePar( FVs,Topo,opt_path,FV_opt );
        
        %% recon. logmap and model
        dbg = opt_path.dbg;
        q0 = FV_opt;
        FVlogs_est = cell(1,1);
        for iter = 1:max_iter
            %opt_fit.alpha = iter;
            fprintf('\n--- iter %d of %d ---\n',iter,max_iter);
            % update logmap shape          
            if opt_path.max_level > 0
                FVlogs_est{1} = geo_path{1}(2);
            else
                FVlogs_est{1} = model;
            end            
            % recon model
            opt_pc.saveFolder = plyfolder;           
            [reconW, FV_recon1, FV_recon2, FV_recon3] = reconViaDual2m(FV_opt, FV_pc_use, FVlogs_est, Topo, opt_pc);
            
            if useTrack
                opt_pc.alphas = reconW{1};
            else
                opt_pc.alphas = [];
            end
            [~, FV_recon3{1}.vertices] = procrustes(model.vertices, FV_recon3{1}.vertices, 'Scaling', false,'reflection',false);
            % update model
            model.vertices = FV_recon3{1}.vertices;
            fit = model;
            % feed model to optimise with landmarks
            x0 = model.vertices(:);
            FVs{1} = model; % similar to previous frame
            fun = @(x) fastShellMarkerDistance(x,FVs,mocap_lm,low_idx,Topo,Ev,Ef,Eo,boundaryedges,opt_fit);
            [xopt] = fminunc(fun,x0,optoptions);
            fit.vertices = reshape(xopt, nverts, 3);
            
            if iter < max_iter && opt_path.max_level > 0
                % feed model to recompute geo path
                % update input shape
                %FVs{1} = model;
                geo_path{1}(end) = fit;
                opt_path.startLogmap = false;
                [ ~,geo_path ] = GeodesicAveragePar( FVs,Topo,opt_path,FV_opt,geo_path );
            end
            % show current fit and model
            if showFig
                figure;
                subplot(1,2,1);
                patch(model, 'FaceColor', modelc, 'EdgeColor', 'none', 'FaceLighting', 'phong',...
                    'FaceAlpha', 0.4);
                hold on;
                plot3(mocap_lm(:,1),mocap_lm(:,2),mocap_lm(:,3), 'bo', 'MarkerSize',5,'MarkerFaceColor',dmarkerc);
                hold on;
                plot3(model.vertices(low_idx,1),model.vertices(low_idx,2),...
                    model.vertices(low_idx,3), 'bo', 'MarkerSize',5,'MarkerFaceColor',mmarkerc);
                
                axis equal; axis tight; axis off; cameratoolbar;light;
                title(['Frame ',num2str(frame_id), ' recon and data']);
                
                subplot(1,2,2);
                patch(fit, 'FaceColor', fitc, 'EdgeColor', 'none', 'FaceLighting', 'phong',...
                    'FaceAlpha', 0.4);
                hold on;
                plot3(mocap_lm(:,1),mocap_lm(:,2),mocap_lm(:,3), 'bo', 'MarkerSize',5,'MarkerFaceColor',dmarkerc);
                hold on;
                plot3(fit.vertices(low_idx,1),fit.vertices(low_idx,2),...
                    fit.vertices(low_idx,3), 'bo', 'MarkerSize',5,'MarkerFaceColor',mmarkerc);
                axis equal; axis tight; axis off; cameratoolbar;light;
                title(['Frame ',num2str(frame_id), ' fit and data']);
            end
        end
        fprintf('Frame %d Done.\n', frame_id);
        if true
            % save mat results
            savename = [matfolder, 'frame_',num2str(frame_id,'%04d'), '.mat'];
            save(savename, 'model','fit', 'mocap_lm', 'low_idx', 'reconW','run_info');
            % save ply results
            modelname = [plyfolder, 'model_',num2str(frame_id,'%04d'), '.ply'];
            plywrite( modelname, model.faces, model.vertices);
            fitname = [plyfolder, 'fit_',num2str(frame_id,'%04d'), '.ply'];
            plywrite( fitname, fit.faces, fit.vertices);
        end
    end
    
end
end
