function demo_mesh_edit(in_eta)
% this script reproduces the results shown in Fig.10
if nargin < 1
    in_eta = 0.001;
end

dataset = 'scape71';
experiment = 'edit';
datafolder = 'data/precomputed/';
savefolder = ['results_',experiment,'_',dataset,'/'];
if ~exist(savefolder, 'dir')
    mkdir(savefolder);
end

global q0;

if strcmp(dataset, 'cat')
    pc_file = 'catPC_N10L2P99Ep1E1e+00_eta5.mat';
    load(pc_file);
    def_model = FV_opt;
    lm_ind = [303 343 390 438 496];
    lm_ind = [lm_ind 94]; % add one pt at back
    ndim = 5;
    max_iter = 1;
    
    
elseif strcmp(dataset, 'scape71')
    pc_file = [datafolder, 'scape_fullPC_N71L2P99.9Ep1E1e-04.mat'];
    init_file = [datafolder, 'low1.ply'];
    load(pc_file);
    mean_model = FV_opt;
    [def_model.faces, def_model.vertices] = plyread(init_file, 'tri');
    %lm_ind = [2,200,451,236,480];% head,L-hand,R-hand,L-foot,R-foot
    ndim = 20;
    max_iter = 1;
    % for fig.10 results, the most similar training data is retrieved as
    % initial guess
    scapedata= [datafolder, 'scape_data.mat'];
    if exist(scapedata, 'file')
        load(scapedata);
        fprintf('load low-res SCAPE data');
    end
end
showFig = true;
datac = [1 1 1];
modelc = [1 1 0];
fitc = [0 1 1];

poolobj = gcp('nocreate'); % If no pool, do not create new one.
if isempty(poolobj)
    parpool(1);
end
for test_id = 1
    model = def_model;
    data = def_model;
    %% compute RBM from landmark
    if strcmp(dataset, 'cat')
        data_lm = def_model.vertices(lm_ind, :);
        if test_id == 1
            data_lm(5,1) = data_lm(5,1) + 40;
            data_lm(5,2) = data_lm(5,2) - 30;
            data_lm(5,3) = data_lm(5,3) - 20;
        elseif test_id == 2
            data_lm(5,1) = data_lm(5,1) + 20;
            data_lm(5,2) = data_lm(5,2) - 50;
            data_lm(5,3) = data_lm(5,3) + 30;
        end
        
    elseif strcmp(dataset, 'scape71')
        if test_id==1 % fig.16 in 2016 paper
            highfile = [datafolder,'1.ply'];
            lowfile = [datafolder,'low1.ply'];
            [highmodel.faces,highmodel.vertices]=plyread(highfile, 'tri');
            [lowmodel.faces,lowmodel.vertices]=plyread(lowfile, 'tri');
            
            high_ind = [1062,1134,5850,5795,12476,9539];
            high_model_lm = highmodel.vertices(high_ind+1, :);
            lm_ind = knnsearch(lowmodel.vertices, high_model_lm);
            
            data_lm(1,:) = [0.116683 -0.0884821 0.299269];
            data_lm(2,:) = [0.110961 0.00129135 -0.202344];
            data_lm(3,:) = [0.144909 -0.0733306 0.0533981];
            data_lm(4,:) = [0.323708 -0.377689 -0.00993826];
            data_lm(5,:) = [0.77033 -0.646099 -0.0983151];
            data_lm(6,:) = [0.534004 -0.296305 0.00375251];
            
            % look for the closest training model using landmarks
            for i=1:length(scape_data)
                [dist_lm(i), ~, ~] = procrustes(data_lm, scape_data{i}.vertices(high_ind+1,:), 'Scaling', false, 'reflection', false);
            end
            [~,close_id] = min(dist_lm);
            model = low_data{close_id};
            
        elseif test_id==2 % fig.18 in 2017 paper           
            highfile = [datafolder,'1_res2.ply'];
            lowfile = [datafolder,'low1.ply'];
            [highmodel.faces,highmodel.vertices]=plyread(highfile, 'tri');
            [lowmodel.faces,lowmodel.vertices]=plyread(lowfile, 'tri');
            
            high_ind = [2100,1069,1122,1296,245,258];
            high_model_lm = highmodel.vertices(high_ind+1, :);
            lm_ind = knnsearch(lowmodel.vertices, high_model_lm);
                     
            data_lm(1,:) = [1.7340   -0.1161   -0.0259];
            data_lm(2,:) = [1.1185   -0.0648    0.6700];
            data_lm(3,:) = [1.2809   -0.3140   -0.2178];
            data_lm(4,:) = [1.1591   -0.1111    0.0546];
            data_lm(5,:) = [0.4419    0.3796    0.2352];
            data_lm(6,:) = [0.1151   -0.0114   -0.1898];
         
        elseif test_id==3 % fig.19 in 2017 paper
            highfile = [datafolder,'1_res2.ply'];
            lowfile = [datafolder,'low1.ply'];
            [highmodel.faces,highmodel.vertices]=plyread(highfile, 'tri');
            [lowmodel.faces,lowmodel.vertices]=plyread(lowfile, 'tri');
            
            high_ind = [2100,1069,1122,1296,245,258];
            high_model_lm = highmodel.vertices(high_ind+1, :);
            lm_ind = knnsearch(lowmodel.vertices, high_model_lm);
            
            data_lm(1,:) = [1.76895 -0.154094 0.0351899];
            data_lm(2,:) = [1.12638  0.21031 0.485901];
            data_lm(3,:) = [1.73305 -0.243183 -0.361392];
            data_lm(4,:) = [1.1591  -0.11109 0.0546433];
            data_lm(5,:) = [0.544733 0.396143 0.235155];
            data_lm(6,:) = [0.115068 -0.0113539  -0.189825];
        end        
    end

    nverts = size(model.vertices, 1);
    % compute optimal R and t using landmarks, model init with average
    model_lm = model.vertices(lm_ind,:);
    [~, Z, TF] = procrustes(data_lm, model_lm, 'Scaling', true, 'reflection', false);
    meanC = mean(TF.c);
    C = repmat(meanC, nverts, 1);
    model.vertices = model.vertices * TF.T + C;
    
    clear FV_ga
    clear FVs
    clear FVtest
    
    % show init model before fitting
    if showFig
        figure;
        patch(model, 'FaceColor', modelc, 'EdgeColor', 'none', 'FaceLighting', 'phong');
        axis equal; axis tight; axis off; cameratoolbar; light;
        hold on;
        plot3(model.vertices(lm_ind,1),model.vertices(lm_ind,2),model.vertices(lm_ind,3), 'ro', 'MarkerSize',10,'MarkerFaceColor',[0 1 0]);
        axis equal; axis tight; axis off; cameratoolbar;
        title('default model and markers');
        hold on;
        plot3(data_lm(:,1),data_lm(:,2),data_lm(:,3), 'ro', 'MarkerSize',10,'MarkerFaceColor',[1 0 0]);
        axis equal; axis tight; axis off; cameratoolbar;
        
        % text
        for i=1:6
            text(model_lm(i,1),model_lm(i,2),model_lm(i,3), ['model',num2str(i)], 'HorizontalAlignment','left','FontSize',8);
            text(data_lm(i,1),data_lm(i,2),data_lm(i,3), ['data',num2str(i)], 'HorizontalAlignment','left','FontSize',8);
        end
        
    end

    %% set up two options
    opt_fit = [];
    opt_path = [];
    opt_fit.mu = 1;
    opt_fit.lambda = 1;
    if strcmp(dataset, 'cat')
        opt_fit.eta = 1;
        opt_fit.alpha = 0.1; % 0.1 for body
        opt_path.eta = 1;
    elseif strcmp(dataset, 'scape71')
        opt_fit.eta = in_eta; % stiffness (the higher, the stiffer), 0.01
        opt_fit.alpha = 0.1; % hard user constraint
        opt_path.eta = 0.0001;
    end
    opt_path.dbg = false;
    opt_path.max_level = 2;
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
    
    %% mesh editing via shell energy and landmarks constraint
    FVs{1} = model;
    Topo = model.faces;
    [Ev, Eo, Ef] = getEdgesFromFaces(Topo);
    boundaryedges = Ef(:,2)==0;
    
    nedge = size(Ev, 1);
    nface = size(Ef, 1);
    vones = ones(nverts,1);
    
    optoptions = optimoptions('fminunc','Algorithm','trust-region','SpecifyObjectiveGradient',true,...
        'CheckGradients',false,'Display','final', 'HessianFcn','objective');
    fun = @(x) fastShellMarkerDistance(x,FVs,data_lm,lm_ind,Topo,Ev,Ef,Eo,boundaryedges,opt_fit);
    
    x0 = model.vertices(:);
    [xopt] = fminunc(fun,x0,optoptions);
    % update model vertices
    model.vertices = reshape(xopt, nverts, 3);
    if showFig
        figure;
        subplot(1,2,1);
        patch(model, 'FaceColor', fitc, 'EdgeColor', 'none', 'FaceLighting', 'phong',...
            'FaceAlpha', 0.4);
        axis equal; axis tight; axis off; cameratoolbar; light;
 
        subplot(1,2,2);
        patch(model, 'FaceColor', fitc, 'EdgeColor', 'none', 'FaceLighting', 'phong',...
            'FaceAlpha', 0.4);
        hold on;
        plot3(data_lm(:,1),data_lm(:,2),data_lm(:,3), 'ro', 'MarkerSize',10,'MarkerFaceColor',[1 0 0]);
        axis equal; axis tight; axis off; cameratoolbar; light;
        title('fit and markers');
    end
    [d,dvec] = geomDSD(model,data,Topo,Ev,Ef,opt_fit.eta,1,vones );
    
    %% init geodesic path, and reconstruct logmap and shoot back
    if opt_path.max_level > 0
        FVs{1} = model;
        [ ~,geo_path ] = GeodesicAveragePar( FVs,Topo,opt_path,FV_opt );
    end
  
    %% recon. logmap and model
    pc_data = load(pc_file);
    FV_pc_ref = pc_data.FV_pc_ref;
    ndim_max = length(FV_pc_ref) / 2;
 
    clear pc_data
    if ndim > 0
        FV_pc_use = cell(ndim*2,1);
        for i=1:ndim
            FV_pc_use{i} = FV_pc_ref{i};
            FV_pc_use{i+ndim} = FV_pc_ref{i+ndim_max};
        end
    else
        FV_pc_use = [];
    end
    clear FV_pc_ref
    clear FVlogs
    
    %%
    q0 = FV_opt;
    FVlogs_est = cell(1,1);
    opt_pc.ndim = ndim;
    opt_pc.eta = opt_path.eta;
    opt_pc.nlength = 2^opt_path.max_level + 1;
    opt_pc.dry_run = false;
    opt_pc.ninput = 1;
    %opt_pc.weights = signed_dist;
    opt_pc.weights = vones;
    opt_pc.useMem = true;
    opt_pc.useLagrange = true;
    opt_pc.alphas = [];
    opt_pc.init = 0;
    opt_pc.useRefLen = true;

    for iter = 1:max_iter
        fprintf('\n--- iter %d of %d ---,fit eta = %f \n',iter,max_iter, opt_fit.eta);
        % update logmap shape      
        if opt_path.max_level > 0
            FVlogs_est{1} = geo_path{1}(2);
        else
            FVlogs_est{1} = model;
        end
        
        %% recon model
        if ndim > 0
            [reconW, FV_recon1, FV_recon2, FV_recon3] = reconViaDual2m(FV_opt, FV_pc_use, FVlogs_est, Topo, opt_pc);
            opt_pc.alphas = reconW{1};
        else
            FV_recon3{1} = mean_model;
        end
   
        [~, FV_recon3{1}.vertices] = procrustes(model.vertices, FV_recon3{1}.vertices, 'Scaling', false, 'reflection', false);
        %% show current fit and model
        if showFig
            figure;        
            patch(model, 'FaceColor', fitc, 'EdgeColor', 'none', 'FaceLighting', 'phong',...
                'FaceAlpha', 1);
            hold on;
            plot3(data_lm(:,1),data_lm(:,2),data_lm(:,3), 'ko', 'MarkerSize',8,'MarkerFaceColor',[1 0 0]);
            hold on;
            plot3(model.vertices(lm_ind,1),model.vertices(lm_ind,2),...
                model.vertices(lm_ind,3), 'ko', 'MarkerSize',8,'MarkerFaceColor',[0 1 0]);
            axis equal; axis tight; axis off; cameratoolbar;light;
            title(['[iter: ',num2str(iter), '] fit and data']);
        end
        %%
        if iter <= max_iter && opt_path.max_level > 0
            % feed model to optimise with landmarks
            x0 = FV_recon3{1}.vertices(:);
            FVs{1} = FV_recon3{1};
            fun = @(x) fastShellMarkerDistance(x,FVs,data_lm,lm_ind,Topo,Ev,Ef,Eo,boundaryedges,opt_fit);
            [xopt] = fminunc(fun,x0,optoptions);
            model.vertices = reshape(xopt, nverts, 3);
            % feed model to recompute geo path
            % update input shape
            FVs{1} = model;
            geo_path{1}(2^opt_path.max_level + 1) = model;
            %geo_init = geo_path;
            opt_path.startLogmap = false;
            [ ~,geo_path ] = GeodesicAveragePar( FVs,Topo,opt_path,FV_opt,geo_path );
 
        end
  
        % save mesh
        save_mesh_fit = [savefolder, '_fit_t',num2str(test_id),'_alpha',num2str(opt_fit.alpha),...
            '_eta',num2str(opt_fit.eta),'_d',num2str(ndim,'%02d'),'_i',num2str(iter,'%02d'), '.obj'];
        objwrite(save_mesh_fit, model.faces, model.vertices);
        save_mesh_proj = [savefolder, '_proj_t',num2str(test_id),'_alpha',num2str(opt_fit.alpha),...
            '_eta',num2str(opt_fit.eta),'_d',num2str(ndim,'%02d'),'_i',num2str(iter,'%02d'), '.obj'];
        objwrite(save_mesh_proj, FV_recon3{1}.faces, FV_recon3{1}.vertices);
        % save other info: reconW{1};
        save_mat = [savefolder, '_t',num2str(test_id),'_alpha',num2str(opt_fit.alpha),...
            '_eta',num2str(opt_fit.eta),'_d',num2str(ndim,'%02d'),'_i',num2str(iter,'%02d'), '.mat'];
        save(save_mat, 'reconW');
  
    end

end
end