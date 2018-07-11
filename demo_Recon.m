function elapTime = demo_Recon(nlevel, cutoff, dataid, eta, recon_train)
% demo script to reconstruct shape with dual caculus approach
addpath(genpath('../clean_code/'));
tic;
if nargin < 1
    nlevel = 2;
    cutoff = 0.99;
    dataid = 2;
    eta = 0.0001;
    recon_train = 1;
end

expid = 'mem';
pct = cutoff*100;
dryrun = false; % no shooting
showFig = false;
if recon_train > 0
    reconTrain = true;
else
    reconTrain = false;
end

if dataid == 1
    dataset = 'faust';
    ntrain = 8;
    ntest = 2;
elseif dataid == 2
    dataset = 'body';
    ntrain = 9;
    ntest = 1;
end

% set id of shapes to be reconstructed
sid = 2;
eid = 2;

K = 2^nlevel;

savefolder = ['results_',expid,'_',dataset, '/'];
if ~exist(savefolder, 'dir')
    mkdir(savefolder);
end
 
pc_name = [savefolder,dataset,'PC_N',num2str(ntrain),'L',num2str(nlevel),'P',num2str(pct),'E',num2str(eta,'%.e'),'.mat'];
geoTrain_name = [savefolder,dataset,'Geo_N',num2str(ntrain),'L',num2str(nlevel),'E',num2str(eta,'%.e'),'.mat'];
geoTest_name = [savefolder,dataset,'Geo_N',num2str(ntrain),'L',num2str(nlevel),'E',num2str(eta,'%.e'),'_test.mat'];


if reconTrain
    recon_file = [savefolder,dataset,'reconTrain_N',num2str(ntrain),'L',num2str(nlevel),'E',num2str(eta,'%.e'),'.mat'];
else
    recon_file = [savefolder,dataset,'reconTest_N',num2str(ntrain),'L',num2str(nlevel),'E',num2str(eta,'%.e'),'.mat'];
end
    
if exist(pc_name, 'file')
    pc_data = load(pc_name);
else
    error('pc file not found.');
end
maxPC = length(pc_data.FV_pc_ref)/2;
ndim_max = maxPC;
ndim_recon_max = 2;
Topology = pc_data.FV_opt.faces;
[Ev, Eo, Ef] = getEdgesFromFaces(Topology);
boundaryedges = Ef(:,2)==0;

if reconTrain
    geo_train = load(geoTrain_name);
    for i=1:ntrain
        FVs{i} = geo_train.FV_path{i}(end);
    end
    ps = pc_data.FVlogs; % load logmap real data
    ninput = ntrain;
else
    % load test geo file
    geo_test = load(geoTest_name);
    for i=1:ntest
        FVs{i} = geo_test.FV_path_test{i}(end);
        ps{i} = geo_test.FV_path_test{i}(2);
    end   
    ninput = ntest;
end

verbose = true;

qs{1} = pc_data.FV_opt;
for i=1:ndim_max
    if verbose; fprintf('set PC %d\n', i); end;
    qs{end+1} = pc_data.FV_pc_ref{i};
    
end

for i=1:ndim_max
    if verbose; fprintf('set PC %d\n', maxPC+i); end;
    qs{end+1} = pc_data.FV_pc_ref{i+maxPC};
    
end

%% visual pcs
FV_mean = qs{1};
clear pc_data;
clear pc_len;
clear geo_test;
clear isRescale;
for k=1:ndim_recon_max
    pc_len2(k) = geomDSD(FV_mean, qs{k+1},Topology,Ev,Ef,eta,1);
    pc_len(k) = sqrt(pc_len2(k));
end
refPCLength = min(pc_len)/2;
refLength = refPCLength;

global q0;

%% define
elapTime = zeros(ninput,ndim_max);
lambda_opt = cell(ninput,ndim_max);

optdual = [];
optdual.useRedu = true;
optdual.useGlobal = true;
optdual.useOld = false;

ratio = zeros(ninput,1);
ratio_out = zeros(ninput, 1);
options = cell(ninput,1);
option0 = cell(ninput,1);
lambda_0 = cell(ninput,1);
pnew = cell(ninput,1);
pcs = cell(ndim_recon_max,1);
fh = cell(ninput, 1);
datweights = cell(ninput, 1);
FV_recon = cell(ninput, ndim_recon_max);% shorten recon
FV_recon2 = cell(ninput, ndim_recon_max);% logmap recon
FV_recon3 = cell(ninput, ndim_recon_max);% input recon

ori_len = zeros(ninput, 1);
short_len = zeros(ninput, 1);
long_len = zeros(ninput, ndim_recon_max);

dis_short = cell(ninput,1);
dis_log = cell(ninput,1);
dis_input=cell(ninput,1);
dis_recon = cell(ninput,ndim_recon_max);
dis_recon2=cell(ninput,ndim_recon_max);
dis_recon3=cell(ninput,ndim_recon_max);

err1 = zeros(ninput, ndim_recon_max);
err2 = zeros(ninput, ndim_recon_max);
err3 = zeros(ninput, ndim_recon_max);
% use fmincon by Aeq = ones(1,n), and beq = 1
optoptions_con = optimoptions('fmincon','SpecifyObjectiveGradient',true,...
    'Display','iter');
% ,'MaxIterations',100,'MaxFunctionEvaluations',100 ,'Algorithm', 'trust-region-reflective'
optoptions_unc = optimoptions('fminunc','SpecifyObjectiveGradient',true,...
    'Display','iter','Algorithm','quasi-newton');

% shortening length of input if necessary
for i=sid:eid
    if reconTrain
        fprintf('rescaling train %d\n', i);
    else
        fprintf('rescaling test %d\n', i);
    end
    isRescale{i} = false;
    % as DSD have been modified, so length no longer valid
    dis2_p{i} = geomDSD(FV_mean, ps{i},Topology,Ev,Ef,eta,1);
    ori_len(i) = sqrt(dis2_p{i});
    % shorten
    if ori_len(i) > refLength % if longer than refLength, shortening to be less than refLength
        %disp('shortening...');
        ratio(i) = refLength/ori_len(i);
        options{i}.useLagrange = true;
        options{i}.eta = eta;
        options{i}.useMem = true;
        [pnew{i},ratio_out(i)] = rescaleShell(FV_mean, ps{i}, Topology, Ev, Ef, Eo, boundaryedges, ratio(i), options{i});
        isRescale{i} = true;
        % new len
        tmp = geomDSD(FV_mean, pnew{i},Topology,Ev,Ef,eta,1);
        short_len(i) = sqrt(tmp);
    else
        pnew{i} = ps{i};
        short_len(i) = ori_len(i);
    end
    if showFig
        figure;
        subplot(121)
        patch(ps{i}, 'FaceColor', [1 1 1], 'EdgeColor', 'none', 'FaceLighting', 'phong');
        axis equal; axis tight; axis off; cameratoolbar; light; view(90,0);
        subplot(122)
        patch(pnew{i}, 'FaceColor', [1 1 0], 'EdgeColor', 'none', 'FaceLighting', 'phong');
        axis equal; axis tight; axis off; cameratoolbar; light; view(90,0);
        title('original to short');
    end
end
% prepare pcs
for dim=1:ndim_recon_max
    %pcs{dim} = cell(2*dim+1, 1);
    pcs{dim}{1} = qs{1};
    for k = 1:dim
        %if verbose; fprintf('setting pc %d with %d\n', k, k);end;
        pcs{dim}{end+1} = qs{k+1};
    end
    for k = 1:dim
        %if verbose; fprintf('setting pc %d with %d\n', dim+k+1, ndim_max+k+1);end;
        pcs{dim}{end+1} = qs{ndim_max+k+1};
    end
end


%%
for i=sid:eid
    for dim = ndim_recon_max % dims used to recon
        if reconTrain
            fprintf('Recon train %d with dim=%d \n', i, dim);
        else
            fprintf('Recon test %d with dim=%d \n', i, dim);
        end
        % init lambda with [1 0 0 ...]
        lambda_0{i} = zeros(1, 2*dim+1);
        if dim > 1 && optdual.useOld
            lambda_0{i}(1:dim) = lambda_opt{i};
        else
            lambda_0{i}(1) = 1;
            q0=FV_mean;
        end
        
        fh{i} = @(x) objDualCaculus2mPlus1(x,pnew{i},pcs{dim,1},Topology,Ev,Ef,Eo,boundaryedges,eta,optdual,i);
        
        % compute optimal lambdas
        if optdual.useRedu
            tic;
            tmp = fminunc(fh{i},lambda_0{i}(2:end),optoptions_unc);
            lambda_opt{i,dim} = [0 tmp];
            lambda_opt{i,dim}(1) = 1 - sum(tmp);
        else
            tic;
            Aeq = ones(1, dim+1); beq = 1;
            lambda_opt{i,dim} = fmincon(fh{i},lambda_0{i},[],[],Aeq,beq,[],[],[],optoptions_con);
        end
        elapTime(i,dim) = toc;
        
        option0{i}.datweights = lambda_opt{i,dim};
        option0{i}.useLagrange = true;
        option0{i}.useMem = true;
        option0{i}.eta = eta;
        FV_recon{i,dim} = MultiResElasticAv( pcs{dim,1},Topology,Ev,Ef,Eo,boundaryedges, option0{i} );
        %% prolongate
        if isRescale{i}
            
            options{i}.verbose = false;
            options{i}.useMem = false;
            [FV_recon2{i,dim}] = rescaleShell(FV_mean, FV_recon{i,dim}, Topology, Ev, Ef, Eo, boundaryedges, 1/ratio(i), options{i});
            % get len
            tmp = geomDSD(FV_mean, FV_recon2{i,dim},Topology,Ev,Ef,eta,1);
            long_len(i,dim) = sqrt(tmp);
            
            [~, FV_recon{i,dim}.vertices] = procrustes(pnew{i}.vertices, FV_recon{i,dim}.vertices, 'Scaling', false);
            
            [~, FV_recon2{i,dim}.vertices] = procrustes(ps{i}.vertices, FV_recon2{i,dim}.vertices, 'Scaling', false);
            
        else
            FV_recon2{i,dim} = FV_recon{i,dim};
            long_len(i) = short_len(i);
        end
        % shoot 3 steps for the case K=4
        opt2exp = [];
        opt2exp.step = K-1;
        opt2exp.verbose = false;
        opt2exp.useLagrange = true;
        opt2exp.eta = eta;
        opt2exp.useMem = false;
        if opt2exp.step > 0 && ~dryrun
            FV_recon3{i,dim} = TestShellExp2( FV_mean, FV_recon2{i,dim},Topology,Ev,Ef,Eo,boundaryedges,opt2exp );
        else
            FV_recon3{i,dim} = FV_recon2{i,dim};
        end
        % error
        dis_short{i} = geomDSD(FV_mean, pnew{i},Topology,Ev,Ef,eta,1);
        dis_log{i} = geomDSD(FV_mean, ps{i},Topology,Ev,Ef,eta,1);
        dis_input{i} = geomDSD(FV_mean, FVs{i},Topology,Ev,Ef,eta,1);
        dis_recon{i,dim} = geomDSD(pnew{i}, FV_recon{i,dim},Topology,Ev,Ef,eta,1);
        dis_recon2{i,dim} = geomDSD(ps{i}, FV_recon2{i,dim},Topology,Ev,Ef,eta,1);
        dis_recon3{i,dim} = geomDSD(FVs{i}, FV_recon3{i,dim},Topology,Ev,Ef,eta,1);
        err1(i,dim) = dis_recon{i,dim} / dis_short{i};
        err2(i,dim) = dis_recon2{i,dim} / dis_log{i};
        err3(i,dim) = dis_recon3{i,dim} / dis_input{i};
        
        if true
            figure;
            subplot(1,3,1);
            patch(FVs{i}, 'FaceColor', [1 1 1], 'EdgeColor', 'none', 'FaceLighting', 'phong');
            axis equal; axis tight; axis off; cameratoolbar; light; view(90,0);
            subplot(1,3,2);
            patch(FV_recon3{i,dim}, 'FaceColor', [0 1 1], 'EdgeColor', 'none', 'FaceLighting', 'phong');
            axis equal; axis tight; axis off; cameratoolbar; light; view(90,0);
            subplot(1,3,3);
            patch(FVs{i}, 'FaceColor', [1 1 1], 'EdgeColor', 'none', 'FaceLighting', 'phong','FaceAlpha', 0.5);
            hold on
            patch(FV_recon3{i,dim}, 'FaceColor', [0 1 1], 'EdgeColor', 'none', 'FaceLighting', 'phong','FaceAlpha', 0.5);
            axis equal; axis tight; axis off; cameratoolbar; light; view(90,0);
        end
        
        fprintf('ShortErr of %d at dim=%d: %f\n', i, dim, err1(i,dim));
        fprintf('LogErr of %d at dim=%d: %f\n', i, dim, err2(i,dim));
        fprintf('InputErr of %d at dim=%d: %f\n',i, dim, err3(i,dim));
        fprintf('----------------------------------------------------\n');
        
    end
end

%% save results
elapTime = toc;
save(recon_file, 'lambda_opt','FV_recon','FV_recon2','FV_recon3','err1','err2','err3','elapTime');

end