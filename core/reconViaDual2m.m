function [reconW, FV_recon, FV_recon2, FV_recon3] = reconViaDual2m(FV_opt, FV_pc_ref, FVlogs,Topology, opt)
%NOTE: actually using 2m+1 rather than 2m
eta = opt.eta;
ndim = opt.ndim;
dry_run = opt.dry_run;
nlength = opt.nlength;

[Ev, Eo, Ef] = getEdgesFromFaces(Topology);
boundaryedges = Ef(:,2)==0;

ndim_max = length(FV_pc_ref)/2;
% FV_pc has 2*ndim + 1
FV_pc{1} = FV_opt;
for i=1:ndim
    FV_pc{i+1} = FV_pc_ref{i};
end
for i=1:ndim
    FV_pc{ndim+1+i} = FV_pc_ref{ndim_max+i};
end
% get pc length
for i=1:ndim*2
    pc_len2(i) = geomDSD(FV_opt,FV_pc{i+1},Topology,Ev,Ef,eta,1,opt.weights);
end
pc_len = sqrt(pc_len2);
refLength = min(pc_len)/2;
pnew = cell(opt.ninput, 1);
reconW = cell(opt.ninput, 1);
FV_recon = cell(opt.ninput, 1);
FV_recon2 = cell(opt.ninput, 1);
FV_recon3 = cell(opt.ninput, 1);

options = cell(opt.ninput, 1);
exp2opt = cell(opt.ninput, 1);
ratio = zeros(opt.ninput, 1);
ratio_out = zeros(opt.ninput, 1);
optoptions = optimoptions('fminunc','SpecifyObjectiveGradient',true,...
        'CheckGradients',false,'Display','iter','Algorithm','quasi-newton');

optdual.useRedu = true;
optdual.useGlobal = true;
optdual.useOld = false;
optdual.weights = opt.weights;
for test_id = 1:opt.ninput
    % shortening length of input
    dis2_p{test_id} = geomDSD(FVlogs{test_id}, FV_opt,Topology,Ev,Ef,eta,1,opt.weights);
    len_p{test_id} = sqrt(dis2_p{test_id});
    isRescale{test_id} = false;
    if (len_p{test_id} > refLength) && opt.useRefLen
        ratio(test_id) = refLength/len_p{test_id};
        options{test_id}.useLagrange = true;
        options{test_id}.eta = eta;
        options{test_id}.useMem = true;
        % compute weighted avearge of two
        [pnew{test_id},ratio_out(test_id)] = rescaleShell(FV_opt, FVlogs{test_id}, Topology, Ev, Ef, Eo, boundaryedges, ratio(test_id), options{test_id});
        %[ pnew{test_id} ] = MultiResElasticAvOfTwo( FV_opt,FVlogs{test_id},Topology,Ev,Ef,Eo,boundaryedges,options{test_id} );        
        isRescale{test_id} = true;
    else
        pnew{test_id} = FVlogs{test_id};
    end
end
    
for test_id = 1:opt.ninput
    fprintf('Reconstructing shape: %d\n', test_id);
    if ~isempty(opt.alphas)
        fprintf('initialise lambda_0 with previous frame...\n');
        lambda_0{test_id} = opt.alphas;
    else
        if opt.init == 0
            fprintf('initialise as [1 0 0 ...]');
            lambda_0{test_id} = zeros(1, ndim*2+1);
            lambda_0{test_id}(1) = 1;
        else
            fprintf('initialise as [0 1/2m 1/2m ...]');
            lambda_0{test_id} = ones(1, ndim*2+1)./(ndim*2);
            lambda_0{test_id}(1) = 0;
        end 
    end
    
    % recon optimal lambdas        
    fun{test_id} = @(x) objDualCaculus2mPlus1(x,pnew{test_id},FV_pc,Topology,Ev,Ef,Eo,boundaryedges,...
        eta,optdual,test_id);
    
    reconW{test_id} = fminunc(fun{test_id},lambda_0{test_id}(2:end),optoptions);
    reconW{test_id} = [0 reconW{test_id}];
    reconW{test_id}(1) = 1 - sum(reconW{test_id});
    % recon logmap
    options{test_id}.datweights = reconW{test_id};
    FV_recon{test_id} = MultiResElasticAv( FV_pc,Topology,Ev,Ef,Eo,boundaryedges, options{test_id} );
    % prolongate
    if isRescale{test_id}
        [FV_recon2{test_id}] = rescaleShell(FV_opt, FV_recon{test_id}, Topology, Ev, Ef, Eo, boundaryedges,...
            1/ratio(test_id), options{test_id});
    else
        FV_recon2{test_id} = FV_recon{test_id};
    end

    if ~dry_run
        % shooting 3 steps
        exp2opt{test_id} = [];
        exp2opt{test_id}.eta = eta;
        exp2opt{test_id}.step = nlength-2;
        exp2opt{test_id}.useLagrange = opt.useLagrange;
        exp2opt{test_id}.id = test_id;
        exp2opt{test_id}.verbose = false;
        exp2opt{test_id}.useMem = opt.useMem;
        
        if exp2opt{test_id}.step > 0
            FV_recon3{test_id} = TestShellExp2( FV_opt, FV_recon2{test_id},Topology,Ev,Ef,Eo,boundaryedges,exp2opt{test_id} );
        else
            FV_recon3{test_id} = FV_recon2{test_id};
        end
    else
        FV_recon3{test_id} = FV_recon2{test_id};
    end    
end

end