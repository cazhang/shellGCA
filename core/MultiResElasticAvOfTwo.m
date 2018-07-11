function [ FV_ea ] = MultiResElasticAvOfTwo( FV1,FV2,Topo,Ev,Ef,Eo,boundaryedges,options )
%MULTIRESELASTICAV Multiresolution elastic average of two or more meshes
%   Compute elastic average by optimisation but where the set of vertices
%   belongs to multiple meshes at different resolutions. Shell energy is
%   optimised at all resolutions simultaneously.
%
% Inputs:
%
% FV1 - shell 1
% FV2 - shell 2
% options - Optional options structure. Fields include:
%              FVinit  - FV structure containing initialisation
%              InitNum - If FVinit not defined, then this specifies which
%                        number shape to initialise with
%              verbose - If true, print out diagnostic information
%              datweights - weights to apply on length only!
% Note: if you want to optimise S so that W[S,S1] / W[S,S2] = w1 / w2, you
% need to pass sqrt(w) rather then w itself, coz the weights works on
% length. 

nmeshes = 2;
if isfield(options, 'verbose')
    verbose = options.verbose;
else
    verbose = false;
end
if ~isfield(options, 'useMem')
    options.useMem = false;
end
if ~isfield(options, 'datweights')
    options.datweights(1:nmeshes) = 1;
else
    if nmeshes ~= length(options.datweights)
        error('wrong datweights size');
    else
        if verbose
            fprintf(['Data weights:\n']);
            for i=1:nmeshes
                fprintf(['weight for input ',num2str(i),' is ',num2str(options.datweights(i)),'\n']);
            end
        end
    end
end
if ~isfield(options, 'mu')
    options.mu = 1;
end
if ~isfield(options, 'lambda')
    options.lambda = 1;
end
if ~isfield(options, 'eta')
    options.eta = 0.0001;
end
if ~isfield(options, 'MaxIter')
    options.MaxIter = 500;
end

% Choose solver type
if isfield(options, 'Solver')
    if verbose; disp(['Solver: ' options.Solver]); end
else
    if options.useLagrange
        options.Solver = 'fsolve';
    else       
        options.Solver = 'fminunc';
    end
    if verbose; disp(['Solver: ' options.Solver]); end
    
end
    
% Choose between user provided initialisation or automatic
if isfield(options, 'FVinit')
    if verbose; disp('Using user-provided initialisation'); end
    FV = options.FVinit;
else
    if isfield(options, 'InitNum')
        if verbose; disp(['Initialising with user specified shape ' num2str(options.InitNum)]); end
        if options.InitNum==1
            FV=FV1;
        else
            FV=FV2;
        end
    else
        [~,InitNum] = max(options.datweights);
        if verbose; disp(['Initialising with shape ' num2str(InitNum), ' by dataweights']); end
        if InitNum==1
            FV=FV1;
        else
            FV=FV2;
        end
    end
end

x0 = FV.vertices(:);

FVs{1}=FV1;
FVs{2}=FV2;


if strcmp(options.Solver, 'fminunc')
    if verbose
        dispOpt = 'iter';
    else
        dispOpt = 'none';
    end
    if verLessThan('matlab','8.6') 
        % for R2015a
        optoptions = optimoptions('fminunc','Algorithm','trust-region','GradObj','on',...
            'Display','final', 'Hessian','on', 'MaxIter', options.MaxIter);
    else
        % for R2016b
        optoptions = optimoptions('fminunc','Algorithm','trust-region','SpecifyObjectiveGradient',true,...
            'CheckGradients',false,'Display',dispOpt,'HessianFcn','objective', 'MaxIter', options.MaxIter);
    end
    if isfield(options, 'useMulRes') && options.useMulRes
        % use multi-resolution
        fun = @(x) multiShellGradDeformed(x,FVs,Topo,Ev,Ef,Eo,boundaryedges,options);
        [xopt] = fminunc(fun,x0,optoptions);
    else        
        % test undeformed
        %fun = @(x) fastShellGradAndHessUnd(x,FVs,Topo,Ev,Ef,Eo,boundaryedges,options);
        if options.useMem
            fprintf('using classical membrane.\n');
            fun = @(x) geomShellGradDef(x,FVs,Topo,Ev,Ef,Eo,boundaryedges,options);
        else
            fprintf('using edge and face.\n');
            fun = @(x) shellGradDeformed(x,FVs,Topo,Ev,Ef,Eo,boundaryedges,options);
        end
        [xopt] = fminunc(fun,x0,optoptions);        
    end  
else
    % Lagrange
    if verbose
        dispOpt = 'iter';
    else
        dispOpt = 'none';
    end
    optoptions = optimoptions(@fsolve,'Display',dispOpt,...
        'SpecifyObjectiveGradient',true,'CheckGradients',false, 'MaxIterations', options.MaxIter);
    if options.useMem
        fprintf('using classical membrane with Lagrange. \n');
        fun = @(x) geomShellGradDefLag(x,FVs,Topo,Ev,Ef,Eo,boundaryedges,options);
    else
        fprintf('using edge and face with Lagrange. \n');
        fun = @(x) shellGradDeformedLag(x,FVs,Topo,Ev,Ef,Eo,boundaryedges,options);
    end
    
    x_in = [x0; ones(6,1)];
    %nverts = length(x0)/3;
    %x_in = x0;
    [x_out] = fsolve(fun,x_in,optoptions);
    xopt = x_out(1:length(x0));
end

FV_ea.vertices = reshape(xopt,length(xopt)/3,3);
if isfield(options, 'useMulRes') && options.useMulRes
    FV_ea.faces = Topo{1};
else
    FV_ea.faces = Topo;
end


%[~, Z] = procrustes(FV.vertices, FV_ea.vertices, 'Scaling', false, 'Reflection',false);
%FV_ea.vertices = Z;
    
end

function [resid] = shellResiduals(x,FVs,Evs,Efs,allboundaryedges,mu,lambda,eta,resweights)

V = reshape(x,length(x)/3,3);
resid = [];

for res = 1:2
    F = FVs{res,1}.faces;
    Ev = Evs{res};
    Ef = Efs{res};
    boundaryedges = allboundaryedges{res};
    % Get the triangle vertices
    v1 = F(:, 1);
    v2 = F(:, 2);
    v3 = F(:, 3);
    
    % Compute the edge vectors
    e1s = V(v2, :) - V(v1, :);
    %e2s = V2(v3, :) - V2(v1, :);
    e3s = V(v2, :) - V(v3, :);
    
    % Compute triangle normals
    Tn    = cross(e1s, e3s, 2);
    Tnlength = sqrt(sum(Tn.^2,2));
    % Compute triangle areas
    at = Tnlength./2;
    % Compute
    Tn = Tn./repmat(Tnlength,1,3);
    
    % Compute dihedral angles for non-boundary edges
    Te = acos(max(-1,min(1,sum(Tn(Ef(~boundaryedges,1),:).*Tn(Ef(~boundaryedges,2),:),2))));
    
    le = sqrt(sum((V(Ev(:,1),:)-V(Ev(:,2),:)).^2,2));
    
    % Compute length scaling terms
    de(~boundaryedges) = (1/3).*(at(Ef(~boundaryedges,1))+at(Ef(~boundaryedges,2)));
    % Not sure next one is valid...
    de(boundaryedges) = (1/3).*(at(Ef(boundaryedges,2))+at(Ef(boundaryedges,2)));
    de = de';
    
    for i=1:size(FVs,2)
        % Edge lengths
        resid = [resid; resweights(res) .* sqrt(mu.*FVs{res,i}.de) .* ((FVs{res,i}.le-le)./FVs{res,i}.le)];
        % Triangle areas
        resid = [resid; resweights(res) .* sqrt(lambda.*FVs{res,i}.at) .* ((FVs{res,i}.at-at)./FVs{res,i}.at)];
        % Dihedral angles
        resid = [resid; resweights(res) .* sqrt(eta).*FVs{res,i}.le(~boundaryedges).*sqrt(1./FVs{res,i}.de(~boundaryedges)) .* (FVs{res,i}.Te-Te)];
    end
end

if ~isreal(resid)
    disp('Complex!');
end

end