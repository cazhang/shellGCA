function [ FV_ea ] = MultiResElasticAv( FVs,Topo,Ev,Ef,Eo,boundaryedges,options )
% compute the elastic average of multiple shapes
% Inputs:
    % FVs -  A cell array, each element of which is a
    %        face/vertex mesh structure, e.g. FVs{1}.vertices and
    %        FVs{1}.faces.
    % options - Optional options structure. Fields include:
    %        FVinit  - FV structure containing initialisation
    %        InitNum - If FVinit not defined, then this specifies which
    %                        number shape to initialise with
    %        datweights - weights of each shape
    %        verbose - If true, print out diagnostic information
    % Ev - i and j
    % Eo - k and l
    % Ef - f1 and f2
% output
    % FV_ea: elastic average

%%
nmeshes = length(FVs);
if ~isfield(options, 'verbose')
    verbose = false;
else
    verbose = options.verbose;
end

if ~isfield(options, 'datweights')
    options.datweights(1:nmeshes) = 1;
    allOne = 1;
else
    allOne = 0;
    nweights = length(options.datweights);
    if nmeshes ~= nweights
        error(['wrong datweights size: ',num2str(nmeshes), ':',num2str(nweights)]);
    else
        if verbose
            fprintf(['data weights:\n']);
            disp(options.datweights);
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
if ~isfield(options, 'useMem')
    options.useMem = true;
end
if ~isfield(options, 'useLagrange')
    options.useLagrange = false;
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
    FVinit = options.FVinit;
else
    if isfield(options, 'InitNum')
        if verbose; disp(['Initialising with user specified shape ' num2str(options.InitNum)]); end
        
        FVinit = FVs{options.InitNum};              
    else
        if nmeshes>2
            if allOne
                if verbose; disp('Finding most central mesh'); end
                energy = zeros(nmeshes, nmeshes);
                for i=1:nmeshes
                    for j=1:nmeshes
                        if i~=j
                            energy(i,j)=geomDSD(FVs{i},FVs{j},Topo,Ev,Ef,options.eta,1);
                        end
                    end
                end
                tmp = sum(energy, 2);
                [val, idx] = min(tmp);
                FVinit=FVs{idx};
                if verbose; disp(['Initialising with shape ', num2str(idx)]); end
            else
                if verbose; disp('Using highest weight mesh'); end
                [val, idx] = max(options.datweights);
                FVinit=FVs{idx};
                if verbose; disp(['Initialising with shape ', num2str(idx)]); end
                
            end
        else
            if verbose; disp('Initialising with shape 1'); end          
            FVinit = FVs{1};         
        end
    end
end

x0 = FVinit.vertices(:);

nverts = size(FVinit.vertices,1);

if strcmp(options.Solver, 'fminunc')

    if verLessThan('matlab','8.6') 
        % for R2015a
        optoptions = optimoptions('fminunc','Algorithm','trust-region','GradObj','on',...
            'Display','final', 'Hessian','on', 'MaxIter', options.MaxIter);
    else
        % for R2016b
        optoptions = optimoptions('fminunc','Algorithm','trust-region','SpecifyObjectiveGradient',true,...
            'CheckGradients',false,'Display','none', 'HessianFcn','objective',...
            'OptimalityTolerance', 1e-20, 'StepTolerance',1e-10);
    end
    
    if options.useMem
        fun = @(x) geomShellGradDef(x,FVs,Topo,Ev,Ef,Eo,boundaryedges,options);
    else
        fun = @(x) shellGradDeformed(x,FVs,Topo,Ev,Ef,Eo,boundaryedges,options);
    end
    [xopt] = fminunc(fun,x0,optoptions);
         
else
    % Lagrange
    optoptions = optimoptions(@fsolve,'Display','none','SpecifyObjectiveGradient',true);
    %'FunctionTolerance', 1e-20, 'StepTolerance',1e-10,'OptimalityTolerance', 1e-20);
    if options.useMem
        fun = @(x) geomShellGradDefLag(x,FVs,Topo,Ev,Ef,Eo,boundaryedges,options);
    else
        fun = @(x) shellGradDeformedLag(x,FVs,Topo,Ev,Ef,Eo,boundaryedges,options);
    end
    x_in = [x0; ones(6,1)];
    [x_out] = fsolve(fun,x_in,optoptions);
    xopt = x_out(1:length(x0));
end

FV_ea.vertices = reshape(xopt,length(xopt)/3,3);
FV_ea.faces = Topo;

end