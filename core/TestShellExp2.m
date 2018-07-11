function [ FV_exp, FV_pre ] = TestShellExp2( FV1,FV2,Topo,Ev,Ef,Eo,boundaryedges,options )
% do geodesic shooting from FV1 towards FV2
% input:
    % FV1: reference shape
    % FV2: variational shape
% output: 
    % FV_exp: (final) shooted shape
    % FV_pre: previous shooted shape, if options.step > 1
    
% test deformed Hessian matrix
if ~isfield(options, 'saveFolder')
    options.saveFolder = '';
end
if ~isfield(options, 'saveShoot')
    options.saveShoot = false;
end
if isfield(options, 'verbose')
    verbose = options.verbose;
else
    verbose = false;
end
if ~isfield(options, 'useMem')
    options.useMem = true;
end
if ~isfield(options, 'useLagrange')
    options.useLagrange = true;
end
% Choose solver type
if isfield(options, 'Solver')
    if verbose; disp(['Solver: ' options.Solver]); end
else
    options.Solver = 'fsolve';
    if verbose; disp(['Solver: ' options.Solver]); end
end
    
if ~isfield(options, 'step')
    options.step = 1;
end
if ~isfield(options, 'MaxIterations')
    options.MaxIterations = 400;
end
% Choose between user provided initialisation or automatic
if isfield(options, 'FVinit')
    if verbose; disp('Using user-provided initialisation'); end
    FV = options.FVinit;
else
    FV = FV1;
end
x0 = FV.vertices(:);

if ~isfield(FV1, 'de')
    FV1 = precompute(FV1,Topo,Ev,Ef,boundaryedges);
end
if ~isfield(FV2, 'de')
    FV2 = precompute(FV2,Topo,Ev,Ef,boundaryedges);
end

if verLessThan('matlab','8.6')
    % for R2015a
    optoptions = optimoptions(@fsolve,'Display','iter','Jacobian','on');
else
    % for R2016b
    optoptions = optimoptions(@fsolve,'Display','none',...
    'SpecifyObjectiveGradient',true,'CheckGradients',false, 'MaxIterations', options.MaxIterations);
end
options.firstDerivWRTDef = true;

if ~isfield(options, 'expweights')
    fprintf('using default weights.');
    options.expweights = [1,1];
end

if verbose 
    fprintf('weights are %5.2f and %5.2f\n', options.expweights(1),options.expweights(2));  
end
for step = 1:options.step
    if verbose; fprintf('shooting step %d of total %d. ', step, options.step); end;  
    if options.useLagrange        
        if options.useMem
            if verbose;fprintf('using classical membrane with Lagrange.\n'); end;
            fun = @(x) geomShellExp2Lag(x,FV1,FV2,Topo,Ev,Ef,Eo,boundaryedges,options);
        else
            if verbose;fprintf('using edge and face with Lagrange.\n'); end;
            fun = @(x) shellExp2Lag(x,FV1,FV2,Topo,Ev,Ef,Eo,boundaryedges,options);
        end
        x_in = [x0; ones(6,1)];            
        [x_out] = fsolve(fun,x_in,optoptions);
        x_est = x_out(1:length(x0));
    else
        if options.useMem
            if verbose;fprintf('Using classical membrane with REG.\n'); end;
            fun = @(x) geomShellExp2(x,FV1,FV2,Topo,Ev,Ef,Eo,boundaryedges,options);
        else
            if verbose;fprintf('Using edge and face with REG.\n'); end;
            fun = @(x) shellExp2(x,FV1,FV2,Topo,Ev,Ef,Eo,boundaryedges,options);
        end

        [x_est] = fsolve(fun,x0,optoptions);
    end
     
    if verbose; fprintf('Done.\n'); end;
    
    % update
    FV1.vertices = FV2.vertices;
    FV2.vertices = reshape(x_est, length(x_est)/3, 3);
    x0 = FV2.vertices(:);
       
    FV_exp.vertices = FV2.vertices;
    FV_exp.faces = FV2.faces;
    
    if nargout > 1 && step == options.step
        FV_pre = FV1;
    end
end

end