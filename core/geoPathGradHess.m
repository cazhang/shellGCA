function [cost,grad,H]=geoPathGradHess( x, FV_ga, FV_input, free_shells,Topo, Ev, Ef, Eo, boundaryedges,options)
% cost: path energy (we omit a factor of K)
% grad: path gradient, (nverts*3* (K-1), 1)
% H: hessian matrix, a (K-1 * K-1) block matrix, where each is a nverts*3
% by nverts*3 sub matrix
% implementation idea: 
% cost - just sum of multiple segments, using DSD function;
% gradient - concatenation of multiple grad_k, and grad_k = D_def W[S_k-1,
% S_k] + D_und W[S_k, S_k+1]
% H - K-1 by K-1 block matrix, each is a nverts*3 by nverts*3 submatrix
% given by mixted hessian
nverts = size(FV_ga.vertices, 1);
nsize = nverts*3;
nshells = free_shells + 2;
if length(x) ~= nverts*3*free_shells
    error('dimension of x not matching.');
end
if ~isfield(options, 'mu')
    options.mu = 1;
end
if ~isfield(options, 'lambda')
    options.lambda = 1;
end
% convert x to FV_path, along with FV_ga, FV_input
% passed in x should be [x1;y1;z1;x2;y2;z2...xk;yk;zk]
FV_path = [];
FV_path{1} = FV_ga;
FV_path{nshells} = FV_input;
for i=1:free_shells
    temp = x(nsize*(i-1)+1:nsize*i);
    FV_path{i+1}.vertices = reshape(temp, nverts, 3);
    FV_path{i+1}.faces = Topo;
end
% Cost
cost = 0;
for i=1:nshells-1
    if options.useMem
        temp_cost(i) = geomDSD(FV_path{i}, FV_path{i+1}, Topo, Ev, Ef, options.eta, 1);
    else
        temp_cost(i) = DSD(FV_path{i}, FV_path{i+1}, Topo, Ev, Ef, options.eta, 1);
    end
    cost=cost+temp_cost(i);
end
%disp(['segments energies: ',num2str(temp_cost)]);
%fprintf('\n');

if nargout > 1
    grad = [];
    H = [];
    %grad = zeros(nsize*free_shells, 1);
    %H = zeros(nsize*free_shells, nsize*free_shells);
    % Grad & Hess
    for i=1:free_shells
        k = i+1;% index in FV_path
        from_ind = nsize*(i-1)+1;
        to_ind = nsize*i;
        % D_def W[S_k-1, S_k]
        x_free = x(from_ind:to_ind);
        FVs{1} = FV_path{k-1};
        if options.useMem
            [~,gdef,Hdef] = geomShellGradDef( x_free, FVs, Topo, Ev, Ef, Eo, boundaryedges,options);
        else
            [~,gdef,Hdef] = shellGradDeformed( x_free, FVs, Topo, Ev, Ef, Eo, boundaryedges,options);
        end
        % D_und W[S_k, S_k+1]
        FVs{1} = FV_path{k+1};
        if options.useMem
            [~,gund,Hund] = geomShellGradUnd( x_free, FVs, Topo, Ev, Ef, Eo, boundaryedges,options);
        else
            [~,gund,Hund] = fastShellGradAndHessUnd( x_free, FVs, Topo, Ev, Ef, Eo, boundaryedges,options);
        end
        grad(from_ind:to_ind, 1) = gdef(:) + gund(:);
        H(from_ind:to_ind, from_ind:to_ind) = Hdef + Hund;
        % mixed hessian D_1 D_2 W[S_k, S_k+1]
        FV_ref = FV_path{k};
        FV_def = FV_path{k+1};
        if i < free_shells
            if options.useMem
                Hmix = geomMixedHessianShell(FV_ref, FV_def, Topo, Ev, Ef, Eo, boundaryedges, options);
            else
                Hmix = fastMixedHessianShell(FV_ref, FV_def, Topo, Ev, Ef, Eo, boundaryedges, options);
            end
            % fill at the right of diag element: row the same, col + nsize
            H(from_ind:to_ind, from_ind+nsize:to_ind+nsize) = Hmix;
            % fill at the down of diag element: col the same, row + nsize
            H(from_ind+nsize:to_ind+nsize, from_ind:to_ind) = Hmix';
        end
    end
end
