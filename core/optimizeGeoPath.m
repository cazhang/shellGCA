function Path = optimizeGeoPath(oldPath, Topo, Ev,Ef,Eo,boundaryedges,options)

num_shells = options.num_shells;
free_shells = num_shells-2;
FV_input = oldPath(num_shells);
FV_bar = oldPath(1);

options.firstDerivWRTDef = true;
x0 = [];

for k = 1:free_shells
    x0 = [x0; oldPath(k+1).vertices(:)];
end

[old_cost, ~, ~] = geoPathGradHess( x0, FV_bar, FV_input, free_shells,Topo, Ev, Ef, Eo,...
    boundaryedges,options);
tic;
optoptions = optimoptions('fminunc','Algorithm','trust-region','SpecifyObjectiveGradient',true,...
    'CheckGradients',false,'Display','iter', 'HessianFcn','objective');
fun = @(x) geoPathGradHess(x,FV_bar,FV_input, free_shells,Topo, Ev, Ef, Eo, boundaryedges,options);

dbg = false;
if dbg
    xopt = x0;
else
    [xopt] = fminunc(fun,x0,optoptions);
end

toc;
Path = makeGeoPathFromVertex(oldPath, xopt);
[new_cost, ~, ~] = geoPathGradHess( xopt, FV_bar, FV_input, free_shells,Topo, Ev, Ef, Eo, boundaryedges,options);

fprintf('Before cost = %5.4f, After cost = %5.4f\n', old_cost, new_cost);
end