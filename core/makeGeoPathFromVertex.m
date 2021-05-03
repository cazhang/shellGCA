function [FV_path] = makeGeoPathFromVertex(geo_path, xVec, aux)
if nargin < 3
    aux = 1;
end
% note: FV_path {1} is average, and FV_path {end} is input. The order
% of xVec should be consistent now. Top is closer to mean,
% bottom closer to input. 

FV_path = geo_path;
nverts = size(geo_path(1).vertices, 1);
nsize = nverts * 3;
sizex = size(xVec);
if sizex(2) ~= 1
    error('x must be column vector.\n');
end
free_shells = sizex(1) / nsize;
num_shells = length(geo_path);
if num_shells == free_shells+2
    fprintf('end shape not treated as free.\n');
elseif num_shells == free_shells+1
    fprintf('end shape treated as free.\n');
end
% fill up input and mean
for i=1:free_shells
    x = xVec(nsize*(i-1)+1:nsize*i);
    FV_path(i*aux+1).vertices = reshape(x, nverts, 3);
    %FV_path{1}(i+1).faces = FV_mean.faces;
end
end