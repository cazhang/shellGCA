function flag = visualizeGeoPath(geoPath, meshid, angle)

useAngle = true;
if nargin < 3
    useAngle = false;
end

if meshid ~= 0
    fprintf('Display meshid = %d only \n', meshid);
    if iscell(geoPath{meshid})
        ncell = length(geoPath{1});
        nshell = length(geoPath{1}{1});
        figure;
        for i=1:ncell
            for j=1:nshell
                subplot(ncell,nshell,(i-1)*nshell+j);
                patch(geoPath{1}{i}(j), 'FaceColor', [1 1 1], 'EdgeColor', 'none', 'FaceLighting', 'phong');
                axis equal; axis tight; axis off; cameratoolbar; light;
                if useAngle 
                    view(angle,0); 
                end;
            end
        end
        flag = 1;
        fprintf('worked on cell in cell.\n');
    elseif isstruct(geoPath{meshid})
        nshell = length(geoPath{meshid});
        figure;
        for i=1:nshell
            subplot(2,nshell,i);
            [~, Z] = procrustes(geoPath{meshid}(1).vertices, geoPath{meshid}(i).vertices,...
                'Scaling', false, 'Reflection',false);
            geoPath{meshid}(i).vertices = Z; 
            if i==1 || i == nshell
                cvec = [1 1 1];
            else
                cvec = [1 1 0];
            end
            patch(geoPath{meshid}(i), 'FaceColor', cvec, 'EdgeColor', 'none', 'FaceLighting', 'phong');
            axis equal; axis tight; axis off; cameratoolbar; light;
            if useAngle
                view(angle,0);
            end
        end
        flag = 2;
        fprintf('worked on struct in cell.\n');
    end
else
    fprintf('Display all meshes.\n');
    nmesh = length(geoPath);
    nshell = length(geoPath{1});
    nverts = size(geoPath{1}(1).vertices, 1);
    alphaVec = 0.2.*ones(nverts, 1);
    figure;
    for i=1:nmesh
        for j=1:nshell
            [~, geoPath{i}(j).vertices] = procrustes(geoPath{1}(nshell).vertices, geoPath{i}(j).vertices,...
                'Scaling', false,'Reflection',false);
            cVal = j/nshell;
            subplot(nmesh,nshell+1,(i-1)*(nshell+1)+j);
            if nmesh==1
                if (j>1) && (j<nshell)
                    patch(geoPath{i}(j), 'FaceColor', [1 153/255 51/255], 'EdgeColor', 'none', 'FaceLighting', 'phong',...
                'FaceVertexAlphaData', 0.2);
                else
                    patch(geoPath{i}(j), 'FaceColor', [0 204/255 102/255], 'EdgeColor', 'none', 'FaceLighting', 'phong',...
                'FaceVertexAlphaData', 0.2);
                end
            else
                patch(geoPath{i}(j), 'FaceColor', [cVal cVal cVal], 'EdgeColor', 'none', 'FaceLighting', 'phong',...
                'FaceVertexAlphaData', 0.2);
            end
            axis equal; axis tight; axis off; cameratoolbar; light;
            if useAngle
                view(angle,0);
            end
            
            subplot(nmesh,nshell+1,i*(nshell+1));
            patch(geoPath{i}(j), 'FaceColor', [cVal cVal cVal], 'EdgeColor', 'none', 'FaceLighting', 'phong',...
                'FaceVertexAlphaData', alphaVec, 'FaceAlpha', 'interp');
            axis equal; axis tight; axis off; cameratoolbar;
            if useAngle
                view(angle,0);
            end
            if j==nshell
                light;
            end
        end
    end
    
end
end