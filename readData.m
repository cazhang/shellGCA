% read in 3 pairs of cactus shapes
function FVs = readData(dataset,ninput,id,isTrain)
if nargin < 3
    id = 1;
end
if nargin < 4
    isTrain = true;
end

%pcinfo=java.net.InetAddress.getLocalHost;
%hostname = pcinfo.getHostName;
[~,hostname]= system('hostname');

if strcmp(hostname(1:2), 're')
    server_folder = '/shared/storage/cs/staffstore/cz679/datasets/';
elseif strcmp(hostname(1:2), 'wv')
    server_folder = '/local/sdb1/datasets/';
else
    server_folder = '/Users/chao/cssvn/shellgca/code/';
end
if strcmp(dataset, 'cactus')   
    for i = 1:ninput
        if ninput == 6
            % [3,4,5,6,7,8]
            input_name = [server_folder,'cactus/coarseCactusPoses', num2str(i+2), '.ply'];
        elseif ninput == 4
            input_name = [server_folder,'cactus/coarseCactusPoses', num2str(i+29), '.ply'];
        else
            % [0,1,..,8,9]
            input_name = [server_folder,'cactus/coarseCactusPoses', num2str(i-1), '.ply'];
        end
        [FVs{i}.faces,FVs{i}.vertices]=plyread(input_name,'tri');
    end
elseif strcmp(dataset, 'cactus2')
    for i=1:ninput
        % [0,1,2,3,5,7]
        indice = [0 1 2 3 5 7];
        input_name = [server_folder,'cactus/coarseCactusPoses', num2str(indice(i)), '.ply'];
        [FVs{i}.faces,FVs{i}.vertices]=plyread(input_name,'tri');
    end
        
elseif strcmp(dataset, 'body') % FAUST data: 100 meshes, 10 id, each 10 pose
    ninput = 9;
    ntest = 10 - ninput;
    if isTrain
        for i=1:ninput
            index = (id-1)*10+(i-1);
            input_name = [server_folder,'Bodies/low_reg_', num2str(index,'%03d'), '.ply'];
            fprintf('Reading %s\n', input_name);
            %input_name = ['veri', num2str(i-1), '.ply'];
            [FVs{i}.faces,FVs{i}.vertices]=plyread(input_name,'tri');
        end
    else
        for i=1:ntest
            index = (id-1)*10+(i-1)+8;
            input_name = [server_folder,'Bodies/low_reg_', num2str(index,'%03d'), '.ply'];
            fprintf('Reading %s\n', input_name);
            %input_name = ['veri', num2str(i-1), '.ply'];
            [FVs{i}.faces,FVs{i}.vertices]=plyread(input_name,'tri');
        end
    end
    
    
elseif strcmp(dataset, 'male100') % FAUST data: 100 meshes, 10 id, each 10 pose
    meshfolder = [server_folder, 'maleExt/deci/'];
    meshdirs = dir([meshfolder,'*.ply']);
    ntotal = length(meshdirs);
    FVs = [];
    if isTrain      
        for i = 1:2:ntotal
            input_name = meshdirs(i).name;
            full_name = [meshfolder, input_name];
            fprintf('Reading %s\n', full_name);
            %input_name = ['veri', num2str(i-1), '.ply'];
            [FVs{end+1}.faces,FVs{end+1}.vertices]=plyread(full_name,'tri');
        end
    else
        for i = 2:2:ntotal
            input_name = meshdirs(i).name;
            full_name = [meshfolder, input_name];
            fprintf('Reading %s\n', full_name);
            %input_name = ['veri', num2str(i-1), '.ply'];
            [FVs{end+1}.faces,FVs{end+1}.vertices]=plyread(full_name,'tri');
        end        
    end
    
elseif strcmp(dataset, 'kick') % FAUST data: 100 meshes, 10 id, each 10 pose
    meshfolder = [server_folder, 'maleKick/deci/'];
    meshdirs = dir([meshfolder,'*.ply']);
    ntotal = length(meshdirs);
    FVs = [];
    if isTrain      
        for i = 1:ntotal
            input_name = meshdirs(i).name;
            full_name = [meshfolder, input_name];
            fprintf('Reading %s\n', full_name);
            %input_name = ['veri', num2str(i-1), '.ply'];
            [FVs{end+1}.faces,FVs{end+1}.vertices]=plyread(full_name,'tri');
        end       
    end    

elseif strcmp(dataset, 'kick2') % FAUST data: 100 meshes, 10 id, each 10 pose
    meshfolder = [server_folder, 'maleKick2/deci/'];
    meshdirs = dir([meshfolder,'*.ply']);
    ntotal = length(meshdirs);
    FVs = [];
    if isTrain      
        for i = 1:ntotal
            input_name = meshdirs(i).name;
            full_name = [meshfolder, input_name];
            fprintf('Reading %s\n', full_name);
            %input_name = ['veri', num2str(i-1), '.ply'];
            [FVs{end+1}.faces,FVs{end+1}.vertices]=plyread(full_name,'tri');
        end       
    end 
elseif strcmp(dataset, 'male3') % male3 of dyna data
    if isTrain
        meshfolder = [server_folder, 'male3/train/deci/'];
    else
        meshfolder = [server_folder, 'male3/test/deci/'];
    end
    meshdirs = dir([meshfolder,'*.ply']);
    ninput = length(meshdirs);
    for i=1:ninput
        input_name = meshdirs(i).name;
        full_name = [meshfolder, input_name];
        fprintf('Reading %s\n', full_name);
        %input_name = ['veri', num2str(i-1), '.ply'];
        [FVs{i}.faces,FVs{i}.vertices]=plyread(full_name,'tri');
    end
    
elseif strcmp(dataset, 'fatty') % 50002
    if isTrain
        meshfolder = [server_folder, 'fatty/deci/'];
    end
    meshdirs = dir([meshfolder,'*.ply']);
    ninput = length(meshdirs);
    for i=1:ninput
        input_name = meshdirs(i).name;
        full_name = [meshfolder, input_name];
        fprintf('Reading %s\n', full_name);
        %input_name = ['veri', num2str(i-1), '.ply'];
        [FVs{i}.faces,FVs{i}.vertices]=plyread(full_name,'tri');
    end

elseif strcmp(dataset, 'fattySim') % fattySim
    if isTrain
        meshfolder = [server_folder, 'fatty/fatty_simult/'];
    end
    meshdirs = dir([meshfolder,'*.ply']);
    ninput = length(meshdirs);
    for i=1:ninput
        input_name = meshdirs(i).name;
        full_name = [meshfolder, input_name];
        fprintf('Reading %s\n', full_name);
        %input_name = ['veri', num2str(i-1), '.ply'];
        [FVs{i}.faces,FVs{i}.vertices]=plyread(full_name,'tri');
    end

elseif strcmp(dataset, '50021') % 50021
    if isTrain
        meshfolder = [server_folder, '50021/deci_simul/'];
    end
    meshdirs = dir([meshfolder,'*.ply']);
    ninput = length(meshdirs);
    for i=1:ninput
        input_name = meshdirs(i).name;
        full_name = [meshfolder, input_name];
        fprintf('Reading %s\n', full_name);
        %input_name = ['veri', num2str(i-1), '.ply'];
        [FVs{i}.faces,FVs{i}.vertices]=plyread(full_name,'tri');
    end
    
elseif strcmp(dataset, 'cat') % cat
    meshfolder = [server_folder, 'toscahires-mat/'];
    meshdirs = dir([meshfolder,'low_cat*.ply']);
    ninput = length(meshdirs);
    for i=1:ninput
        %input_name = [meshfolder, 'low_cat', num2str(i), '.ply'];
        input_name = [meshfolder, meshdirs(i).name];
        %full_name = [meshfolder, input_name];
        fprintf('Reading %s\n', input_name);
        %input_name = ['veri', num2str(i-1), '.ply'];
        [FVs{i}.faces,FVs{i}.vertices]=plyread(input_name,'tri');
    end

elseif strcmp(dataset, 'cat2') % cat
    meshfolder = [server_folder, 'cats2/'];
    meshdirs = dir([meshfolder,'simp*.ply']);
    ninput = length(meshdirs);
    for i=1:ninput
        %input_name = [meshfolder, 'low_cat', num2str(i), '.ply'];
        input_name = [meshfolder, meshdirs(i).name];
        %full_name = [meshfolder, input_name];
        fprintf('Reading %s\n', input_name);
        %input_name = ['veri', num2str(i-1), '.ply'];
        [FVs{i}.faces,FVs{i}.vertices]=plyread(input_name,'tri');
    end
elseif strcmp(dataset, 'hc_male3') % male3 of dyna data
       
    if isTrain
        meshfolder = [server_folder, 'hc_male3/train/deci/'];
        meshdirs = dir([meshfolder,'*.ply']);
    else
        meshfolder = [server_folder, 'male3/test/deci/'];
        meshdirs = dir([meshfolder,'*.ply']);
    end
    ninput = length(meshdirs);
    for i=1:ninput
        input_name = meshdirs(i).name;
        full_name = [meshfolder, input_name];
        fprintf('Reading %s\n', full_name);
        %input_name = ['veri', num2str(i-1), '.ply'];
        [FVs{i}.faces,FVs{i}.vertices]=plyread(full_name,'tri');
    end
elseif strcmp(dataset, 'dynaMale') % meshlab to 3k
    meshfolder = [server_folder, 'dynaMale/'];
    meshdirs = dir([meshfolder,'*.ply']);
    ninput = length(meshdirs);
    for i=1:ninput
        input_name = meshdirs(i).name;
        full_name = [meshfolder, input_name];
        fprintf('Reading %s\n', full_name);
        %input_name = ['veri', num2str(i-1), '.ply'];
        [FVs{i}.faces,FVs{i}.vertices]=plyread(full_name,'tri');
    end
    
elseif strcmp(dataset, 'dynaFemale') % meshlab to 3k
    meshfolder = [server_folder, 'dynaFemale/'];
    meshdirs = dir([meshfolder,'*.ply']);
    ninput = length(meshdirs);
    for i=1:ninput   
        input_name = meshdirs(i).name;
        full_name = [meshfolder, input_name];
        fprintf('Reading %s\n', full_name);
        %input_name = ['veri', num2str(i-1), '.ply'];
        [FVs{i}.faces,FVs{i}.vertices]=plyread(full_name,'tri');
    end
elseif strcmp(dataset, 'face') % faceware data: 2k faces
    if isTrain
        meshfolder = [server_folder, 'faces/train/'];
        meshdirs = dir([meshfolder,'*.ply']);
    else
        meshfolder = [server_folder, 'faces/test/'];
        meshdirs = dir([meshfolder,'*.ply']);
    end
    ninput = length(meshdirs);
    for i=1:ninput
        input_name = meshdirs(i).name;
        full_name = [meshfolder, input_name];
        fprintf('Reading %s\n', full_name);
        %input_name = ['veri', num2str(i-1), '.ply'];
        [FVs{i}.faces,FVs{i}.vertices]=plyread(full_name,'tri');
    end
    
elseif strcmp(dataset, 'faceware') % faceware data: 5k faces
    if isTrain
        meshfolder = [server_folder, 'faceware/low5k/train/'];
        meshdirs = dir([meshfolder,'*.ply']);
    else
        meshfolder = [server_folder, 'faceware/low5k/test/'];
        meshdirs = dir([meshfolder,'*.ply']);
    end
    ninput = length(meshdirs);
    for i=1:ninput
        input_name = meshdirs(i).name;
        full_name = [meshfolder, input_name];
        fprintf('Reading %s\n', full_name);
        %input_name = ['veri', num2str(i-1), '.ply'];
        [FVs{i}.faces,FVs{i}.vertices]=plyread(full_name,'tri');
    end

elseif strcmp(dataset, 'casear') % casear
    if isTrain
        meshfolder = [server_folder, 'casear/train_align/deci/'];
        meshdirs = dir([meshfolder,'*.ply']);
    else
        meshfolder = [server_folder, 'casear/test_align/deci/'];
        meshdirs = dir([meshfolder,'*.ply']);
    end
    ninput = length(meshdirs);
    for i=1:ninput
        input_name = meshdirs(i).name;
        full_name = [meshfolder, input_name];
        fprintf('Reading %s\n', full_name);
        %input_name = ['veri', num2str(i-1), '.ply'];
        [FVs{i}.faces,FVs{i}.vertices]=plyread(full_name,'tri');
    end
    
elseif strcmp(dataset, 'casear30') % casear30
    if isTrain
        meshfolder = [server_folder, 'casear/train_align/deci/'];
        meshdirs = dir([meshfolder,'*.ply']);
    else
        meshfolder = [server_folder, 'casear/test_align/deci/'];
        meshdirs = dir([meshfolder,'*.ply']);
    end
    for i=1:ninput
        input_name = meshdirs(i).name;
        full_name = [meshfolder, input_name];
        fprintf('Reading %s\n', full_name);
        %input_name = ['veri', num2str(i-1), '.ply'];
        [FVs{i}.faces,FVs{i}.vertices]=plyread(full_name,'tri');
    end
    
elseif strcmp(dataset, 'smpl_f50') % smpl, sythetic 100 female
    if isTrain
        meshfolder = [server_folder, 'femaleSMPL/train/deci/'];
        meshdirs = dir([meshfolder,'*.ply']);
    else
        meshfolder = [server_folder, 'femaleSMPL/test/deci/'];
        meshdirs = dir([meshfolder,'*.ply']);
    end
    for i=1:ninput
        input_name = meshdirs(i).name;
        full_name = [meshfolder, input_name];
        fprintf('Reading %s\n', full_name);
        %input_name = ['veri', num2str(i-1), '.ply'];
        [FVs{i}.faces,FVs{i}.vertices]=plyread(full_name,'tri');
    end
    
elseif strcmp(dataset, 'scape71')
    meshfolder = [server_folder, 'scape_obj/deci/'];
    savetrain = [server_folder, 'scape_obj/train/'];
    savetest = [server_folder, 'scape_obj/test/'];
    [train_list, test_list] = prepare_index_scape;
    if isTrain
        ind_list = train_list;
    else
        ind_list = test_list;
    end
    for i=1:length(ind_list)
        input_name = [num2str(ind_list(i)),'.ply'];
        full_name = [meshfolder, input_name];
        fprintf('Reading %s\n', full_name);
        %input_name = ['veri', num2str(i-1), '.ply'];
        [FVs{i}.faces,FVs{i}.vertices]=plyread(full_name,'tri');
%         if isTrain
%             savename = [savetrain,num2str(ind_list(i)),'.obj'];
%             objwrite(savename, FVs{i}.faces, FVs{i}.vertices);
%         else
%             savename = [savetest,num2str(ind_list(i)),'.obj'];
%             objwrite(savename, FVs{i}.faces, FVs{i}.vertices);
%         end
    end
    
elseif strcmp(dataset, 'scape_full')
    meshfolder = [server_folder, 'scape_obj/deci/'];
    
    for i=1:ninput
        input_name = [num2str(i),'.ply'];
        full_name = [meshfolder, input_name];
        fprintf('Reading %s\n', full_name);
        %input_name = ['veri', num2str(i-1), '.ply'];
        [FVs{i}.faces,FVs{i}.vertices]=plyread(full_name,'tri');
    end
    
elseif strcmp(dataset, 'scape_smpl')
    meshfolder = [server_folder, 'scape_smpl/deci/'];
    
    for i=1:ninput
        input_name = [num2str(i),'.ply'];
        full_name = [meshfolder, input_name];
        fprintf('Reading %s\n', full_name);
        %input_name = ['veri', num2str(i-1), '.ply'];
        [FVs{i}.faces,FVs{i}.vertices]=plyread(full_name,'tri');
    end
    
elseif strcmp(dataset, 'face40')
    meshfolder = [server_folder, 'face_iccv/deci/'];
    
    for i=1:ninput
        input_name = ['f',num2str(i-1),'.ply'];
        full_name = [meshfolder, input_name];
        fprintf('Reading %s\n', full_name);
        %input_name = ['veri', num2str(i-1), '.ply'];
        [FVs{i}.faces,FVs{i}.vertices]=plyread(full_name,'tri');
    end
    
    
elseif strcmp(dataset, 'horse')
    for i=1:ninput
        input_name = ['horse/horse-gallop-', num2str(i-1), '1.ply'];
        %input_name = ['veri', num2str(i-1), '.ply'];
        [FVs{i}.faces,FVs{i}.vertices]=plyread(input_name,'tri');
    end
end
end

% support function
function [train_list, test_list] = prepare_index_scape
fulllist = [1:71];
test_list = [47	69	17	37	59	45	42	66	70	16	8	29	46	52	24	31	51 ...
    60	55	1	5	6	33	49	27	50	19	7	61	44	20	43	23	25	56];
test_list = sort(test_list + 1);
train_list = setdiff(fulllist, test_list);
end