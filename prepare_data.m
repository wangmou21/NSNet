


clear all;
close all;
clc;

type_struct = 'SpherePacks';              % 'SpherePacks', 'Fiber', 'QSGS'
rwdata(type_struct)

function [] = rwdata(type_struct)
    dir_data = ['J:/Database/NS_data/'];
    path_original = [dir_data,type_struct, '_Mass/'];
    path_output = ['../data/',type_struct, '/'];
    dir_original = dir(path_original);
    dir_original(1:2)=[];
    num_case = length(dir_original);
    
    list_poros = [];

    for i = 1:num_case
    
        poros_tmp = strsplit(dir_original(i).name,'_');
        
        % Structure information
        path_case = [path_original,dir_original(i).name,'/Input_DL.dat'];
        data_temp = readmatrix(path_case);
        
        data = squeeze(data_temp(:,1));                            %结构信息
        dim_s = nthroot(length(data),3);
        data = reshape(data,[dim_s,dim_s,dim_s]);
        data = data(1:200,1:200,1:200);
        data = data(:);
        path_out_tmp = [path_output,poros_tmp{2},'/structure.mat'];
        mkdir([path_output,poros_tmp{2}]);
        save(path_out_tmp,'data');
        
%         dim_s = nthroot(length(data),3);
%         s_3d = reshape(data,[dim_s,dim_s,dim_s]);                              %porous structure,3D
%         [x1,y1,z1] = meshgrid(1:dim_s,1:dim_s,1:dim_s);
%         xs = 100;
%         ys = xs;
%         zs = xs;
%         h = slice(x1,y1,z1,s_3d,xs,ys,zs);
        
        data = squeeze(data_temp(:,2));                            % Mass信息
        data = reshape(data,[dim_s,dim_s,dim_s]);
        data = data(1:200,1:200,1:200);
        data = data(:);
        path_out_tmp = [path_output,poros_tmp{2},'/Mass.mat'];
        save(path_out_tmp,'data');
        
%         m_3d = reshape(data,[dim_s,dim_s,dim_s]);                              %porous structure,3D
%         figure;
%         h = slice(x1,y1,z1,m_3d,xs,ys,zs);
    
        path_case = [dir_data,type_struct, '_Temp/',dir_original(i).name,'/TOVER.dat'];
        data_temp = readmatrix(path_case);
    
        data = squeeze(data_temp(:,4));                             %温度信息
        data = reshape(data,[dim_s,dim_s,dim_s]);
        data = data(1:200,1:200,1:200);
        data = data(:);
        path_out_tmp = [path_output,poros_tmp{2},'/Temp.mat'];
        save(path_out_tmp,'data');
        
%         t_3d = reshape(data,[dim_s,dim_s,dim_s]);                              %porous structure,3D
%         figure;
%         h = slice(x1,y1,z1,t_3d,xs,ys,zs);
        
        path_case = [dir_data,type_struct, '_Flow/',dir_original(i).name,'/Vel_UZ.mat'];
        load(path_case);
        data = permute(imageVolUz,[3 1 2]);
        data = data(1:200,1:200,1:200);
        data = data(:);
        path_out_tmp = [path_output,poros_tmp{2},'/Flow.mat'];
        save(path_out_tmp,'data');
        
        list_poros = [list_poros; str2num(poros_tmp{2})];
    end

    name_save = ['../data/',type_struct,'/list_poros.mat'];
    save(name_save,'list_poros');

end
