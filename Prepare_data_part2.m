
clear all;
close all;
clc;

type_struct = 'SpherePacks';
path_output = ['../data/',type_struct, '_2/'];

path_data = 'J:/CFD_total/Data_zip/SpherePacks_Flow_2/';
dir_data = dir(path_data);
dir_data(1:2) = [];
dir_data(end) = [];

num_case = length(dir_data);
for i = 1:num_case
     
    path_file = [path_data,dir_data(i).name,'/S_porosity.mat'];
    load(path_file);
    
    poros_tmp = strsplit(dir_data(i).name,'_');
    path_out_tmp = [path_output,poros_tmp{2},'/structure.mat'];
    mkdir([path_output,poros_tmp{2}]);
    S = S(:);
    save(path_out_tmp,'S');
        
%     subplot(1,2,1);
%     dim_s = 201;
%         [x1,y1,z1] = meshgrid(1:dim_s,1:dim_s,1:dim_s);
%         xs = 100;
%         ys = xs;
%         zs = xs;
%         h = slice(x1,y1,z1,S,xs,ys,zs);
%         shading flat
%shading interp

    path_file = [path_data,dir_data(i).name,'/Uz_porosity.mat'];
    load(path_file);
%        subplot(1,2,2);
%     h = slice(x1,y1,z1,imageVolUz,xs,ys,zs);
%     shading flat
    imageVolUz = imageVolUz(:);
    path_out_tmp = [path_output,poros_tmp{2},'/Flow.mat'];
    save(path_out_tmp,'imageVolUz');
    
end