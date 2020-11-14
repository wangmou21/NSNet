# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 19:23:14 2020

@author: Silence
"""

import os
import time
import torch
import h5py
import logging
import numpy as np
import scipy.io as sio


import config


def loss_mre(output, target):
    error = abs(output - target)
    error = error/target
    if isinstance(error, torch.Tensor):
        error = torch.mean(error)
    else:
        error = np.mean(error)      
    return error


def loss_mre3d(output, target, refer, l_type='a', is_norm=True):
    output = output[refer==0]
    target = target[refer==0]
      
    error = abs(output - target)
    if l_type=='r':
        if is_norm:
            error[error<1E-4]=0
        else:
            error[error<0.5*1E-5]=0   
        error = error/(abs(target)+1E-4)
    if isinstance(error, torch.Tensor):
        error = torch.mean(error)
    else:
        error = np.mean(error) 
    return error


def compute_norm_Flow(data_type, poros):

    if data_type == 'SpherePacks':
        norm_target3d = 2**(15.69*poros**3-23.87*poros**2 + 18.36*poros-9.38)
    elif data_type == 'QSGS':
        norm_target3d = 2**(17.02*poros**3-27.4*poros**2 + 16.83*poros-8.23)
    elif data_type == 'Fiber':
        norm_target3d = 2**(29.48*poros**3-53.44*poros**2 + 39.15*poros-16.6)    
    return norm_target3d

def computing_heat(material,temper):
    
    if isinstance(material, torch.Tensor):
        material = material.detach().cpu().numpy()
    if isinstance(temper, torch.Tensor):
        temper = temper.detach().cpu().numpy()
        
    
    shape_material = material.shape
    DAORE1, DAORE2 = 0.023, 100
    material[material==0] = DAORE1
    material[material==1] = DAORE2
    
    ndim_mat = np.size(shape_material)
    if ndim_mat==4:
        temper = np.pad(temper,((0,0),(1,1),(0,0),(0,0)),'edge')
        QT = np.mean((temper[:,0:shape_material[1],1:shape_material[1]-1,1:shape_material[1]-1]-temper[:,2:shape_material[1]+2,1:shape_material[1]-1,1:shape_material[1]-1])*material[:,:,1:shape_material[1]-1,1:shape_material[1]-1],(3,2))/2.
        keff = np.sum(QT,ndim_mat-3)/(np.max(temper,(3,2,1))-np.min(temper,(3,2,1)))
    elif ndim_mat==3:
        temper = np.pad(temper,((1,1),(0,0),(0,0)),'edge')
        QT = np.mean((temper[0:shape_material[1]]-temper[2:shape_material[1]+2])*material,(2,1))/2.
        keff = np.sum(QT,ndim_mat-3)/(np.max(temper,(2,1))-np.min(temper,(2,1)))
    
    return keff


def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)
        
def move_data_to_gpu(x, cuda=True):
    if 'float' in str(x.dtype):
        x = torch.Tensor(x)
    elif 'int' in str(x.dtype):
        x = torch.LongTensor(x)
    else:
        raise Exception("Error!")

    if cuda:
        x = x.cuda()

    return x 

def append_to_dict(dict, key, value):
    if key in dict.keys():
        dict[key].append(value)
    else:
        dict[key] = [value]


def create_logging(log_dir, filemode):
    create_folder(log_dir)
    i1 = 0

    while os.path.isfile(os.path.join(log_dir, '{:04d}.log'.format(i1))):
        i1 += 1
        
    log_path = os.path.join(log_dir, '{:04d}.log'.format(i1))
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
        datefmt='%a, %d %b %Y %H:%M:%S',
        filename=log_path,
        filemode=filemode)

    # Print to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    
    return logging



def read_data(data_type):
    
    sub_task = config.sub_task
    workspace = config.workspace
    dir_data = config.dir_data
    dim_s = config.dim_s
    
    
    feature_path = os.path.join(workspace, 'features', 
        'data_{}_{}.h5'.format(data_type, sub_task))
    create_folder(os.path.dirname(feature_path))
    
    print('Reading {}_{} data ...'.format(data_type, sub_task))
    extract_time = time.time()
    
    path_porosity = os.path.join(dir_data, data_type, 'list_infor.mat')
    data_temp = sio.loadmat(path_porosity)
    list_poros = data_temp['data'][:,0]
    list_Mass = data_temp['data'][:,1]
    list_Flow = data_temp['data'][:,2]
    list_Temp = data_temp['data'][:,3]
    num_poros = len(list_poros)
    
    
    hf = h5py.File(feature_path, 'w')
    
    hf.create_dataset(
        name='input', 
        shape=(num_poros, dim_s, dim_s, dim_s), 
        maxshape=(None, dim_s, dim_s, dim_s), 
        dtype=np.int)
    
    hf.create_dataset(
        name='porosity', 
        shape=(num_poros, 1), 
        maxshape=(None, 1), 
        dtype=np.float64)
    
    hf.create_dataset(
        name='target', 
        shape=(num_poros, 1), 
        maxshape=(None, 1), 
        dtype=np.float64)
        
    hf.create_dataset(
        name='target3d', 
        shape=(num_poros, dim_s, dim_s, dim_s), 
        maxshape=(None, dim_s, dim_s, dim_s), 
        dtype=np.float64)
    
    for i, poros_tmp in enumerate(list_poros):
        
        path_data = os.path.join(dir_data,data_type, '{}'.format(poros_tmp), 'structure.mat')
        data_temp = sio.loadmat(path_data)
        hf['input'][i] = data_temp['data'].reshape(dim_s,dim_s,dim_s).T
        hf['porosity'][i] = poros_tmp
        
        path_data = os.path.join(dir_data,data_type, '{}'.format(poros_tmp), '{}.mat'.format(sub_task))
        data_temp = sio.loadmat(path_data)
        hf['target3d'][i] = data_temp['data'].reshape(dim_s,dim_s,dim_s).T
        
        list_value = eval('list_'+sub_task)
        hf['target'][i] = list_value[i]        
        
    #if (sub_task=='Flow') &  (data_type=='SpherePacks'):
            
    hf.close()
    
    print('Write hdf5 file to {} using {:.3f} s'.format(
        feature_path, time.time() - extract_time))