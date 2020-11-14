# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 10:37:20 2020

@author: Silence
"""


import os
import logging
import time
import h5py
import numpy as np
import scipy.io as sio

from utils import compute_norm_Flow, computing_heat

import config
    
class DataGenerator_Mass(object):
    
    def __init__(self, seed=1234):
        '''Data generator for training and validation. 
        
        Args:
          feature_hdf5_path: string, path of hdf5 feature file
          batch_size: int
        '''

        self.batch_size = config.batch_size
        self.num_slice_per_dim = config.num_slice_per_dim
        self.random_state = np.random.RandomState(seed)
        self.size_slice_per_dim = config.size_slice_per_dim
        dir_data = config.dir_data
        dim_s = config.dim_s
        
        ratio_train_test = config.ratio_train_test
        
        # Load training data
        load_time = time.time()
        
        path_porosity = os.path.join(dir_data, config.type_train, 'list_infor.mat')
        infor_temp = sio.loadmat(path_porosity)
        list_poros = infor_temp['data'][:,0]
        
        self.num_poros = len(list_poros)
        self.value_range = config.value_range
        
        
        self.data_dict = {}
        self.data_dict['input'] = np.zeros((self.num_poros,dim_s, dim_s, dim_s)).astype(np.float32)
        self.data_dict['porosity'] = np.zeros((self.num_poros)).astype(np.float32)
        self.data_dict['target3d'] = np.zeros((self.num_poros,dim_s, dim_s, dim_s)).astype(np.float32)
        self.data_dict['target'] = np.zeros((self.num_poros)).astype(np.float32)
        
        for i, poros_tmp in enumerate(list_poros):
        
            path_data = os.path.join(dir_data, config.type_train, '{}'.format(poros_tmp), 'structure.mat')
            data_temp = sio.loadmat(path_data)
            self.data_dict['input'][i] = data_temp['data'].reshape(dim_s,dim_s,dim_s).T
            self.data_dict['porosity'][i] = poros_tmp
        
            path_data = os.path.join(dir_data, config.type_train, '{}'.format(poros_tmp), '{}.mat'.format(config.sub_task))
            data_temp = sio.loadmat(path_data)
            self.data_dict['target3d'][i] = data_temp['data'].reshape(dim_s,dim_s,dim_s).T
            self.data_dict['target'][i] = infor_temp['data'][i,1]       
        
       
        index_tmp = np.zeros((self.num_poros,self.num_slice_per_dim**3),dtype = int)
        for i in range(self.num_poros):
            index_tmp[i,:] = np.random.permutation(self.num_slice_per_dim**3)
        
        self.train_index = index_tmp[:,0:round(ratio_train_test*np.size(index_tmp,1))]
        self.validate_index = index_tmp[:,round(ratio_train_test*np.size(index_tmp,1)):]        
           
        self.num_train = len(self.train_index.flatten())
        self.num_validate = len(self.validate_index.flatten())
        
        self.train_slice_indexes = np.random.permutation(self.num_train)
        self.validate_slice_indexes = np.random.permutation(self.num_validate)
               
        logging.info('Load data time: {:.3f} s'.format(time.time() - load_time))
        logging.info('Training slice num: {}'.format(self.num_train))            
        logging.info('Validation slice num: {}'.format(self.num_validate))
        
        self.pointer = 0
        
    def generate_train(self):
        '''Generate mini-batch data for training. 
        
        Returns:
          batch_data_dict: dict containing 
        '''
        
        while True:
            # Reset pointer
            if self.pointer > self.num_train-self.batch_size:
                self.pointer = 0
                self.random_state.shuffle(self.train_slice_indexes)

            # Get batch slice indexes
            batch_indexes_tmp = self.train_slice_indexes[
                self.pointer: self.pointer + self.batch_size]
                
            self.pointer += self.batch_size
            
            batch_indexes_s = batch_indexes_tmp % self.num_poros
            batch_indexes_l = batch_indexes_tmp // self.num_poros
            batch_indexes = self.train_index[batch_indexes_s, batch_indexes_l]
            
            batch_data_dict = {}
            
            batch_indexes_x = batch_indexes // (self.num_slice_per_dim**2)
            batch_indexes_y = batch_indexes % (self.num_slice_per_dim**2)// (self.num_slice_per_dim)
            batch_indexes_z = batch_indexes % (self.num_slice_per_dim**1)
            
            
            index_step = (config.dim_s-self.size_slice_per_dim)//(self.num_slice_per_dim-1)
            
            batch_data_dict['input'] = np.array([self.data_dict['input'][batch_indexes_s[i],
                                            batch_indexes_x[i]*index_step:batch_indexes_x[i]*index_step+self.size_slice_per_dim,
                                            batch_indexes_y[i]*index_step:batch_indexes_y[i]*index_step+self.size_slice_per_dim,
                                            batch_indexes_z[i]*index_step:batch_indexes_z[i]*index_step+self.size_slice_per_dim]
                                           for i in range(self.batch_size) ])
            
            batch_data_dict['porosity'] = self.data_dict['porosity'][batch_indexes_s]

            target3d_tmp = np.array([self.data_dict['target3d'][batch_indexes_s[i],
                                    batch_indexes_x[i]*index_step:batch_indexes_x[i]*index_step+self.size_slice_per_dim,
                                    batch_indexes_y[i]*index_step:batch_indexes_y[i]*index_step+self.size_slice_per_dim,
                                    batch_indexes_z[i]*index_step:batch_indexes_z[i]*index_step+self.size_slice_per_dim]
                                    for i in range(self.batch_size) ])
            
            # computing deff before normaliztion            
            target = np.mean(target3d_tmp,(1,2,3))
            target = target/config.dCdX/config.Db

            batch_data_dict['target'] = target.astype(np.float32)
            
            batch_data_dict['target3d'] = target3d_tmp/config.scale_Mass
            
            #data_mass_tmp = np.log(self.value_range*np.clip(data_mass_tmp,0,1)+1)-np.log(1-self.value_range*np.clip(data_mass_tmp,-1,0))
            
            yield batch_data_dict

class DataGenerator_Flow(object):
    
    def __init__(self, seed=1234):
        '''Data generator for training and validation. 
        
        Args:
          feature_hdf5_path: string, path of hdf5 feature file
          batch_size: int
        '''

        self.batch_size = config.batch_size
        self.num_slice_per_dim = config.num_slice_per_dim
        self.random_state = np.random.RandomState(seed)
        self.size_slice_per_dim = config.size_slice_per_dim
        dir_data = config.dir_data
        dim_s = config.dim_s
        
        ratio_train_test = config.ratio_train_test
        
        # Load training data
        load_time = time.time()
        
        path_porosity = os.path.join(dir_data, config.type_train, 'list_infor.mat')
        infor_temp = sio.loadmat(path_porosity)
        list_poros = infor_temp['data'][:,0]
        
        if config.type_train=='SpherePacks':
            path_porosity = os.path.join(dir_data, config.type_train+'_2', 'list_infor.mat')
            infor_temp2 = sio.loadmat(path_porosity)
            list_poros2 = infor_temp2['data'][:,0]
            list_poros = np.hstack((list_poros,list_poros2))
            infor_temp['data'] = np.vstack((infor_temp['data'],infor_temp2['data']))
        

        list_poros = list_poros[2:]        
        self.num_poros = len(list_poros)
        self.value_range = config.value_range
        
        
        self.data_dict = {}
        self.data_dict['input'] = np.zeros((self.num_poros,dim_s, dim_s, dim_s)).astype(np.float32)
        self.data_dict['porosity'] = np.zeros((self.num_poros)).astype(np.float32)
        self.data_dict['target3d'] = np.zeros((self.num_poros,dim_s, dim_s, dim_s)).astype(np.float32)
        self.data_dict['target'] = np.zeros((self.num_poros)).astype(np.float32)
        
        
        for i, poros_tmp in enumerate(list_poros):
            if i<24:
                path_data = os.path.join(dir_data, config.type_train, '{}'.format(poros_tmp), 'structure.mat')
            else:
                path_data = os.path.join(dir_data, config.type_train+'_2', '{}'.format(poros_tmp), 'structure.mat')
            data_temp = sio.loadmat(path_data)
            self.data_dict['input'][i] = data_temp['data'].reshape(dim_s,dim_s,dim_s).T
            self.data_dict['porosity'][i] = poros_tmp
            
            if i<24:
                path_data = os.path.join(dir_data, config.type_train, '{}'.format(poros_tmp), '{}.mat'.format(config.sub_task))
            else:
                path_data = os.path.join(dir_data, config.type_train+'_2', '{}'.format(poros_tmp), '{}.mat'.format(config.sub_task))
            data_temp = sio.loadmat(path_data)
            self.data_dict['target3d'][i] = data_temp['data'].reshape(dim_s,dim_s,dim_s).T
            self.data_dict['target'][i] = infor_temp['data'][i,2]    
        
       
        index_tmp = np.zeros((self.num_poros,self.num_slice_per_dim**3),dtype = int)
        for i in range(self.num_poros):
            index_tmp[i,:] = np.random.permutation(self.num_slice_per_dim**3)
        
        self.train_index = index_tmp[:,0:round(ratio_train_test*np.size(index_tmp,1))]
        self.validate_index = index_tmp[:,round(ratio_train_test*np.size(index_tmp,1)):]        
           
        self.num_train = len(self.train_index.flatten())
        self.num_validate = len(self.validate_index.flatten())
        
        self.train_slice_indexes = np.random.permutation(self.num_train)
        self.validate_slice_indexes = np.random.permutation(self.num_validate)
               
        logging.info('Load data time: {:.3f} s'.format(time.time() - load_time))
        logging.info('Training slice num: {}'.format(self.num_train))            
        logging.info('Validation slice num: {}'.format(self.num_validate))
        
        self.pointer = 0
        
    def generate_train(self):
        '''Generate mini-batch data for training. 
        
        Returns:
          batch_data_dict: dict containing 
        '''
        
        while True:
            # Reset pointer
            if self.pointer > self.num_train-self.batch_size:
                self.pointer = 0
                self.random_state.shuffle(self.train_slice_indexes)

            # Get batch slice indexes
            batch_indexes_tmp = self.train_slice_indexes[
                self.pointer: self.pointer + self.batch_size]
                
            self.pointer += self.batch_size
            
            batch_indexes_s = batch_indexes_tmp % self.num_poros
            batch_indexes_l = batch_indexes_tmp // self.num_poros
            batch_indexes = self.train_index[batch_indexes_s, batch_indexes_l]
            
            batch_data_dict = {}
            
            batch_indexes_x = batch_indexes // (self.num_slice_per_dim**2)
            batch_indexes_y = batch_indexes % (self.num_slice_per_dim**2)// (self.num_slice_per_dim)
            batch_indexes_z = batch_indexes % (self.num_slice_per_dim**1)
            
            
            index_step = (config.dim_s-self.size_slice_per_dim)//(self.num_slice_per_dim-1)
            
            batch_data_dict['input'] = np.array([self.data_dict['input'][batch_indexes_s[i],
                                            batch_indexes_x[i]*index_step:batch_indexes_x[i]*index_step+self.size_slice_per_dim,
                                            batch_indexes_y[i]*index_step:batch_indexes_y[i]*index_step+self.size_slice_per_dim,
                                            batch_indexes_z[i]*index_step:batch_indexes_z[i]*index_step+self.size_slice_per_dim]
                                           for i in range(self.batch_size) ])
            
            poros_tmp = 1-np.mean(batch_data_dict['input'],(3,2,1))
            batch_data_dict['porosity'] = self.data_dict['porosity'][batch_indexes_s]
            batch_data_dict['porosity'] = poros_tmp

            target3d_tmp = np.array([self.data_dict['target3d'][batch_indexes_s[i],
                                    batch_indexes_x[i]*index_step:batch_indexes_x[i]*index_step+self.size_slice_per_dim,
                                    batch_indexes_y[i]*index_step:batch_indexes_y[i]*index_step+self.size_slice_per_dim,
                                    batch_indexes_z[i]*index_step:batch_indexes_z[i]*index_step+self.size_slice_per_dim]
                                    for i in range(self.batch_size) ])
            
            # computing deff before normaliztion       
            target = np.mean(target3d_tmp,(1,2,3))
            target = target*1E7/6

            batch_data_dict['target'] = target.astype(np.float32)
                        
            norm_target3d = compute_norm_Flow('SpherePacks', poros_tmp)
            #norm_target3d = np.max(target3d_tmp,(3,2,1))
            
            batch_data_dict['target3d'] = target3d_tmp/norm_target3d[:,None, None, None]
            
            #data_mass_tmp = np.log(self.value_range*np.clip(data_mass_tmp,0,1)+1)-np.log(1-self.value_range*np.clip(data_mass_tmp,-1,0))
            
            yield batch_data_dict        
            
            
class DataGenerator_Temp(object):
    
    def __init__(self, seed=1234):
        '''Data generator for training and validation. 
        
        Args:
          feature_hdf5_path: string, path of hdf5 feature file
          batch_size: int
        '''

        self.batch_size = config.batch_size
        self.num_slice_per_dim = config.num_slice_per_dim
        self.random_state = np.random.RandomState(seed)
        self.size_slice_per_dim = config.size_slice_per_dim
        dir_data = config.dir_data
        dim_s = config.dim_s
        
        ratio_train_test = config.ratio_train_test
        
        # Load training data
        load_time = time.time()
        
        path_porosity = os.path.join(dir_data, config.type_train, 'list_infor.mat')
        infor_temp = sio.loadmat(path_porosity)
        list_poros = infor_temp['data'][:,0]
          
        self.num_poros = len(list_poros)
        self.value_range = config.value_range
        
        
        self.data_dict = {}
        self.data_dict['input'] = np.zeros((self.num_poros,dim_s, dim_s, dim_s)).astype(np.float32)
        self.data_dict['porosity'] = np.zeros((self.num_poros)).astype(np.float32)
        self.data_dict['target3d'] = np.zeros((self.num_poros,dim_s, dim_s, dim_s)).astype(np.float32)
        self.data_dict['target'] = np.zeros((self.num_poros)).astype(np.float32)
        
        
        for i, poros_tmp in enumerate(list_poros):

            path_data = os.path.join(dir_data, config.type_train, '{}'.format(poros_tmp), 'structure.mat')
            data_temp = sio.loadmat(path_data)
            self.data_dict['input'][i] = data_temp['data'].reshape(dim_s,dim_s,dim_s).T
            self.data_dict['porosity'][i] = poros_tmp
            
            path_data = os.path.join(dir_data, config.type_train, '{}'.format(poros_tmp), '{}.mat'.format(config.sub_task))
            data_temp = sio.loadmat(path_data)
            self.data_dict['target3d'][i] = data_temp['data'].reshape(dim_s,dim_s,dim_s).T
            self.data_dict['target'][i] = infor_temp['data'][i,2]    
        
       
        index_tmp = np.zeros((self.num_poros,self.num_slice_per_dim**2),dtype = int)
        for i in range(self.num_poros):
            index_tmp[i,:] = np.random.permutation(self.num_slice_per_dim**2)
        
        self.train_index = index_tmp[:,0:round(ratio_train_test*np.size(index_tmp,1))]
        self.validate_index = index_tmp[:,round(ratio_train_test*np.size(index_tmp,1)):]        
           
        self.num_train = len(self.train_index.flatten())
        self.num_validate = len(self.validate_index.flatten())
        
        self.train_slice_indexes = np.random.permutation(self.num_train)
        self.validate_slice_indexes = np.random.permutation(self.num_validate)
               
        logging.info('Load data time: {:.3f} s'.format(time.time() - load_time))
        logging.info('Training slice num: {}'.format(self.num_train))            
        logging.info('Validation slice num: {}'.format(self.num_validate))
        
        self.pointer = 0
        
    def generate_train(self):
        '''Generate mini-batch data for training. 
        
        Returns:
          batch_data_dict: dict containing 
        '''
        
        while True:
            # Reset pointer
            if self.pointer > self.num_train-self.batch_size:
                self.pointer = 0
                self.random_state.shuffle(self.train_slice_indexes)

            # Get batch slice indexes
            batch_indexes_tmp = self.train_slice_indexes[
                self.pointer: self.pointer + self.batch_size]
                
            self.pointer += self.batch_size
            
            batch_indexes_s = batch_indexes_tmp % self.num_poros
            batch_indexes_l = batch_indexes_tmp // self.num_poros
            batch_indexes = self.train_index[batch_indexes_s, batch_indexes_l]
            
            batch_data_dict = {}
            
            batch_indexes_x = batch_indexes // (self.num_slice_per_dim**2)
            batch_indexes_y = batch_indexes % (self.num_slice_per_dim**2)// (self.num_slice_per_dim)
            batch_indexes_z = batch_indexes % (self.num_slice_per_dim**1)
            
            
            index_step = (config.dim_s-self.size_slice_per_dim)//(self.num_slice_per_dim-1)
            
            batch_data_dict['input'] = np.array([self.data_dict['input'][batch_indexes_s[i],:,
                                            batch_indexes_x[i]*index_step:batch_indexes_x[i]*index_step+self.size_slice_per_dim,
                                            batch_indexes_y[i]*index_step:batch_indexes_y[i]*index_step+self.size_slice_per_dim]
                                           for i in range(self.batch_size) ])
            
            batch_data_dict['porosity'] = self.data_dict['porosity'][batch_indexes_s]

            target3d_tmp = np.array([self.data_dict['target3d'][batch_indexes_s[i],:,
                                    batch_indexes_x[i]*index_step:batch_indexes_x[i]*index_step+self.size_slice_per_dim,
                                    batch_indexes_y[i]*index_step:batch_indexes_y[i]*index_step+self.size_slice_per_dim]
                                    for i in range(self.batch_size) ])
            
                 
            target = computing_heat(batch_data_dict['input'],target3d_tmp)

            batch_data_dict['target'] = target.astype(np.float32)
                        
            batch_data_dict['target3d'] = target3d_tmp/100
            
            yield batch_data_dict        
            
class TestDataGenerator_Mass(object):
    
    def __init__(self, data_type,seed=1234):
        self.batch_size = config.batch_size_test
        dir_data = config.dir_data
        dim_s = config.dim_s
        self.data_type = data_type
        
        path_porosity = os.path.join(dir_data, data_type, 'list_infor.mat')
        infor_temp = sio.loadmat(path_porosity)
        list_poros = infor_temp['data'][:,0]
        
        self.num_poros = len(list_poros)
        self.value_range = config.value_range
        
        self.data_dict = {}
        self.data_dict['input'] = np.zeros((self.num_poros,dim_s, dim_s, dim_s)).astype(np.float32)
        self.data_dict['porosity'] = np.zeros((self.num_poros)).astype(np.float32)
        self.data_dict['target3d'] = np.zeros((self.num_poros,dim_s, dim_s, dim_s)).astype(np.float32)
        self.data_dict['target'] = np.zeros((self.num_poros)).astype(np.float32)
        
        for i, poros_tmp in enumerate(list_poros):
        
            path_data = os.path.join(dir_data, data_type, '{}'.format(poros_tmp), 'structure.mat')
            data_temp = sio.loadmat(path_data)
            self.data_dict['input'][i] = data_temp['data'].reshape(dim_s,dim_s,dim_s).T
            self.data_dict['porosity'][i] = poros_tmp
        
            path_data = os.path.join(dir_data, data_type, '{}'.format(poros_tmp), '{}.mat'.format(config.sub_task))
            data_temp = sio.loadmat(path_data)
            self.data_dict['target3d'][i] = data_temp['data'].reshape(dim_s,dim_s,dim_s).T
            self.data_dict['target'][i] = infor_temp['data'][i,1]       
        
        
    def generate_test(self, max_iteration=None):
        
        batch_size = self.batch_size
        
        num_data = len(self.data_dict['porosity'])
        
        iteration = 0
        pointer = 0
        
        while True:
            if iteration == max_iteration:
                break

            # Reset pointer
            if pointer >= num_data:
                break
            
            batch_data_dict = {}
            
            batch_data_dict['input'] = np.array(self.data_dict['input'][pointer:pointer+batch_size])
            batch_data_dict['porosity'] = np.array(self.data_dict['porosity'][pointer:pointer+batch_size])
            target3d_tmp = np.array(self.data_dict['target3d'][pointer:pointer+batch_size])
            target = np.mean(target3d_tmp,(1,2,3))
            target = target/config.dCdX/config.Db
            batch_data_dict['target'] = target.astype(np.float32)
                     
            #data_mass_tmp = np.log(self.value_range*np.clip(data_mass_tmp,0,1)+1)-np.log(1-self.value_range*np.clip(data_mass_tmp,-1,0))
            batch_data_dict['target3d'] = target3d_tmp/config.scale_Mass
            
            pointer += batch_size
            iteration += 1
                      
            yield batch_data_dict
            
class TestDataGenerator_Flow(object):
    
    def __init__(self, data_type,seed=1234):
        self.batch_size = config.batch_size_test
        dir_data = config.dir_data
        dim_s = config.dim_s
        self.data_type = data_type
        
        if data_type=='SpherePacks':
            index_bg = 2
        elif data_type=='QSGS':
            index_bg = 1
        else:
            index_bg = 0
                
        
        path_porosity = os.path.join(dir_data, data_type, 'list_infor.mat')
        infor_temp = sio.loadmat(path_porosity)
        list_poros = infor_temp['data'][:,0]
        
        if data_type=='SpherePacks':
            path_porosity = os.path.join(dir_data, config.type_train+'_2', 'list_infor.mat')
            infor_temp2 = sio.loadmat(path_porosity)
            list_poros2 = infor_temp2['data'][:,0]
            list_poros = np.hstack((list_poros,list_poros2))
            infor_temp['data'] = np.vstack((infor_temp['data'],infor_temp2['data']))


        list_poros = list_poros[index_bg:]        
        self.num_poros = len(list_poros)
        self.value_range = config.value_range
        
        self.data_dict = {}
        self.data_dict['input'] = np.zeros((self.num_poros,dim_s, dim_s, dim_s)).astype(np.float32)
        self.data_dict['porosity'] = np.zeros((self.num_poros)).astype(np.float32)
        self.data_dict['target3d'] = np.zeros((self.num_poros,dim_s, dim_s, dim_s)).astype(np.float32)
        self.data_dict['target'] = np.zeros((self.num_poros)).astype(np.float32)
        
        for i, poros_tmp in enumerate(list_poros):
            
            if i<26-index_bg:
                path_data = os.path.join(dir_data, data_type, '{}'.format(poros_tmp), 'structure.mat')
            else:
                path_data = os.path.join(dir_data, data_type+'_2', '{}'.format(poros_tmp), 'structure.mat')
            data_temp = sio.loadmat(path_data)
            self.data_dict['input'][i] = data_temp['data'].reshape(dim_s,dim_s,dim_s).T
            self.data_dict['porosity'][i] = poros_tmp
            
            if i<26-index_bg:
                path_data = os.path.join(dir_data, data_type, '{}'.format(poros_tmp), '{}.mat'.format(config.sub_task))
            else:
                path_data = os.path.join(dir_data, data_type+'_2', '{}'.format(poros_tmp), '{}.mat'.format(config.sub_task))
            data_temp = sio.loadmat(path_data)
            self.data_dict['target3d'][i] = data_temp['data'].reshape(dim_s,dim_s,dim_s).T
            self.data_dict['target'][i] = infor_temp['data'][i,2]       
        
        
    def generate_test(self, max_iteration=None):
        
        batch_size = self.batch_size
        
        num_data = len(self.data_dict['porosity'])
        
        iteration = 0
        pointer = 0
        
        while True:
            if iteration == max_iteration:
                break

            # Reset pointer
            if pointer >= num_data:
                break
            
            batch_data_dict = {}
            
            batch_data_dict['input'] = np.array(self.data_dict['input'][pointer:pointer+batch_size])
            poros_tmp = np.array(self.data_dict['porosity'][pointer:pointer+batch_size])            
            batch_data_dict['porosity'] = poros_tmp
            target3d_tmp = np.array(self.data_dict['target3d'][pointer:pointer+batch_size])
            target = np.mean(target3d_tmp,(1,2,3))
            target = target*1E7/6
            batch_data_dict['target'] = target.astype(np.float32)
                     
            #data_mass_tmp = np.log(self.value_range*np.clip(data_mass_tmp,0,1)+1)-np.log(1-self.value_range*np.clip(data_mass_tmp,-1,0))
            norm_target3d = compute_norm_Flow(self.data_type, poros_tmp)                
            batch_data_dict['target3d'] = target3d_tmp/norm_target3d[:,None, None, None]
            
            
            pointer += batch_size
            iteration += 1
                      
            yield batch_data_dict
            
class TestDataGenerator_Temp(object):
    
    def __init__(self, data_type,seed=1234):
        self.batch_size = config.batch_size_test
        dir_data = config.dir_data
        dim_s = config.dim_s
        self.data_type = data_type
        
        path_porosity = os.path.join(dir_data, data_type, 'list_infor.mat')
        infor_temp = sio.loadmat(path_porosity)
        list_poros = infor_temp['data'][:,0]
        
        self.num_poros = len(list_poros)
        self.value_range = config.value_range
        
        self.data_dict = {}
        self.data_dict['input'] = np.zeros((self.num_poros,dim_s, dim_s, dim_s)).astype(np.float32)
        self.data_dict['porosity'] = np.zeros((self.num_poros)).astype(np.float32)
        self.data_dict['target3d'] = np.zeros((self.num_poros,dim_s, dim_s, dim_s)).astype(np.float32)
        self.data_dict['target'] = np.zeros((self.num_poros)).astype(np.float32)
        
        for i, poros_tmp in enumerate(list_poros):
        
            path_data = os.path.join(dir_data, data_type, '{}'.format(poros_tmp), 'structure.mat')
            data_temp = sio.loadmat(path_data)
            self.data_dict['input'][i] = data_temp['data'].reshape(dim_s,dim_s,dim_s).T
            self.data_dict['porosity'][i] = poros_tmp
        
            path_data = os.path.join(dir_data, data_type, '{}'.format(poros_tmp), '{}.mat'.format(config.sub_task))
            data_temp = sio.loadmat(path_data)
            self.data_dict['target3d'][i] = data_temp['data'].reshape(dim_s,dim_s,dim_s).T
            self.data_dict['target'][i] = infor_temp['data'][i,1]       
        
        
    def generate_test(self, max_iteration=None):
        
        batch_size = self.batch_size
        
        num_data = len(self.data_dict['porosity'])
        
        iteration = 0
        pointer = 0
        
        while True:
            if iteration == max_iteration:
                break

            # Reset pointer
            if pointer >= num_data:
                break
            
            batch_data_dict = {}
            
            batch_data_dict['input'] = np.array(self.data_dict['input'][pointer:pointer+batch_size])
            batch_data_dict['porosity'] = np.array(self.data_dict['porosity'][pointer:pointer+batch_size])
            target3d_tmp = np.array(self.data_dict['target3d'][pointer:pointer+batch_size])
            target = computing_heat(batch_data_dict['input'],target3d_tmp)
            batch_data_dict['target'] = target.astype(np.float32)
                    
            batch_data_dict['target3d'] = target3d_tmp/100
            
            pointer += batch_size
            iteration += 1
                      
            yield batch_data_dict