# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 21:20:48 2020

@author: Silence
"""

import os
import numpy as np
import pandas as pd
import scipy.io as sio

import config

data_type = 'SpherePacks'             # 'SpherePacks', 'QSGS'



dir_data = config.dir_data


# Setting the parameter


num_dim = 200



TIN, TOUT = 100, 25
DAORE1, DAORE2 = 0.023, 100
DD=3/20.
NX, NY, NZ = num_dim, num_dim, num_dim

path_poros = os.path.join(dir_data, data_type+'_2', 'list_poros.mat')
list_porosity = sio.loadmat(path_poros)
list_porosity = np.squeeze(list_porosity['list_poros'])

list_keff = np.zeros(list_porosity.shape)
list_diff = np.zeros(list_porosity.shape)
list_flow = np.zeros(list_porosity.shape)
i = 0
for poros_tmp in list_porosity:
    
    path_data = os.path.join(dir_data,data_type+'_2', '{}'.format(poros_tmp), 'Flow.mat')
    data_temp = sio.loadmat(path_data)
    data_m = data_temp['data'].reshape(num_dim,num_dim,num_dim)
    
    list_flow[i] = np.mean(data_m,(2,1,0))*1E7/6

    i +=1
    
list_infor = np.vstack((list_porosity, list_diff, list_flow, list_keff)).T
path_infor = os.path.join(dir_data, data_type+'_2', 'list_infor.mat')
sio.savemat(path_infor, {'data':list_infor})





