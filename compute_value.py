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

data_type = 'QSGS'              # 'SpherePacks', 'QSGS'



dir_data = config.dir_data


# Setting the parameter


num_dim = 200



TIN, TOUT = 100, 25
DAORE1, DAORE2 = 0.023, 100
DD=3/20.
NX, NY, NZ = num_dim, num_dim, num_dim

path_poros = os.path.join(dir_data, data_type, 'list_poros.mat')
list_porosity = sio.loadmat(path_poros)
list_porosity = np.squeeze(list_porosity['list_poros'])

list_keff = np.zeros(list_porosity.shape)
list_diff = np.zeros(list_porosity.shape)
list_flow = np.zeros(list_porosity.shape)
i = 0
for poros_tmp in list_porosity:

    path_data = os.path.join(dir_data,data_type, '{}'.format(poros_tmp), 'structure.mat')
    data_temp = sio.loadmat(path_data)
    data_s = data_temp['data'].reshape(num_dim,num_dim,num_dim).T
    
    DAORE = np.zeros((NX,NY,NZ),dtype=float)
    DAORE[data_s==0] = DAORE1
    DAORE[data_s==1] = DAORE2
    
    del data_s
    
    path_data = os.path.join(dir_data,data_type, '{}'.format(poros_tmp), 'Temp.mat')
    data_temp = sio.loadmat(path_data)
    data_t = data_temp['data'].reshape(num_dim,num_dim,num_dim).T
    
    # ------------------------计算终了时的热流密度------------------
    T1 = np.pad(data_t,((1,1),(0,0),(0,0)),'edge')
    QT = np.mean((T1[0:NX,1:NY-1,1:NZ-1]-T1[2:NX+2,1:NY-1,1:NZ-1])*DAORE[0:NX,1:NY-1,1:NZ-1],(2,1))/DD/2.
    #QT = np.mean((T1[0:NX]-T1[2:NX+2])*DAORE[0:NX,0:NY,0:NZ],(2,1))/DD/2
    list_keff[i] = DD*np.sum(QT)/(TIN-TOUT)
    
    
    path_data = os.path.join(dir_data,data_type, '{}'.format(poros_tmp), 'Mass.mat')
    data_temp = sio.loadmat(path_data)
    data_m = data_temp['data'].reshape(num_dim,num_dim,num_dim)
    
    list_diff[i] = np.mean(data_m,(2,1,0))*6*(num_dim-1)
    
    path_data = os.path.join(dir_data,data_type, '{}'.format(poros_tmp), 'Flow.mat')
    data_temp = sio.loadmat(path_data)
    data_m = data_temp['data'].reshape(num_dim,num_dim,num_dim)
    
    list_flow[i] = np.mean(data_m,(2,1,0))*1E7/6

    
    
    i +=1
    
list_infor = np.vstack((list_porosity, list_diff, list_flow, list_keff)).T
path_infor = os.path.join(dir_data, data_type, 'list_infor.mat')
sio.savemat(path_infor, {'data':list_infor})





