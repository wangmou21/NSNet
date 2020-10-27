# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 11:39:08 2020

@author: Silence
"""

import platform


## Set the path
sys = platform.system()
if sys == 'Windows':
    dir_data = 'L:/NSNet/NSNet/data/'
    workspace = 'L:/NSNet/NSNet/space'
elif sys == "Linux":
    dir_data = '/home/mouwang/Work/NSNet/data/'
    workspace = '/home/mouwang/Work/NSNet/space'
    
## Set the task and data
sub_task = 'Temp'                   # option for task, 'Temp' for temperature 
type_train = 'SpherePacks'          # option for training data
type_test = 'SpherePacks'

## Data setting
size_slice_per_dim = 104
num_slice_per_dim = 12

dim_s = 200

scale_Temp = 100

ratio_train_test = 0.92
batch_size = 1
batch_size_test = 1
iteration_max = 30000

cuda = True
reduce_lr = True