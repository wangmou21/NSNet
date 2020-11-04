# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 11:39:08 2020

@author: Silence
"""

import platform


## Set the path
sys = platform.system()
if sys == 'Windows':
    dir_data = 'L:/NSNet/data/'
    workspace = 'L:/NSNet/space'
elif sys == "Linux":
    dir_data = '/home/mouwang/Work/NSNet/data/'
    workspace = '/home/mouwang/Work/NSNet/space'
    
## Set the task and data
sub_task = 'Mass'                   # option for task, 'Temp' for temperature 
type_train = 'SpherePacks'          # option for training data Fiber
type_test = 'SpherePacks'

## Data setting
size_slice_per_dim = 104
num_slice_per_dim = 12

dim_s = 200

scale_Temp = 100

ratio_train_test = 0.92
batch_size = 20
batch_size_test = 1
iteration_max = 30000

cuda = True
reduce_lr = True
half_lr = True
early_stop = True

cuda = True

# Mass
dCdX = 1/(dim_s-1)
Db = 1/6
scale_Mass = 1E-2
value_range_Mass = 1E-4
