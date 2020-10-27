# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 11:39:08 2020

@author: Silence
"""

import platform


## Set the path
sys = platform.system()
if sys == 'Windows':
    dir_data = 'J:/CFD_Temp_final/data/'
    workspace = 'J:/CFD_Temp_final/space'
elif sys == "Linux":
    dir_data = '/home/mouwang/Work/NSNet/data/'
    workspace = '/home/mouwang/Work/NSNet/space'