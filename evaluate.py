# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 20:34:06 2020

@author: Silence
"""

import os
import pickle
import numpy as np
import logging
import datetime
import torch

from sklearn.metrics import mean_absolute_error


import config

from utils import move_data_to_gpu, loss_mre, loss_mre3d, append_to_dict, compute_norm_Flow, computing_heat


class Evaluator(object):
    def __init__(self, model, data_generator):
        '''Evaluator to evaluate prediction performance. 
        
        Args: 
          model: object
          data_generator: object
        '''
        
        self.model = model
        self.data_generator = data_generator


    def evaluate(self, data_type, max_iteration=None):
        '''Evaluate the performance. 
        
        Args: 
          data_type: 'train' | 'validate'
          max_iteration: None | int, maximum iteration to run to speed up evaluation
          verbose: bool
        '''

        generate_func = self.data_generator.generate_test(max_iteration=max_iteration)
        
        # Forward
        output_dict = forward(
            model=self.model, 
            generate_func=generate_func, 
            return_input=True,
            return_target=True)
        
        error_mre = loss_mre(output= output_dict['output'], target=output_dict['target'])
        error_mae3d = loss_mre3d(output=output_dict['output3d'], target=output_dict['target3d'], refer=output_dict['input'], l_type='a')
        error_mre3d = loss_mre3d(output=output_dict['output3d'], target=output_dict['target3d'], refer=output_dict['input'], l_type='r')
     
        
        logging.info('Data type: {}'.format(data_type))
        logging.info('Relative error of coefficient: {:.4f}'.format(error_mre))
        logging.info('Absolute error of field: {:.5f}'.format(error_mae3d))
        logging.info('Relative error of field: {:.4f}'.format(error_mre3d))

        statistics = {
            'mre': error_mre,
            'mae3d': error_mae3d, 
            'mre3d': error_mre3d
            }

        return statistics



class StatisticsContainer(object):
    def __init__(self, statistics_path):
        '''Container of statistics during training. 
        
        Args:
          statistics_path: string, path to write out
        '''
        self.statistics_path = statistics_path

        self.backup_statistics_path = '{}_{}.pickle'.format(
            os.path.splitext(self.statistics_path)[0], 
                datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

        # Statistics
        self.statistics_dict = {'data': []}

    def append_and_dump(self, iteration, statistics):
        '''Append statistics to container and dump the container. 
        
        Args:
          iteration: int
          statistics: dict of statistics
        '''
        statistics['iteration'] = iteration
        self.statistics_dict['data'].append(statistics)

        pickle.dump(self.statistics_dict, open(self.statistics_path, 'wb'))
        pickle.dump(self.statistics_dict, open(self.backup_statistics_path, 'wb'))
        logging.info('Dump statistics to {}'.format(self.statistics_path))
        
def forward(model, generate_func, return_input=False, return_target=False):
    '''Forward data to model in mini-batch. 
    
    Args: 
      model: object
      generate_func: function
      cuda: bool
      return_input: bool
      return_target: bool
      max_validate_num: None | int, maximum mini-batch to forward to speed up validation
    '''
    output_dict = {}
    
    # Evaluate on mini-batch
    for batch_data_dict in generate_func:

        # Predict
        batch_feature = move_data_to_gpu(batch_data_dict['input'], config.cuda)
        
        with torch.no_grad():
            model.eval()
            batch_output_3d = model(batch_feature)
            if config.sub_task == 'Flow':
                norm_target3d = compute_norm_Flow(config.type_train, batch_data_dict['porosity'])
                batch_output = torch.mean(batch_output_3d,(3,2,1))*norm_target3d*1E7/6
            elif config.sub_task == 'Temp':
                batch_output = computing_heat(batch_data_dict['input'],batch_output_3d*100)  
      
        append_to_dict(output_dict, 'output3d', batch_output_3d.data.cpu().numpy())
        if config.sub_task == 'Temp':
            append_to_dict(output_dict, 'output', batch_output)
        else:
            append_to_dict(output_dict, 'output', batch_output.data.cpu().numpy())
            
        if return_input:
            append_to_dict(output_dict, 'input', batch_data_dict['input'])
            
        if return_target:
            if 'target' in batch_data_dict.keys():
                append_to_dict(output_dict, 'target', batch_data_dict['target'])
            if 'target3d' in batch_data_dict.keys():
                append_to_dict(output_dict, 'target3d', batch_data_dict['target3d'])
                
                
    for key in output_dict.keys():
        output_dict[key] = np.concatenate(output_dict[key], axis=0)

    return output_dict    
