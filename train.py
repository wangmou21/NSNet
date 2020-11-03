# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 19:23:29 2020

@author: Silence
"""

import os
import time
import logging
import torch
import torch.optim as optim

from models import MassNET
from data_generator import DataGenerator_Mass, TestDataGenerator_Mass
from utils import read_data, create_folder, create_logging, move_data_to_gpu, loss_func_3d

import config







def train():
    
    sub_task = config.sub_task
    workspace = config.workspace
    iteration_max = config.iteration_max
        
    logs_dir = os.path.join(workspace, 'logs', 
        'train_{}_{}'.format(config.type_train, sub_task))
    create_logging(logs_dir, 'w')
        
    checkpoints_dir = os.path.join(workspace, 'checkpoints', 
        'checkpoints_{}_{}'.format(config.type_train, sub_task))
    create_folder(checkpoints_dir)   
    
    
    validate_statistics_path_SpherePacks = os.path.join(workspace, 'statistics', 
            'statistics_{}_{}_ball'.format(config.type_train, sub_task), 'statistics.pickle')  
    create_folder(os.path.dirname(validate_statistics_path_SpherePacks))
        
    validate_statistics_path_fiber = os.path.join(workspace, 'statistics', 
            'statistics_{}_{}_fiber'.format(config.type_train, sub_task),'statistics.pickle')  
    create_folder(os.path.dirname(validate_statistics_path_fiber))
        
    validate_statistics_path_QSGS = os.path.join(workspace, 'statistics', 
            'statistics_{}_{}_QSGS'.format(config.type_train, sub_task), 'statistics.pickle')  
    create_folder(os.path.dirname(validate_statistics_path_QSGS))
    
    
    logging.info('Task:{}'.format(sub_task))
        
    # Model
    Model = eval(sub_task+'NET')
    model = Model()
    #print(model)
    
    if config.cuda:
        logging.info('Using GPU.') 
        model.cuda()
    if config.sys == "Linux":
        model = torch.nn.DataParallel(model)
        
    criterion = torch.nn.L1Loss()
        
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999),
                            eps=1e-08, weight_decay=0., amsgrad=True)
    
    # Data generator
    
    DataGenerator = eval('DataGenerator_'+sub_task)
    data_generator = DataGenerator()
    TestDataGenerator = eval('TestDataGenerator_'+sub_task)
    data_generator_SpherePacks = TestDataGenerator_Mass(data_type='SpherePacks')
    data_generator_QSGS = TestDataGenerator_Mass(data_type='QSGS')
    data_generator_Fiber = TestDataGenerator_Mass(data_type='Fiber')

    train_bgn_time = time.time()
    iteration = 0
    prev_val_loss = float("inf")
    # Train on mini batches
    for batch_data_dict in data_generator.generate_train():

        if iteration % 100 == 0:
            logging.info('------------------------------------')
            logging.info('Iteration: {}'.format(iteration))

            train_fin_time = time.time()
            
            
        # Save model
        if iteration % 100 == 0 and iteration > 1000:
            checkpoint = {
                'iteration': iteration, 
                'model': model.state_dict(), 
                'optimizer': optimizer.state_dict()}

            checkpoint_path = os.path.join(
                checkpoints_dir, '{}_iterations.pth'.format(iteration))
                
            torch.save(checkpoint, checkpoint_path)
            logging.info('Model saved to {}'.format(checkpoint_path))   
        
        # Move data to GPU
        for key in batch_data_dict.keys():
            if key in ['input', 'target', 'porosity','target3d']:
                batch_data_dict[key] = move_data_to_gpu(batch_data_dict[key], config.cuda)
                
        # Train
        model.train()
        batch_output_3d, batch_output = model(batch_data_dict['input'])
            
        loss_target3d = loss_func_3d(output=batch_output_3d, target=batch_data_dict['target3d'], refer=batch_data_dict['input'], l_type='a')    
            
            # # Adjust learning rate (halving)
            # if half_lr:
            #     if val_loss >= prev_val_loss:
            #         val_no_impv += 1
            #         if val_no_impv >= 2:
            #             halving = True
            #         if self.val_no_impv >= 5 and self.early_stop:
            #             print("No imporvement for 10 epochs, early stopping.")
            #             break
            #     else:
            #         self.val_no_impv = 0
            
            # prev_val_loss = val_loss
        loss_target = criterion(batch_output, batch_data_dict['target'])
        loss = loss_target3d

        logging.info('iteration: {:d}, loss_t: {:.6f}, loss_t3d: {:.8f}'.format(iteration, loss_target, loss_target3d))
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Stop learning
        if iteration == config.iteration_max:
            break
            
        iteration += 1






if __name__ == '__main__':

    train()