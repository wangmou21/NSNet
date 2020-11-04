# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 19:23:29 2020

@author: Silence
"""

import os
import time
import logging
import numpy as np
import torch
import torch.optim as optim

from models import MassNET
from data_generator import DataGenerator_Mass, TestDataGenerator_Mass
from evaluate import Evaluator, StatisticsContainer
from utils import create_folder, create_logging, move_data_to_gpu, loss_mre, loss_mre3d

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
            'statistics_{}_{}_SpherePacks'.format(config.type_train, sub_task), 'statistics.pickle')  
    create_folder(os.path.dirname(validate_statistics_path_SpherePacks))
        
    validate_statistics_path_Fiber = os.path.join(workspace, 'statistics', 
            'statistics_{}_{}_Fiber'.format(config.type_train, sub_task),'statistics.pickle')  
    create_folder(os.path.dirname(validate_statistics_path_Fiber))
        
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
        
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999),
                            eps=1e-08, weight_decay=0., amsgrad=True)
    
    # Data generator
    
    DataGenerator = eval('DataGenerator_'+sub_task)
    data_generator = DataGenerator()
    TestDataGenerator = eval('TestDataGenerator_'+sub_task)
    data_generator_SpherePacks = TestDataGenerator(data_type='SpherePacks')
    data_generator_QSGS = TestDataGenerator(data_type='QSGS')
    data_generator_Fiber = TestDataGenerator(data_type='Fiber')
    
    evaluator_SpherePacks = Evaluator(model=model, data_generator=data_generator_SpherePacks) 
    evaluator_QSGS = Evaluator(model=model, data_generator=data_generator_QSGS)
    evaluator_Fiber = Evaluator(model=model, data_generator=data_generator_Fiber)
    
    validate_statistics_container_SpherePacks = StatisticsContainer(validate_statistics_path_SpherePacks)
    validate_statistics_container_QSGS = StatisticsContainer(validate_statistics_path_QSGS)
    validate_statistics_container_Fiber = StatisticsContainer(validate_statistics_path_Fiber)
    

    train_bgn_time = time.time()
    iteration = 0
    prev_loss_validate = float("inf")
    val_no_impv = 0
    halving = False
    best_loss_validate = float("inf")
    iters_record = 0
    cv_loss = np.zeros((config.iteration_max//100))
    
    # Train on mini batches
    for batch_data_dict in data_generator.generate_train():
        
        if iteration % 10 == 0:
            logging.info('------------------------------------')
            logging.info('Iteration: {}'.format(iteration))
            logging.info('Current learning rate: {lr:.6f}'.format(lr=optimizer.param_groups[0]['lr']))
            
            train_fin_time = time.time()
            validate_statistics_SpherePacks = evaluator_SpherePacks.evaluate(data_type='validate', max_iteration=None)          
            validate_statistics_container_SpherePacks.append_and_dump(iteration, validate_statistics_SpherePacks)
            
            validate_statistics_QSGS = evaluator_QSGS.evaluate(data_type='validate', max_iteration=None)          
            validate_statistics_container_QSGS.append_and_dump(iteration, validate_statistics_QSGS)
            
            validate_statistics_Fiber = evaluator_Fiber.evaluate(data_type='validate', max_iteration=None)          
            validate_statistics_container_Fiber.append_and_dump(iteration, validate_statistics_Fiber)
            
            
            loss_validate = validate_statistics_SpherePacks['mre']+validate_statistics_QSGS['mre']+validate_statistics_Fiber['mre']
            loss_validate = loss_validate/3
            
            train_time = train_fin_time - train_bgn_time
            validate_time = time.time() - train_fin_time
            
            logging.info('Train time: {:.3f} s, vildate time: {:.3f} s'.format(train_time, validate_time))

            train_bgn_time = time.time()           
            
            # Adjust learning rate (halving)
            if config.half_lr:
                if loss_validate >= prev_loss_validate:
                    val_no_impv += 1
                    logging.info(val_no_impv)
                    if val_no_impv >= 2:
                        halving = True
                    if val_no_impv >= 3 and config.early_stop:
                        logging.info("No imporvement for 1500 iteration, early stopping.")
                        break
                else:
                    val_no_impv = 0
            logging.info(halving)
                        
            if halving:
                optim_state = optimizer.state_dict()
                optim_state['param_groups'][0]['lr'] = optim_state['param_groups'][0]['lr'] / 2.0
                optimizer.load_state_dict(optim_state)
                logging.info('Learning rate adjusted to: {lr:.6f}'.format(lr=optim_state['param_groups'][0]['lr']))
                halving = False
            prev_loss_validate = loss_validate

            # Save the best model
            cv_loss[iters_record] = loss_validate
            if loss_validate < best_loss_validate:
                best_loss_validate = loss_validate
                
                checkpoint = {
                    'iteration': iteration, 
                    'model': model.state_dict(), 
                    'optimizer': optimizer.state_dict()}

                # checkpoint_path = os.path.join(
                #     checkpoints_dir, '{}_iterations.pth'.format(iteration))
                checkpoint_path = os.path.join(checkpoints_dir, 'best.pth')
                
                torch.save(checkpoint, checkpoint_path)
                logging.info('Find better validated model, saving to saved to {}'.format(checkpoint_path))
        
        
        # Move data to GPU
        for key in batch_data_dict.keys():
            if key in ['input', 'target', 'porosity','target3d']:
                batch_data_dict[key] = move_data_to_gpu(batch_data_dict[key], config.cuda)
                
        # Train
        model.train()
        batch_output_3d, batch_output = model(batch_data_dict['input'])
            
        loss_target3d = loss_mre3d(output=batch_output_3d, target=batch_data_dict['target3d'], refer=batch_data_dict['input'], l_type='a')    
            

        loss_target = loss_mre(batch_output, batch_data_dict['target'])
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