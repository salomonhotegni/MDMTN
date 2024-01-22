import sys
import os
import argparse
import numpy as np 
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from PIL import Image
import cv2

###################################
####### MULTI-TASK Training #######
###################################


def compute_loss(outputs, targets, criterion, lmbd, device):
    
        num_tasks = len(outputs)
        train_loss = []
        for i in range(num_tasks):
            loss_ti = criterion(outputs[i], targets[:, i]).to(device)
            train_loss.append(loss_ti)
        loss = torch.mean(sum(lmbd[i] * train_loss[i] for i in range(num_tasks)))
        
        return loss
    

from src.utils.lsuv import lsuv_init

def train_multitask_mtan_model(train_loader, val_loader, model,
                          params_init, init_model = True):
    
        search_lambda, search_lr, data_search = params_init["search_lambda"], params_init["search_lr"], params_init["data_search"]
        device, lmbd = params_init["device"], params_init["lmbd"]
        num_tasks, num_outs = params_init["num_tasks"], params_init["num_outs"]
        max_epochs, tol_epochs, num_epochs_search = params_init["num_epochs"], params_init["tol_epochs"], params_init["num_epochs_search"]
        num_model, main_dir, mod_logdir = params_init["num_model"], params_init["main_dir"], params_init["mod_logdir"]
        criterion = params_init["criterion"]

        # file names
        if not os.path.exists("%s/%s"%(main_dir, mod_logdir)):
                os.makedirs("%s/%s"%(main_dir, mod_logdir))
        MODEL_FILE = str("%s/%s/model%03d.pth"%(main_dir, mod_logdir, num_model))

        # global variables
        violation_epochs = 0
        
        model = model.to(device)
        # Initialize the networks
        if init_model:
            model = lsuv_init(model, train_loader, needed_std=1.0, std_tol=0.1,
                                  max_attempts=10, do_orthonorm=True, device=device) 
        
        lr, lr_sched_coef, lr_sched_step_size = params_init["lr"], params_init["lr_sched_coef"], params_init["lr_sched_step_size"]
        
        optimizer = optim.Adam(model.mt_parameters(), lr=lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_sched_step_size, gamma=lr_sched_coef)

        act_bst_accu = 0.0
        TRAIN_LOSS = []
        VAL_ACCU = []
        
        if (search_lambda[0] or search_lr[0]):
            max_epochs = num_epochs_search
        
        for i in range(max_epochs):
            print("----------------------------------")
            print(f"-------- EPOCH {i+1} --------")
            print("----------------------------------")
            #--------------------------------------------------------------------------#
            # Train process                                                             #
            #--------------------------------------------------------------------------#
            model.train()
            train_loss = 0.0
            for batch_idx, (data, target) in enumerate(train_loader):
                all_targets = []
                for j in range(num_tasks):
                    target_i = target[j]
                    all_targets.append(target_i)
                all_targets = torch.stack(all_targets).transpose(0, 1)
                data, targets = data.to(device), all_targets.to(device, dtype=torch.int64)

                outputs = model(data)
                loss = compute_loss(outputs, targets, criterion, lmbd, device)
                torch.autograd.set_detect_anomaly(True)
                model.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss = train_loss + loss.item()

                if batch_idx % 50 == 0:
                    print('[BATCH ({}) ({:.0f}%)]\tLoss: {:.6f}'.format(
                                batch_idx+1, 100. * (batch_idx+1) / len(train_loader),  loss.item()))

            train_loss = train_loss/len(train_loader.dataset)
            TRAIN_LOSS.append(train_loss)
            scheduler.step()
            #--------------------------------------------------------------------------#
            # Evaluation process                                                             #
            #--------------------------------------------------------------------------#
            model.eval()
            val_CORR = torch.zeros(num_tasks)
            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(val_loader):
                    all_targets = []
                    for j in range(num_tasks):
                        target_i = target[j]
                        all_targets.append(target_i)
                    all_targets = torch.stack(all_targets).transpose(0, 1)
                    data, targets = data.to(device), all_targets.to(device, dtype=torch.int64)

                    outputs = model(data)
                    actu_test_CORR = []
                    for  j in range(num_tasks):
                        test_pred_j = outputs[j].argmax(dim=1, keepdim=True)
                        test_corr_j = test_pred_j.eq(targets[:, j].view_as(test_pred_j)).sum().item()
                        actu_test_CORR.append(test_corr_j)
                    actu_test_CORR = torch.tensor(actu_test_CORR)
                    val_CORR = val_CORR + actu_test_CORR
            
            val_accuracy = 100 * val_CORR / len(val_loader.dataset)
            print('\nTest set: Average Accuracy: ({:.2f}%)\n'.format(
                    val_accuracy.mean().item()))
            ######################################

            #--------------------------------------------------------------------------#
            # output                                                                   #
            #--------------------------------------------------------------------------#
            # print("val_accuracy before: ", val_CORR / len(val_loader.dataset))
            # val_accuracy = 100 * val_CORR / len(val_loader.dataset)
            # print("val_accuracy after: ", val_accuracy)
            VAL_ACCU.append(val_accuracy)
            if act_bst_accu < val_accuracy.mean().item():
                act_bst_accu = val_accuracy.mean().item()
                violation_epochs = 0
                torch.save(model.state_dict(), MODEL_FILE)
                print("Best global performance (Accuracy)!")
                print('\nValidation set: Average Accuracy: {:.2f}%    (Best: {:.2f}%)\n'.format(
                        val_accuracy.mean().item(), act_bst_accu))
                if (num_tasks <= 3):# print out per-task accuracies for problem with small number of main tasks
                    for i in range(num_tasks):
                        print("Accuracy Task {}: {:.2f}%".format(i+1, val_accuracy[i].item()))
            else:
                print('\nValidation set: Average Accuracy: ({:.2f}%)\n'.format(
                        val_accuracy.mean().item()))
                violation_epochs += 1
                if tol_epochs is not None:
                    if (violation_epochs > tol_epochs):
                        print(f"No improvement in accuracy after {tol_epochs} more epochs. END!!!")  
                        break;
                        
        model_val_accuracy = act_bst_accu
        model.load_state_dict(torch.load(MODEL_FILE))
        return model, TRAIN_LOSS, VAL_ACCU, model_val_accuracy


import random

def full_training_mtan(train_loader, val_loader, model, params_init, init_model = True):

    data_search = params_init["data_search"]
    search_lambda = params_init["search_lambda"]
    search_lr = params_init["search_lr"]
    num_classes = params_init["num_outs"]
    Best_lmbd = search_lambda[1][0]
    if search_lambda[0]:
        print("################################")
        print(f"#### Searching for Lambda ... ####")
        print("################################")

        Best_lmbd_accu = 0.0
        for lmbd in search_lambda[1]:
            print(f"******** For Lambda = {lmbd} ********")
            params_init["lmbd"] = lmbd
            _, _, _, model_val_accuracy = train_multitask_mtan_model(data_search, val_loader, model,
                                                                params_init, init_model = init_model)
            if Best_lmbd_accu < model_val_accuracy:
                Best_lmbd = lmbd
                Best_lmbd_accu = model_val_accuracy
        print("Best Lambda value: ", Best_lmbd)
        
    params_init["lmbd"] = Best_lmbd
    search_lambda[0] = False
    
    if search_lr[0]:
        print("################################")
        print(f"#### Turning lr ... ####")
        print("################################")

        Best_lr_accu = 0.0
        for lr in search_lr[1]:
            print(f"******** For lr = {lr} ********")
            params_init["lr"] = lr
            _, _, _, model_val_accuracy = train_multitask_mtan_model(data_search, val_loader, model,
                                                                params_init, init_model = init_model)
            if Best_lr_accu < model_val_accuracy:
                Best_lr = lr
                Best_lr_accu = model_val_accuracy
        print("Best lr value: ", Best_lr)
        
    params_init["lr"] = Best_lr
    search_lr[0] = False
    
    print("###############################")
    print(f"#### TRAINING started ! ####")
    print("###############################")
    model, TRAIN_LOSS, VAL_ACCU, model_val_accuracy = train_multitask_mtan_model(train_loader, val_loader, model, params_init, init_model = init_model)
    
    print("Lambda used: ", Best_lmbd)
    print("Learning rate used: ", Best_lr)
    return model, TRAIN_LOSS, VAL_ACCU, model_val_accuracy