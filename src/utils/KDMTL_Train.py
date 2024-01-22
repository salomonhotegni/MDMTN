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

from src.utils.KDMTL_utils import compute_loss

from src.utils.lsuv import lsuv_init

def train_multitask_kdmtl_model(train_loader, val_loader, model,
                          params_init, init_model = True):
    
        search_lambda, data_search = params_init["search_lambda"], params_init["data_search"]
        device, lmbd = params_init["device"], params_init["lmbd"]
        num_tasks, num_outs = params_init["num_tasks"], params_init["num_outs"]
        max_epochs, tol_epochs, num_epochs_search = params_init["num_epochs"], params_init["tol_epochs"], params_init["num_epochs_search"]
        num_model, main_dir, mod_logdir = params_init["num_model"], params_init["main_dir"], params_init["mod_logdir"]
        sing_mod_logdir = params_init["sing_mod_logdir"]
        criterion = params_init["criterion"]
        KDMTL_single = params_init["KDMTL_single"]

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
        
        single_model = {}
        for i in range(num_tasks):
            single_model[i] = KDMTL_single(NUM_OUT = num_outs[i]).to(device)
            SINGLE_MODEL_FILE = str("%s/%s/KDMTL_single_%03d.pth"%(main_dir, sing_mod_logdir, int(i)))
            single_model[i].load_state_dict(torch.load(SINGLE_MODEL_FILE))
        
        lr, lr_sched_coef, lr_sched_step_size = params_init["lr"], params_init["lr_sched_coef"], params_init["lr_sched_step_size"]
        a_lr, a_weight_decay = params_init["a_lr"], params_init["a_weight_decay"]
        
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_sched_step_size, gamma=lr_sched_coef)
        params = []
        for i in range(num_tasks):
            params += model.adaptors[i].parameters()
        adaptor_optimizer = optim.Adam(params, lr=a_lr, weight_decay=a_weight_decay)

        act_bst_accu = 0.0
        TRAIN_LOSS = []
        VAL_ACCU = []
        
        if search_lambda[0]:
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

                outputs, shared_feats = model(data)
                #print("len(outputs) = ", len(outputs))
                feats_single = []
                for i in range(num_tasks):
                    with torch.no_grad():
                        _, feat_ti = single_model[i](data)
                        feats_single.append(feat_ti)
                loss = compute_loss(model, outputs, targets, criterion, shared_feats, feats_single, lmbd, device)
                torch.autograd.set_detect_anomaly(True)
                model.zero_grad()
                adaptor_optimizer.zero_grad()
                loss.backward()
                adaptor_optimizer.step()
                optimizer.step()
                
                train_loss = train_loss + loss.item()

                if batch_idx % 50 == 0:
                    print('[BATCH ({}) ({:.0f}%)]\tLoss: {:.6f}'.format(
                                batch_idx+1, 100. * (batch_idx+1) / len(train_loader),  loss.item()))

            train_loss = train_loss/len(train_loader.dataset)
            TRAIN_LOSS.append(train_loss)
            scheduler.step()
            
            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(val_loader):
                    all_targets = []
                    for j in range(num_tasks):
                        target_i = target[j]
                        all_targets.append(target_i)
                    all_targets = torch.stack(all_targets).transpose(0, 1)
                    data, targets = data.to(device), all_targets.to(device, dtype=torch.int64)

                    outputs, _ = model(data)
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

def full_training_kdmtl(train_loader, val_loader, model, params_init, init_model = True):

    data_search = params_init["data_search"]
    search_lambda = params_init["search_lambda"]
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
            _, _, _, model_val_accuracy = train_multitask_kdmtl_model(data_search, val_loader, model,
                                                                params_init, init_model = init_model)
            if Best_lmbd_accu < model_val_accuracy:
                Best_lmbd = lmbd
                Best_lmbd_accu = model_val_accuracy
        print("Best Lambda value: ", Best_lmbd)
    
    print("###############################")
    print(f"#### TRAINING started ! ####")
    print("###############################")
    search_lambda[0] = False
    params_init["lmbd"] = Best_lmbd
    model, TRAIN_LOSS, VAL_ACCU, model_val_accuracy = train_multitask_kdmtl_model(train_loader, val_loader, model, params_init, init_model = init_model)
    
    return model, TRAIN_LOSS, VAL_ACCU, model_val_accuracy




###################################
####### SINGLE-TASK Training ######
###################################

def train_single_model_kdmtl(train_loader, val_loader, model, params_sg):
    
    device, ind_task = params_sg["device"], params_sg["ind_task"]
    num_epochs, tol_epochs = params_sg["num_epochs"], params_sg["tol_epochs"]
    main_dir, mod_logdir = params_sg["main_dir"], params_sg["mod_logdir"]

    # number of epochs
    NUM_EPOCHS = num_epochs

    # file names
    if not os.path.exists("%s/%s"%(main_dir, mod_logdir)):
            os.makedirs("%s/%s"%(main_dir, mod_logdir))
    MODEL_FILE = str("%s/%s/KDMTL_single_%03d.pth"%(main_dir, mod_logdir, ind_task))

    # global variables
    if tol_epochs is None: tol_epochs = num_epochs
    violation_epochs = 0
    criterion = params_sg["criterion"] 
    
    lr = params_sg["lr"]
    optimizer = optim.Adam(model.parameters(), lr=lr)
    lr_sched_coef = params_sg["lr_sched_coef"]
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_sched_coef)
    
    # global variables
    g_step = 0
    max_correct = 0
    model = model.to(device)

    # training and evaluation loop
    for epoch in range(NUM_EPOCHS):
        print("----------------------------------")
        print(f"-------- EPOCH {epoch+1} --------")
        print("----------------------------------")
        #--------------------------------------------------------------------------#
        # train process                                                            #
        #--------------------------------------------------------------------------#
        model.train()
        train_loss = 0
        train_corr = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target[ind_task].to(device, dtype=torch.int64)
            optimizer.zero_grad()
            output, _ = model(data)
            loss = criterion(output, target)
            train_pred = output.argmax(dim=1, keepdim=True)
            train_corr += train_pred.eq(target.view_as(train_pred)).sum().item()
            train_loss += criterion(output, target, reduction='sum').item()
            loss.backward()
            optimizer.step()
            g_step += 1

            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{:05d}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
        train_loss /= len(train_loader.dataset)
        #train_accuracy = 100 * train_corr / len(train_loader.dataset)
        scheduler.step()  
        #--------------------------------------------------------------------------#
        # test process                                                             #
        #--------------------------------------------------------------------------#
        model.eval()
        val_loss = 0
        correct = 0
        total_pred = np.zeros(0)
        total_target = np.zeros(0)
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target[ind_task].to(device,  dtype=torch.int64)
                output, _ = model(data)
                val_loss += criterion(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                total_pred = np.append(total_pred, pred.cpu().numpy())
                total_target = np.append(total_target, target.cpu().numpy())
                correct += pred.eq(target.view_as(pred)).sum().item()
            if(max_correct < correct):
                torch.save(model.state_dict(), MODEL_FILE)
                max_correct = correct
                print("Best accuracy! correct images: %5d"%correct)

        #--------------------------------------------------------------------------#
        # output                                                                   #
        #--------------------------------------------------------------------------#
        val_loss /= len(val_loader.dataset)
        val_accuracy = 100 * correct / len(val_loader.dataset)
        best_val_accuracy = 100 * max_correct / len(val_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%) (best: {:.2f}%)\n'.format(val_loss, correct, len(val_loader.dataset), val_accuracy, best_val_accuracy))

    model.load_state_dict(torch.load(MODEL_FILE))
        
    return model, train_loss, val_accuracy
