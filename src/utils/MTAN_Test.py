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
######## MULTI-TASK testing #######
###################################

# Define function for testing a model
def test_multitask_mtan_model(test_loader, model, params_init, load = True):
    
        num_model, device = params_init["num_model"], params_init["device"]
        num_tasks, main_dir, mod_logdir = params_init["num_tasks"], params_init["main_dir"] , params_init["mod_logdir"] 
    
        # Set model to evaluation mode
        model1 = model.to(device)
        if load:
            if not os.path.exists("%s/%s"%(main_dir, mod_logdir)):
                    os.makedirs("%s/%s"%(main_dir, mod_logdir))
            MODEL_FILE = str("%s/%s/model%03d.pth"%(main_dir, mod_logdir, num_model))
            model1.load_state_dict(torch.load(MODEL_FILE))

        model.eval()
        test_CORR = torch.zeros(num_tasks)
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
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
                test_CORR = test_CORR + actu_test_CORR
        
        test_accuracy = 100 * test_CORR / len(test_loader.dataset)
        perc_wrong_pred = 100-test_accuracy
        print('\nTest set: Average Accuracy: ({:.2f}%)\n'.format(
                test_accuracy.mean().item()))
        #if (num_tasks <= 3):
        for i in range(num_tasks):
            print("Accuracy Task {}: {:.04f}%".format(i+1, test_accuracy[i].item()))
                    
        return test_accuracy, perc_wrong_pred