import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

###################################
######## MULTI-TASK MODELS ########
###################################

class KDMTL_Network_I(nn.Module):
    def __init__(self, num_classes):
        super(KDMTL_Network_I, self).__init__()
        self.num_tasks = len(num_classes) + 1
        self.num_classes = num_classes

        # Shared Networks            
        self.Shared_block = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(10, 20, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(320, 50),
            nn.ReLU(inplace=True),
        )
        
        # Task-specific Output Networks
        self.task_blocks = nn.ModuleList()
        for i in range(self.num_tasks-1):
            task_output = nn.Sequential(
                nn.Linear(50, num_classes[i]),
            )
            self.task_blocks.append(task_output)
        
        # Adaptors
        self.adaptors = nn.ModuleList()
        for i in range(self.num_tasks-1):
            adaptor = nn.Sequential(
                nn.Linear(50, 50),
                nn.ReLU(inplace=True),
            )
            self.adaptors.append(adaptor)
    
    def forward(self, x):
        # Shared output
        sh_out = self.Shared_block(x)
        
        # Task-specific output
        task_outputs = []
        #task_feats = []
        for t in range(self.num_tasks-1):
            sh_out_t = sh_out.clone()
            task_out = self.task_blocks[t](sh_out_t)
            task_outputs.append(F.log_softmax(task_out, dim=1))
        
        return task_outputs, sh_out

    def zero_grad(self):
        self.Shared_block.zero_grad()
        for t in range(self.num_tasks-1):
            self.adaptors[t].zero_grad()
            self.task_blocks[t].zero_grad()  

    def mt_parameters(self):
        params = list(self.Shared_block.parameters())
        for t in range(self.num_tasks-1):
            params += list(self.task_blocks[t].parameters())
            params += list(self.adaptors[t].parameters())
        return params
    
######################################################################
    
class KDMTL_Network_II(nn.Module):
    def __init__(self, num_classes):
        super(KDMTL_Network_II, self).__init__()
        self.num_tasks = len(num_classes) + 1
        self.num_classes = num_classes
        #self.batch_size = batch_size

        # Shared Networks            
        self.Shared_block = nn.Sequential(
            nn.Conv2d(3, 16, 3), 
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3), 
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),    
            nn.Linear(1152, 32), 
            nn.ReLU(),
        )
        
        # Task-specific Output Networks
        self.task_blocks = nn.ModuleList()
        for i in range(self.num_tasks-1):
            task_output = nn.Sequential(
                nn.Linear(32, num_classes[i]),
            )
            self.task_blocks.append(task_output)
        
        # Adaptors
        self.adaptors = nn.ModuleList()
        for i in range(self.num_tasks-1):
            adaptor = nn.Sequential(
                nn.Linear(32, 32),
                nn.ReLU(inplace=True),
            )
            self.adaptors.append(adaptor)
    
    def forward(self, x):
        # Shared output
        sh_out = self.Shared_block(x)
        
        # Task-specific output
        task_outputs = []
        task_feats = []
        for t in range(self.num_tasks-1):
            sh_out_t = sh_out.clone()
            task_out = self.task_blocks[t](sh_out_t)
            task_outputs.append(F.log_softmax(task_out, dim=1))
        
        return task_outputs, sh_out

    def zero_grad(self):
        self.Shared_block.zero_grad()
        for t in range(self.num_tasks-1):
            self.adaptors[t].zero_grad()
            self.task_blocks[t].zero_grad()  

    def mt_parameters(self):
        params = list(self.Shared_block.parameters())
        for t in range(self.num_tasks-1):
            params += list(self.task_blocks[t].parameters())
            params += list(self.adaptors[t].parameters())
        return params
    
###################################
######## SINGLE-TASK MODELS #######
###################################


class KDMTL_SingleNetwork_I(nn.Module):
    def __init__(self, NUM_OUT = 10):
        super(KDMTL_SingleNetwork_I, self).__init__()
        self.conv_sp1 = nn.Conv2d(1, 10, kernel_size=5)
        self.max_pool_sp1 = nn.MaxPool2d(kernel_size=2)
        self.conv_sp2 = nn.Conv2d(10, 20, kernel_size=5)
        self.max_pool_sp2 = nn.MaxPool2d(kernel_size=2)
        self.fc1_sp1 = nn.Linear(320, 50)
        self.fc1_sp2 = nn.Linear(50, 10)

    def forward(self, x):
        conv_sp1 = F.relu(self.max_pool_sp1(self.conv_sp1(x)))
        conv_sp2 = F.relu(self.max_pool_sp2(self.conv_sp2(conv_sp1)))
        conv_sp2 = torch.flatten(conv_sp2, 1)
        fc1_sp1 = F.relu(self.fc1_sp1(conv_sp2))
        return F.log_softmax(self.fc1_sp2(fc1_sp1), dim=1), fc1_sp1
    
########################################################################

class KDMTL_SingleNetwork_II(nn.Module):
    def __init__(self, NUM_OUT = 10):
        super(KDMTL_SingleNetwork_II, self).__init__()
        self.conv_sp1 = nn.Conv2d(3, 16, 3)
        self.Bnorm_sp1 = nn.BatchNorm2d(16)
        self.conv_sp2 = nn.Conv2d(16, 32, 3)
        self.Bnorm_sp2 = nn.BatchNorm2d(32)
        self.fc_sp1 = nn.Linear(1152, 32)
        self.fc_sp2 = nn.Linear(32, NUM_OUT)
        
        self.MaxPool_sp = nn.MaxPool2d(kernel_size=2)
        
    def forward(self, x):
        conv_sp1 = F.relu(self.Bnorm_sp1(self.MaxPool_sp(self.conv_sp1(x))))
        conv_sp2 = F.relu(self.Bnorm_sp2(self.MaxPool_sp(self.conv_sp2(conv_sp1))))
        conv_sp2 = torch.flatten(conv_sp2, 1)
        fc_sp1 = F.relu(self.fc_sp1(conv_sp2))
        return F.log_softmax(self.fc_sp2(fc_sp1), dim=1), fc_sp1
