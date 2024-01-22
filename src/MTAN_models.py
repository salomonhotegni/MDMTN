import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

###################################
######## MULTI-TASK MODEL #########
###################################

class MTAN_Network_I(nn.Module):

    def __init__(self, num_classes):
        super(MTAN_Network_I, self).__init__()
        self.num_tasks = len(num_classes) + 1
        self.num_classes = num_classes
        self.zero_layers = []
        filter = [1, 10, 20, 320, 50]
        
        # Shared Network
        sh_module1 = self.sh_layer_conv([filter[0], filter[1]])
        sh_module2 = self.sh_layer_conv([filter[1], filter[2]])
        sh_module3 = self.sh_layer_lin([filter[3], filter[4]])
        self.Shared_block = nn.ModuleList([sh_module1, sh_module2, sh_module3])
        
        # Task-specific Networks
        self.task_blocks = nn.ModuleList()
        for i in range(self.num_tasks-1):
            task_output = nn.Sequential(
                nn.Linear(50, num_classes[i]),
            )
            self.task_blocks.append(task_output)

        # Task-specific Attention Networks
        att_module = self.att_layer_conv([filter[1], filter[2]])
        feat_module = self.feat_layer_lin([filter[3], filter[4]])
        self.att_blocks = nn.ModuleList()
        for i in range(self.num_tasks-1):
            task_att = nn.ModuleList([att_module, feat_module])
            self.att_blocks.append(task_att)

    ###################################################################################
    def sh_layer_conv(self, channel):
        sh_block = nn.Sequential(
            nn.Conv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(inplace=True),
        )
        return sh_block
    
    def sh_layer_lin(self, channel):
        sh_block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channel[0], channel[1]),
            nn.ReLU(inplace=True),
        )
        return sh_block
    ###################################################################################
    def att_layer_conv(self, channel):
        att_block = nn.Sequential(
            nn.Conv2d(in_channels=channel[0], out_channels=channel[0], kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.Sigmoid(),
        )
        return att_block
        
    def feat_layer_lin(self, channel):
        feat_block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channel[0], channel[0]),
            nn.ReLU(inplace=True),
            nn.Linear(channel[0], channel[1]),
            nn.Sigmoid(),
        )
        return feat_block
    ###################################################################################
        
    def forward(self, x):
        # Shared output
        sh_out1 = self.Shared_block[0](x)
        sh_out2 = self.Shared_block[1](sh_out1)
        sh_out3 = self.Shared_block[2](sh_out2)
        sh_out = [sh_out1, sh_out2, sh_out3]
        
        # Task-specific output
        task_outputs = []
        for i in range(self.num_tasks-1):
            t_out1 = self.att_blocks[i][0](sh_out[0])
            t_out2 = sh_out[1]*t_out1
            t_out3 = sh_out[2]*self.att_blocks[i][1](t_out2)
            task_out = self.task_blocks[i](t_out3)
            task_outputs.append(F.log_softmax(task_out, dim=1))
        return task_outputs

    def zero_grad(self):
        self.Shared_block.zero_grad()
        for t in range(self.num_tasks-1):
            self.task_blocks[t].zero_grad()  

    def mt_parameters(self):
        params = list(self.Shared_block.parameters())
        for t in range(self.num_tasks-1):
            params += list(self.task_blocks[t].parameters())
            params += list(self.att_blocks[t].parameters())
        return params
    
    
##################################################################################
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
##################################################################################

class MTAN_Network_II(nn.Module):

    def __init__(self, num_classes):
        super(MTAN_Network_II, self).__init__()
        self.num_tasks = len(num_classes) + 1
        self.num_classes = num_classes
        self.zero_layers = []
        filter = [3, 16, 32, 1152, 32]
        
        # Shared Network
        sh_module1 = self.sh_layer_conv([filter[0], filter[1]])
        sh_module2 = self.sh_layer_conv([filter[1], filter[2]])
        sh_module3 = self.sh_layer_lin([filter[3], filter[4]])
        self.Shared_block = nn.ModuleList([sh_module1, sh_module2, sh_module3])
        
        # Task-specific Networks
        self.task_blocks = nn.ModuleList()
        for i in range(self.num_tasks-1):
            task_output = nn.Sequential(
                nn.Linear(32, num_classes[i]),
            )
            self.task_blocks.append(task_output)

        # Task-specific Attention Networks
        att_module = self.att_layer_conv([filter[1], filter[2]])
        feat_module = self.feat_layer_lin([filter[3], filter[4]])
        self.att_blocks = nn.ModuleList()
        for i in range(self.num_tasks-1):
            task_att = nn.ModuleList([att_module, feat_module])
            self.att_blocks.append(task_att)

    ###################################################################################
    
    def sh_layer_conv(self, channel):
        sh_block = nn.Sequential(
            nn.Conv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=3),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(channel[1]),
            nn.ReLU(inplace=True),
        )
        return sh_block
    
    def sh_layer_lin(self, channel):
        sh_block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channel[0], channel[1]),
            nn.ReLU(inplace=True),
        )
        return sh_block
    ###################################################################################
    def att_layer_conv(self, channel):
        att_block = nn.Sequential(
            nn.Conv2d(in_channels=channel[0], out_channels=channel[0], kernel_size=1),
            nn.BatchNorm2d(channel[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=3),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(channel[1]),
            nn.Sigmoid(),
        )
        return att_block
        
    def feat_layer_lin(self, channel):
        att_block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channel[0], channel[0]),
            nn.ReLU(inplace=True),
            nn.Linear(channel[0], channel[1]),
            nn.Sigmoid(),
        )
        return att_block
    ###################################################################################
        
    def forward(self, x):
        # Shared output
        sh_out1 = self.Shared_block[0](x)
        sh_out2 = self.Shared_block[1](sh_out1)
        sh_out3 = self.Shared_block[2](sh_out2)
        sh_out = [sh_out1, sh_out2, sh_out3]
        
        # Task-specific output
        task_outputs = []
        for i in range(self.num_tasks-1):
            t_out1 = self.att_blocks[i][0](sh_out[0])
            t_out2 = sh_out[1]*t_out1
            t_out3 = sh_out[2]*self.att_blocks[i][1](t_out2)
            task_out = self.task_blocks[i](t_out3)
            task_outputs.append(F.log_softmax(task_out, dim=1))
        return task_outputs

    def zero_grad(self):
        self.Shared_block.zero_grad()
        for t in range(self.num_tasks-1):
            self.task_blocks[t].zero_grad()  

    def mt_parameters(self):
        params = list(self.Shared_block.parameters())
        for t in range(self.num_tasks-1):
            params += list(self.task_blocks[t].parameters())
            params += list(self.att_blocks[t].parameters())
        return params
    