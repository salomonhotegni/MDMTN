import torch
import torch.nn as nn
import torch.nn.functional as F

###################################
######## SINGLE-TASK MODEL ########
###################################


class SingleTaskModel_II(nn.Module):
    def __init__(self, NUM_OUT = 10):
        super(SingleTaskModel_II, self).__init__()
        self.NUM_OUT = NUM_OUT
        self.conv_sp1 = nn.Conv2d(3, 16, 3)
        self.Bnorm_sp1 = nn.BatchNorm2d(16)
        self.conv_sp2 = nn.Conv2d(16, 32, 3)
        self.Bnorm_sp2 = nn.BatchNorm2d(32)
        self.fc_sp1 = nn.Linear(1152, 32)
        self.fc_sp2 = nn.Linear(32, self.NUM_OUT)
        
        self.MaxPool_sp = nn.MaxPool2d(kernel_size=2)
        
    def forward(self, x):
        conv_sp1 = F.relu(self.Bnorm_sp1(self.MaxPool_sp(self.conv_sp1(x))))
        conv_sp2 = F.relu(self.Bnorm_sp2(self.MaxPool_sp(self.conv_sp2(conv_sp1))))
        conv_sp2 = torch.flatten(conv_sp2, 1)
        fc_sp1 = F.relu(self.fc_sp1(conv_sp2))
        return F.log_softmax(self.fc_sp2(fc_sp1), dim=1)
