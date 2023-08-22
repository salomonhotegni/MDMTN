import torch
import torch.nn as nn
import torch.nn.functional as F
    
###################################
######## SINGLE-TASK MODEL ########
###################################


class SingleTaskModel_I(nn.Module):
    def __init__(self, NUM_OUT = 10):
        super(SingleTaskModel_I, self).__init__()
        self.NUM_OUT = NUM_OUT
        self.conv_sp1 = nn.Conv2d(1, 10, kernel_size=5)
        self.max_pool_sp1 = nn.MaxPool2d(kernel_size=2)
        self.conv_sp2 = nn.Conv2d(10, 20, kernel_size=5)
        self.max_pool_sp2 = nn.MaxPool2d(kernel_size=2)
        self.fc1_sp1 = nn.Linear(320, 50)
        self.fc1_sp2 = nn.Linear(50, self.NUM_OUT)

        
    def forward(self, x):
        conv_sp1 = F.relu(self.max_pool_sp1(self.conv_sp1(x)))
        conv_sp2 = F.relu(self.max_pool_sp2(self.conv_sp2(conv_sp1)))
        conv_sp2 = torch.flatten(conv_sp2, 1)
        fc1_sp1 = F.relu(self.fc1_sp1(conv_sp2))
        return F.log_softmax(self.fc1_sp2(fc1_sp1), dim=1)
