import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import AffinityPropagation

from src.utils.GrOWL_utils import create_GrOWL_params

###################################
######## MULTI-TASK MODEL #########
###################################

class HardPS_MultiTaskNetwork_II(nn.Module):
    """
        This is a Hard Parameter Sharing model that connects three neural network components with each other:
        The first component, Shared_block, computes the shared intermediary result of inference.
        The second and third component are bound to their task and process the intermediary result according to their task.
        This model has the ability to induce sparsity through the GrOWL function (obj_sparsity).
    """
    def __init__(self,  GrOWL_parameters, num_classes):
        super(HardPS_MultiTaskNetwork_II, self).__init__()
        self.num_tasks = len(num_classes) + 1
        self.num_classes = num_classes
        self.zero_layers = []
        self.GrOWL_parameters = GrOWL_parameters
        # WC scalarization variable "t"
        self.wc_variable = nn.Parameter(torch.randn(1, requires_grad = True))
        
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

        # Task-specific Networks
        self.task_blocks = nn.ModuleList()
        for i in range(self.num_tasks-1):
            task_output = nn.Sequential(
                nn.Linear(32, num_classes[i]), 
            )
            self.task_blocks.append(task_output)
        
    
    def forward(self, x):
        # Shared output
        sh_out = self.Shared_block(x)
        
        # Task-specific output
        task_outputs = []
        for t in range(self.num_tasks-1):
            sh_out_t = sh_out.clone()
            task_out = self.task_blocks[t](sh_out_t)
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
        return params
    
    # Get the secondary objective function for sparsity
    def obj_sparsity(self,):
            model_weights = self.state_dict()
            GrOWL_regularizer = 0.0
            for name, weight in model_weights.items():
                if ('weight' in name) and (len(weight.shape)>1): 
                    # Reshape the weight tensor
                    # if conv: lx(l-1)xwxh ----> (l-1)x(l.w.h)
                    # if fc: lx(l-1) ----> (l-1)xl
                    # Therefore: Weight matrix's ROWS ==> previous layer's NEURONS
                    org_shape = weight.shape
                    #print(f"{name}: {weight.shape}")
                    if ("task_blocks" in name) or (org_shape[1] == self.GrOWL_parameters["skip_layer"]): # Don't use GrOWL for the input and output layers
                        continue

                    if len(org_shape) == 2:
                        reshaped_weight = weight.T
                    else:
                        reshaped_weight = weight.view(weight.shape[1], -1)
                    # Compute the norm-2 of each row
                    norm_2_rows = torch.norm(reshaped_weight, p=2, dim=1)
                    # Sort the norms in ascending order
                    sorted_norm_2_rows, _ = torch.sort(norm_2_rows)

                    layer_GrOWL_penalty = 0.0
                    n = len(sorted_norm_2_rows)
                    theta_is = create_GrOWL_params(self, n)
                    for i in range(n):
                        # Compute per-layer GrOWL penalty
                        layer_GrOWL_penalty = layer_GrOWL_penalty + theta_is[i]*sorted_norm_2_rows[i]
                    # Get GrOWL regularizer
                    GrOWL_regularizer = GrOWL_regularizer + layer_GrOWL_penalty

            return GrOWL_regularizer
            ###############################  

