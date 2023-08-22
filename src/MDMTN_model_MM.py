import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import AffinityPropagation

from src.utils.GrOWL_utils import create_GrOWL_params

###################################
######## MULTI-TASK MODEL #########
###################################

class SparseMonitoredMultiTaskNetwork_I(nn.Module):
    """
        This is a Monitored Deep Multi-Task Network model that connects five neural network components with each other:
        The first component, Shared_block, computes the shared intermediary result of inference.
        The second and third components (task_blocks) are bound to their task and process the intermediary result according to their task.
        The fourth and fifth components (monitors) are task-specific monitors that correct the output from the shared component before going to the task-specific output components.
        This model has the ability to induce sparsity through the GrOWL function (obj_sparsity).
    """
    
    def __init__(self, GrOWL_parameters, num_classes, static_a = [True, None]):
        super(SparseMonitoredMultiTaskNetwork_I, self).__init__()
        self.num_tasks = len(num_classes) + 1
        self.num_classes = num_classes
        self.static_a = static_a
        self.zero_layers = []
        self.GrOWL_parameters = GrOWL_parameters
        # WC scalarization variable "t"
        self.wc_variable = nn.Parameter(torch.randn(1, requires_grad = True))
        if self.static_a[0]:
            if self.static_a[1] is None:
                self.task_coef = 0.5 - 0.5/(self.num_tasks - 1)
                self.alpha = dict([(f"alpha_{t}", nn.Parameter(torch.tensor([1-self.task_coef, self.task_coef] ))) for t in range(self.num_tasks-1)])
            else:
                self.task_coef = static_a[1]
                self.alpha = dict([(f"alpha_{t}", nn.Parameter(torch.tensor([self.task_coef[t][0], self.task_coef[t][1]]))) for t in range(self.num_tasks-1)])
            
        else:
            self.alpha = dict([(f"alpha_{t}", nn.Parameter(torch.tensor([1.0, 0.0], requires_grad = True))) for t in range(self.num_tasks-1)])
        
        
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
        )#.to(device)
        
        # Task-specific Networks
        self.task_blocks = nn.ModuleList()
        for i in range(self.num_tasks-1):
            task_output = nn.Sequential(
                nn.Linear(50, num_classes[i]),
            )#.to(device)
            self.task_blocks.append(task_output)
        
        # Monitors
        self.monitors = nn.ModuleList()
        for i in range(self.num_tasks-1):
            monitor = nn.Sequential(
                nn.Conv2d(1, 20, kernel_size=5),
                nn.MaxPool2d(kernel_size=2),
                nn.ReLU(inplace=True),
                nn.Flatten(),
                nn.Linear(2880, 50),
                nn.ReLU(inplace=True),
            )#.to(device)
            self.monitors.append(monitor)
    
    def forward(self, x):
        # Shared output
        sh_out = self.Shared_block(x)
        
        # Task-specific output
        task_outputs = []
        for t in range(self.num_tasks-1):
            sh_out_t = sh_out.clone()
            mon_out = self.monitors[t](x)
            alpha_t = self.alpha[f"alpha_{t}"] #/(self.alpha[f"alpha_{t}"].norm())
            task_out = alpha_t[0]*sh_out_t + alpha_t[1]*mon_out
            task_out = self.task_blocks[t](task_out)
            task_outputs.append(F.log_softmax(task_out, dim=1))
        
        return task_outputs

    def zero_grad(self):
        self.Shared_block.zero_grad()
        for t in range(self.num_tasks-1):
            self.monitors[t].zero_grad()
            self.task_blocks[t].zero_grad()  

    def mt_parameters(self):
        params = list(self.Shared_block.parameters())
        params.append(self.wc_variable)
        for t in range(self.num_tasks-1):
            params += list(self.task_blocks[t].parameters())
            params += list(self.monitors[t].parameters())
            if self.static_a[0]:
                continue
            params.append(self.alpha[f"alpha_{t}"])
        return params
    
    def get_alphas(self):
        alphas = []
        for t in range(self.num_tasks-1):
            a_t = self.alpha[f"alpha_{t}"] #/(self.alpha[f"alpha_{t}"].norm())
            a_t = a_t.detach().numpy().tolist()
            alphas.append(a_t)
        return alphas
    

    # Get the secondary objective function for sparsity
    def obj_sparsity(self,):
            model_weights = self.state_dict()
            GrOWL_regularizer = 0.0
            for name, weight in model_weights.items():
                if 'weight' in name: 
                    # Reshape the weight tensor
                    # if conv: lx(l-1)xwxh ----> (l-1)x(l.w.h)
                    # if fc: lx(l-1) ----> (l-1)xl
                    # Therefore: Weight matrix's ROWS ==> previous layer's NEURONS
                    org_shape = weight.shape
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
