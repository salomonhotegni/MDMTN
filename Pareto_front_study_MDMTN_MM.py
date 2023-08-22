import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from src.MDMTN_model_MM import SparseMonitoredMultiTaskNetwork_I
from src.utils.projectedOWL_utils import proxOWL
from src.utils.Preference_vectors import get_pref_vects
from Train_and_Test import load_MultiMnist_data
from src.utils.PFstudy_utils import Train_Test_PFstudy


MultiMNISt_params = {"a": torch.zeros(3),
                           "epsilon": 0.0001, "num_tasks": 3, "num_outs": [10, 10],
                        "max_iter": 12, "max_iter_search": 3, "max_iter_retrain": 10, 
                        "lr": 0.0025, "lr_sched_coef": 0.5, "LR_scheduler": True, 
                      "num_epochs": 3, "tol_epochs": None,
                     "num_model": 0,"main_dir": "logs/MDMTN_MM_logs/Pareto_Front_Study", "mod_logdir": "MDMTN_model_MM_PF_onek",
                      "mu": 2.5e-05,  "rho": 0.5, "min_sparsRate": 20.00, # (%)
                     "base_optimizer": optim.Adam, "is_search": False, "Sparsity_study": False,
                      "criterion": torch.nn.functional.nll_loss,}
    

GrOWL_parameters = {"tp": "spike", #"Dejiao", #"linear", 
                    "beta1": 0.8,  
                    "beta2": 0.2, 
                   "proxOWL": proxOWL,
                   "skip_layer": 1, # Skip layer with "1" neuron
                    "sim_preference": 0.7, 
                    "max_layerSRate": 0.8,
                   }

if __name__ == "__main__":

    SPARSE_MODEL_FILE = "src/Sparse_models/Sparse_MDMTN_model_I.pth"

    if not os.path.exists(SPARSE_MODEL_FILE):
        raise ValueError("No sparse model found ! First, find the preference vector k that yields the best model performance and save the obtained model in the directory `src/Sparse_models/")
    else:
        # Choose device
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        cudnn.benchmark = True
        if use_cuda == False:
            print("WARNING: CPU will be used for training.")

        MultiMNISt_params["device"] = device

        train_loader, val_loader, test_loader = load_MultiMnist_data()

        k0 = 0.001 # the sparsity coefficient in the preference vector that produced the considered sparse model.
        ws = get_pref_vects(k0)
        data_name = "MultiMnist"
        archi_name = "MDMTN"

        if not os.path.exists("%s/%s"%("Images", data_name)):
            os.makedirs("%s/%s"%("Images", data_name))

        inst_model = SparseMonitoredMultiTaskNetwork_I(GrOWL_parameters, MultiMNISt_params["num_outs"], static_a = [False, None])

        Train_Test_PFstudy(ws, train_loader, val_loader, test_loader, MultiMNISt_params,
                        SPARSE_MODEL_FILE, data_name, archi_name, inst_model)

        
        #####################################################################################
        

    