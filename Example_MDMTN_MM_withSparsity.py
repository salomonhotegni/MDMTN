import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from src.utils.projectedOWL_utils import proxOWL
from src.MDMTN_model_MM import SparseMonitoredMultiTaskNetwork_I
from Train_and_Test import train_and_test_model_MM

k = [1e-3, 0.2, 0.799]

MultiMNISt_params = {"w": k, "a": torch.zeros(3),
                           "epsilon": 0.0001, "num_tasks": 3, "num_outs": [10, 10],
                        "max_iter": 12, "max_iter_search": 3, "max_iter_retrain": 10, 
                        "lr": 0.0025, "lr_sched_coef": 0.5, "LR_scheduler": True, 
                      "num_epochs": 3, "tol_epochs": None,
                     "num_model": 0,"main_dir": "logs/MDMTN_MM_logs", "mod_logdir": "MDMTN_model_MM_onek",
                     "mu": 2.5e-05,  "rho": 0.5, "min_sparsRate": 20.00, # (%)
                     "base_optimizer": optim.Adam, "is_search": True, "Sparsity_study": True,
                      "criterion": torch.nn.functional.nll_loss,}
    
GrOWL_parameters = {"tp": "spike", #"Dejiao", #"linear", 
                    "beta1": 0.8,  
                    "beta2": 0.2, 
                   "proxOWL": proxOWL,
                   "skip_layer": 1, # Skip layer with "1" neuron
                    "sim_preference": 0.7, 
                    "max_layerSRate": 0.80,
                   }
    
if __name__ == "__main__":

    # import os
    # directory_path = MultiMNISt_params["main_dir"] 

    # if os.path.exists(directory_path):
    #     print(f"The directory '{directory_path}' exists.")
    # else:
    #     print(f"The directory '{directory_path}' does not exist.")

    # Choose device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    cudnn.benchmark = True
    if use_cuda == False:
        print("WARNING: CPU will be used for training.")

    MultiMNISt_params["device"] = device

    model = SparseMonitoredMultiTaskNetwork_I(GrOWL_parameters, MultiMNISt_params["num_outs"], static_a = [False, None])
    Test_accuracy, prec_wrong_images, TR_metrics, ALL_TRAIN_LOSS, ALL_VAL_ACCU, ALL_ORIG_losses, MODEL_VAL_ACCU, BEST_val_accu, Best_iter = train_and_test_model_MM(model, MultiMNISt_params)

    # print("Test Accuracy = ", Test_accuracy.mean().item())
    # print("Left digits Accuracy = ", Test_accuracy[0].item())
    # print("Right digits Accuracy = ", Test_accuracy[1].item())

    import pickle

    ## Save the results lists to a file
    with open(f'logs/MDMTN_MM_logs/MultiMNIST_results_k1_{k[0]}_k2_{k[1]}_k3_{k[2]}.pkl', 'wb') as f:
        pickle.dump(([Test_accuracy, prec_wrong_images], [TR_metrics, Best_iter], ALL_TRAIN_LOSS, ALL_VAL_ACCU,
                     ALL_ORIG_losses, MODEL_VAL_ACCU, BEST_val_accu), f)
