import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from src.MDMTN_model_CM import SparseMonitoredMultiTaskNetwork_II
from src.utils.projectedOWL_utils import proxOWL
from Train_and_Test import train_and_test_model_CM

k = [0.0, 0.55, 0.45]

Cifar10mnist_params = {"w": k, "a": torch.zeros(3),
                           "epsilon": 0.0001, "num_tasks": 3, "num_outs": [10, 10],
                        "max_iter": 12, "max_iter_search": 3, "max_iter_retrain": 10,  
                        "lr": 0.0001, "lr_sched_coef": 0.98, "LR_scheduler": True, 
                      "num_epochs": 3, "tol_epochs": None,
                     "num_model": 0,"main_dir": "logs/MDMTN_CM_logs", "mod_logdir": "MDMTN_model_CM_k0is0",
                     "mu": 0.0001,  "rho": 0.5, "min_sparsRate": 10.00, # (%)
                     "base_optimizer": optim.Adam, "is_search": False, "Sparsity_study": False,
                      "criterion": torch.nn.functional.nll_loss,}
    

GrOWL_parameters = {"tp": "spike", #"Dejiao", #"linear", 
                    "beta1": 0.8,  
                    "beta2": 0.2, 
                   "proxOWL": proxOWL,
                   "skip_layer": 3, # Skip layer with "3" neuron
                    "sim_preference": 0.8, 
                    "max_layerSRate": 0.3,
                   }

if __name__ == "__main__":

    # Choose device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    cudnn.benchmark = True
    if use_cuda == False:
        print("WARNING: CPU will be used for training.")

    Cifar10mnist_params["device"] = device

    model = SparseMonitoredMultiTaskNetwork_II(GrOWL_parameters, Cifar10mnist_params["num_outs"], static_a = [False, None])
    Test_accuracy, prec_wrong_images, TR_metrics, ALL_TRAIN_LOSS, ALL_VAL_ACCU, ALL_ORIG_losses, MODEL_VAL_ACCU, BEST_val_accu, Best_iter = train_and_test_model_CM(model, Cifar10mnist_params)

    import pickle

    ## Save the results lists to a file
    with open(f'logs/MDMTN_CM_logs/Cifar10Mnist_results_k1_{k[0]}_k2_{k[1]}_k3_{k[2]}.pkl', 'wb') as f:
        pickle.dump(([Test_accuracy, prec_wrong_images], [TR_metrics, Best_iter], ALL_TRAIN_LOSS, ALL_VAL_ACCU,
                     ALL_ORIG_losses, MODEL_VAL_ACCU, BEST_val_accu), f)
        