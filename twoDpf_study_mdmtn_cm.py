import os
import torch
import torch.backends.cudnn as cudnn

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from src.utils.Preference_vectors import get_pref_vects
from Train_and_Test import load_Cifar10Mnist_data
# from Train_and_Test import load_MultiMnist_data # for MultiMnist dataset
from src.utils.PFstudy_utils import Train_Test_PFstudy

from config import get_params

main_dir = "logs/MDMTN_CM_logs/Pareto_Front_Study"
mod_logdir = "MDMTN_model_CM_PF_onek"
archi_name = "MDMTN"
data_name = "Cifar10Mnist"
Sparsity_study = False
num_model = 0

k0 = 0.01 # the sparsity coefficient in the preference vector that produced the considered sparse model.
ws = get_pref_vects(k0)

if __name__ == "__main__":

    SPARSE_MODEL_FILE = "src/Sparse_models/Sparse_MDMTN_model_II.pth"

    if not os.path.exists(SPARSE_MODEL_FILE):
        raise ValueError("No sparse model found ! First, find the preference vector k that yields the best model performance and save the obtained model in the directory `src/Sparse_models/")
    else:
        # Choose device
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        if use_cuda == False:
            print("WARNING: CPU will be used for training.")

        inst_model, Cifar10mnist_params, GrOWL_parameters = get_params(ws[0], archi_name, data_name, main_dir, mod_logdir, num_model, Sparsity_study)

        Cifar10mnist_params["device"] = device

        train_loader, val_loader, test_loader = load_Cifar10Mnist_data()

        if not os.path.exists("%s/%s"%("Images", data_name)):
            os.makedirs("%s/%s"%("Images", data_name))

        Train_Test_PFstudy(ws, train_loader, val_loader, test_loader, Cifar10mnist_params,
                        SPARSE_MODEL_FILE, data_name, archi_name, inst_model)

        
        #####################################################################################
        

 
