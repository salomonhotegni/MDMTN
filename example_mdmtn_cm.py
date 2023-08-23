import torch
import torch.backends.cudnn as cudnn

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from Train_and_Test import train_and_test_model_CM # for Cifar10Mnist dataset
# from Train_and_Test import train_and_test_model_MM # for MultiMnist dataset
from config import get_params

k = [1e-2, 0.8, 0.19] # (k_0, k_1, k_2)
main_dir = "logs/MDMTN_CM_logs"
mod_logdir = "MDMTN_model_CM_onek"
archi_name = "MDMTN"
data_name = "Cifar10Mnist"
Sparsity_study = True #(use Sparsity_study = False, for k_0 = 0)
num_model = 0

if __name__ == "__main__":

    # Choose device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    cudnn.benchmark = True
    if use_cuda == False:
        print("WARNING: CPU will be used for training.")

    model, Cifar10mnist_params, GrOWL_parameters = get_params(k, archi_name, data_name, main_dir, mod_logdir, num_model, Sparsity_study)

    Cifar10mnist_params["device"] = device

    Test_accuracy, prec_wrong_images, TR_metrics, ALL_TRAIN_LOSS, ALL_VAL_ACCU, ALL_ORIG_losses, MODEL_VAL_ACCU, BEST_val_accu, Best_iter = train_and_test_model_CM(model, Cifar10mnist_params)

    import pickle

    ## Save the results lists to a file
    with open(f'logs/MDMTN_CM_logs/Cifar10Mnist_results_k1_{k[0]}_k2_{k[1]}_k3_{k[2]}.pkl', 'wb') as f:
        pickle.dump(([Test_accuracy, prec_wrong_images], [TR_metrics, Best_iter], ALL_TRAIN_LOSS, ALL_VAL_ACCU,
                     ALL_ORIG_losses, MODEL_VAL_ACCU, BEST_val_accu), f)
        