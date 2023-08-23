import torch
import torch.backends.cudnn as cudnn
import torch.optim

from src.utils.MGDA_utils import train_test_MGDA

from config import get_params_mgda

from torch import Tensor
from typing import Iterable, Union
_params_t = Union[Iterable[Tensor], Iterable[dict]]


model_dir_path = "logs/MDMTN_CM_logs/MGDA_model_logs/model_states"
archi_name = "MDMTN"
data_name = "Cifar10Mnist"


if __name__ == "__main__":

    # Choose device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    cudnn.benchmark = True
    if use_cuda == False:
        print("WARNING: CPU will be used for training.")

    model, Cifar10mnist_params_mgda = get_params_mgda(archi_name.lower(), data_name, model_dir_path, device)

    train_losses, test_accuracies = train_test_MGDA(model, data_name, Cifar10mnist_params_mgda, device)

    import pickle

    ## Save the results lists to a file
    with open(f'logs/MDMTN_CM_logs/MGDA_model_logs/Cifar10Mnist_results_MDMTNmgda.pkl', 'wb') as f:
        pickle.dump((train_losses, test_accuracies), f)

    print("Done!")