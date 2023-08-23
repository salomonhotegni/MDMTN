import torch
import torch.backends.cudnn as cudnn

#plt.style.use('ggplot')

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from Train_and_Test import train_and_test_STL_model_CM
# from Train_and_Test import train_and_test_STL_model_MM # for MultiMnist dataset

from config import get_params_singleModel

# ```
#    Run this script to train a Single Task Learning model on the Cifar10Mnist dataset.
#    The parameter 'ind_task' represents the index of the desired task (0 or 1).
#```


main_dir = "logs/STL_CM_logs"
data_name = "Cifar10Mnist"
num_model = 0
ind_task = 0
mod_logdir = "STL_model_CM_cifar"

if __name__ == "__main__":

    # Choose device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    cudnn.benchmark = True
    if use_cuda == False:
        print("WARNING: CPU will be used for training.")

    model, Cifar10mnist_params_STL = get_params_singleModel(data_name, main_dir, mod_logdir, num_model, ind_task)

    Cifar10mnist_params_STL["device"] = device
    
    Test_accuracy, prec_wrong_images, ALL_TRAIN_LOSS, ALL_VAL_ACCU = train_and_test_STL_model_CM(model, Cifar10mnist_params_STL)

    import pickle

    ## Save the results lists to a file
    with open(f'logs/STL_CM_logs/Cifar10Mnist_results_STL_{Cifar10mnist_params_STL["mod_logdir"]}.pkl', 'wb') as f:
        pickle.dump(([Test_accuracy, prec_wrong_images], ALL_TRAIN_LOSS, ALL_VAL_ACCU), f)
        