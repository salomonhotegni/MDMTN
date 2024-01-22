import torch
import torch.backends.cudnn as cudnn

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from Train_and_Test import train_and_test_KDMTLmodel_CM, train_and_test_STL_model_CM
#from Train_and_Test import train_and_test_KDMTLmodel_MM, train_and_test_STL_model_MM #(for MultiMNIST data)

from src.utils.KDMTL_Train import train_single_model_kdmtl
from src.utils.KDMTL_Test import test_single_model_kdmtl


from config import get_params_kdmtl, get_params_singleModel


data_name = "Cifar10Mnist"
lmbds = [[0.01, 0.04], [0.04, 0.01],
         [0.02, 0.03], [0.03, 0.02],
         [0.025, 0.025]]
num_tasks = 2

##########################
### SINGLE TASK-MODELS ###
##########################

train_test_func_sg = train_and_test_STL_model_CM
training_func_sg = train_single_model_kdmtl
testing_func_sg = test_single_model_kdmtl
main_dir_sg = "logs/KDMTL_logs"
mod_logdir_sg = "KDMTL_sg_models_CM"

################################
### KDMTL (MULTI-TASK MODEL) ###
################################

main_dir = "logs/KDMTL_logs"
mod_logdir = "KDMTL_model_CM"
sing_mod_logdir = mod_logdir_sg

if __name__ == "__main__":

    # Choose device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cpu")#"cuda" if use_cuda else "cpu")
    cudnn.benchmark = True
    if use_cuda == False:
        print("WARNING: CPU will be used for training.")
        
    from time import time
    # Start timer
    import datetime
    print("KDMTL: ", datetime.datetime.now())
    t_kdmtl_0 = time()
        
    #####################
    for i in range(num_tasks):
        ind_task = i
        model_sg, params_STL = get_params_singleModel(data_name, main_dir_sg, mod_logdir_sg, num_model = i, ind_task = ind_task)

        params_STL["device"] = device
        
        Test_accuracy, prec_wrong_images, ALL_TRAIN_LOSS, ALL_VAL_ACCU = train_test_func_sg(model_sg, params_STL,
                                                                                            train_func= training_func_sg,
                                                                                            test_func= testing_func_sg)

        import pickle

        ## Save the results lists to a file
        with open(f'{main_dir}/{data_name}_results_STL_{params_STL["mod_logdir"]}.pkl', 'wb') as f:
            pickle.dump(([Test_accuracy, prec_wrong_images], ALL_TRAIN_LOSS, ALL_VAL_ACCU), f)
    #############################

    num_model = 0
    model, Cifar10mnist_params = get_params_kdmtl(lmbds, data_name, main_dir, mod_logdir, num_model)

    Cifar10mnist_params["device"] = device
    Cifar10mnist_params["sing_mod_logdir"] = mod_logdir_sg
    Cifar10mnist_params["data_search"] = device

    Test_accuracy, prec_wrong_images, ALL_TRAIN_LOSS, ALL_VAL_ACCU, MODEL_VAL_ACCU = train_and_test_KDMTLmodel_CM(model, Cifar10mnist_params)

    import pickle

    ## Save the results lists to a file
    with open(f'{main_dir}/{data_name}_KDMTLresults.pkl', 'wb') as f:
        pickle.dump(([Test_accuracy, prec_wrong_images], ALL_TRAIN_LOSS, ALL_VAL_ACCU,
                    MODEL_VAL_ACCU), f)
        
    t_kdmtl_1 = time()-t_kdmtl_0
    # Print computation time
    print('\nFull KDMTL Training Computation time: {} minutes'.format(t_kdmtl_1/60))
    print(datetime.datetime.now())
