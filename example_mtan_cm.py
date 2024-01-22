import torch
import torch.backends.cudnn as cudnn

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from Train_and_Test import train_and_test_MTANmodel_CM
#from Train_and_Test import train_and_test_MTANmodel_MM

from config import get_params_mtan

data_name = "Cifar10Mnist"
num_tasks = 2
lmbds = [[0.01, 0.09], [0.09, 0.01],
         [0.02, 0.08], [0.08, 0.02],
         [0.03, 0.07], [0.07, 0.03],
         [0.04, 0.06], [0.06, 0.04],
         [0.05, 0.05]]

lr_s = [0.00001, 
        0.0001, 
        0.001, 0.01, 
        0.1
        ]

################################
### MTAN (MULTI-TASK MODEL) ###
################################

main_dir = "logs/MTAN_logs"
mod_logdir = "MTANL_model_CM"

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
    print("MTAN: ", datetime.datetime.now())
    t_mtan_0 = time()

    num_model = 0
    model, Cifar10mnist_params = get_params_mtan(lmbds, lr_s, data_name, main_dir, mod_logdir, num_model)

    Cifar10mnist_params["device"] = device
    
    Test_accuracy, prec_wrong_images, ALL_TRAIN_LOSS, ALL_VAL_ACCU, MODEL_VAL_ACCU = train_and_test_MTANmodel_CM(model, Cifar10mnist_params)

    import pickle

    ## Save the results lists to a file
    with open(f'{main_dir}/{data_name}_MTANresults.pkl', 'wb') as f:
        pickle.dump(([Test_accuracy, prec_wrong_images], ALL_TRAIN_LOSS, ALL_VAL_ACCU,
                    MODEL_VAL_ACCU), f)
        
    t_mtan_1 = time()-t_mtan_0
    # Print computation time
    print('\nFull MTAN Training Computation time: {} minutes'.format(t_mtan_1/60))
    print(datetime.datetime.now())

#############################################
    
# Lambda used:  [0.01, 0.09]
# Learning rate used:  0.001
# Training completed !

# Computation time: 26.364986793200174 minutes
# 2024-01-18 13:16:11.922208
# Testing ...

# Test set: Average Accuracy: (69.26%)

# Accuracy Task 1: 45.3000%
# Accuracy Task 2: 93.2200%

# Full MTAN Training Computation time: 26.375362888971964 minutes
        


