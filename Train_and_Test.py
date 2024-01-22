import numpy as np 
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms
from PIL import Image

import matplotlib.pyplot as plt

#plt.style.use('ggplot')

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from src.utils.WCsAL_Train import full_training, train_single_model
from src.utils.WCsAL_Test import test_multitask_model, test_single_model
from Data_loaders.MultiMnistLoaders import MultiMnist_loaders
from Data_loaders.Cifar10MnistLoaders import Cifar10Mnist_loaders

from src.utils.KDMTL_Train import full_training_kdmtl
from src.utils.KDMTL_Test import test_multitask_kdmtl_model

from src.utils.MTAN_Train import full_training_mtan
from src.utils.MTAN_Test import test_multitask_mtan_model

def load_Cifar10Mnist_data():
    data_path = "Data/Cifar10Mnist"
    split_rate = 0.8
    batch_size = [256, 256] # (train, test)

    train_transform = transforms.Compose([#transforms.Lambda(lambda x: Image.fromarray(np.uint8(x))),
                                        #transforms.RandomRotation(20),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                        transforms.Lambda(lambda x: x.permute(0, 1, 2))
                                        ])

    test_transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                        transforms.Lambda(lambda x: x.permute(0, 1, 2))
                                        ])

    transformers = [train_transform, test_transform] # [None, None]
    train_loader, val_loader, test_loader = Cifar10Mnist_loaders(data_path, split_rate, transformers, batch_size)
    print("Data loaded!")

    cifar10_classes = ('Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
            'Dog', 'Frog', 'Horse', 'Ship', 'Truck')

    print("Show sample image...")
    # Get the first batch from the train loader
    train_dataiter = iter(train_loader)
    images, targets = next(train_dataiter)
    img = images[0]
    plt.figure(figsize=(5, 5))
    img = np.transpose(img, (1, 2, 0))
    plt.imshow(img)
    plt.title(f'{(cifar10_classes[targets[0][0]], targets[1][0].item())}')
    plt.axis('off')
    plt.show()

    return train_loader, val_loader, test_loader 

def train_and_test_model_CM(model, Cifar10mnist_params):

    from time import time
    # Start timer
    import datetime
    print(datetime.datetime.now())
    t0 = time()

    train_loader, val_loader, test_loader = load_Cifar10Mnist_data()

    # # Choose device
    device = Cifar10mnist_params["device"]
    print(f"Training... [--- running on {device} ---]")
    
    final_model, TR_metrics, ALL_TRAIN_LOSS, ALL_VAL_ACCU, ALL_ORIG_losses, MODEL_VAL_ACCU, BEST_val_accu, Best_iter = full_training(train_loader, val_loader, model,
                          Cifar10mnist_params, init_model = True)
       
    print("Training completed !") 

    T_norm_1 = time()-t0
    # Print computation time
    print('\nComputation time: {} minutes'.format(T_norm_1/60))
    print(datetime.datetime.now())

    print("Testing ...") 
    Test_accuracy, prec_wrong_images = test_multitask_model(test_loader, final_model, Cifar10mnist_params, TR_metrics) 

    return Test_accuracy, prec_wrong_images, TR_metrics, ALL_TRAIN_LOSS, ALL_VAL_ACCU, ALL_ORIG_losses, MODEL_VAL_ACCU, BEST_val_accu, Best_iter


#################################################################################

def load_MultiMnist_data():
    data_path = "Data/MultiMnist"
    split_rate = 0.83334
    batch_size = [256, 100] # (train, test)

    train_transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))
                                        ])

    test_transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))
                                        ])

    transformers = [train_transform, test_transform] # [None, None]
    train_loader, val_loader, test_loader = MultiMnist_loaders(data_path, split_rate, transformers, batch_size)
    print("Data loaded!")

    print("Show sample image...")
    # Get the first batch from the train loader
    train_dataiter = iter(train_loader)
    images, targets = next(train_dataiter)
    img = images[0]
    plt.figure(figsize=(5, 5))
    img = np.transpose(img, (1, 2, 0))
    plt.imshow(img)
    plt.title(f"({targets[0][0].item()}, {targets[1][0].item()})")
    plt.axis('off')
    plt.show()

    return train_loader, val_loader, test_loader


def train_and_test_model_MM(model, MultiMNISt_params):

    from time import time
    # Start timer
    import datetime
    print(datetime.datetime.now())
    t0 = time()

    train_loader, val_loader, test_loader = load_MultiMnist_data()
    
    device = MultiMNISt_params["device"]
    print(f"Training... [--- running on {device} ---]")
    
    final_model, TR_metrics, ALL_TRAIN_LOSS, ALL_VAL_ACCU, ALL_ORIG_losses, MODEL_VAL_ACCU, BEST_val_accu, Best_iter = full_training(train_loader, val_loader, model,
                          MultiMNISt_params, init_model = True)
    
    print("Training completed !") 

    T_norm_1 = time()-t0
    # Print computation time
    print('\nComputation time: {} minutes'.format(T_norm_1/60))
    print(datetime.datetime.now())

    print("Testing ...") 
    Test_accuracy, prec_wrong_images = test_multitask_model(test_loader, final_model, MultiMNISt_params, TR_metrics) 

    return Test_accuracy, prec_wrong_images, TR_metrics, ALL_TRAIN_LOSS, ALL_VAL_ACCU, ALL_ORIG_losses, MODEL_VAL_ACCU, BEST_val_accu, Best_iter

#################################################################################

def train_and_test_KDMTLmodel_CM(model, Cifar10mnist_params):

    from time import time
    # Start timer
    import datetime
    print(datetime.datetime.now())
    t0 = time()

    train_loader, val_loader, test_loader = load_Cifar10Mnist_data()
    
    size_data_for_search = int(0.1 * len(train_loader.dataset))
    _ = len(train_loader.dataset) - size_data_for_search
    dataset_for_search, _ = torch.utils.data.random_split(train_loader.dataset, [size_data_for_search, _])
    dataloader_for_search = torch.utils.data.DataLoader(dataset_for_search, batch_size=256, shuffle=True)


    device = Cifar10mnist_params["device"]
    print(f"Training... [-- running on {device} --]")
    
    Cifar10mnist_params["data_search"] = dataloader_for_search
    
    final_model, ALL_TRAIN_LOSS, ALL_VAL_ACCU, MODEL_VAL_ACCU = full_training_kdmtl(train_loader, val_loader, model,
                          Cifar10mnist_params, init_model = True)
       
    print("Training completed !") 

    T_norm_1 = time()-t0
    # Print computation time
    print('\nComputation time: {} minutes'.format(T_norm_1/60))
    print(datetime.datetime.now())

    print("Testing ...") 
    Test_accuracy, prec_wrong_images = test_multitask_kdmtl_model(test_loader, final_model, Cifar10mnist_params, load = True) 

    return Test_accuracy, prec_wrong_images, ALL_TRAIN_LOSS, ALL_VAL_ACCU, MODEL_VAL_ACCU

################################################################################


def train_and_test_KDMTLmodel_MM(model, MultiMNISt_params):

    from time import time
    # Start timer
    import datetime
    print(datetime.datetime.now())
    t0 = time()

    train_loader, val_loader, test_loader = load_MultiMnist_data()
    
    size_data_for_search = int(0.1 * len(train_loader.dataset))
    _ = len(train_loader.dataset) - size_data_for_search
    dataset_for_search, _ = torch.utils.data.random_split(train_loader.dataset, [size_data_for_search, _])
    dataloader_for_search = torch.utils.data.DataLoader(dataset_for_search, batch_size=256, shuffle=True)
    
    device = MultiMNISt_params["device"]
    print(f"Training... [-- running on {device} --]")
    
    MultiMNISt_params["data_search"] = dataloader_for_search
    
    final_model, ALL_TRAIN_LOSS, ALL_VAL_ACCU, MODEL_VAL_ACCU = full_training_kdmtl(train_loader, val_loader, model,
                          MultiMNISt_params, init_model = True)
    
    print("Training completed !") 

    T_norm_1 = time()-t0
    # Print computation time
    print('\nComputation time: {} minutes'.format(T_norm_1/60))
    print(datetime.datetime.now())

    print("Testing ...") 
    Test_accuracy, prec_wrong_images = test_multitask_kdmtl_model(test_loader, final_model, MultiMNISt_params, load = True) 

    return Test_accuracy, prec_wrong_images, ALL_TRAIN_LOSS, ALL_VAL_ACCU, MODEL_VAL_ACCU

#################################################################################

def train_and_test_MTANmodel_CM(model, Cifar10mnist_params):

    from time import time
    # Start timer
    import datetime
    print(datetime.datetime.now())
    t0 = time()

    train_loader, val_loader, test_loader = load_Cifar10Mnist_data()
    
    size_data_for_search = int(0.1 * len(train_loader.dataset))
    _ = len(train_loader.dataset) - size_data_for_search
    dataset_for_search, _ = torch.utils.data.random_split(train_loader.dataset, [size_data_for_search, _])
    dataloader_for_search = torch.utils.data.DataLoader(dataset_for_search, batch_size=256, shuffle=True)


    device = Cifar10mnist_params["device"]
    print(f"Training... [-- running on {device} --]")
    
    Cifar10mnist_params["data_search"] = dataloader_for_search
    
    final_model, ALL_TRAIN_LOSS, ALL_VAL_ACCU, MODEL_VAL_ACCU = full_training_mtan(train_loader, val_loader, model,
                          Cifar10mnist_params, init_model = True)
       
    print("Training completed !") 

    T_norm_1 = time()-t0
    # Print computation time
    print('\nComputation time: {} minutes'.format(T_norm_1/60))
    print(datetime.datetime.now())

    print("Testing ...") 
    Test_accuracy, prec_wrong_images = test_multitask_mtan_model(test_loader, final_model, Cifar10mnist_params, load = True) 

    return Test_accuracy, prec_wrong_images, ALL_TRAIN_LOSS, ALL_VAL_ACCU, MODEL_VAL_ACCU

#################################################################################

def train_and_test_MTANmodel_MM(model, MultiMNISt_params):

    from time import time
    # Start timer
    import datetime
    print(datetime.datetime.now())
    t0 = time()

    train_loader, val_loader, test_loader = load_MultiMnist_data()
    
    size_data_for_search = int(0.1 * len(train_loader.dataset))
    _ = len(train_loader.dataset) - size_data_for_search
    dataset_for_search, _ = torch.utils.data.random_split(train_loader.dataset, [size_data_for_search, _])
    dataloader_for_search = torch.utils.data.DataLoader(dataset_for_search, batch_size=256, shuffle=True)
    
    device = MultiMNISt_params["device"]
    print(f"Training... [-- running on {device} --]")
    
    MultiMNISt_params["data_search"] = dataloader_for_search
    
    final_model, ALL_TRAIN_LOSS, ALL_VAL_ACCU, MODEL_VAL_ACCU = full_training_mtan(train_loader, val_loader, model,
                          MultiMNISt_params, init_model = True)
    
    print("Training completed !") 

    T_norm_1 = time()-t0
    # Print computation time
    print('\nComputation time: {} minutes'.format(T_norm_1/60))
    print(datetime.datetime.now())

    print("Testing ...") 
    Test_accuracy, prec_wrong_images = test_multitask_mtan_model(test_loader, final_model, MultiMNISt_params, load = True) 

    return Test_accuracy, prec_wrong_images, ALL_TRAIN_LOSS, ALL_VAL_ACCU, MODEL_VAL_ACCU

#################################################################################

##########################
######### STL ############
##########################

def train_and_test_STL_model_CM(model, Cifar10mnist_params_STL, train_func = None, test_func = None):

    from time import time
    # Start timer
    import datetime
    print(datetime.datetime.now())
    t0 = time()

    train_loader, val_loader, test_loader = load_Cifar10Mnist_data()
    
    print("Training ...") 
    
    if train_func is None:
        train_func = train_single_model
    
    final_model, ALL_TRAIN_LOSS, ALL_VAL_ACCU = train_func(train_loader, val_loader, model, Cifar10mnist_params_STL)

    print("Training completed !") 
    

    T_norm_1 = time()-t0
    # Print computation time
    print('\nComputation time: {} minutes'.format(T_norm_1/60))
    print(datetime.datetime.now())

    print("Testing ...") 
    
    if test_func is None:
        test_func = test_single_model
        
    Test_accuracy, prec_wrong_images = test_func(test_loader, final_model, Cifar10mnist_params_STL) 

    return Test_accuracy, prec_wrong_images, ALL_TRAIN_LOSS, ALL_VAL_ACCU



def train_and_test_STL_model_MM(model, MultiMNISt_params_STL, train_func = None, test_func = None):

    from time import time
    # Start timer
    import datetime
    print(datetime.datetime.now())
    t0 = time()

    train_loader, val_loader, test_loader = load_MultiMnist_data()
    
    print("Training ...") 
    
    if train_func is None:
        train_func = train_single_model
    
    final_model, ALL_TRAIN_LOSS, ALL_VAL_ACCU = train_func(train_loader, val_loader, model, MultiMNISt_params_STL)

    print("Training completed !") 
    

    T_norm_1 = time()-t0
    # Print computation time
    print('\nComputation time: {} minutes'.format(T_norm_1/60))
    print(datetime.datetime.now())

    print("Testing ...") 
    
    if test_func is None:
        test_func = test_single_model
        
    Test_accuracy, prec_wrong_images = test_func(test_loader, final_model, MultiMNISt_params_STL) 

    return Test_accuracy, prec_wrong_images, ALL_TRAIN_LOSS, ALL_VAL_ACCU

###############################################

