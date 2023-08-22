import os
from os import path
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import torch.optim
import numpy as np

from Data_loaders.MGDA_dataLoaders_utils import Cifar10Mnist_dataset
from src.MDMTN_MGDA_model_CM import MDMTNmgda_MultiTaskNetwork_II
from src.utils.MGDA_utils import build_MGDA_optimizer, train_multi, test_multi, one_hot_encode_data 

from torch import device, nn, Tensor
from typing import Iterable, Union
import matplotlib.pyplot as plt
from typing import Iterable, Union
_params_t = Union[Iterable[Tensor], Iterable[dict]]


if __name__ == "__main__":

    from time import time
    # Start timer
    import datetime
    print(datetime.datetime.now())
    t0 = time()

    # Choose device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    #cudnn.benchmark = True
    if use_cuda == False:
        print("WARNING: CPU will be used for training.")

    print("Retrieving data...")

    import pickle

    cifar_labels = {
        0: "Airplane", 1: "Automobile", 2: "Bird", 3: "Cat", 4: "Deer",
        5: "Dog", 6: "Frog", 7: "Horse", 8: "Ship", 9: "Truck"
    }

    data_train, data_test = Cifar10Mnist_dataset()

    X_train, y_train = zip(*data_train)
    X_test, y_test = zip(*data_test)

    X_train = np.array(X_train)
    X_test = np.array(X_test)

    plt.imshow(data_train[0][0])
    plt.title(f'{(cifar_labels[y_train[0][0]], y_train[0][1])}')
    plt.show()

    y_train = np.array(one_hot_encode_data(y_train))
    y_test = np.array(one_hot_encode_data(y_test))

    print("Data is loaded")

    # Training starts here
    model_repetitions = 5
    training_epochs = 100
    batch_size = 256
    archi = "mdmtn"

    train_losses = []

    # Testing stuff
    test_accuracies = []

    # dd/mm/YY H:M:S
    dt_string = datetime.datetime.now().strftime("%d-%m-%Y--%H-%M-%S")
    model_dir_path = "logs/MDMTN_CM_logs/MGDA_model_logs/model_states" + "/" + dt_string
    os.mkdir(model_dir_path)

    for i in range(model_repetitions):
        print(f"######## Repetition {i+1}/{model_repetitions} ########")
        model_multi = MDMTNmgda_MultiTaskNetwork_II(batch_size, device=device, static_a = [False, None])
        loss_fn = nn.CrossEntropyLoss()

        MTLOptimizerClass = build_MGDA_optimizer(torch.optim.SGD)
        # Start learning rate: 1e-3
        mtl_optim = MTLOptimizerClass(model_multi.parameters(), lr=1e-2, momentum=0.9)

        for epoch in tqdm(range(training_epochs)):
            #print("Training...")
            model_multi.train()
            train_loss = train_multi(torch.tensor(X_train, dtype=torch.float32, device=device),
                                                          torch.tensor(y_train, dtype=torch.float32, device=device),
                                                          model_multi, mtl_optim,
                                                          loss_fn, batch_size, device=device, archi = archi, img_shp = (32, 32, 3))
            train_losses.extend(train_loss)

            # Halve learning rate every 30 epochs
            if epoch > 0 and epoch % 30 == 0:
                for optim_param in mtl_optim.param_groups:
                    optim_param['lr'] = optim_param['lr'] / 2

        # Save model iteration
        model_multi.save_model(model_dir_path + "/" + f"model_{i}")

        print("Testing...")
        model_multi.train(mode=False)
        test_acc_task = test_multi(torch.tensor(X_test, dtype=torch.float32, device=device),
                                                              torch.tensor(y_test, dtype=torch.float32, device=device),
                                                              model_multi, loss_fn, batch_size,
                                                              device=device, img_shp = (32, 32, 3))

        test_accu_t1 = 100*sum(test_acc_task[0])/len(test_acc_task[0])
        test_accu_t2 = 100*sum(test_acc_task[1])/len(test_acc_task[1])
        test_accuracies.append([test_accu_t1, test_accu_t2])

        print(f"Finised Repetition {i+1} with Accuracy Task 1: {test_accu_t1}")
        print(f"Finised Repetition {i+1} with Accuracy Task 2: {test_accu_t2}")

    mean_acc = np.array(test_accuracies).mean(axis = 0)
    print("Mean Accuracy Task 1: ", mean_acc[0])
    print("Mean Accuracy Task 2: ", mean_acc[1])
    T_norm_1 = time()-t0
    # Print computation time
    print('\nComputation time: {} minutes'.format(T_norm_1/60))
    print(datetime.datetime.now())

    import pickle

    ## Save the results lists to a file
    with open(f'logs/MDMTN_CM_logs/MGDA_model_logs/Cifar10Mnist_results_MDMTNmgda.pkl', 'wb') as f:
        pickle.dump((train_losses, test_accuracies), f)

    print("Done!")