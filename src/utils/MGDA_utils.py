import torch
from tqdm import tqdm
import torch.optim
import numpy as np
import os
from os import path

from torch import nn, Tensor
from typing import Iterable, Union
import matplotlib.pyplot as plt
from typing import Iterable, Union
_params_t = Union[Iterable[Tensor], Iterable[dict]]

from Data_loaders.MGDA_dataLoaders_utils import MGDA_Data
from Data_loaders.MGDA_dataLoaders_utils import Cifar10Mnist_dataset
from Data_loaders.MGDA_dataLoaders_utils import MultiMnist_dataset

#######################################
#####  Helper functions for MGDA #####
#######################################


def build_MGDA_optimizer(optimizer_type: torch.optim.Optimizer):
    """
    Builds a MGDAoptimizer Class derived from optimizer_type, e.g Adam or SGD.

    :param optimizer_type: class of the optimizer to use, e.g torch.optim.Adam
    :return: MGDAoptimizer: class
    """
    class MGDAoptimizer(optimizer_type):
        underlying_optimizer = optimizer_type

        @staticmethod
        def frank_wolfe_solver(gradients: list,
                               termination_threshold: float = 1e-4,
                               max_iterations: int = 10,
                               device: str = "cpu") -> Tensor:
            """
            Applies Frank-Wolfe-Solver to a list of (shared) gradients on t-many tasks
            :param gradients: list of (shared) gradients
            :param termination_threshold: termination condition
            :param max_iterations: #iterations before algorithm termination
            :return: Tensor of shape [t]
            """

            # Amount of tasks
            T = len(gradients)
            # Amount of layers
            L = len(gradients[0])

            # Initialize alpha
            alpha = torch.tensor([1 / T for _ in range(T)], device=device)

            M = torch.zeros(size=(T, T), dtype=torch.float32, device=device)


            for i in range(T):
                flat_gradient_i = torch.concat([torch.flatten(gradients[i][layer]) for layer in range(L)])
                for j in range(T):
                    flat_gradient_j = torch.concat([torch.flatten(gradients[j][layer]) for layer in range(L)])
                    if M[j][i] != 0:
                        M[i][j] = M[j][i]
                    else:
                        M[i][j] = torch.dot(flat_gradient_i, flat_gradient_j)

            # Initialize gamma
            gamma = float('inf')
            iteration = 0

            while gamma > termination_threshold and iteration <= max_iterations:
                alpha_m_sum = torch.matmul(alpha, M)
                t_hat = torch.argmin(alpha_m_sum)

                g_1 = torch.zeros_like(alpha, device=device)
                g_2 = alpha

                g_1[t_hat] = 1

                g1_Mg1 = torch.matmul((g_1), torch.matmul(M, g_1))
                g2_Mg2 = torch.matmul((g_2), torch.matmul(M, g_2))
                g1_Mg2 = torch.matmul((g_1), torch.matmul(M, g_2))

                if g1_Mg1 <= g1_Mg2:
                    gamma = 1
                elif g1_Mg2 >= g2_Mg2:
                    gamma = 0
                else:
                    dir_a = g2_Mg2 - g1_Mg2
                    dir_b = g1_Mg1 - 2*g1_Mg2 + g2_Mg2
                    gamma = dir_a / dir_b

                alpha = (1 - gamma) * alpha + gamma * g_1
                iteration += 1

                if T <= 2:
                    break
            return alpha

    return MGDAoptimizer

###############################################

# Helper functions for evaluation
def compute_accuracy(predictions, targets) -> float:
    return float(
        sum([pred.argmax() == target.argmax() for pred, target in zip(predictions, targets)]) / len(predictions))

def train_multi(X, y, model, optimizer, loss_fn, batch_size, device, archi, img_shp = (32, 32, 3)):
    # X is a torch Variable
    permutation = torch.randperm(X.size()[0])
    losses = [[] for _ in range(model.num_of_tasks)]

    for i in range(0, X.size()[0], batch_size):
        indices = permutation[i:i + batch_size]
        batch_x, batch_y = X[indices].to(device), y[indices].to(device)
        # Only accept full-size batches
        if batch_x.shape[0] != batch_size:
            continue
        batch_x = batch_x.reshape([batch_size, 1, img_shp[0], img_shp[1], img_shp[2]])

        shared_gradients = [[] for _ in range(model.num_of_tasks)]
        for t in range(model.num_of_tasks):
            model.zero_grad()

            # Full forward pass
            output = model.forward(batch_x)

            # Compute t-specific loss
            t_output = output[:, t * model.task_out_n:(t + 1) * model.task_out_n]
            t_label = batch_y[:, t]
            t_loss = loss_fn(t_output, t_label)

            # Backward pass
            t_loss.backward()

            for param in model.shared.parameters():
                if param.grad is not None:
                    _grad = param.grad.data.detach().clone()
                    shared_gradients[t].append(_grad)

        alphas = optimizer.frank_wolfe_solver(shared_gradients, device=device)

        # Collect task specific gradients regarding task specific loss
        z = model.shared_forward(batch_x)

        # aggregate loss
        loss = torch.zeros(1, device=device)
        for t in range(model.num_of_tasks):
            if archi == "hps":
                loss_t = loss_fn(model.taskOutput[f"task_{t}"].forward(z), batch_y[:, t]) * alphas[t]
            elif archi == "mdmtn":
                loss_t = loss_fn(model.taskOutput[f"task_{t}"].forward(z[t]), batch_y[:, t]) * alphas[t]
            else: raise ValueError("Model Architecture should be 'hps' or 'mdmtn' !")
            loss += loss_t
            losses[t].append(loss_t.detach().item())

        loss.backward()

        optimizer.step()
    return losses

def test_multi(X, y, model, loss_fn, batch_size, device, img_shp = (32, 32, 3)):
    model.eval()
    permutation = torch.randperm(X.size()[0])
    task_accuracies = [[] for _ in range(model.num_of_tasks)]

    with torch.no_grad():
        for i in tqdm(range(0, X.size()[0], batch_size)):
            indices = permutation[i:i + batch_size]
            batch_x, batch_y = X[indices].to(device), y[indices].to(device)
            if batch_x.shape[0] != batch_size:
                continue
            batch_x = batch_x.reshape([batch_size, 1, img_shp[0], img_shp[1], img_shp[2]])
            pred = model(batch_x)

            for t in range(model.num_of_tasks):
                t_output = pred[:, t * model.task_out_n:(t + 1) * model.task_out_n]
                # Compute classification accuracy per task
                task_accuracies[t].append(compute_accuracy(t_output, batch_y[:, t]))

    return task_accuracies

def one_hot_encode_data(array):
    """One hot encodes the target labels of a list of data-labels"""
    res = []
    for index in range(len(array)):
        res.append([MGDA_Data.one_hot_encode(array[index][0]),
                         MGDA_Data.one_hot_encode(array[index][1])]
                   )
    return res

#############################

def load_Cifar10Mnist_mgda():

    print("Retrieving data...")

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

    return X_train, X_test, y_train, y_test

def load_MultiMnist_mgda():

    print("Retrieving data...")

    data_train, data_test = MultiMnist_dataset()

    X_train, y_train = zip(*data_train)
    X_test, y_test = zip(*data_test)

    X_train = np.array(X_train)
    X_test = np.array(X_test)

    plt.imshow(data_train[0][0])
    plt.title(f'{(y_train[0][0], y_train[0][1])}')
    plt.show()

    y_train = np.array(one_hot_encode_data(y_train))
    y_test = np.array(one_hot_encode_data(y_test))

    print("Data is loaded")

    return X_train, X_test, y_train, y_test

def train_test_MGDA(model, data_name, mod_params_mgda, device):
    model_repetitions = mod_params_mgda["model_repetitions"]
    training_epochs = mod_params_mgda["training_epochs"]
    archi = mod_params_mgda["archi"]
    batch_size = mod_params_mgda["batch_size"]
    img_shp = mod_params_mgda["img_shp"]
    momentum = mod_params_mgda["momentum"]
    lr = mod_params_mgda["lr"]
    model_dir_path = mod_params_mgda["model_dir_path"]

    if data_name == "Cifar10Mnist":
        X_train, X_test, y_train, y_test = load_Cifar10Mnist_mgda()
    elif data_name == "MultiMnist":
        X_train, X_test, y_train, y_test = load_MultiMnist_mgda()
    else: raise ValueError(f"Unknown dataset {data_name} !")
    
    train_losses = []

    # Testing stuff
    test_accuracies = []

    from time import time
    # Start timer
    import datetime
    print(datetime.datetime.now())
    t0 = time()

    # dd/mm/YY H:M:S
    dt_string = datetime.datetime.now().strftime("%d-%m-%Y--%H-%M-%S")
    model_dir_path = model_dir_path + "/" + dt_string
    os.mkdir(model_dir_path)

    for i in range(model_repetitions):
        print(f"######## Repetition {i+1}/{model_repetitions} ########")
        model_multi = model
        loss_fn = nn.CrossEntropyLoss()

        MTLOptimizerClass = build_MGDA_optimizer(torch.optim.SGD)

        mtl_optim = MTLOptimizerClass(model_multi.parameters(), lr=lr, momentum=momentum)

        for epoch in tqdm(range(training_epochs)):
            #print("Training...")
            model_multi.train()
            train_loss = train_multi(torch.tensor(X_train, dtype=torch.float32, device=device),
                                                          torch.tensor(y_train, dtype=torch.float32, device=device),
                                                          model_multi, mtl_optim,
                                                          loss_fn, batch_size, device=device, archi = archi, img_shp=img_shp)
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
                                                              device=device, img_shp = img_shp)

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

    return train_losses, test_accuracies



    

