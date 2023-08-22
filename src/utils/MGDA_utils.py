import torch
from tqdm import tqdm
import torch.optim
from torch import device, Tensor
import torch.optim
from torch import Tensor

from typing import Iterable, Union
_params_t = Union[Iterable[Tensor], Iterable[dict]]

from Data_loaders.MGDA_dataLoaders_utils import MGDA_Data

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