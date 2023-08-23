import torch
from torch import nn
import torch.nn.functional as F


class HPSmgda_MultiTaskNetwork_I(nn.Module):
    """
    This is a Hard Parameter Sharing model that connects three neural network components with each other:
    The first component, MultiTNShared, computes the shared intermediary result of inference.
    The second and third component are bound to their task and process the intermediary result according to their task.
    In the forward method, the task specific outputs are concatenated and returned as one complete tensor.
    """

    def __init__(self, batch_size, device=torch.device("cpu")):
        super().__init__()
        self.num_of_tasks = 2
        self.device = device
        self.shared = MultiTNShared(batch_size).to(self.device)
        self.taskOutput = dict([(f"task_{t}", MultiTNTaskSpecific().to(self.device)) for t in range(self.num_of_tasks)])
        # Intermediary representation
        self.z_rep = None
        self.task_out_n = 10
        self.batch_size = batch_size

    def shared_forward(self, x):
        self.z_rep = self.shared(x)
        return self.z_rep

    def task_specific_forward(self, x=None):
        if x is None:
            x = self.z
        outputs = []
        for t in range(self.num_of_tasks):
            out = self.taskOutput[f"task_{t}"](x)
            outputs.append(out)
        return torch.cat(outputs, 1)

    def forward(self, x):
        z = self.shared_forward(x)
        return self.task_specific_forward(z)

    def zero_grad(self, set_to_none: bool = False) -> None:
        self.shared.zero_grad()
        for t in range(self.num_of_tasks):
            self.taskOutput[f"task_{t}"].zero_grad()

    def parameters(self):
        params = []
        params += self.shared.parameters()
        for t in range(self.num_of_tasks):
            params += self.taskOutput[f"task_{t}"].parameters()
        return params

    def save_model(self, path: str):
        torch.save(self.shared.state_dict(), path + "model_shared.pth")
        for t in range(self.num_of_tasks):
            torch.save(self.taskOutput[f"task_{t}"].state_dict(), path + f"model_task_{t}.pth")

    # Assuming the same architecture
    def load_model(self, path: str):
        self.shared.load_state_dict(torch.load(path + "model_shared.pth"))
        self.shared.batch_size = self.batch_size
        for t in range(self.num_of_tasks):
            self.taskOutput[f"task_{t}"].load_state_dict(torch.load(path + f"model_task_{t}.pth"))

    def __repr__(self):
        repr = ""
        repr += "Shared component: " + super().__repr__() + "\n"
        for t in range(self.num_of_tasks):
            repr += f'Task specific component of task {t}: {self.taskOutput[f"task_{t}"].__repr__()}' + "\n"
        return repr

class MultiTNShared(nn.Module):
    """Shared component"""
    def __init__(self, batch_size):
        self.batch_size = batch_size
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, 5)
        self.conv2 = nn.Conv2d(10, 20, 5)
        self.fc = nn.Linear(320, 50)

        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        x = x.transpose(4, 2)
        x = x.reshape([self.batch_size, 1, 28, 28])
        x = self.conv1(x)
        x = self.relu(self.max_pool(x))
        x = self.conv2(x)
        x = self.relu(self.max_pool(x))
        x = x.reshape([self.batch_size, 320])
        return self.relu(self.fc(x))


class MultiTNTaskSpecific(nn.Module):
    """ Task specific component """

    def __init__(self):
        super(MultiTNTaskSpecific, self).__init__()
        self.fc1 = nn.Linear(50, 10)

    def forward(self, x):
        return F.log_softmax(self.fc1(x), dim=1)