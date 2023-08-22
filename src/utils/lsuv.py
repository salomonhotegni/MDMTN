#######################################################
#### Layer-Sequential Unit-Variance Initialization ####
#######################################################
# Introduced by Mishkin et al. in "All you need is a good init" (https://arxiv.org/abs/1511.06422v7)
 
# Layer-Sequential Unit-Variance Initialization (LSUV) is a simple method for weight initialization for deep net learning. The initialization strategy involves the following two step:

# 1) First, pre-initialize weights of each convolution or inner-product layer with orthonormal matrices.

# 2) Second, proceed from the first to the final layer, normalizing the variance of the output of each layer to be equal to one.



# Adapted from https://github.com/shunk031/LSUV.pytorch
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class LSUVInit(object):

    def __init__(self,
                 model: nn.Module,
                 data_loader: DataLoader,
                 needed_std: float = 1.0,
                 std_tol: float = 0.1,
                 max_attempts: int = 10,
                 do_orthonorm: bool = True,
                 device: torch.device = 'str') -> None:
        self._model = model
        self.data_loader = data_loader
        self.needed_std = needed_std
        self.std_tol = std_tol
        self.max_attempts = max_attempts
        self.do_orthonorm = do_orthonorm
        self.device = device

        self.eps = 1e-8
        self.hook_position = 0
        self.total_fc_conv_layers = 0
        self.done_counter = -1
        self.hook = None
        self.act_dict: np.ndarray = None
        self.counter_to_apply_correction = 0
        self.correction_needed = False
        self.current_coef = 1.0

    def svd_orthonormal(self, w: np.ndarray) -> np.ndarray:
        shape = w.shape
        if len(shape) < 2:
            raise RuntimeError("Only shapes of length 2 or more are supported.")
        flat_shape = (shape[0], np.prod(shape[1:]))
        a = np.random.normal(0.0, 1.0, flat_shape)  # w;
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v
        print(shape, flat_shape)
        q = q.reshape(shape)
        return q.astype(np.float32)

    def count_conv_fc_layers(self, m: nn.Module) -> None:
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            self.total_fc_conv_layers += 1

    def orthogonal_weights_init(self, m: nn.Module) -> None:
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            if hasattr(m, 'weight_v'):
                w_ortho = self.svd_orthonormal(m.weight_v.data.cpu().numpy())
                m.weight_v.data = torch.from_numpy(w_ortho)
                try:
                    nn.init.constant_(m.bias, 0)
                except Exception:
                    pass
            else:
                w_ortho = self.svd_orthonormal(m.weight.data.cpu().numpy())
                m.weight.data = torch.from_numpy(w_ortho)
                try:
                    nn.init.constant_(m.bias, 0)
                except Exception:
                    pass

    def store_activations(self,
                          module: nn.Module,
                          data: torch.Tensor,
                          output: torch.Tensor) -> None:
        self.act_dict = output.detach().cpu().numpy()

    def add_current_hook(self, m: nn.Module) -> None:
        if self.hook is not None:
            return
        if (isinstance(m, nn.Conv2d)) or (isinstance(m, nn.Linear)):
            if self.hook_position > self.done_counter:
                self.hook = m.register_forward_hook(self.store_activations)
            else:
                self.hook_position += 1

    def apply_weights_correction(self, m: nn.Module) -> None:
        if self.hook is None:
            return
        if not self.correction_needed:
            return
        if (isinstance(m, nn.Conv2d)) or (isinstance(m, nn.Linear)):
            if self.counter_to_apply_correction < self.hook_position:
                self.counter_to_apply_correction += 1
            else:
                if hasattr(m, 'weight_g'):
                    m.weight_g.data *= float(self.current_coef)
                    self.correction_needed = False
                else:
                    m.weight.data *= self.current_coef
                    self.correction_needed = False

    def initialize(self) -> nn.Module:
        model = self._model
        model.eval()

        model.apply(self.count_conv_fc_layers)
        if self.do_orthonorm:
            model.apply(self.orthogonal_weights_init)

        model = model.to(self.device)
        for layer_idx in range(self.total_fc_conv_layers):
            print(layer_idx)
            model.apply(self.add_current_hook)
            data = next(iter(self.data_loader))
            data, _ = [d for d in data]
            data = data.to(self.device)
            model(data)
            current_std = self.act_dict.std()
            print('std at layer ', layer_idx, ' = ', current_std)

            attempts = 0
            while (np.abs(current_std - self.needed_std) > self.std_tol):
                self.current_coef = self.needed_std / (current_std + self.eps)
                self.correction_needed = True
                model.apply(self.apply_weights_correction)

                model = model.to(self.device)
                model(data)
                current_std = self.act_dict.std()
                print('std at layer ', layer_idx, ' = ', current_std, 'mean = ', self.act_dict.mean())
                attempts += 1
                if attempts > self.max_attempts:
                    break

            if self.hook is not None:
                self.hook.remove()

            self.done_counter += 1
            self.counter_to_apply_correction = 0
            self.hook_position = 0
            self.hook = None
            print('finish at layer', layer_idx)

        print('LSUV init done!')
        return model


def lsuv_init(model: nn.Module,
              data_loader: DataLoader,
              needed_std: float,
              std_tol: float,
              max_attempts: int,
              do_orthonorm: bool,
              device: torch.device) -> nn.Module:

    return LSUVInit(
        model, data_loader, needed_std, std_tol,
        max_attempts, do_orthonorm, device).initialize()
