# Multi-Objective Optimization for Sparse Deep Neural Network Training

Contained within this repository is the source code corresponding to our paper titled "Multi-Objective Optimization for Sparse Deep Neural Network Training". A new Multi-Task Learning model architecture is introduced, namely the Monitored Deep Multi-Task Network (MDMTN). The framework for conducting experiments is based on PyTorch. The code has been tested within the Python 3 environment using JupiterLab, and necessitates the use of some Python packages, namely ```pytorch, torchvision, numpy, Pillow, scipy,```
and ```sklearn.``` 
We provide the (code to generate the) newly introduced Cifar10Mnist dataset in ```/Data_loaders/Create_Cifar10Mnist_dataset.py```.

<div style="text-align:center;">
    <img src="/Images/MDMTN_diag_new.jpg" alt="Monitored Deep Multi-Task Network" width="550" height="300">
</div>

<figure class="image">
  <img src="/Images/MDMTN_diag_new.jpg">
  <figcaption>Figure 1. Diagram of the Monitored Deep Multi-Task Network (MDMTN) with 2 main tasks. The task-specific monitors are designed to capture task-specific information that the shared network may miss.</figcaption>
</figure>

## USAGE

The script ```Train_and_Test.py```
gathers the helper functions from ```/src/utils/``` for training and testing a model on a dataset, given a preference vector.

To train a model for a specific preference vector $k$ while inducing sparsity, execute the scripts named ```“Example_{architecture}_{dataset}_withSparsity.py”```. 
The architectures are denoted as “MDMTN” and “HPS” (Hard Parameter Sharing), and the acronyms “MM” and “CM” represent the MultiMNIST and Cifar10Mnist datasets, respectively. STL stands for Single Task Learning model architectures. To exclude the secondary objective that induces sparsity, execute the scripts labeled as ```“Example_{architecture}_{dataset}_noSparsity.py”```.

For training a model with the Multiple Gradient Descent Algorithm (MGDA), run the scripts named ```“Example_{architecture}_MGDA_{dataset}.py”```.

Use the scripts named `"Pareto_front_study_{architecture}_{dataset}"` for the additional (2D) Pareto front study we conducted on both model architectures. This required first finding the preference vector that yields the best model performance and saving the obtained model in the directory `src/Sparse_models/`.

#### Parameters used for the helper functions:
- `w: (list)` - Preference vector k.
- `a: (list)` - Reference point.
- `epsilon: (real: 0-1)` - Augmentation term coefficient in the modified Weighted Chebyshev scalarization method.
- `num_tasks: (int)` - Number of tasks considered.
- `max_iter: (int)` - Number of iterations by default.
- `max_iter_search: (int)` - Number of iterations during the first phase of the training algorithm.
- `max_iter_retrain: (int)` - Number of iterations during the second phase of the training algorithm.
- `num_epochs: (int)` - Number of epochs per iteration.
- `tol_epochs: (int)` - Maximum number of epochs to wait if there is no improvement in model performance.
- `lr: (real)` - Initialization of the learning rate.
- `LR_scheduler: (True/False)` - Whether to reduce the learning rate after a certain period.
- `lr_sched_coef: (real)` - Reduction coefficient of the learning rate.
- `mu: (real)` - Lagrangian multiplier μ.
- `rho: (real)` - Coefficient for updating the Lagrangian multiplier μ.
- `min_sparsRate: (%)` - Minimum sparsity rate for a model to be saved.
- `max_layerSRate: (real: 0-1)` - Maximum sparsity rate for a layer.
- `sim_preference: (real: 0-1)` - Similarity preference in the Affinity Propagation method.
- `skip_layer: (int)` - The layers that have this number of neurons are skipped when applying GrOWL (preferably the input layer).
- `is_search: (True/False)` - True: for the first training phase (searching for sparse model); False: for the second phase (Forcing parameter sharing).
- `Sparsity_study: (True/False)` - True: for sparsity study; False: for Pareto front study
- `base_optimizer: (torch.optim)` - Optimizer.
- `criterion: (torch.nn.functional)` - Criterion.
- `num_batchEpoch: (int)` - Number of batches to use for an epoch.
- `num_model: (int)` - Model number.
- `main_dir: (directory)` - Main directory for saving the training results.
- `mod_logdir: (directory)` - Directory for saving the model.


## REFERENCES

We utilize specific code segments from Sener Ozan and Vladlen Koltun's work 'Multi-task learning as multi-objective optimization' to train a model using the Multiple Gradient Descent Algorithm (MGDA). Additionally, we incorporate code from Zhang Dejiao et al.'s paper 'Learning to Share: Simultaneous Parameter Tying and Sparsification in Deep Learning' to implement GrOWL as the secondary objective function used. Moreover, we draw from the work of Mishkin Dmytro and Jiri Matas in 'All You Need is a Good Init' for the initialization of our models.

