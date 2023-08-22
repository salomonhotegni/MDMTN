from os import path
import torch
from torch import Tensor
from torchvision import transforms
import pickle
from Data_loaders.Create_Cifar10Mnist_dataset import PaddedDataset, create_cifar10mnist

def Cifar10Mnist_loaders(data_path, split_rate, transformers = [None, None], batch_size = [256, 256]):

    if not path.exists(path.join(data_path, "test_10k_CIFAR_MNIST.pkl")) or not path.exists(path.join(data_path, "train_50k_CIFAR_MNIST.pkl")) or not path.exists(path.join(data_path, "train_60k_CIFAR_MNIST.pkl")):
         train_60k_images, train_50k_images, test_10k_images = create_cifar10mnist()
    else:
        with open(f'{data_path}/train_60k_CIFAR_MNIST.pkl', 'rb') as f:
            train_60k_images = pickle.load(f)
        with open(f'{data_path}/train_50k_CIFAR_MNIST.pkl', 'rb') as f:
            train_50k_images = pickle.load(f)
        with open(f'{data_path}/test_10k_CIFAR_MNIST.pkl', 'rb') as f:
            test_10k_images = pickle.load(f)

    def Transform(transform):
        if transform is None:
            return transforms.Compose([ transforms.ToTensor(),transforms.Lambda(lambda x: x.permute(0, 1, 2))])
        else:
            return transform

    # Create PyTorch datasets for the padded images
    train_60k_dataset = PaddedDataset(train_60k_images, transform = Transform(transformers[0]))
    train_50k_dataset = PaddedDataset(train_50k_images, transform = Transform(transformers[1]))
    test_10k_dataset = PaddedDataset(test_10k_images, transform = Transform(transformers[1]))

    # Split the train_loader into train and validation loaders
    if (split_rate <= 0) or (split_rate >= 1):
        raise ValueError(" Use 0<split_rate<1")
    train_size = int(split_rate * len(train_50k_dataset))
    val_size = len(train_50k_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_50k_dataset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size[0], shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size[1], shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_10k_dataset, batch_size=batch_size[1], shuffle=False)

    return train_loader, val_loader, test_loader
