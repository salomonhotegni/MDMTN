import torch
from torchvision import transforms
from Data_loaders.Load_MultiMnist import MNIST

def global_transformer():
    return transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))])


def get_dataset(params, configs, transformers = [None, None]):

    train_transform = transformers[0]
    valid_transform = transformers[1]

    if train_transform is None: train_transform = global_transformer()
    if valid_transform is None: valid_transform = global_transformer()

    if 'dataset' not in params:
        print('ERROR: No dataset is specified')

    if 'mnist' in params['dataset']:
        train_dst = MNIST(root=configs['mnist']['path'], train=True, download=True, transform=train_transform, multi=True)
        train_loader = torch.utils.data.DataLoader(train_dst, batch_size=params['batch_size'][0], shuffle=True, num_workers=4)

        val_dst = MNIST(root=configs['mnist']['path'], train=False, download=True, transform=valid_transform, multi=True)
        val_loader = torch.utils.data.DataLoader(val_dst, batch_size=params['batch_size'][1], shuffle=True, num_workers=4)
        return train_loader, train_dst, val_loader, val_dst