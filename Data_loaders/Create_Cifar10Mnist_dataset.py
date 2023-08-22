from PIL import Image, ImageEnhance
import numpy as np
import torch
import pickle
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10, MNIST

def pad_image(cifar_image, mnist_image, alpha=128, brightness=1.0):
    '''
    As a MNIST digit is in white color with a completely black background, it is enough to make all black pixel of the images of MNIST digits transparent before the padding.
    
    When a MNIST digit image is pasted onto a new RGBA image with `mask = alpha_mask`, the black pixels in the MNIST digit image (which have a grayscale value of 0) are treated as completely transparent in the new image, while the white pixels (which have a grayscale value of 255) are treated as slightly opaque. The pixels with grayscale values between 0 and 255 are treated as partially transparent, with the level of transparency depending on their brightness value.    
    '''
    
    # Convert the CIFAR image to a PIL image
    pil_image = Image.fromarray(np.array(cifar_image))

    # Convert the MNIST image to a PIL image and resize it to fit the CIFAR image
    mnist_image = Image.fromarray(np.array(mnist_image), mode='L')
    mnist_image = mnist_image.resize((32, 32))

    # Create a new RGB image
    padded_image = Image.new('RGB', (32, 32), (0, 0, 0))

    # Create an alpha channel mask with the white pixels having a slightly transparent alpha value
    alpha_mask = ImageEnhance.Brightness(mnist_image).enhance(brightness).point(lambda x: alpha if x > 0 else 0)

    # Paste the CIFAR image onto the new image
    padded_image.paste(pil_image, (0, 0))

    # Paste the MNIST image onto the new image, with the black pixels made transparent and the white pixels slightly transparent
    padded_image.paste(mnist_image, (0, 0), mask=alpha_mask)

    # Convert the padded image to a numpy array and return it
    padded_image = np.array(padded_image)
    return padded_image

# Pad the MNIST digits over the CIFAR images
def train_val_padding(cifar_dataset, mnist_dataset, all_mnist = False):
    '''
    As MNIST data has 60000 images in the train set against 50000 for CIFAR10 data, we can do the padding of the first 50000 MNIST digits on the 50000 CIFAR10 images. Then, we will pad the remaining 10000 MNIST digits on 10000 random CIFAR10 images (with a fixed seed). On the validation set, this has no effect because they have the same number of images: 10000.

    This function gives the choice between: using 60000 images (all_mnist = True) for training and using 50000 images (all_mnist = False).
    '''
    
    # Set the random seed for reproducibility
    np.random.seed(123)
    padded_images = []
    N = len(cifar_dataset)
    if all_mnist:
        N = len(mnist_dataset)
    for i in range(N):
        mnist_image, mnist_label = mnist_dataset[i]
        if i < len(cifar_dataset):
            cifar_image, cifar_label = cifar_dataset[i]
        else:
            # If we've already padded all the CIFAR images, then pad a random one
            j = np.random.randint(len(cifar_dataset))
            cifar_image, cifar_label = cifar_dataset[j]
        padded_image = pad_image(cifar_image, mnist_image)
        # The label of the new image is a couple of the two labels
        padded_images.append((padded_image, (cifar_label, mnist_label)))

    return padded_images

# Define a custom dataset class for the padded images
class PaddedDataset(torch.utils.data.Dataset):
    def __init__(self, padded_images, transform=None):
        self.padded_images = padded_images
        self.transform = transform

    def __getitem__(self, index):
        image, label = self.padded_images[index]
        if self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.padded_images)


def create_cifar10mnist():
    # Load the CIFAR and MNIST datasets'
    train_cifar_dataset = CIFAR10(root='Data/Cifar10Mnist/', train=True, download= True)
    test_cifar_dataset = CIFAR10(root='Data/Cifar10Mnist/', train=False, download= True)
    train_mnist_dataset = MNIST(root='Data/Cifar10Mnist/', train=True, download= True)
    test_mnist_dataset = MNIST(root='Data/Cifar10Mnist/', train=False, download= True)

    train_60k_images = train_val_padding(train_cifar_dataset, train_mnist_dataset, all_mnist = True)
    train_50k_images = train_val_padding(train_cifar_dataset, train_mnist_dataset)
    test_10k_images = train_val_padding(test_cifar_dataset, test_mnist_dataset)

    # Save the padded images to a file
    with open('Data/Cifar10Mnist/train_60k_CIFAR_MNIST.pkl', 'wb') as f:
        pickle.dump(train_60k_images, f)
    with open('Data/Cifar10Mnist/train_50k_CIFAR_MNIST.pkl', 'wb') as f:
        pickle.dump(train_50k_images, f)
    with open('Data/Cifar10Mnist/test_10k_CIFAR_MNIST.pkl', 'wb') as f:
        pickle.dump(test_10k_images, f)
    
    return train_60k_images, train_50k_images, test_10k_images

if __name__ == "__main__":
    
    train_60k_images, train_50k_images, test_10k_images = create_cifar10mnist()
    
    # Classes for CIFAR-10
    cifar10_classes = ('Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
            'Dog', 'Frog', 'Horse', 'Ship', 'Truck')

    import matplotlib.pyplot as plt

    # Get the first 10 images from the train dataset
    images, labels = zip(*train_50k_images[:10])

    # Create a figure with subplots for each image
    fig, axes = plt.subplots(nrows=2, ncols=5)

    # Loop through the images and labels and display them in the subplots
    for i, (image, label) in enumerate(zip(images, labels)):
        row = i // 5
        col = i % 5
        ax = axes[row, col]
        ax.imshow(image)
        ax.set_title(f'{(cifar10_classes[label[0]], label[1])}')
        ax.axis('off')
    plt.tight_layout()
    plt.savefig("Data/Cifar10Mnist/first_10_images.png")
    plt.show()
