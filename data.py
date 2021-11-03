from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os

def get_mnist(batch_size=64, image_size=224) :
    # download data
    mnist_train_data = datasets.MNIST(root="./MNIST_data",
                                      train=True,
                                      transform=transforms.Compose([
                                          transforms.ToTensor(),
                                          transforms.Resize(image_size)
                                      ]),
                                      download=True)
    mnist_val_data = datasets.MNIST(root="./MNIST_data",
                                    train=False,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Resize(image_size)
                                    ]),
                                    download=True)

    # DataLoad
    mnist_train_dataloader = DataLoader(dataset=mnist_train_data,
                                        shuffle=True,
                                        batch_size=batch_size,
                                        drop_last=True,
                                        num_workers=8)

    mnist_val_dataloader = DataLoader(dataset=mnist_val_data,
                                      shuffle=True,
                                      batch_size=batch_size,
                                      drop_last=True,
                                      num_workers=8)


    return mnist_train_dataloader, mnist_val_dataloader

def get_imagenet(root: str, img_size: tuple, batch_size: int, num_worker=4):
    """
        parameters: root: str, img_size: int, batch_size: int
        returns: (train_dataset, val_dataset)

        root:
            dataset file structure must be same as below
        ===================
            root/
                |-train
                |-val
        ===================

        img_size:
            Resize images as of img_size

        batch_size:
            batch_size
    """

    train_path = os.path.join(root, 'train')
    val_path = os.path.join(root, 'val')

    imagenet_train = ImageFolder(
        root=train_path,
        transform=transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor()
        ])
    )

    imagenet_val = ImageFolder(
        root=val_path,
        transform=transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor()
        ])
    )

    train_dataloader = DataLoader(
        batch_size=batch_size,
        dataset=imagenet_train,
        shuffle=True,
        num_workers=num_worker,
        drop_last=True,
    )

    val_dataloader = DataLoader(
        batch_size=batch_size,
        dataset=imagenet_val,
        shuffle=True,
        num_workers=num_worker,
        drop_last=True,
    )

    return (train_dataloader, val_dataloader)

def get_damegenet(root: str, img_size: tuple, batch_size: int, num_worker=4):
    """
        parameters: root: str, img_size: int, batch_size: int
        returns: adversarial_val_dataset

        root:
            dataset file structure must be same as below
        ===================
            root/
                |-DAmagenet
        ===================

        img_size:
            Resize images as of (img_size, img_size)

        batch_size:
            batch_size
    """
    val_path = os.path.join(root, 'DAmageNet')

    imagenet_val = ImageFolder(
        root=val_path,
        transform=transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor()
        ])
    )

    val_dataloader = DataLoader(
        batch_size=batch_size,
        dataset=imagenet_val,
        shuffle=True,
        num_workers=num_worker,
        drop_last=True,
    )

    return val_dataloader


if __name__ == "__main__" :
    print(torch.cuda.is_available())











