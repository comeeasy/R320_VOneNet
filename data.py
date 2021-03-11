from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets

def get_mnist(batch_size=64) :
    # download data
    mnist_train_data = datasets.MNIST(root="./MNIST_data",
                                      train=True,
                                      transform=transforms.ToTensor(),
                                      download=False)
    mnist_val_data = datasets.MNIST(root="./MNIST_data",
                                      train=False,
                                      transform=transforms.ToTensor(),
                                      download=False)

    # DataLoad
    mnist_train_dataloader = DataLoader(dataset=mnist_train_data,
                                        shuffle=True,
                                        batch_size=batch_size,
                                        drop_last=True)

    mnist_val_dataloader = DataLoader(dataset=mnist_val_data,
                                      batch_size=batch_size,
                                      drop_last=True)


    return mnist_train_dataloader, mnist_val_dataloader









