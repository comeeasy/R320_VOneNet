from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets

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
                                        drop_last=True)

    mnist_val_dataloader = DataLoader(dataset=mnist_val_data,
                                      shuffle=True,
                                      batch_size=batch_size,
                                      drop_last=True)


    return mnist_train_dataloader, mnist_val_dataloader

if __name__ == "__main__" :
    get_mnist()









