# Fast Gradient Sign Method using simple CNN and MNIST
#
#
# https://www.pyimagesearch.com/2021/03/01/adversarial-attacks-with-fgsm-fast-gradient-sign-method/
# The above blog used Keras & TensorFlow

# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torch.utils.data as utils
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn.functional as f
import numpy as np
import matplotlib.pyplot as plt
import random
import tqdm

# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# Hyper-parameters
learning_rate = 1e-3
batch_size = 128
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
eps = 0.35

# Load datasets(MNIST)
path = './'
MNIST_t = dsets.MNIST(path, download=True, train=True, transform=transforms.ToTensor())
MNIST_val = dsets.MNIST(path, download=True, train=False, transform=transforms.ToTensor())

data_t = utils.DataLoader(MNIST_t, shuffle=True, batch_size=batch_size)
data_val = utils.DataLoader(MNIST_val, shuffle=True, batch_size=batch_size, drop_last=True)

class_name = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# Network
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        '''
        # 1st conv layer = 32 3*3 filters, with 2*2 stride
        # CNN => ReLU => Batch Normalization(which is skipped in this code)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), stride=(2, 2))
        # No padding
        # No Pooling

        # 2nd conv layer = 64 3*3 filters, with 2*2 stride
        # CNN => ReLU => Batch Normalization(which is skipped in this code)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(2, 2))
        # No padding
        # No Pooling

        # 1st and only FC
        # FC => ReLU (=> Batch Normalization)=> Dropout(0.5)
        # Since the data is MNIST, the model has to output 10 values
        # Linear( W_out * H_out * N_channel, N_classes)
        self.fc = nn.Linear(6 * 6 * 64, 10)


        # The blog used MSE and Softmax as loss function
        # In this code, I will use CrossEntropyLoss to do them at the same time
        '''

        # For the sake of studying ML, I will use deeper network
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5), stride=(2, 2), padding=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(3 * 3 * 128, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 10)

    def forward(self, x):
        x = self.pool(f.relu(self.conv1(x)))
        # Batch Normalization(x)
        x = self.pool(f.relu(self.conv2(x)))
        x = self.conv3(x)
        # Batch Normalization(x)

        # Flattening
        x = x.view(x.size(0), -1)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)

        return x


# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


# Training loop
def training_loop(n_epoch, network, learning_rate, loss_fn, data_t, adv=False):
    optimizer = optim.Adam(network.parameters(), lr=learning_rate)

    if adv == False:
        for i in range(0, n_epoch):
            loss_sum = 0
            for target in tqdm.tqdm(data_t):
                img, label = target
                img = img.to(device)
                label = label.to(device)
                # forward
                prediction = network(img)
                loss_train = loss_fn(prediction, label)
                loss_sum += loss_train.item()
                # backward
                optimizer.zero_grad()
                loss_train.backward()
                optimizer.step()

                # print(f"Epoch: {i + 1}, Batch: {n * batch_size}/60000, Training loss {loss_train.item():.4f},")
            print(f'\naverage loss in {i + 1}th epoch: {loss_sum / ((batch_size) + 1)}')
    else:
        for i in range(0, n_epoch):
            loss_sum = 0
            for target in tqdm.tqdm(data_t):
                _, label = target
                adv_img, label = generate_image_adversary(data=target, loss_fn=loss_fn, model=network)
                adv_img = adv_img.to(device)
                label = label.to(device)
                # forward
                prediction = network(adv_img)
                loss_train = loss_fn(prediction, label)
                loss_sum += loss_train.item()
                # backward
                optimizer.zero_grad()
                loss_train.backward()
                optimizer.step()

                # print(f"Epoch: {i + 1}, Batch: {n * batch_size}/60000, Training loss {loss_train.item():.4f},")
            print(f'\naverage loss in {i + 1}th epoch: {loss_sum / ((batch_size) + 1)}')
# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


# We pass an image through the trained-model, and calculate the loss of the image
# Then, we use the sign of the gradiant of loss to generate adversarial attack

# adv=original+ϵ∗sign(∇xJ(x,θ))
def generate_image_adversary(model, loss_fn, data, eps=0.03):

    img, label = data

    # Adding randomness

    len = img.size(0)
    rand_idx = torch.randperm(len)
    tmp = img.clone()
    tmp_l = label.clone()

    for i in range(len):
        tmp[i] = img[rand_idx[i]]
        tmp_l[i] = label[rand_idx[i]]

    tmp = tmp.to(device)
    tmp_l = tmp_l.to(device)

    tmp.requires_grad = True


    pred = model(tmp).to(device)

    loss = loss_fn(pred, tmp_l).to(device)
    # we need to calculate ∇xJ(x,θ)
    loss.backward()
    tmp.requires_grad = False

    tmp = tmp + eps * tmp.grad.data.sign()
    tmp = torch.clamp(tmp, 0, 1)

    return tmp, tmp_l


# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


def accuracy(model, data, classes, attack=False, eps=eps):
    print('Calculating Accuracy...')
    if attack == False:
        with torch.no_grad():
            n_correct = 0
            n_samples = 0
            acc_arr = [0 for i in range(classes)]
            acc = 0

            n_class_correct = [0 for i in range(10)]
            n_class_samples = [0 for i in range(10)]

            for target in tqdm.tqdm(data):
                images, labels = target
                images = images.to(device)
                labels = labels.to(device)
                with torch.no_grad():
                    outputs = model(images).to(device)
                    # max returns (value ,index)
                    _, predicted = torch.max(outputs, 1)

                    n_samples += len(labels)
                    n_correct += (predicted == labels).sum().item()

                    for i in range(images.size(0)):
                        label = labels[i]
                        pred = predicted[i]

                        if (label == pred):
                            n_class_correct[label] += 1
                        n_class_samples[label] += 1

            acc = 100.0 * n_correct / n_samples
            for i in range(10):
                acc_arr[i] = 100.0 * n_class_correct[i] / n_class_samples[i]
            return acc, acc_arr
    else:
        n_correct = 0
        n_samples = 0
        acc_arr = [0 for i in range(classes)]
        acc = 0

        n_class_correct = [0 for i in range(10)]
        n_class_samples = [0 for i in range(10)]

        for target in tqdm.tqdm(data):
            adv_img, labels = generate_image_adversary(model=model, loss_fn=loss, data=target, eps=eps)
            with torch.no_grad():
                adv_img = adv_img.to(device)
                labels = labels.to(device)
                outputs = model(adv_img).to(device)
                # max returns (value ,index)
                _, predicted = torch.max(outputs, 1)

                n_samples += len(labels)
                n_correct += (predicted == labels).sum().item()

                for i in range(adv_img.size(0)):
                    label = labels[i]
                    pred = predicted[i]

                    if (label == pred):
                        n_class_correct[label] += 1
                    n_class_samples[label] += 1

        acc = 100.0 * n_correct / n_samples
        for i in range(10):
            acc_arr[i] = 100.0 * n_class_correct[i] / n_class_samples[i]
        return acc, acc_arr


# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# Optimizer & loss_fn
model = ConvNet().to(device)
loss = nn.CrossEntropyLoss()

print('\nTraining the model')
n_epochs = 1
training_loop(
    n_epoch=n_epochs,
    network=model,
    learning_rate=learning_rate,
    loss_fn=loss,
    data_t=data_t,
    adv=False)

# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


tot_acc, idv_acc = accuracy(model, data_val, 10, attack=False)
print(f'Accuracy of the network: {tot_acc} %')
#for i in range(10):
 #   print(f'Accuracy of {class_name[i]}: {idv_acc[i]} %')


tot_acc, idv_acc = accuracy(model, data_val, 10, attack=True)
print(f'Accuracy of the network after attacked: {tot_acc} %')
#for i in range(10):
 #   print(f'Accuracy of {class_name[i]}: {idv_acc[i]} %')

learning_rate = 1e-4
print('\nre-Training the model')
n_epochs = 10
training_loop(
    n_epoch=n_epochs,
    network=model,
    learning_rate=learning_rate,
    loss_fn=loss,
    data_t=data_t,
    adv=True)

tot_acc, idv_acc = accuracy(model, data_val, 10, attack=True)
print(f'Accuracy of the fine-turned network after attacked: {tot_acc} %')
#for i in range(10):
 #   print(f'Accuracy of {class_name[i]}: {idv_acc[i]} %')