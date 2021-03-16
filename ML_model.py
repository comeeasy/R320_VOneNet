import torch
import torch.optim as optim
import torch.nn as nn

# https://github.com/dicarlolab/vonenet
# part of vonenet/vonenet/back_ends.py
class AlexNetBackEnd(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 192, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class MNIST_net(nn.Module):
    def __init__(self, layers=3 ,num_classes=10):
        super().__init__()

        if layers == 1:
            self.features = nn.Sequential(
                nn.Linear(28 * 28, 28 * 28, bias=True),
                nn.ReLU(inplace=True)
            )
            self.classifier = nn.Sequential(
                nn.Linear(28 * 28, num_classes),
            )

        elif layers == 3:
            self.features = nn.Sequential(
                nn.Linear(28 * 28, 28 * 28, bias=True),
                nn.ReLU(inplace=True),
                nn.Linear(28 * 28, 28 * 28, bias=True),
                nn.ReLU(inplace=True),
                nn.Linear(28 * 28, 28 * 28, bias=True),
                nn.ReLU(inplace=True),
            )
            self.classifier = nn.Sequential(
                nn.Linear(28 * 28, num_classes)
            )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# 병철 model
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

