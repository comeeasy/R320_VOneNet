
import numpy as np
import torch
from torch import nn
from collections import OrderedDict


# AlexNet Back-End architecture
# Based on Torchvision implementation in
# https://github.com/pytorch/vision/blob/master/torchvision/models/alexnet.py
class AlexNetBackEnd(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(32, 192, kernel_size=5, stride=2, padding=2),
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


# ResNet Back-End architecture
# Based on Torchvision implementation in
# https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Identity(nn.Module):
    def forward(self, x):
        return x




    def forward(self, inp):
        x = self.conv_input(inp)

        for t in range(self.times):
            if t == 0:
                skip = self.norm_skip(self.skip(x))
                self.conv2.stride = (2, 2)
            else:
                skip = x
                self.conv2.stride = (1, 1)

            x = self.conv1(x)
            x = getattr(self, f'norm1_{t}')(x)
            x = self.nonlin1(x)

            x = self.conv2(x)
            x = getattr(self, f'norm2_{t}')(x)
            x = self.nonlin2(x)

            x = self.conv3(x)
            x = getattr(self, f'norm3_{t}')(x)

            x += skip
            x = self.nonlin3(x)
            output = self.output(x)

        return output

import torch.functional as F
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # For the sake of studying ML, I will use deeper network
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=3)
        self.fc1 = nn.Sequential(
            nn.Linear(512, 512),
            # nn.ReLU(inplace=8
            nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True)
            # nn.Tanh()
        )
        self.fc3 = nn.Linear(512, 10)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        # print(x.shape)

        x = self.pool(self.relu(self.conv1(x)))
        # Batch Normalization(x)
        x = self.pool(self.relu(self.conv2(x)))
        x = self.conv3(x)

        # Flattening
        before_fc = x.size(0)
        x = x.view(x.size(0), -1)

        # print(x.shape)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x

class Basic_Linear_Regression(nn.Module):
    def __init__(self):
        super(Basic_Linear_Regression, self).__init__()

        self.voneblock_connector = nn.Linear(1 * 28 * 28, 28 * 28, bias=True)
        self.fc1 = nn.Linear(28 * 28, 28 * 28, bias=True)
        self.fc2 = nn.Linear(28 * 28, 10, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.flatten = nn.Flatten()


    def forward(self, x):
        x = self.flatten(x)

        # for VOneBlock
        x = self.voneblock_connector(x)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x

class Basic_CNN(nn.Module):
    def __init__(self):
        super(Basic_CNN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc = nn.Linear(7 * 7 * 64, 10, bias=True)

        nn.init.xavier_uniform(self.fc.weight)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out