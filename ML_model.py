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
                nn.Identity(),
                nn.ReLU(inplace=True)
            )
            self.classifier = nn.Sequential(
                nn.Linear(28 * 28, num_classes),
            )

        elif layers == 3:
            self.features = nn.Sequential(
                nn.Linear(28 * 28, 2 * 28 * 28, bias=True),
                nn.ReLU(inplace=True),
                nn.Linear(2 * 28 * 28, 3 * 28 * 28, bias=True),
                nn.ReLU(inplace=True),
                nn.Linear(3 * 28 * 28, 2 * 28 * 28, bias=True),
                nn.ReLU(inplace=True)
            )
            self.classifier = nn.Sequential(
                nn.Linear(2 * 28 * 28, num_classes)
            )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

