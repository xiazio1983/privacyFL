# -*- coding: utf-8 -*-
# @Time : 2022/9/26 20:07
# @Author : jklujklu
# @Email : jklujklu@126.com
# @File : model.py
# @Software: PyCharm
import torchvision
from torch import nn
import torch.nn.functional as F

from utils import model_structure


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # (3,28,28) -> (32,26,26)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(5, 5), stride=1, padding=1)
        # (32,) -> (32,14,14)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # (32,14,14) -> (64,12,12)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5), stride=1, padding=1)
        # (64,12,12) -> (64,6,6)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # (64,6,6) -> (64,4,4)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5, 5), stride=1, padding=1)
        self.fl = nn.Flatten()
        self.fc1 = nn.Linear(4 * 4 * 64, 64)
        self.fc2 = nn.Linear(64, 43)

    def forward(self, inputs):
        tensor = inputs.view(-1, 3, 30, 30)
        tensor = F.relu(self.conv1(tensor))
        # print(tensor.shape)
        tensor = self.pool1(tensor)
        # print(tensor.shape)
        tensor = F.dropout(tensor, p=0.25)
        # print(tensor.shape)
        tensor = F.relu(self.conv2(tensor))
        # print(tensor.shape)
        tensor = self.pool2(tensor)
        # print(tensor.shape)
        tensor = F.dropout(tensor, p=0.25)
        tensor = F.relu(self.conv3(tensor))
        # tensor = self.fl(tensor
        #                  )
        # print(tensor.shape)
        tensor = tensor.view(-1, 4 * 4 * 64)
        tensor = F.relu(self.fc1(tensor))
        tensor = self.fc2(tensor)
        return tensor


class Linear(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        # (3,30,30) -> (32,28,28)
        self.bn = nn.BatchNorm1d(input_size)
        self.fc1 = nn.Linear(5, 255)
        self.fc2 = nn.Linear(255, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, inputs):
        tensor = inputs.view(-1, 5)
        tensor = self.bn(tensor)
        tensor = F.relu(self.fc1(tensor))
        tensor = F.dropout(tensor, p=0.25)

        tensor = F.relu(self.fc2(tensor))
        tensor = F.dropout(tensor, p=0.25)

        tensor = F.sigmoid(self.fc3(tensor))
        # tensor = tensor.squeeze()
        return tensor.squeeze()


class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            # nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            # nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            # nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(3 * 3 * 64, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


class CIFAR100Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = torchvision.models.resnet34()
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, 100)

    def forward(self, xb):
        return self.network(xb)


if __name__ == '__main__':
    net = ConvNet()
    model_structure(net)
