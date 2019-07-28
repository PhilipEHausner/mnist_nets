"""
Model definitions
"""

import torch as t
import torch.nn as nn


class OVRNet(nn.Module):
    """

    """
    def __init__(self, kernel_size=3, num_filters=8, num_classes=10):

        super(OVRNet, self).__init__()

        self.kernel_size = kernel_size
        self.num_filters = num_filters
        self.padding = kernel_size // 2

        self.num_classes = num_classes

        self.conv1 = [] # 1 in, 8 out
        self.conv2 = [] # 8 in, 16 out
        self.conv3 = [] # 16 in, 32 out
        self.fc1 = [] # 49*32 in, 128 out
        self.fc2 = [] # 128, 1

        for _ in range(self.num_classes):
            self.conv1.append(nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=self.num_filters, kernel_size=self.kernel_size,
                          padding=self.padding),
                nn.MaxPool2d(2),
                nn.BatchNorm2d(self.num_filters),
                nn.ReLU()
            ))

            self.conv2.append(nn.Sequential(
                nn.Conv2d(in_channels=self.num_filters, out_channels=self.num_filters*2, kernel_size=self.kernel_size,
                          padding=self.padding),
                nn.MaxPool2d(2),
                nn.BatchNorm2d(self.num_filters*2),
                nn.ReLU()
            ))

            self.conv3.append(nn.Sequential(
                nn.Conv2d(in_channels=self.num_filters*2, out_channels=self.num_filters*4, kernel_size=self.kernel_size,
                          padding=self.padding),
                nn.MaxPool2d(2),
                nn.BatchNorm2d(self.num_filters*4),
                nn.ReLU()
            ))

            self.fc1.append(nn.Sequential(
                nn.Linear(self.num_filters*4*(3*3), 128),
                nn.ReLU()
            ))

            self.fc2.append(nn.Sequential(
                nn.Linear(128, 1),
                nn.Sigmoid()
            ))

        self.conv1 = nn.ModuleList(self.conv1)
        self.conv2 = nn.ModuleList(self.conv2)
        self.conv3 = nn.ModuleList(self.conv3)
        self.fc1 = nn.ModuleList(self.fc1)
        self.fc2 = nn.ModuleList(self.fc2)

    def forward(self, x):
        batch_size, _, _, _ = x.shape
        temp = [0] * self.num_classes

        output = t.zeros([batch_size, self.num_classes])

        # self.conv1out = [self.conv1[i](x) for i in range(self.num_classes)]
        for i in range(self.num_classes):
            temp[i] = self.conv1[i](x)
            temp[i] = self.conv2[i](temp[i])
            temp[i] = self.conv3[i](temp[i])
            temp[i] = temp[i].view(temp[i].size(0), -1)

            temp[i] = self.fc1[i](temp[i])

            output[:, i] = self.fc2[i](temp[i]).squeeze(dim=1)

        return output


class ReferenceNet(nn.Module):
    def __init__(self, kernel=3, num_filters=8):
        super(ReferenceNet, self).__init__()
        self.padding = kernel // 2

        self.conv1 = t.nn.Sequential(
            t.nn.Conv2d(in_channels=1, out_channels=num_filters,
                        kernel_size=kernel, padding=self.padding),
            t.nn.MaxPool2d(2),
            t.nn.BatchNorm2d(num_filters),
            t.nn.ReLU())

        self.conv2 = t.nn.Sequential(
            t.nn.Conv2d(in_channels=num_filters, out_channels=num_filters * 2,
                        kernel_size=kernel, padding=self.padding),
            t.nn.MaxPool2d(2),
            t.nn.BatchNorm2d(num_filters * 2),
            t.nn.ReLU())

        self.conv3 = t.nn.Sequential(
            t.nn.Conv2d(in_channels=num_filters * 2, out_channels=num_filters * 4,
                        kernel_size=kernel, padding=self.padding),
            t.nn.MaxPool2d(2, padding=1),
            t.nn.BatchNorm2d(num_filters * 4),
            t.nn.ReLU())

        self.fc1 = t.nn.Sequential(
            t.nn.Linear(num_filters * 4 * (4 * 4), 128),
            t.nn.ReLU())

        self.fc2 = t.nn.Sequential(
            t.nn.Linear(128, 128),
            t.nn.ReLU())

        self.soft = t.nn.Linear(128, 10)

    def forward(self, x):
        self.convout1 = self.conv1(x)
        self.convout2 = self.conv2(self.convout1)
        self.convout3 = self.conv3(self.convout2)
        self.flatview = self.convout3.view(self.convout3.size(0), -1)
        self.fcout1 = self.fc1(self.flatview)
        self.fcout2 = self.fc2(self.fcout1)
        self.final = self.soft(self.fcout2)

        return self.final



