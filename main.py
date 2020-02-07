"""
Training of a CNN. Works on MNIST-like datasets.
"""

from torchvision import datasets
import torchvision as tv
from torch.utils.data import DataLoader
import torch as t
import time
import torch.nn as nn
import numpy as np

from torch.autograd import Variable

from model import OVRNet, ReferenceNet


def validation_loss(test_loader, net, loss, device="cpu"):
    net.eval()
    losses = []

    correct = 0
    total = 0

    for image, label in test_loader:
        images = image.to(device)
        labels = label.to(device)

        optimizer.zero_grad()
        outputs = net(images)

        _, predicted = t.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels.data).sum()

        current_loss = loss(outputs, labels)
        temp = current_loss.data.tolist()
        if type(temp) != list:
            temp = [temp]
        losses.extend(temp)

    print("Validation loss: {:.6f}".format(np.mean(losses)))
    print("Validation accuracy: {:.2f}%".format(100*correct / total))


if __name__ == "__main__":

    train = datasets.MNIST(root="./data", transform=tv.transforms.ToTensor(), train=True, download=True)
    test = datasets.MNIST(root="./data", transform=tv.transforms.ToTensor(), train=False, download=True)

    loader = DataLoader(train, batch_size=64, shuffle=True)

    test_loader = DataLoader(test, batch_size=16, shuffle=False)
    device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
    net = OVRNet()
    net.to(device)
    loss = nn.CrossEntropyLoss()
    loss.to(device)
    optimizer = t.optim.Adam(net.parameters(), lr=0.001)

    print("Starting with training.")

    for epoch in range(50):
        net.train()

        start = time.time()
        best = 10e6

        for (image, label) in loader:
             images = image.to(device)
             labels = label.to(device)
        
             optimizer.zero_grad()
             outputs = net(images)
        
             current_loss = loss(outputs.to(device), labels)
             current_loss.backward()
        
             optimizer.step()

        print("Finished with training set, starting validation.")
        validation_loss(test_loader, net, loss, device)
        end = time.time()
        print("Epoch {} took {:2f} s".format(epoch, end-start))
