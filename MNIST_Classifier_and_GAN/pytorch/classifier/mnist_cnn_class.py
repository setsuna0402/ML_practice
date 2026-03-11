import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim # 建立優化器用
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import random

# input image size: 28x28
# after conv1: (28-5)/1 + 1 = 24
# after conv2: (24-5)/1 + 1 = 20
# after maxpool: 20/2 = 10
# fully connected layer input size: 16*10*10 = 1600
# output size: 10 (class score)
class CNN_classifier(nn.Module):
    def __init__(self):
        super(CNN_classifier, self).__init__()
        # python 3 style
        # super().__init__() is used to call the constructor of the parent class (nn.Module)
        # input channel = 1, output channel = 8
        self.conv1 = nn.Conv2d(1, 8, kernel_size=5, stride=1) 
        self.conv2 = nn.Conv2d(8, 16, kernel_size=5, stride=1) 
        # dropout layer with p=0.25
        self.dropout = nn.Dropout(p=0.25)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)  # max pooling layer
        self.fc1 = nn.Linear(16*10*10, 128)  # fully connected layer, output 128 features
        self.fc2 = nn.Linear(128, 64)  # fully connected layer, output 64 features
        self.fc3 = nn.Linear(64, 10)   # fully connected layer, output 10 features (class score)
        self.relu = nn.ReLU()          # ReLU activation function
        # self.softmax = nn.Softmax()    # softmax activation function

    def forward(self, x):
        # x shape: (batch_size, 1, 28, 28)
        x = self.conv1(x)      # conv1
        x = self.relu(x)       # ReLU
        x = self.dropout(x)    # dropout
        x = self.conv2(x)      # conv2
        x = self.relu(x)       # ReLU
        x = self.maxpool(x)    # maxpool
        x = x.view(-1, self.num_flat_features(x))  # flatten the tensor for fully connected layer
        x = self.fc1(x)        # fc1
        x = self.relu(x)       # ReLU
        x = self.dropout(x)    # dropout
        x = self.fc2(x)        # fc2
        x = self.relu(x)       # ReLU
        x = self.fc3(x)        # fc3
        #x = self.softmax(x)   # softmax, not needed if using CrossEntropyLoss (pytorch will apply softmax internally)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

def train_model(model, train_loader, criterion, optimizer, num_epochs=5, device=torch.device("cpu")):
    print("train model in device:", device)
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            # Clear the gradients (otherwise they will accumulate from previous batches)
            optimizer.zero_grad()
            inputs, labels = inputs.to(device), labels.to(device) 
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels) # compute the loss

            # Backward pass (compute gradients)
            loss.backward()
            # optimisation (one step of gradient descent)
            optimizer.step()

            # total loss for one epoch
            running_loss += loss.item() # item() convert a tensor to a scalar
            '''
            if i % 100 == 0:    # print every 100 mini-batches
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss/100:.4f}')
                running_loss = 0.0
            '''
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')
    print('Finished Training')

