# Learn WGAN with gradient penalty for MNIST dataset using CNN
# We use the same generator and discriminator architecture as in mnist_gan_class.py
# The main difference is the loss function and the training procedure
# We use the Wasserstein loss with gradient penalty
# Reference: https://arxiv.org/abs/1704.00028
# activation functions: ReLU for generator, LeakyReLU for discriminator

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 
import torchvision
import torchvision.transforms as transforms
from torchsummary import summary
import matplotlib.pyplot as plt
import numpy as np
import random

# mnist_generator
class CNN_Generator(nn.Module):
    def __init__(self):
        super(CNN_Generator, self).__init__()
        # input channel = 100, output channel = 256
        self.latent_dim = 100
        self.img_rows = 28
        self.img_cols = 28
        self.img_channels = 1
        self.img_shape = (self.img_channels, self.img_rows, self.img_cols)
        # deconvolution layers
        # input: (100, 1, 1)
        # after deconv1: (256, 7, 7)
        # after deconv2: (128, 14, 14)
        # after deconv3: (64, 28, 28)
        # after deconv4: (1, 28, 28)
        self.deconv1 = nn.ConvTranspose2d(self.latent_dim , 256, kernel_size=7, stride=1)
        self.batchnorm1 = nn.BatchNorm2d(256) 
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(128)
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.batchnorm3 = nn.BatchNorm2d(64)
        self.deconv4 = nn.ConvTranspose2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()          # ReLU activation function
        self.tanh = nn.Tanh()          # Tanh activation 
        self.leaky_relu = nn.LeakyReLU(0.2) # Leaky ReLU
        self.dropout = nn.Dropout(p=0.25)

    def forward(self, x):
        # x shape: (batch_size, 100)
        x = x.view(-1, self.latent_dim, 1, 1)     # reshape to (batch_size, 100, 1, 1)
        x = self.deconv1(x)           # deconv1
        x = self.batchnorm1(x)        # batchnorm1
        x = self.relu(x)              # ReLU
        x = self.dropout(x)           # dropout
        x = self.deconv2(x)           # deconv2
        x = self.batchnorm2(x)        # batchnorm2
        x = self.relu(x)              # ReLU
        x = self.dropout(x)           # dropout
        x = self.deconv3(x)           # deconv3
        x = self.batchnorm3(x)        # batchnorm3
        x = self.relu(x)              # ReLU
        x = self.dropout(x)           # dropout
        x = self.deconv4(x)           # deconv4
        x = self.tanh(x)              # Tanh to get output in range [-1, 1]
        return x

# mnist discriminator
# input: 28x28x1 image, output: real/fake (1)
# structure similar to CNN_classifier, but with different number of channels and output
class CNN_Discriminator(nn.Module):
    def __init__(self):
        super(CNN_Discriminator, self).__init__()
        # input channel = 1, output channel = 64
        # input: (1, 28, 28)
        # after conv1: (64, 24, 24)
        # after conv2: (128, 20, 20)
        # after maxpool: (128, 10, 10)
        # after conv3: (128, 10, 10)
        # after fc1: (1024)
        # after fc2: (128)
        # after fc3: (1)
        # output: (1) real/fake
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=1)
        self.batchnorm1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=1) 
        self.batchnorm2 = nn.BatchNorm2d(128)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)  # max pooling layer
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1) 
        # dropout layer with p=0.25
        self.dropout = nn.Dropout(p=0.25)
        self.fc1 = nn.Linear(128*10*10, 1024)  # fully connected layer, output 1024 features
        self.fc2 = nn.Linear(1024, 128)        # fully connected layer, output 128 features
        self.fc3 = nn.Linear(128, 1)           # fully connected layer, output 1 features (true/false)
        # self.relu = nn.ReLU()                  # ReLU activation function
        self.leaky_relu = nn.LeakyReLU(0.2)    # Leaky ReLU
        # self.softmax = nn.Softmax(dim=1)       # softmax activation function

    def forward(self, x):
        # x shape: (batch_size, 1, 28, 28)
        x = self.conv1(x)      # conv1
        x = self.batchnorm1(x) # batchnorm1
        x = self.leaky_relu(x)       # Leaky ReLU
        x = self.dropout(x)    # dropout
        x = self.conv2(x)      # conv2
        x = self.batchnorm2(x) # batchnorm2
        x = self.leaky_relu(x)       # Leaky ReLU
        x = self.maxpool(x)    # maxpool
        x = self.conv3(x)      # conv3
        x = self.leaky_relu(x)       # Leaky ReLU
        x = self.dropout(x)    # dropout
        x = x.view(-1, self.num_flat_features(x))  # flatten the tensor for fully connected layer
        x = self.fc1(x)        # fc1
        x = self.leaky_relu(x)       # Leaky ReLU
        x = self.dropout(x)    # dropout
        x = self.fc2(x)        # fc2
        x = self.leaky_relu(x)       # Leaky ReLU
        x = self.dropout(x)    # dropout
        x = self.fc3(x)        # fc3
        # for WGAN, we need to avoid using log in the loss function
        # so we do not apply sigmoid or softmax here
        # use binary cross entropy with logits loss, so no need to apply sigmoid here
        # x = self.leaky_relu(x) # Leaky ReLU
        # x = self.softmax(x)   # softmax, not needed if using BCEWith
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

def gradient_penalty(discriminator, real_data, fake_data, device=torch.device("cpu")):
    batch_size = real_data.size(0)
    # Sample epsilon uniformly from [0, 1]
    epsilon = torch.rand(batch_size, 1, 1, 1).to(device)
    # Create interpolated data
    interpolated = epsilon * real_data + (1 - epsilon) * fake_data
    interpolated = Variable(interpolated, requires_grad=True)
    # Compute discriminator output on interpolated data
    d_interpolated = discriminator(interpolated)
    # Compute gradients of discriminator output w.r.t. interpolated data
    gradients = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones(d_interpolated.size()).to(device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]  # Why [0]? Because autograd.grad returns a tuple. In this case, autograd.grad returns (tensor([a,b,c,d,...]),) 
    
    # Reshape gradients to (batch_size, -1)
    gradients = gradients.view(batch_size, -1)  # why -1? because we want to flatten the gradients for each sample in the batch
    # Compute gradient norm
    gradient_norm = gradients.norm(2, dim=1)
    # Compute gradient penalty
    gradient_penalty = ((gradient_norm - 1) ** 2).mean()
    return gradient_penalty

# Train both generator and discriminator
# step_gen: number of steps to train generator
# step_disc: number of steps to train discriminator
# In each epoch, we train the discriminator step_disc times and then train the generator step_gen
def train_both_models(generator, discriminator, dataloader, g_optimizer, d_optimizer, lambda_gp=10, num_epochs=5, step_g=5, step_d=2, device=torch.device("cpu")):
    print("train GAN model in device:", device)
    for epoch in range(num_epochs):
        g_running_loss = 0.0
        d_running_loss = 0.0
        for i, (images, _) in enumerate(dataloader):
            # Get real images and labels
            real_images = images
            real_images = real_images.to(device)  # real images
            real_labels = torch.ones(real_images.size(0), 1).to(device)  # real labels = 1
            # The number of fake images is the same as the number of real images
            fake_labels = torch.zeros(real_images.size(0), 1).to(device)  # fake labels = 0

            batch_size = real_images.size(0)
             # Train Discriminator (use the same batch of real images to train discriminator multiple times)
            for _ in range(step_d):
                # Train Discriminator
                d_optimizer.zero_grad()  # clear gradients
                output_real = discriminator(real_images)
                d_loss_real = torch.mean(output_real)

                # Generate fake images (Its size is identical to the real images)
                z = torch.randn(batch_size, generator.latent_dim, 1, 1).to(device)  # random noise
                fake_images = generator(z)
                # fake_labels = torch.zeros(fake_images.size(0), 1).to(device)  # fake labels = 0
                outputs_fake = discriminator(fake_images.detach()).view(-1, 1)  # detach to avoid compute the gradients for generator
                d_loss_fake = torch.mean(outputs_fake)
                # Compute gradient penalty
                gp = gradient_penalty(discriminator, real_images, fake_images, device)
                # Gradient penalty term
                d_loss = -d_loss_real + d_loss_fake + lambda_gp * gp
                d_loss.backward()
                # Note that optimiser only updates the parameters we gave it when it was created. (So it only updates the discriminator parameters here)
                d_optimizer.step()

            # Train Generator
            for _ in range(step_g):
                g_optimizer.zero_grad()
                z = torch.randn(batch_size, generator.latent_dim, 1, 1).to(device)  # random noise
                fake_images = generator(z)
                outputs_fake = discriminator(fake_images).view(-1, 1) 
                g_loss = -torch.mean(outputs_fake)
                g_loss.backward()
                # Note that optimiser only updates the parameters we gave it when it was created. (So it only updates the generator parameters here)
                g_optimizer.step() 

            g_running_loss += g_loss.item()
            d_running_loss += d_loss.item()

        print(f'Epoch [{epoch+1}/{num_epochs}], Generator Loss: {g_running_loss/len(dataloader):.4f}, Discriminator Loss: {d_running_loss/len(dataloader):.4f}')

    print('Finished Training WGAN Model with Gradient Penalty')

'''
model = CNN_Generator()
print(model)  # print the model architecture
summary(model.cuda(), (100, 1, 1))  # print the model summary

model = CNN_Discriminator()
print(model)  # print the model architecture
summary(model.cuda(), (1, 28, 28))  # print the model summary
'''