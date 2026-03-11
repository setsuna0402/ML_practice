from time import time
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
from mnist_gan_class import *
import time
# from mnist_cnn_class import CNN_classifier, train_model

# Set random seed for reproducibility
torch.cuda.manual_seed_all(42)
torch.manual_seed(42)
np.random.seed(42)
# Define hyperparameters
num_epochs = 10
batch_size = 64
learning_rate = 0.0002
# Below are defined in the classes
# image_size = 28
# image_channels = 1
# latent_dim = 100


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = "cpu"
print(device)

# torchvision.transforms: use to transform the input data
# ToTensor: convert a PIL Image or numpy.ndarray to tensor
# Normalize: normalize a tensor image with mean and standard deviation
# MNIST images are grayscale, ranging from 0 to 1. We normalize them to the range (-1, 1)
# Compose is a class and it can be called like a function.
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = torchvision.datasets.MNIST(root='../data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='../data', train=False, transform=transform, download=True)

# In pytorch, dataset defines the data structure and dataloader defines how to iterate over the dataset
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


generator = CNN_Generator()
discriminator = CNN_Discriminator()
# Move the models to the device (GPU, if available)
if torch.cuda.is_available():
    generator_d = generator.to(device)
    discriminator_d = discriminator.to(device)

# Show the model architecture and summary
'''
print(generator)  # print the model architecture
summary(generator.cuda(), (100, 1, 1))  # print the model summary
print(discriminator)  # print the model architecture
summary(discriminator.cuda(), (1, 28, 28))  # print the model
'''

# Loss function and optimizer
criterion = nn.BCEWithLogitsLoss() # use BCEWithLogitsLoss for stability
# Binary Cross Entropy loss combines a Sigmoid layer and the BCELoss in one single class.
g_optimizer = optim.Adam(generator_d.parameters(), lr=0.0002)  # use Adam optimizer
d_optimizer = optim.Adam(discriminator_d.parameters(), lr=0.0002)  # use Adam optimizer
# record training time
start_time = time.time()
train_both_models(generator_d, discriminator_d, train_loader, criterion, g_optimizer, d_optimizer, num_epochs=num_epochs, step_g=1, step_d=2, device=device)
end_time = time.time()
print("Training time: {:.2f} seconds".format(end_time - start_time))
# Validation on test dataset (first batch)
noise = torch.randn(batch_size, generator.latent_dim, 1, 1)  # random noise
generator = generator_d.cpu()  # move model to CPU
fake_images = generator(noise)  # forward pass
fake_images = fake_images.detach()  # detach to allow numpy conversion
# Here we illustrate the forst 64 images and their predicted labels
fig, axs = plt.subplots(8, 8, figsize=(15, 15))
axs = axs.flatten()  # flatten the 2D array to 1D array
for i in range(batch_size):
    img = fake_images[i].numpy()
    img = img[0, :, :]  # remove the channel dimension
    axs[i].imshow(img, cmap='gray')
    axs[i].axis('off')
plt.savefig('mnist_gan_generated_images.png')

# Save the generator model
torch.save(generator.state_dict(), 'mnist_generator_batch_{}_epoch_{}.pth'.format(batch_size, num_epochs))
# Save the discriminator model
discriminator = discriminator_d.cpu()  # move model to CPU
torch.save(discriminator.state_dict(), 'mnist_discriminator_batch_{}_epoch_{}.pth'.format(batch_size, num_epochs))

