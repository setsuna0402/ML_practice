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
from mnist_cnn_class import CNN_classifier, train_model


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=256, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=256, shuffle=False)

'''
# illustrate how dataset works, also how transform works
print(test_loader.dataset.data.size())
y = test_loader.dataset.data.numpy()  # convert to numpy array
print(torch.min(transform(y[0])))   # The defined compose convert data to tensor....
print(transform)
'''

'''
# illustrate how dataloader works
dataloader_iter = iter(test_loader)  # create an iterator
images, labels = dataloader_iter.__next__()  # get the first batch of data
# images, labels = dataloader_iter.__next__()  # get the second batch of data
fig, axs = plt.subplots(8, 8, figsize=(15, 15))
axs = axs.flatten()  # flatten the 2D array to 1D array
for i in range(64):
    img = images[i].numpy() 
    img = img[0, :, :]  # remove the channel dimension
    label = labels[i]
    axs[i].imshow(img, cmap='gray')
    axs[i].set_title(f'Label: {label}')
    axs[i].axis('off')
plt.show()
'''

model = CNN_classifier()
if torch.cuda.is_available():
    model.to(device)  # move model to GPU
# model.cuda()  # move model to GPU
# Show the model architecture and summary
'''
print(model)  # print the model architecture
summary(model.cuda(), (1, 28, 28))  # print the model summary
'''

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()  # use CrossEntropyLoss for multi-class classification
optimizer = optim.Adam(model.parameters(), lr=0.001)  # use Adam optimizer
train_model(model, train_loader, criterion, optimizer, num_epochs=20, device=device)

# Validation on test dataset (first batch)
dataloader_iter = iter(test_loader)  # create an iterator
images, labels = dataloader_iter.__next__()  # get the first batch of data
model.cpu()  # move model to CPU
outputs = model(images)  # forward pass
_, predicted = torch.max(outputs, 1)
# print(predicted, labels)  # print the output logits
# Here we illustrate the forst 64 images and their predicted labels
fig, axs = plt.subplots(8, 8, figsize=(15, 15))
axs = axs.flatten()  # flatten the 2D array to 1D array
for i in range(64):
    img = images[i].numpy() 
    img = img[0, :, :]  # remove the channel dimension
    label = labels[i]
    axs[i].imshow(img, cmap='gray')
    axs[i].set_title(f'Label: {label}')
    axs[i].axis('off')
plt.show()

# Calculate accuracy on the whole test dataset
correct = np.zeros(10) # correct predictions for each class
total = np.zeros(10) # total samples for each class
all_total = 0
with torch.no_grad():  # no need to calculate gradients
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        all_total += labels.size(0)
        # count number of correct predictions for each class
        for i in range(labels.size(0)):
            total[labels[i].item()] += 1
            correct[labels[i].item()] += (predicted[i] == labels[i]).item()
for i in range(10):
    print('Accuracy of number {}: {:.2f}%'.format(i, 100 * correct[i] / total[i]))
print('-----------------------------------------')
print('Overall Accuracy: {:.2f}%'.format(100 * np.sum(correct) / all_total))

# Save the model
torch.save(model.state_dict(), 'mnist_cnn_batch_256_epoch_20.pth')

