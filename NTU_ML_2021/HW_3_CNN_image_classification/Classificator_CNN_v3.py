'''
Propose:
Train a Convolutional Neural Network (CNN) for image classification tasks.
The data includes labeled images and unlabelled images for semi-supervised learning.
In the version 0.1, we focus on building and training the CNN using only the labeled images.
Author: Dr. Ka Hou Leong
Date: 4/2/2026
Version: 0.1
ML library: PyTorch
'''

# Import necessary packages.
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
# "ConcatDataset" and "Subset" are possibly useful when doing semi-supervised learning.
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision.datasets import DatasetFolder
from torchinfo import summary

# This is for the progress bar.
from tqdm.auto import tqdm
import time
from Model_Class import Deep_Classifier

show_model_summary = False # Whether to print the model summary. You may set it to False if you don't want to see the model summary.
allow_device = True  # Set to False if you want to force using CPU.
use_pin_memory = True  # Set to True if you use GPU. False for CPU and MPS.
n_epochs = 80 # The number of training epochs for classifier. 
do_semi = False # Whether to do semi-supervised learning.
# Batch size for training, validation, and testing.
# A greater batch size usually gives a more stable gradient.
# But the GPU memory is limited, so please adjust it carefully.
batch_size = 128
# 0 means only the main process will load data. Greater than 0 means number of subprocesses to use for data loading.
# If you use cuda, you may set it to a greater value like 4 or 8 to accelerate data loading.
num_workers = 8  # You may change this value based on your system configuration.
file_path = "./project_data_food_11" # Used to load data
# It is important to do data augmentation in training.
# However, not every augmentation is useful.
# Please think about what kind of augmentation is helpful for food recognition.
train_tfm = transforms.Compose([
    # Resize the image into a fixed shape (height = width = 128)
    transforms.Resize((128, 128)),
    # You may add some transforms here.
    # ToTensor() should be the last one of the transforms.
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), # Normalize the image with mean and std of ImageNet dataset.
])

# We don't need augmentations in testing and validation.
# All we need here is to resize the PIL image and transform it into Tensor.
test_tfm = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), # Normalize the image with mean and std of ImageNet dataset.
])

# Construct datasets.
# The argument "loader" tells how torchvision reads the data.
train_set = DatasetFolder(file_path + "/training/labeled", loader=lambda x: Image.open(x), extensions="jpg", transform=train_tfm)
valid_set = DatasetFolder(file_path + "/validation", loader=lambda x: Image.open(x), extensions="jpg", transform=test_tfm)
unlabeled_set = DatasetFolder(file_path + "/training/unlabeled", loader=lambda x: Image.open(x), extensions="jpg", transform=train_tfm)
test_set = DatasetFolder(file_path + "/testing", loader=lambda x: Image.open(x), extensions="jpg", transform=test_tfm)

# Construct data loaders.
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=use_pin_memory)
valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=use_pin_memory)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)



# This func doesn't be used in version 0.1. It is designed for semi-supervised learning.    
def get_pseudo_labels(dataset, model, threshold=0.65):
    # This functions generates pseudo-labels of a dataset using given model.
    # It returns an instance of DatasetFolder containing images whose prediction confidences exceed a given threshold.
    # You are NOT allowed to use any models trained on external data for pseudo-labeling.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Construct a data loader.
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Make sure the model is in eval mode.
    model.eval()
    # Define softmax function.
    # In Pytorch, cross-entropy loss function already includes softmax operation.
    # So, we don't define softmax in the model. We define it here for obtaining probability distributions.
    softmax = nn.Softmax(dim=-1) 

    # Iterate over the dataset by batches.
    for batch in tqdm(data_loader):
        img, _ = batch

        # Forward the data
        # Using torch.no_grad() accelerates the forward process.
        with torch.no_grad():
            logits = model(img.to(device)) # logits: [batch_size, n_classes] No gradient calculation

        # Obtain the probability distributions by applying softmax on logits.
        probs = softmax(logits)

        # ---------- TODO ----------
        # Filter the data and construct a new dataset.

    # # Turn off the eval mode.
    model.train()
    return dataset

# Automatically choose the device to use.
if allow_device:
# Move the network to the appropriate device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS")
    else:
        device = torch.device("cpu")
        print("Using CPU")
else:
    device = torch.device("cpu")
    print("Using CPU")


# Initialize a model, and put it on the device specified.
model = Deep_Classifier().to(device)
model.device = device

if show_model_summary:
    print(model)
    summary(model, input_size=(1, 3, 128, 128))
    print("Model summary shown. Please set 'show_model_summary' to False if you don't want to see the model summary.")
    print("Code execution stopped here. Please set 'show_model_summary' to False to continue.")
    exit()

# For the classification task, we use cross-entropy as the measurement of performance.
criterion = nn.CrossEntropyLoss()

# Initialize optimizer, you may fine-tune some hyperparameters such as learning rate on your own.
optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=1e-5)

# start time record
start_time = time.time()
for epoch in range(n_epochs):
    # ---------- TODO ----------
    # In each epoch, relabel the unlabeled dataset for semi-supervised learning.
    # Then you can combine the labeled dataset and pseudo-labeled dataset for the training.
    if do_semi:
        # Obtain pseudo-labels for unlabeled data using trained model.
        pseudo_set = get_pseudo_labels(unlabeled_set, model)

        # Construct a new dataset and a data loader for training.
        # This is used in semi-supervised learning only.
        concat_dataset = ConcatDataset([train_set, pseudo_set])
        train_loader = DataLoader(concat_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=use_pin_memory)

    # ---------- Training ----------
    # Make sure the model is in train mode before training.
    model.train()

    # These are used to record information in training.
    train_loss = []
    train_accs = []

    # Iterate the training set by batches.
    for batch in tqdm(train_loader):

        # A batch consists of image data and corresponding labels.
        imgs, labels = batch

        # Forward the data. (Make sure data and model are on the same device.)
        logits = model(imgs.to(device))

        # Calculate the cross-entropy loss.
        # We don't need to apply softmax before computing cross-entropy as it is done automatically.
        loss = criterion(logits, labels.to(device))

        # Gradients stored in the parameters in the previous step should be cleared out first.
        optimizer.zero_grad()

        # Compute the gradients for parameters.
        loss.backward()

        # Clip the gradient norms for stable training.
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)

        # Update the parameters with computed gradients.
        optimizer.step()

        # Compute the accuracy for current batch.
        acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

        # Record the loss and accuracy.
        train_loss.append(loss.item())
        train_accs.append(acc)

    # The average loss and accuracy of the training set is the average of the recorded values.
    train_loss = sum(train_loss) / len(train_loss)
    train_acc = sum(train_accs) / len(train_accs)

    # Print the information.
    print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")
    model.eval()

    # These are used to record information in validation.
    valid_loss = []
    valid_accs = []

    # Iterate the validation set by batches.
    for batch in tqdm(valid_loader):

        # A batch consists of image data and corresponding labels.
        imgs, labels = batch

        # We don't need gradient in validation.
        # Using torch.no_grad() accelerates the forward process.
        with torch.no_grad():
            logits = model(imgs.to(device))

        # We can still compute the loss (but not the gradient).
        loss = criterion(logits, labels.to(device))

        # Compute the accuracy for current batch.
        acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

        # Record the loss and accuracy.
        valid_loss.append(loss.item())
        valid_accs.append(acc)

    # The average loss and accuracy for entire validation set is the average of the recorded values.
    valid_loss = sum(valid_loss) / len(valid_loss)
    valid_acc = sum(valid_accs) / len(valid_accs)
    # Print the information.
    print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")

# ---------- Validation ----------
# Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.
model.eval()

# These are used to record information in validation.
valid_loss = []
valid_accs = []

# Iterate the validation set by batches.
for batch in tqdm(valid_loader):

    # A batch consists of image data and corresponding labels.
    imgs, labels = batch

    # We don't need gradient in validation.
    # Using torch.no_grad() accelerates the forward process.
    with torch.no_grad():
        logits = model(imgs.to(device))

    # We can still compute the loss (but not the gradient).
    loss = criterion(logits, labels.to(device))

    # Compute the accuracy for current batch.
    acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

    # Record the loss and accuracy.
    valid_loss.append(loss.item())
    valid_accs.append(acc)

# The average loss and accuracy for entire validation set is the average of the recorded values.
valid_loss = sum(valid_loss) / len(valid_loss)
valid_acc = sum(valid_accs) / len(valid_accs)

# Print the information.
# print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")
print(f"[ Valid ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")

# end time record
end_time = time.time()
print("Training time: {:.2f} minutes".format((end_time - start_time)/60))
# Save the trained model.
model.eval()
model.to(torch.device("cpu"))  # Move the model to CPU before saving.
model_dict = { 
            "classifier_network" : model.state_dict(),
            "classifier_optimizer" : optimizer.state_dict(),
        }   
torch.save(model_dict, "classifier_cnn_v3_batch_{}_epoch_{}.pth".format(batch_size, n_epochs))
print("Model saved.")
