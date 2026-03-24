'''
Propose:
Train a Convolutional Neural Network (CNN) for image classification tasks.
The data includes labeled images and unlabelled images for semi-supervised learning.
In the version 0.7, we use the pretrained resnet18/34 architecture. 
However, we consider the unlabeled data for training by generating pseudo-labels for them. (semi-supervised learning)
Using pretrained model because I need a good model to finish HW13. In HW13, you need a good tearch model to train a small student model.

Author: Dr. Ka Hou Leong
Date: 25/3/2026
Version: 0.7
ML library: PyTorch
'''

# Import necessary packages.
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.transforms import v2
from PIL import Image
# "ConcatDataset" and "Subset" are possibly useful when doing semi-supervised learning.
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision.datasets import DatasetFolder
import torchvision.models as models
from torchinfo import summary
import matplotlib.pyplot as plt

# This is for the progress bar.
from tqdm.auto import tqdm
import time
from Model_Class_pretrained import Classifier_Resnet18, Classifier_Resnet34, SubsetWithPseudoLabels
from Transform_func_hw13 import *

run_in_background = False # make tqdm to be silent mode
show_model_summary = False # Whether to print the model summary. You may set it to False if you don't want to see the model summary.
allow_device = True  # Set to False if you want to force using CPU.
use_pin_memory = True  # Set to True if you use GPU. False for CPU and MPS.
n_epochs = 160 # The number of training epochs for classifier. 
do_semi = True # Whether to do semi-supervised learning.
n_semi_redo = 5 # Redo pseudo labelling every n_semi_redo epochs
n_threshold = 40 # Start to do semi-supervise after n_threshold epochs
# Mixup data. This may work for image classification
do_mixup = False
mixup_alpha = 0.2
# Cut and Mix images
do_cutmix = True # In principle, mixup and cutup can work together. However, we don't support it in this version. You need to choice one!
cutmix_alpha = 1.0

# Batch size for training, validation, and testing.
# A greater batch size usually gives a more stable gradient.
# But the GPU memory is limited, so please adjust it carefully.
batch_size = 128
# 0 means only the main process will load data. Greater than 0 means number of subprocesses to use for data loading.
# If you use cuda, you may set it to a greater value like 4 or 8 to accelerate data loading.
num_workers = 8  # You may change this value based on your system configuration.
file_path = "./project_data_food_11" # the location of the dataset

use_pretrain_model = False # Load a pretrained model and use it as the initial condition of the classifier
model_path = "./classifier_cnn_v5_batch_128_epoch_80.pth"


train_tfm = train_transform
test_tfm = test_transform
# Construct datasets.
# The argument "loader" tells how torchvision reads the data.
train_set = DatasetFolder(file_path + "/training/labeled", loader=lambda x: Image.open(x), extensions="jpg", transform=train_tfm)
valid_set = DatasetFolder(file_path + "/validation", loader=lambda x: Image.open(x), extensions="jpg", transform=test_tfm)
unlabeled_set = DatasetFolder(file_path + "/training/unlabeled", loader=lambda x: Image.open(x), extensions="jpg", transform=train_tfm)
unlabeled_set_pseudo = DatasetFolder(file_path + "/training/unlabeled", loader=lambda x: Image.open(x), extensions="jpg", transform=test_tfm)
test_set = DatasetFolder(file_path + "/testing", loader=lambda x: Image.open(x), extensions="jpg", transform=test_tfm)

# Construct data loaders.
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=use_pin_memory)
valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=use_pin_memory)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

size_train_set = len(train_set)

# This func is designed for semi-supervised learning.    
def get_pseudo_labels(dataset_infer, dataset_augment, model, device, threshold=0.9):
    # This functions generates pseudo-labels of a dataset using given model.
    # It returns an instance of DatasetFolder containing images whose prediction confidences exceed a given threshold.
    
    # Construct a data loader. P.S. dataset_infer is free from augmentation! except resize and normalisation
    data_loader_infer = DataLoader(dataset_infer, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=use_pin_memory)
    # data_loader_augment = DataLoader(dataset_augment, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=use_pin_memory)

    # Make sure the model is in eval mode.
    model.eval()
    # Define softmax function.
    # In Pytorch, cross-entropy loss function already includes softmax operation.
    # So, we don't define softmax in the model. We define it here for obtaining probability distributions.
    softmax = nn.Softmax(dim=-1) 
    
    offset = 0 # This is used to calculate the index of the data in the entire dataset.
    subset_indices = [] # This is used to select the deserved data from the whole dataset
    subset_pseudo_label = [] # Record the predicted label and use in the pseudo subdataset
    print("Start to get pseudo labels for the unlabelled dataset")
    # Iterate over the dataset by batches.
    for batch in tqdm(data_loader_infer, disable=run_in_background):
        img, _ = batch
        img_d = img.to(device) # Move the data to the same device as model.
        size = img_d.size(0) # batch size for current batch, which may be smaller than the specified batch size for the last batch.
        # Calculate the index of the data in this batch 
        local_index = torch.arange(size, dtype=torch.int32, device=device, requires_grad=False) + offset
        # Forward the data
        # Using torch.no_grad() accelerates the forward process.
        with torch.no_grad():
            logits = model(img_d) # logits: [batch_size, n_classes] No gradient calculation

        # Obtain the probability distributions by applying softmax on logits.
        probs = softmax(logits)

        # Filter the data and construct a new dataset.
        pred_prob, pred_label = probs.max(dim=-1) # [batch_size], [batch_size]
        mask = pred_prob > threshold # mask: [batch_size] bool tensor
        # local_index[mask]  # [n_select] tensor
        subset_indices.extend(local_index[mask].cpu().tolist())
        subset_pseudo_label.extend(pred_label[mask].cpu().tolist())

        offset += size

    # # Turn off the eval mode.
    model.train()
    selected_subset = Subset(dataset_augment, subset_indices)
    pseudo_dataset = SubsetWithPseudoLabels(selected_subset, subset_pseudo_label)
    # Here, we randomly select len(train_set) data from the pseudo subset
    # Ensuring the number of pseudo samples is not higher than the number of labelled samples
    perm = torch.randperm(len(pseudo_dataset))[:size_train_set]
    balanced_subset = Subset(pseudo_dataset, perm.tolist())
    
    print("Labelling is done!")
    # return pseudo_dataset
    return balanced_subset

def mixing_data(image, label, device, alpha=0.2):
    # Mix up data from the same batch
    # image shape: [B, channel, H, W]
    # label shape: [B, n_classes]
    # alpha 
    if alpha > 0:
        # use beta distribution since we want lam close to 0 or 1. 
        # When lamda close 0, 1, the mixed img is liks the real imgs with small perturbation
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    image = image.to(device)
    label = label.to(device)
    batch_size = image.size(0)
    # randomly change the order
    index = torch.randperm(batch_size, device=device)
    # mixup images
    mixed_x = lam * image + (1 - lam) * image[index]
    label_a, label_b = label, label[index]

    return mixed_x, label_a, label_b, lam

# Automatically choose the device to use.
if allow_device:
# Move the network to the appropriate device
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
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

if use_pretrain_model:
    # Load the pretrained classifier
    model = Classifier_Resnet18()
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint["classifier_network"])
    model.to(device)
else:
    # Initialize a model, and put it on the device specified.
    model = Classifier_Resnet18().to(device)
    model.device = device

if show_model_summary:
    print(model)
    summary(model, input_size=(1, 3, 256, 256))
    print("Model summary shown. Please set 'show_model_summary' to False if you don't want to see the model summary.")
    print("Code execution stopped here. Please set 'show_model_summary' to False to continue.")
    exit()

# For the classification task, we use cross-entropy as the measurement of performance.
# criterion = nn.CrossEntropyLoss()
criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.0)
# For Cut Mix
if do_cutmix:
    cutmix = v2.CutMix(alpha=cutmix_alpha, num_classes=11)

# Initialize optimizer, you may fine-tune some hyperparameters such as learning rate on your own.
optimizer = torch.optim.AdamW(model.parameters(), lr=0.00003, weight_decay=1e-3)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.03, momentum=0.9, weight_decay=1e-3) # Resnet is deep. Use SGD to accelerate the training process
# This is for SGD. Dynamically decay the lr. 
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

# Record validation accuracy across epochs
valid_acc_epoch = []

# start time record
start_time = time.time()
for epoch in range(n_epochs):
    # ---------- TODO ----------
    # In each epoch, relabel the unlabeled dataset for semi-supervised learning.
    # Then you can combine the labeled dataset and pseudo-labeled dataset for the training.
    if do_semi and (epoch % n_semi_redo == 0) and (epoch >= n_threshold):
        # Obtain pseudo-labels for unlabeled data using trained model.
        pseudo_set = get_pseudo_labels(unlabeled_set_pseudo, unlabeled_set, model, device)

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
    for batch in tqdm(train_loader, disable=run_in_background):

        # A batch consists of image data and corresponding labels.
        imgs, labels = batch
        # Send data to device
        imgs = imgs.to(device)
        labels = labels.to(device)
        if do_mixup:
            # perform Mix up
            imgs_mixed, label_a, label_b, lam = mixing_data(imgs, labels, device, alpha=mixup_alpha)
            logits = model(imgs_mixed) # Make sure data and model are on the same device.
            loss = lam * criterion(logits, label_a) + (1 - lam) * criterion(logits, label_b)
        elif do_cutmix:
            # Perform cut mix
            imgs, labels = cutmix(imgs, labels)
            logits = model(imgs) # Make sure data and model are on the same device.
            loss = criterion(logits, labels)
        else:
            # Forward the data.
            logits = model(imgs)
            # Calculate the cross-entropy loss.
            # We don't need to apply softmax before computing cross-entropy as it is done automatically.
            loss = criterion(logits, labels)

        # Gradients stored in the parameters in the previous step should be cleared out first.
        optimizer.zero_grad()

        # Compute the gradients for parameters.
        loss.backward()

        # Clip the gradient norms for stable training.
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)

        # Update the parameters with computed gradients.
        optimizer.step()
        # Calculate the accuracy only when we don't implement Mixup and CutMix
        if (not do_cutmix) and (not do_mixup):
            # Compute the accuracy for current batch.
            acc = (logits.argmax(dim=-1) == labels).float().mean()
            train_accs.append(acc)

        # Record the loss and accuracy.
        train_loss.append(loss.item())

    # The average loss and accuracy of the training set is the average of the recorded values.
    train_loss = sum(train_loss) / len(train_loss)

    #
    if do_mixup:
        print("We use mixup augmentation. So, accuracy for training is not well defined. Loss value is more informative.")
        print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}")
    elif do_cutmix:
        print("We use cutmix augmentation. So, accuracy for training is not well defined. Loss value is more informative.")   
        print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}")
    else:
        train_acc = sum(train_accs) / len(train_accs)     
        print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")
    model.eval()

    # These are used to record information in validation.
    valid_loss = []
    valid_accs = []

    # Iterate the validation set by batches.
    for batch in tqdm(valid_loader, disable=run_in_background):

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
        valid_accs.append(acc.item())

    # scheduler.step() # update learning rate
    current_lr = optimizer.param_groups[0]["lr"]

    # The average loss and accuracy for entire validation set is the average of the recorded values.
    valid_loss = sum(valid_loss) / len(valid_loss)
    valid_acc = sum(valid_accs) / len(valid_accs)
    valid_acc_epoch.append(valid_acc)
    # Print the information.
    print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}, lr = {current_lr:.5f}")

# ---------- Validation ----------
# Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.
model.eval()

# These are used to record information in validation.
valid_loss = []
valid_accs = []

# Iterate the validation set by batches.
for batch in tqdm(valid_loader, disable=run_in_background):

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

# Convert valid_acc_epoch to numpy array for making plot
valid_acc_epoch = np.array(valid_acc_epoch)
plt.plot(np.arange(n_epochs), valid_acc_epoch, "b-", label="Validation")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.savefig("classifier_v7_accuracy_batch_{}_epoch_{}_CutMiX_semi_supervised_AdamW.png".format(batch_size, n_epochs))
plt.close()


# Save the trained model.
model.eval()
model.to(torch.device("cpu"))  # Move the model to CPU before saving.
model_dict = { 
            "classifier_network" : model.state_dict(),
            "classifier_optimizer" : optimizer.state_dict(),
        }   
# torch.save(model_dict, "classifier_cnn_v5_batch_{}_epoch_{}_semi_supervised_adamW_b.pth".format(batch_size, n_epochs))
# torch.save(model_dict, "classifier_cnn_v5_batch_{}_epoch_{}_mixup_semi_supervised_SGD_low_threshold.pth".format(batch_size, n_epochs))
torch.save(model_dict, "classifier_cnn_v7_batch_{}_epoch_{}_CutMiX_semi_supervised_AdamW.pth".format(batch_size, n_epochs))
print("Model saved.")
