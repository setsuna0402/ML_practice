'''
Propose:
Distilation of a large neural network (teacher) into a smaller one (student) for image classification tasks. 
The teacher model is a model with ResNet-34 trained in HW3, and the number of parameters in the student model must be less than 0.1M.
The code includes data loading, model definition, training loop, and evaluation metrics.

Author: Dr. Ka Hou Leong
Date: 25/3/2026
Version: 0.1
ML library: PyTorch
'''

# Import necessary packages.
from xml.parsers.expat import model

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
from Model_Class import Classifier_Resnet34, SubsetWithPseudoLabels
from Transform_func import *
from Student_Model_Class import StudentNet

run_in_background = False # make tqdm to be silent mode
show_model_summary = False # Whether to print the model summary. You may set it to False if you don't want to see the model summary.
allow_device = True  # Set to False if you want to force using CPU.
use_pin_memory = True  # Set to True if you use GPU. False for CPU and MPS.
n_epochs = 160 # The number of training epochs for classifier. 
do_semi = True # Whether to do semi-supervised learning.
n_semi_redo = 5 # Redo pseudo labelling every n_semi_redo epochs
n_threshold = 40 # Start to do semi-supervise after n_threshold epochs
do_cutmix = False # Whether to do CutMix augmentation
cutmix_alpha = 1.0 # The alpha for CutMix augmentation. 
alpha = 0.5 # The weight for the soft loss in knowledge distillation. The weight for the hard loss is (1 - alpha).
temperature = 4.0 # The temperature for softening the probability distributions in knowledge distillation

# Batch size for training, validation, and testing.
# A greater batch size usually gives a more stable gradient.
# But the GPU memory is limited, so please adjust it carefully.
batch_size = 128
# 0 means only the main process will load data. Greater than 0 means number of subprocesses to use for data loading.
# If you use cuda, you may set it to a greater value like 4 or 8 to accelerate data loading.
num_workers = 8  # You may change this value based on your system configuration.
file_path = "./project_data_food_11" # the location of the dataset

use_pretrain_model = False # Load a pretrained model and use it as the initial condition of the classifier
teacher_model_path = "./classifier_cnn_v7_batch_128_epoch_160_CutMiX_semi_supervised_AdamW_SGD_Resnet34.pth"


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
    # perm = torch.randperm(len(pseudo_dataset))[:size_train_set]
    perm = torch.randperm(len(pseudo_dataset))[:] # teacher model is good, so we use all the pseudo-labeled data.
    balanced_subset = Subset(pseudo_dataset, perm.tolist())
    
    print("Labelling is done!")
    # return pseudo_dataset
    return balanced_subset

# This is the loss function for knowledge distillation. You may modify it as you like.
def loss_fn_kd(student_outputs, labels, teacher_outputs, alpha=0.5, temperature=1.0):
    '''
    Calculate the knowledge distillation loss.
    student_outputs: the output logits of the student model, shape [batch_size, n_classes]
    labels: the ground truth labels, shape [batch_size]
    teacher_outputs: the output logits of the teacher model, shape [batch_size, n_classes]
    alpha: the weight for the soft loss. The weight for the hard loss is (1 - alpha).
    temperature: the temperature for softening the probability distributions. A greater temperature produces softer probability distributions.
    '''
    # Calculate the hard loss using cross-entropy loss function.
    hard_loss = F.cross_entropy(student_outputs, labels) * (1. - alpha)
    # Complete soft loss in knowledge distillation
    soft_loss = 0
    # both outpots are logits and stored in device, so we can directly calculate the soft loss without moving data.
    student_softmax = F.log_softmax(student_outputs / temperature, dim=1)
    teacher_softmax = F.log_softmax(teacher_outputs / temperature, dim=1)
    # KD divergence loss
    soft_loss = F.kl_div(student_softmax, teacher_softmax, reduction='batchmean') * (alpha * temperature * temperature)
    return hard_loss + soft_loss


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


# Load the pretrained classifier
teacher_model = Classifier_Resnet34()
checkpoint = torch.load(teacher_model_path)
teacher_model.load_state_dict(checkpoint["classifier_network"])
teacher_model.to(device)

student_model = StudentNet()
student_model.to(device)


if show_model_summary:
    print(teacher_model)
    summary(teacher_model, input_size=(1, 3, 142, 142))
    print("Teacher Model summary shown.")
    print(student_model)
    summary(student_model, input_size=(1, 3, 142, 142))
    print("Student Model summary shown.")
    print("Code execution stopped here. Please set 'show_model_summary' to False to continue.")
    exit()

# For the classification task, we use cross-entropy as the measurement of performance.
criterion = nn.CrossEntropyLoss() # it is not used. We define our loss function for knowledge distillation
# For Cut Mix
if do_cutmix:
    cutmix = v2.CutMix(alpha=cutmix_alpha, num_classes=11)
# Initialize optimizer, you may fine-tune some hyperparameters such as learning rate on your own.
optimizer = torch.optim.AdamW(student_model.parameters(), lr=0.0003, weight_decay=1e-5)

# Record validation accuracy across epochs
valid_acc_epoch = []

best_valid_acc = -1.0 # This is used to save the best model during training.

# Semi-supervised learning with pseudo-labeling.
if do_semi:
    # Obtain pseudo-labels from the teacher model and construct a new dataset for training.
    pseudo_set = get_pseudo_labels(unlabeled_set_pseudo, unlabeled_set, teacher_model, device, threshold=0.8)

    # Construct a new dataset and a data loader for training.
    # This is used in semi-supervised learning only.
    concat_dataset = ConcatDataset([train_set, pseudo_set])
    train_loader = DataLoader(concat_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=use_pin_memory)

# start time record
start_time = time.time()
for epoch in range(n_epochs):
    teacher_model.eval() # Set the teacher model to evaluation mode. We don't need to calculate gradients for the teacher model.

    # ---------- Training ----------
    # Make sure the model is in train mode before training.
    student_model.train()
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
        if do_cutmix:
            # Perform cut mix
            imgs, labels = cutmix(imgs, labels)
        # Forward the data.
        with torch.no_grad():
            # no gradient calculation for teacher model
            teacher_logits = teacher_model(imgs) # [batch_size, n_classes]
        student_logits = student_model(imgs) # [batch_size, n_classes]
        # Calculate the cross-entropy loss.
        # We don't need to apply softmax before computing cross-entropy as it is done automatically.
        loss = loss_fn_kd(student_logits, labels, teacher_logits, alpha=alpha, temperature=temperature)

        # Gradients stored in the parameters in the previous step should be cleared out first.
        optimizer.zero_grad()

        # Compute the gradients for parameters.
        loss.backward()

        # Clip the gradient norms for stable training.
        grad_norm = nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=10)

        # Update the parameters with computed gradients.
        optimizer.step()
        # Calculate the accuracy 
        # Calculate the accuracy only when we don't implement CutMix
        if (not do_cutmix):
            acc = (student_logits.argmax(dim=-1) == labels).float().mean()
            train_accs.append(acc)

        # Record the loss and accuracy.
        train_loss.append(loss.item())

    # The average loss and accuracy of the training set is the average of the recorded values.
    train_loss = sum(train_loss) / len(train_loss)

    if do_cutmix:
        print("We use cutmix augmentation. So, accuracy for training is not well defined. Loss value is more informative.")   
        print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}")
    else:
        train_acc = sum(train_accs) / len(train_accs)     
        print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")
    student_model.eval()

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
            logits = student_model(imgs.to(device))

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
    print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}, lr = {current_lr:.8f}")
    if valid_acc > best_valid_acc and epoch >= 80: # Save the model with best validation accuracy after epoch 80. 
        best_valid_acc = valid_acc
        student_model.eval()
        student_model.to(torch.device("cpu"))  # Move the model to CPU before saving.
        # Mark the best model
        model_dict = { 
            "classifier_network" : student_model.state_dict(),
            "classifier_optimizer" : optimizer.state_dict(),
        }   
        torch.save(model_dict, "nn_compression_v1_batch_{}_epoch_{}_semi_supervised_AdamW.pth".format(batch_size, epoch))
        student_model.to(device)  # Move the model back to the original device after saving.
        student_model.train() # Set the model back to train mode after saving.

# ---------- Validation ----------
# Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.
student_model.eval()

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
        logits = student_model(imgs.to(device))

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
plt.savefig("nn_compression_v1_accuracy_batch_{}_epoch_{}_semi_supervised_AdamW.png".format(batch_size, n_epochs))
plt.close()


# Save the trained model.
student_model.eval()
student_model.to(torch.device("cpu"))  # Move the model to CPU before saving.
model_dict = { 
            "classifier_network" : student_model.state_dict(),
            "classifier_optimizer" : optimizer.state_dict(),
        }   
torch.save(model_dict, "nn_compression_v1_batch_{}_epoch_{}_semi_supervised_AdamW.pth".format(batch_size, n_epochs))
print("Model saved.")



