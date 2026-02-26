'''
Propose:
Train a Transformer Network for speaker classification tasks.
The voice data is preprocessed into mel-spectrograms (provided by the NTU course), which are fed into the network for training and inference.
In the version 0.1, we use the NTU deflaut model. This help us to get understand how the self-attention mechanism works in the speaker classification task.
Author: Dr. Ka Hou Leong
Date: 26/2/2026
Version: 0.1
ML library: PyTorch
'''

# Import necessary packages.
import math

import numpy as np
import torch
import torch.nn as nn
# "ConcatDataset" and "Subset" are possibly useful when doing semi-supervised learning.
from torch.utils.data import ConcatDataset, DataLoader, Subset, random_split, Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torchvision.datasets import DatasetFolder
import torchvision.models as models
from torchinfo import summary
import matplotlib.pyplot as plt 

# This is for the progress bar.
from tqdm.auto import tqdm
import time
import os
import json
import random
from pathlib import Path
from Load_Data_Module import *
from Schedule_Module import get_cosine_schedule_with_warmup

run_in_background = False # make tqdm to be silent mode
show_model_summary = False # Whether to print the model summary. You may set it to False if you don't want to see the model summary.
allow_device = True  # Set to False if you want to force using CPU.
use_pin_memory = True  # Set to True if you use GPU. False for CPU and MPS.
n_total_step = 140000 # The total number of training steps. 
n_valid_step = 2000 # The number of steps for validation. 
n_warmup_step = 1000 # The number of steps for learning rate warmup.
do_save_model = True # Whether to save the model during training. You may set it to False if you don't want to save the model.
n_save_step = 10000 # The number of steps for saving the model. You may adjust it based on your needs.
ratio_train = 0.9 # 90% of the dataset are allocated into training set. 10% of them go to validation set. 

# Batch size for training, validation, and testing.
# A greater batch size usually gives a more stable gradient.
# But the GPU memory is limited, so please adjust it carefully.
batch_size = 32
# 0 means only the main process will load data. Greater than 0 means number of subprocesses to use for data loading.
# If you use cuda, you may set it to a greater value like 4 or 8 to accelerate data loading.
num_workers = 8  # You may change this value based on your system configuration.
file_path = "./project_data_voice" # the location of the dataset

use_pretrain_model = False # Load a pretrained model and use it as the initial condition of the classifier
model_path = "./classifier_cnn_v5_batch_128_epoch_80.pth"

class Classifier(nn.Module):
    def __init__(self, d_model=80, n_spks=600, dropout=0.1):
        super().__init__()
        # Project the dimension of features from that of input into d_model.
        self.prenet = nn.Linear(40, d_model)
        # TODO:
        #   Change Transformer to Conformer.
        #   https://arxiv.org/abs/2005.08100
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, dim_feedforward=256, nhead=2)
        # self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)

        # Project the the dimension of features from d_model into speaker nums.
        self.pred_layer = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, n_spks),
        )

    def forward(self, mels):
        """
        args:
        mels: (batch size, length, 40)
        return:
        out: (batch size, n_spks)
        """
        # out: (batch size, length, d_model)
        out = self.prenet(mels)
        # out: (length, batch size, d_model)
        out = out.permute(1, 0, 2)
        # The encoder layer expect features in the shape of (length, batch size, d_model).
        out = self.encoder_layer(out)
        # out: (batch size, length, d_model)
        out = out.transpose(0, 1)
        # mean pooling
        stats = out.mean(dim=1)

        # out: (batch, n_spks)
        out = self.pred_layer(stats)
        return out

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
        num_workers = 0 # MPS does not support multi-process data loading, so we set num_workers to 0.
        use_pin_memory = False # MPS does not support pin_memory, so we set it to False.
    else:
        device = torch.device("cpu")
        print("Using CPU")
        use_pin_memory = False
        num_workers = 0 # You do not multiple process to move data when using CPU, so we set num_workers to 0.
else:
    device = torch.device("cpu")
    print("Using CPU")
    use_pin_memory = False
    num_workers = 0 # You do not multiple process to move data when using CPU, so we set num_workers to 0.

train_loader, valid_loader, speaker_num = get_dataloader(file_path, batch_size, num_workers, use_pin_memory, ratio_train)
train_iterator = iter(train_loader)
model = Classifier(n_spks=speaker_num).to(device)

if show_model_summary:
    batch = next(train_iterator)
    mels, labels = batch
    print(model)
    summary(model, input_size=(1, 128, 40)) # Adjust input size to match the expected input of the model
    print("Model summary shown. Please set 'show_model_summary' to False if you don't want to see the model summary.")
    print("Input batch shape: (batch size, length, feature dimension) = {}".format(mels.shape))
    print("Code execution stopped here. Please set 'show_model_summary' to False to continue.")
    exit()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
scheduler = get_cosine_schedule_with_warmup(optimizer, n_warmup_step, n_total_step, num_cycles=0.5)

best_accuracy = -1.0 # current best validation accuracy.
best_state_dict = None # To save the best model state dict based on validation accuracy.
save_threshold = 0.8 # Save the model if the validation accuracy exceeds this threshold.

pbar = tqdm(total=n_valid_step, ncols=0, desc="Train", unit=" step", disable=run_in_background)

start_time = time.time()

for step in range(n_total_step):
    # Get data
    try:
        batch = next(train_iterator)
    except StopIteration:
        # iter creates an iterator object from the train_loader.
        # Since, shuffle is set to True, iter(train_loader) creates an new iterator following different random order. 
        train_iterator = iter(train_loader)
        batch = next(train_iterator)
    model.train()
    mels, labels = batch
    # Transfer to device
    mels = mels.to(device)
    labels = labels.to(device)

    # Calculate loss
    logits = model(mels) # sharp: [batch, speaker_num]
    loss = criterion(logits, labels)
    batch_loss = loss.item() # loss is a tensor, loss.item() is a scalar detached from graph.

    # Pick the label with highest prob as the predicted label
    pred_label = logits.argmax(1) # sharp: [batch, 1]
    # Compute accuracy
    acc = (pred_label == labels).float().mean()
    batch_acc = acc.item() # acc is a tensor, acc.item() is a scalar.

    #Update model parameters
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    current_lr = optimizer.param_groups[0]["lr"]
    # Log
    pbar.update()
    pbar.set_postfix(
        loss = f"{batch_loss:.5f}",
        accuracy = f"{batch_acc:.5f}",
        lr = f"{current_lr:.5e}",
        step = step + 1,
    )

    # Validation
    if (step + 1) % n_valid_step == 0:
        pbar.close() # Close the progress bar for training.
        model.eval()
        valid_acc = []
        valid_loss = []
        with torch.no_grad():
            for batch in tqdm(valid_loader, desc="Valid", ncols=0, unit=" batch", disable=run_in_background):
                mels, labels = batch
                mels = mels.to(device)
                labels = labels.to(device)
                logits = model(mels)
                loss = criterion(logits, labels).item()
                pred_label = logits.argmax(1)
                acc = (pred_label == labels).float().mean()
                valid_acc.append(acc.item())
                valid_loss.append(loss)
        avg_valid_acc = sum(valid_acc) / len(valid_acc)
        avg_valid_loss = sum(valid_loss) / len(valid_loss)
        print("Validation: Step: {} / {}, loss = {}, acc = {}".format(step + 1, n_total_step, avg_valid_loss, avg_valid_acc))
        if best_accuracy < avg_valid_acc:
            print(f"New best validation accuracy: {avg_valid_acc:.5f} at step {step+1}")
            best_accuracy = avg_valid_acc
        # Record the best model based on validation accuracy.
        if best_accuracy > save_threshold:
            print("Step: {} / {}, best_valid_acc = {:.5f}, save_threshold = {:.5f}".format(step + 1, n_total_step, best_accuracy, save_threshold))
            best_accuracy = avg_valid_acc
            best_state_dict = model.state_dict()
            save_threshold = best_accuracy # Update the save threshold to the current best accuracy to save only better models in the future.
            print("save_threshold update to {:.5f}".format(save_threshold))
        pbar = tqdm(total=n_valid_step, ncols=0, desc="Train", unit=" step", disable=run_in_background)
    # Save model
    if (step + 1) % n_save_step == 0 and do_save_model:
        if best_state_dict is not None:
            torch.save(best_state_dict, f"specker_classifier_v1_step_{step+1}_acc_{best_accuracy:.5f}.pth")

end_time = time.time()
elapsed_time = end_time - start_time
print("Training completed in {:.2f} mins.".format(elapsed_time / 60))

