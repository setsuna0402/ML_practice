'''
Train a neural network classifier for player engagement level classification based on the online gaming behavior dataset.
Date: 21/3/2026
Author: Dr. Edward
'''

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix   
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchinfo import summary
import matplotlib.pyplot as plt 
from data_load_and_preprocessing import load_file, check_data, prepare_ml_data
from classifier_NN import GamingEngagementNN     
import time
# This is for the progress bar.
from tqdm.auto import tqdm


run_in_background = False # make tqdm to be silent mode
show_model_summary = False # Whether to print the model summary. You may set it to False if you don't want to see the model summary.
allow_device = True  # Set to False if you want to force using CPU.
use_pin_memory = True  # Set to True if you use GPU. False for CPU and MPS.
do_save = False
num_epochs = 20
batch_size = 32
learning_rate = 0.001
num_workers = 8  # You may change this value based on your system configuration.
file_name = "./project_data/online_gaming_behavior_dataset.csv"

# Automatically choose the device to use.
if allow_device:
# Move the network to the appropriate device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS")
        use_pin_memory = False  # MPS does not support pin_memory, so we set it to False when using MPS.
        num_workers = 0  # MPS does not support num_workers > 0, so we set it to 0 when using MPS.
    else:
        device = torch.device("cpu")
        print("Using CPU")
else:
    device = torch.device("cpu")
    print("Using CPU")

# load and preprocess data
data = load_file(file_name)
# check_data(data)
X_train, X_valid, y_train, y_valid, preprocessor, label_mapping, input_feature_names = prepare_ml_data(data)
input_dim = X_train.shape[1]
print("Input number of features after preprocessing: {}".format(input_dim))
model = GamingEngagementNN(input_dim=input_dim, num_classes=3, dropout_rate=0.3).to(device)

if show_model_summary:
    print(model)
    summary(model, input_size=(1, input_dim))
    print("Model summary shown. Please set 'show_model_summary' to False if you don't want to see the model summary.")
    print("Code execution stopped here. Please set 'show_model_summary' to False to continue.")
    exit()

# criterion and optimizer
# if label_smoothing = 0.1, 0.9 for the true class, 0.1 is the total confidence for all the other classes.
criterion = nn.CrossEntropyLoss(label_smoothing=0.0)
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4) # weight_decay is the L2 regularization term.

# Convert the training and validation data into PyTorch tensors and create DataLoader for batch processing.
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_valid_tensor = torch.tensor(X_valid, dtype=torch.float32)

y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_valid_tensor = torch.tensor(y_valid, dtype=torch.long)

# Create TensorDataset and DataLoader for training and validation sets
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
valid_dataset = TensorDataset(X_valid_tensor, y_valid_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=use_pin_memory)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=use_pin_memory)

validation_losses = []
valid_accuracies = []
best_valid_acc = 0.0

start_time = time.time()
# training loop
for epoch in range(num_epochs):
    model.train()
    train_loss = []
    train_acc = []

    for batch in tqdm(train_loader, disable=run_in_background):
        input_feature, labels = batch
        # pass the input features and labels to the device (GPU/MPS/CPU)
        input_feature = input_feature.to(device)
        labels = labels.to(device)
        logits = model(input_feature)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())
        acc = (logits.argmax(dim=-1) == labels).float().mean()
        train_acc.append(acc.item())
    avg_train_loss = np.mean(train_loss)
    avg_train_acc = np.mean(train_acc)
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Train Accuracy: {avg_train_acc:.4f}")

    # validation loop
    model.eval()
    valid_loss = []
    valid_acc = []
    with torch.no_grad():
        for batch in tqdm(valid_loader, disable=run_in_background):
            input_feature, labels = batch
            input_feature = input_feature.to(device)
            labels = labels.to(device)
            logits = model(input_feature)
            loss = criterion(logits, labels)
            valid_loss.append(loss.item())
            acc = (logits.argmax(dim=-1) == labels).float().mean()
            valid_acc.append(acc.item())
    avg_valid_loss = np.mean(valid_loss)
    avg_valid_acc = np.mean(valid_acc)
    if avg_valid_acc > best_valid_acc:
        best_valid_acc = avg_valid_acc
        model_dict = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_valid_acc": best_valid_acc,
            }
        if do_save:
            torch.save(model_dict, "classifier_batch_{}_epoch_{}.pth".format(batch_size, epoch))
    validation_losses.append(avg_valid_loss)
    valid_accuracies.append(avg_valid_acc)
    print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {avg_valid_loss:.4f}, Validation Accuracy: {avg_valid_acc:.4f}")
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Training completed in {elapsed_time:.2f} seconds.")

# Plot the validation loss and accuracy curves
plt.plot(np.arange(1, num_epochs + 1), validation_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Validation Loss Curve')
plt.legend()
plt.savefig('validation_loss_curve.png')
plt.close()
plt.plot(np.arange(1, num_epochs + 1), valid_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy Curve')
plt.legend()
plt.savefig('validation_accuracy_curve.png')
plt.close()

# confusion_matrix and classification_report
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for batch in valid_loader:
        input_feature, labels = batch
        input_feature = input_feature.to(device)
        labels = labels.to(device)
        logits = model(input_feature)
        preds = logits.argmax(dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
print("Classification Report:")
print(classification_report(all_labels, all_preds, target_names=label_mapping.keys()))
print("Confusion Matrix:")
print(confusion_matrix(all_labels, all_preds))
