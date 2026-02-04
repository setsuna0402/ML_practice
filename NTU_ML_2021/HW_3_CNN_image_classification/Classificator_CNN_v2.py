'''
Propose:
Train a Convolutional Neural Network (CNN) for image classification tasks.
Instead of a simple CNN as in version 0.1, we implement a autoencoder to retrieve features.
Then, we use the extracted features for classification.
The data includes labeled images and unlabelled images for semi-supervised learning.
In the version 0.2, we focus on building and training the CNN using only the labeled images.
Author: Dr. Ka Hou Leong
Date: 4/2/2026
Version: 0.2
ML library: PyTorch
'''

# Import necessary packages.
from xml.parsers.expat import model
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
# "ConcatDataset" and "Subset" are possibly useful when doing semi-supervised learning.
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision.datasets import DatasetFolder

# This is for the progress bar.
from tqdm.auto import tqdm
import time
from Model_Class import AutoEncoder, Classifier_Autoencoder

allow_device = True  # Set to False if you want to force using CPU.
use_pin_memory = False  # Set to True if you use GPU. False for CPU and MPS.
n_epochs = 80 # The number of training epochs for classifier. Currently, autoencoder and classifier are trained separately.
n_epochs_ae = 40 # The number of training epochs for autoencoder.
do_semi = False # Whether to do semi-supervised learning.
# Batch size for training, validation, and testing.
# A greater batch size usually gives a more stable gradient.
# But the GPU memory is limited, so please adjust it carefully.
batch_size = 128
# 0 means only the main process will load data. Greater than 0 means number of subprocesses to use for data loading.
# If you use cuda, you may set it to a greater value like 4 or 8 to accelerate data loading.
num_workers = 8  # You may change this value based on your system configuration.
file_path = "./project_data_food_11"

# It is important to do data augmentation in training.
# However, not every augmentation is useful.
# Please think about what kind of augmentation is helpful for food recognition.
train_tfm = transforms.Compose([
    # Resize the image into a fixed shape (height = width = 128)
    transforms.Resize((128, 128)),
    # You may add some transforms here.
    # ToTensor() should be the last one of the transforms.
    transforms.ToTensor(),
])

# We don't need augmentations in testing and validation.
# All we need here is to resize the PIL image and transform it into Tensor.
test_tfm = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
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


# This func doesn't be used in current version. It is designed for semi-supervised learning.    
def get_pseudo_labels(dataset, ae_model, classifier_model, threshold=0.65, device=device):
    # This functions generates pseudo-labels of a dataset using given model.
    # It returns an instance of DatasetFolder containing images whose prediction confidences exceed a given threshold.
    # You are NOT allowed to use any models trained on external data for pseudo-labeling.

    # Construct a data loader.
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Make sure the model is in eval mode.
    ae_model.eval()
    classifier_model.eval()
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
            features_vec = ae_model.encoder(img.to(device))
            logits = classifier_model(features_vec) # logits: [batch_size, n_classes] No gradient calculation

        # Obtain the probability distributions by applying softmax on logits.
        probs = softmax(logits)

        # ---------- TODO ----------
        # Filter the data and construct a new dataset.

    # # Turn off the eval mode.
    classifier_model.train()
    return dataset



# Initialize a autoencoder model, and put it on the device specified.
autoencoder_model = AutoEncoder().to(device)
autoencoder_model.device = device

# Initialize the classifier model, and put it on the device specified.
classifier_model = Classifier_Autoencoder().to(device)
classifier_model.device = device

# Define the loss function.
# For the autoencoder, we use mean squared error (MSE) as the measurement of performance.
autoencoder_criterion = nn.MSELoss()
# For the classification task, we use cross-entropy as the measurement of performance.
classifier_criterion = nn.CrossEntropyLoss()

# Initialize optimizer for autoencoder, you may fine-tune some hyperparameters such as learning rate on your own.
autoencoder_optimizer = torch.optim.Adam(autoencoder_model.parameters(), lr=0.0003)

# Initialize optimizer for classifier, you may fine-tune some hyperparameters such as learning rate on your own.
classifier_optimizer = torch.optim.Adam(classifier_model.parameters(), lr=0.0003, weight_decay=1e-5)

# start time record
start_time_ae = time.time()

# Train the autoencoder first.
for epoch in range(n_epochs_ae):
    # ---------- Training ----------
    # Make sure the model is in train mode before training.
    autoencoder_model.train()

    # These are used to record information in training.
    train_loss = []
    # We use both labeled and unlabeled data to train the autoencoder.
    combined_dataset = ConcatDataset([train_set, unlabeled_set])
    train_ae_loader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=use_pin_memory)
    # Iterate the training set by batches.
    for batch in tqdm(train_ae_loader):

        # A batch consists of image data and corresponding labels.
        imgs, _ = batch

        # Forward the data. (Make sure data and model are on the same device.)
        reconstructed = autoencoder_model(imgs.to(device))

        # Calculate the MSE loss.
        loss = autoencoder_criterion(reconstructed, imgs.to(device))

        # Gradients stored in the parameters in the previous step should be cleared out first.
        autoencoder_optimizer.zero_grad()

        # Compute the gradients for parameters.
        loss.backward()

        # Update the parameters with computed gradients.
        autoencoder_optimizer.step()

        # Record the loss.
        train_loss.append(loss.item())

    # The average loss of the training set is the average of the recorded values.
    train_loss = sum(train_loss) / len(train_loss)

    # Print the information.
    print(f"[ Autoencoder Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}")
# end time record
end_time_ae = time.time()
print("Autoencoder Training time: {:.2f} minutes".format((end_time_ae - start_time_ae)/60))

# start time record
start_time_classifier = time.time()

for epoch in range(n_epochs):
    # ---------- TODO ----------
    # In each epoch, relabel the unlabeled dataset for semi-supervised learning.
    # Then you can combine the labeled dataset and pseudo-labeled dataset for the training.
    if do_semi:
        # Obtain pseudo-labels for unlabeled data using trained model.
        pseudo_set = get_pseudo_labels(unlabeled_set, classifier_model)

        # Construct a new dataset and a data loader for training.
        # This is used in semi-supervised learning only.
        concat_dataset = ConcatDataset([train_set, pseudo_set])
        train_loader = DataLoader(concat_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=use_pin_memory)

    # ---------- Training ----------
    # Make sure the model is in train mode before training.
    classifier_model.train()

    # These are used to record information in training.
    train_loss = []
    train_accs = []
    autoencoder_model.eval()  # Set autoencoder to eval mode
    # Iterate the training set by batches.
    for batch in tqdm(train_loader):

        # A batch consists of image data and corresponding labels.
        imgs, labels = batch

        # Forward the data through the autoencoder to get features.
        with torch.no_grad():
            features_vec = autoencoder_model.encoder(imgs.to(device))

        # Forward the data. (Make sure data and model are on the same device.)
        logits = classifier_model(features_vec)

        # Calculate the cross-entropy loss.
        # We don't need to apply softmax before computing cross-entropy as it is done automatically.
        loss = classifier_criterion(logits, labels.to(device))

        # Gradients stored in the parameters in the previous step should be cleared out first.
        classifier_optimizer.zero_grad()

        # Compute the gradients for parameters.
        loss.backward()

        # Clip the gradient norms for stable training.
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)

        # Update the parameters with computed gradients.
        classifier_optimizer.step()

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
# end time record
end_time_classifier = time.time()
print("Classifier Training time: {:.2f} minutes".format((end_time_classifier - start_time_classifier)/60))

# record validation time
start_time_validation = time.time()
# ---------- Validation ----------
# Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.
autoencoder_model.eval()  # Set autoencoder to eval mode
classifier_model.eval()

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
        features_vec = autoencoder_model.encoder(imgs.to(device))
        logits = classifier_model(features_vec)

    # We can still compute the loss (but not the gradient).
    loss = classifier_criterion(logits, labels.to(device))

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
end_time_validation = time.time()
print("Validation time: {:.2f} minutes".format((end_time_validation - start_time_validation)/60))

# Show all time records
print("Total Training time: {:.2f} minutes".format((end_time_ae - start_time_ae + end_time_classifier - start_time_classifier)/60))
print("Total Validation time: {:.2f} minutes".format((end_time_validation - start_time_validation)/60))
print("Time cost: Autoencoder {:.2f} mins, Classifier {:.2f} mins".format((end_time_ae - start_time_ae)/60, (end_time_classifier - start_time_classifier)/60))

# Save the trained model.
autoencoder_model.eval()
classifier_model.eval()
autoencoder_model.to(torch.device("cpu"))  # Move the model to CPU before saving.
classifier_model.to(torch.device("cpu"))  # Move the model to CPU before saving.
model_dict = { 
            "autoencoder_network" : autoencoder_model.state_dict(),
            "autoencoder_optimizer" : autoencoder_optimizer.state_dict(),
            "classifier_network" : classifier_model.state_dict(),
            "classifier_optimizer" : classifier_optimizer.state_dict(),
        }   
torch.save(model_dict, "classifier_cnn_v2_batch_{}_epoch_{}.pth".format(batch_size, n_epochs))
print("Model saved.")