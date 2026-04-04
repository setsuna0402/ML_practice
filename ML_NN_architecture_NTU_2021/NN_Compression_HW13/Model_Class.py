'''
Define models used in this project.
AutoEncoder: The autoencoder model for feature extraction.
Classifier_Autoencoder: The classifier model that takes features from the autoencoder as input.
Classifier: The CNN classifier model.
Author: Dr. Ka Hou Leong
Date: 4/2/2026
'''
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, Subset
import torchvision.models as models


# Define a dataset class which is a subset of a dataset, and the labels are psedo
class SubsetWithPseudoLabels(Dataset):
    """A Subset that returns (img, pseudo_label) instead of (img, original_label)."""
    def __init__(self, subset: Subset, pseudo_labels: list[int]): # type hint
        self.subset = subset
        self.pseudo_labels = pseudo_labels  # aligned with subset order

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, _ = self.subset[idx]          # ignore original label
        return img, self.pseudo_labels[idx]

# Resnet18
class Classifier_Resnet18(nn.Module):
    def __init__(self):
        super().__init__()
        # input image size: [3, 128, 128]
        self.resnet18_layers = models.resnet18(weights=None)
        # Get the number of input features for the final layer
        num_ftrs = self.resnet18_layers.fc.in_features
        self.resnet18_layers.fc  = nn.Sequential(
            nn.BatchNorm1d(num_ftrs),
            nn.Linear(num_ftrs, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 11)
        )
        # self.resnet18_layers.fc = nn.Linear(num_ftrs, 11)  # replace the classifier head
    def forward(self, x):
        # input (x): [batch_size, 3, 128, 128]
        # output: [batch_size, 11]

        # Extract features by convolutional layers.
        x = self.resnet18_layers(x)

        return x
    
# Resnet34
class Classifier_Resnet34(nn.Module):
    def __init__(self):
        super().__init__()
        # input image size: [3, 128, 128]
        self.resnet34_layers = models.resnet34(weights=None)
        # Get the number of input features for the final layer
        num_ftrs = self.resnet34_layers.fc.in_features
        self.resnet34_layers.fc  = nn.Sequential(
            nn.BatchNorm1d(num_ftrs),
            nn.Linear(num_ftrs, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 11)
        )
        # self.resnet34_layers.fc = nn.Linear(num_ftrs, 11)  # replace the classifier head
    def forward(self, x):
        # input (x): [batch_size, 3, 128, 128]
        # output: [batch_size, 11]

        # Extract features by convolutional layers.
        x = self.resnet34_layers(x)

        return x


# Define the autocoder-based classifier.
class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # input image size: [3, 128, 128]
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, 1),  # [64, 64, 64]
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, 3, 2, 1),  # [128, 32, 32]
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, 3, 2, 1),  # [256, 16, 16]
            nn.LeakyReLU(),
            nn.Conv2d(256, 512, 3, 2, 1),  # [512, 8, 8]
            # nn.LeakyReLU(),
            # nn.Flatten(),  # [512*8*8]
            # nn.Linear(512 * 8 * 8, 1024), # [Batch_size, 1024]
            # nn.LeakyReLU(),
            # nn.Linear(1024, 256), # [Batch_size, 256] feature vector
        )
        # Decoder
        self.decoder = nn.Sequential(
            # nn.Linear(256, 1024),
            # nn.LeakyReLU(),
            # nn.Linear(1024, 512 * 8 * 8),
            # nn.LeakyReLU(),
            # nn.Unflatten(1, (512, 8, 8)),  # [512, 8, 8]
            nn.ConvTranspose2d(512, 256, 3, 2, 1, output_padding=1),  # [256, 16, 16]
            nn.LeakyReLU(),
            nn.ConvTranspose2d(256, 128, 3, 2, 1, output_padding=1),  # [128, 32, 32]
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 64, 3, 2, 1, output_padding=1),   # [64, 64, 64]
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 3, 3, 2, 1, output_padding=1),     # [3, 128, 128]
            nn.Tanh(), # Output pixel values in range [-1, 1]
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class Classifier_Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        # The classifier uses the feature vector extracted by the autoencoder's encoder.
        # input vector size: [512, 8, 8]
        self.fc_layers = nn.Sequential(
            # nn.AdaptiveAvgPool2d(1),  # [B, 512, 1, 1]
            # nn.Flatten(),             # [B, 512]
            nn.Conv2d(512, 256, 3, 1, 1),  # [256, 8, 8]
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2, 0),              # [256, 4, 4]

            nn.Conv2d(256, 128, 3, 1, 1),  # [128, 4, 4]
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2, 0),         # [128, 2, 2]
            nn.Flatten(),                 # [B, 128*2*2 = 512]

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(0.25),  # Dropout for regularization

            # nn.Linear(256, 1024),
            # nn.BatchNorm1d(1024),
            # nn.LeakyReLU(),
            # nn.Dropout(0.25),  # Dropout for regularization

            # nn.Linear(1024, 256),
            # nn.BatchNorm1d(256),
            # nn.LeakyReLU(),
            # nn.Dropout(0.25),  # Dropout for regularization

            nn.Linear(256, 128),
            # nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128, 11),  # 11 classes for food classification
        )

    def forward(self, x):
        # input (x): [batch_size, 256]
        # output: [batch_size, 11]

        # Classification layers
        x = self.fc_layers(x)

        return x

class JointEncoderClassifier(nn.Module):
    """
    Joint end-to-end model:
    image -> encoder -> classifier -> logits
    """
    def __init__(self, encoder: nn.Module, classifier: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.classifier = classifier

    def forward(self, x):
        feats = self.encoder(x)
        logits = self.classifier(feats)
        return logits

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        # The arguments for commonly used modules:
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)

        # input image size: [3, 128, 128]
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(4, 4, 0),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(256 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 11)
        )

    def forward(self, x):
        # input (x): [batch_size, 3, 128, 128]
        # output: [batch_size, 11]

        # Extract features by convolutional layers.
        x = self.cnn_layers(x)

        # The extracted feature map must be flatten before going to fully-connected layers.
        x = x.flatten(1)

        # The features are transformed by fully-connected layers to obtain the final logits.
        x = self.fc_layers(x)
        return x

class Deep_Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        # The arguments for commonly used modules:
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)

        # input image size: [3, 128, 128]
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(3, 64, 5, 1, 2), # [64, 128, 128]
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2, 0), # [64, 64, 64]

            nn.Conv2d(64, 128, 4, 2, 2), # [128, 33, 33]
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),

            nn.Conv2d(128, 256, 4, 1, 1), # [256, 32, 32]
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.MaxPool2d(4, 4, 0), # [256, 8, 8]

            nn.Conv2d(256, 512, 4, 1, 2), # [512, 9, 9]
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2, 0), # [512, 4, 4]
            
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(512 * 4 * 4, 2048), # [8192] -> [2056]
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(),
            nn.MaxPool1d(2, 2, 0), # [2048] -> [1024]
            nn.Dropout(0.25),

            nn.Linear(1024, 512), # [1024] -> [512]
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.MaxPool1d(2, 2, 0), # [512] -> [256]
            nn.Dropout(0.25),

            nn.Linear(256, 64), # [256] -> [64]
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Dropout(0.25),
            nn.Linear(64, 11) # [64] -> [11]
        )

    def forward(self, x):
        # input (x): [batch_size, 3, 128, 128]
        # output: [batch_size, 11]

        # Extract features by convolutional layers.
        x = self.cnn_layers(x)

        # The extracted feature map must be flatten before going to fully-connected layers.
        x = x.flatten(1)

        # The features are transformed by fully-connected layers to obtain the final logits.
        x = self.fc_layers(x)
        return x