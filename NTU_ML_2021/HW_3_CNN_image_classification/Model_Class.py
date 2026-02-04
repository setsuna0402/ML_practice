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
            nn.LeakyReLU(),
            nn.Flatten(),  # [512*8*8]
            nn.Linear(512 * 8 * 8, 1024), # [Batch_size, 1024]
            nn.LeakyReLU(),
            nn.Linear(1024, 256), # [Batch_size, 256] feature vector
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(256, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 512 * 8 * 8),
            nn.LeakyReLU(),
            nn.Unflatten(1, (512, 8, 8)),  # [512, 8, 8]
            nn.ConvTranspose2d(512, 256, 3, 2, 1, output_padding=1),  # [256, 16, 16]
            nn.LeakyReLU(),
            nn.ConvTranspose2d(256, 128, 3, 2, 1, output_padding=1),  # [128, 32, 32]
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 64, 3, 2, 1, output_padding=1),   # [64, 64, 64]
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 3, 3, 2, 1, output_padding=1),     # [3, 128, 128]
            nn.Sigmoid(),  # To ensure the output is in [0, 1]
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class Classifier_Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        # The classifier uses the feature vector extracted by the autoencoder's encoder.
        # input vector size: [256]
        self.fc_layers = nn.Sequential(
            nn.Linear(256, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Dropout(0.25),  # Dropout for regularization

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Dropout(0.25),  # Dropout for regularization

            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128, 11),  # 11 classes for food classification
        )

    def forward(self, x):
        # input (x): [batch_size, 256]
        # output: [batch_size, 11]

        # Classification layers
        x = self.fc_layers(x)

        return x

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