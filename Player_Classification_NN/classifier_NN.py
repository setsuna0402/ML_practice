'''
Define the model architecture for classifying player engagement levels using a neural network.
Date: 21/3/2026
Author: Dr. Edward
'''
import torch
import torch.nn as nn


# Define the neural network architecture for player engagement classification
# Fullly connected feedforward neural network with 3 hidden layers and dropout for regularization
class GamingEngagementNN(nn.Module):
    def __init__(self, input_dim: int, num_classes: int = 3, dropout_rate: float = 0.3):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout_rate),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout_rate),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.model(x)