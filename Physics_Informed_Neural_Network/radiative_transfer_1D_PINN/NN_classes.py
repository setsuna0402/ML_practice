import torch
import torch.nn as nn
from torchinfo import summary

class FCN(nn.Module):
    def __init__(self, N_hidden=64, dropout_rate=0.1):
        super().__init__()
        self.activation = nn.ReLU()
        self.network = nn.Sequential(
            nn.Linear(1, N_hidden),
            nn.Tanh(),
            nn.Dropout(dropout_rate),
            nn.Linear(N_hidden, N_hidden),
            nn.Tanh(),
            nn.Dropout(dropout_rate),
            nn.Linear(N_hidden, N_hidden),
            nn.Tanh(),
            nn.Dropout(dropout_rate),
            nn.Linear(N_hidden, 1)
        )
        

    def forward(self, x):
        return self.network(x)
    
class FCN_two(nn.Module):
    def __init__(self, N_hidden=64, dropout_rate=0.1):
        super().__init__()
        self.activation = nn.ReLU()
        self.network = nn.Sequential(
            nn.Linear(2, N_hidden),
            nn.Tanh(),
            nn.Dropout(dropout_rate),
            nn.Linear(N_hidden, N_hidden),
            nn.Tanh(),
            nn.Dropout(dropout_rate),
            nn.Linear(N_hidden, N_hidden),
            nn.Tanh(),
            nn.Dropout(dropout_rate),
            nn.Linear(N_hidden, 1)
        )
        

    def forward(self, x):
        return self.network(x)
    

class FCN_four(nn.Module):
    def __init__(self, N_hidden=64, dropout_rate=0.1):
        super().__init__()
        self.activation = nn.ReLU()
        self.network = nn.Sequential(
            nn.Linear(4, N_hidden),
            nn.Tanh(),
            nn.Dropout(dropout_rate),
            nn.Linear(N_hidden, N_hidden),
            nn.Tanh(),
            nn.Dropout(dropout_rate),
            nn.Linear(N_hidden, N_hidden),
            nn.Tanh(),
            nn.Dropout(dropout_rate),
            nn.Linear(N_hidden, N_hidden),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
            nn.Linear(N_hidden, 1)
        )
        

    def forward(self, x):
        return self.network(x)


class FCN_four_short(nn.Module):
    def __init__(self, N_hidden=64, dropout_rate=0.1):
        super().__init__()
        self.activation = nn.ReLU()
        self.network = nn.Sequential(
            nn.Linear(4, N_hidden),
            nn.Tanh(),
            nn.Dropout(dropout_rate),
            nn.Linear(N_hidden, N_hidden),
            nn.Tanh(),
            nn.Dropout(dropout_rate),
            nn.Linear(N_hidden, N_hidden),
            nn.Tanh(),
            nn.Dropout(dropout_rate),
            nn.Linear(N_hidden, 1)
        )
        

    def forward(self, x):
        return self.network(x)
