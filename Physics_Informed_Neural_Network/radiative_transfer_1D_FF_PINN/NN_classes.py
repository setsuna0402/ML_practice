import torch
import torch.nn as nn
from torchinfo import summary

'''
class FF_FCN_four(nn.Module):
    def __init__(self, input_dim=4, N_hidden=64, ff_dim=64, sigma=8.0):
        super().__init__()
        # This might look like parammeter B will be different for each input, 
        # but B are fixed after initialization, and will not be updated during training.
        self.B = nn.Parameter(torch.randn(input_dim, ff_dim) * sigma, requires_grad=False)
        self.activation = nn.Tanh()
        self.network = nn.Sequential(
            nn.Linear(ff_dim*2, N_hidden), # *2 because we concatenate sin and cos features
            nn.Tanh(),
            nn.Linear(N_hidden, N_hidden),
            nn.Tanh(),
            nn.Linear(N_hidden, N_hidden),
            nn.Tanh(),
            nn.Linear(N_hidden, N_hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(N_hidden, 1)
        )

    def forward(self, x):
        # x: (batch_size, input_dim)
        # B: (input_dim, ff_dim)
        # x_proj: (batch_size, ff_dim)
        x_proj = 2 * torch.pi * x @ self.B
        ff_feat = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        out = self.network(ff_feat)
        return out
'''

class FF_FCN_four(nn.Module):
    def __init__(self, input_dim=4, N_hidden=64, ff_dim=64, sigma=4.0):
        super().__init__()
        # only embed the spatial coordinate x, not the other 3 conditions
        self.Bx = nn.Parameter(torch.randn(1, ff_dim) * sigma, requires_grad=False)
        self.act = nn.Tanh()
        
        in_feat_dim = 2 * ff_dim + input_dim  # x's FF embedding +  (x, Io, xc, freq)
        self.network = nn.Sequential(
            nn.Linear(in_feat_dim, N_hidden),
            nn.Tanh(),
            nn.Linear(N_hidden, N_hidden),
            nn.Tanh(),
            nn.Linear(N_hidden, N_hidden),
            nn.Tanh(),
            nn.Linear(N_hidden, N_hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(N_hidden, 1)
        )

    def forward(self, u):
        # u: [N,4] = [x, Io, xc, freq]
        x = u[:, 0:1]    # spatial coordinate x
        # cond = u[:, 1:] # Io, xc, freq 
        
        # FF embedding for x
        x_proj = 2 * torch.pi * x @ self.Bx
        x_ff = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        
        # concatenate FF embedding of x with the other conditions
        all_feat = torch.cat([u, x_ff], dim=-1)
        out = self.network(all_feat)
        return out