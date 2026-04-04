'''
Define student models used in this project.
Author: Dr. Ka Hou Leong
Date: 25/3/2026
'''
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, Subset
import torchvision.models as models


class StudentNet(nn.Module):
    def __init__(self):
      super(StudentNet, self).__init__()

      self.cnn = nn.Sequential(
        nn.Conv2d(3, 32, 3),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        
        nn.Conv2d(32, 32, 3),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2, 2, 0),

        nn.Conv2d(32, 64, 3),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2, 2, 0),

        nn.Conv2d(64, 100, 3),
        nn.BatchNorm2d(100),
        nn.ReLU(),
        nn.MaxPool2d(2, 2, 0),

        # Here we adopt Global Average Pooling for various input size.
        nn.AdaptiveAvgPool2d((1, 1)),
      )
      self.fc = nn.Sequential(
        nn.Linear(100, 11),
      )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)    

class StudentNet_DPC(nn.Module):
    '''
    Use depthwise and pointwise convolution to reduce the number of parameters.
    '''
    def __init__(self):
      super(StudentNet_DPC, self).__init__()

      self.cnn = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, groups=3),  # depthwise convolution
        nn.BatchNorm2d(3),
        nn.ReLU(),
        nn.Conv2d(in_channels=3, out_channels=32, kernel_size=1),            # pointwise convolution
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2, 2, 0),

        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, groups=32),  # depthwise convolution
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1),            # pointwise convolution
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2, 2, 0),

        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, groups=32),  # depthwise convolution
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1),            # pointwise convolution
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2, 2, 0),

        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, groups=64),  # depthwise convolution
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Conv2d(in_channels=64, out_channels=100, kernel_size=1),            # pointwise convolution
        nn.BatchNorm2d(100),
        nn.ReLU(),
        nn.MaxPool2d(2, 2, 0),


        # Here we adopt Global Average Pooling for various input size.
        nn.AdaptiveAvgPool2d((1, 1)),
      )
      self.fc = nn.Sequential(
        nn.Linear(100, 11),
      )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)

class StudentNet_DPGC(nn.Module):
    '''
    Use depthwise, pointwise and group convolution to reduce the number of parameters.
    '''
    def __init__(self):
      super(StudentNet_DPGC, self).__init__()

      self.cnn = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, groups=3),  # depthwise convolution
        nn.BatchNorm2d(3),
        nn.ReLU(),
        nn.Conv2d(in_channels=3, out_channels=32, kernel_size=1),            # pointwise convolution
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, groups=4),  # group convolution
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2, 2, 0),

        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, groups=32),  # depthwise convolution
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1),            # pointwise convolution
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, groups=4),  # group convolution
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2, 2, 0),

        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, groups=32),  # depthwise convolution
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1),            # pointwise convolution
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, groups=4),  # group convolution
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2, 2, 0),

        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, groups=64),  # depthwise convolution
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Conv2d(in_channels=64, out_channels=100, kernel_size=1),            # pointwise convolution
        nn.BatchNorm2d(100),
        nn.ReLU(),
        nn.Conv2d(in_channels=100, out_channels=100, kernel_size=3, groups=4),  # group convolution
        nn.BatchNorm2d(100),
        nn.ReLU(),
        nn.MaxPool2d(2, 2, 0),


        # Here we adopt Global Average Pooling for various input size.
        nn.AdaptiveAvgPool2d((1, 1)),
      )
      self.fc = nn.Sequential(
        nn.Linear(100, 11),
      )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)

class DPGCResidualBlock(nn.Module):
    """
    Depthwise + Pointwise + GroupConv residual block.
    Uses padding=1 to preserve spatial size.
    Adds a projection shortcut when in_channels != out_channels.
    """
    def __init__(self, in_channels, out_channels, group_conv_groups=4):
        super().__init__()

        # make sure groups divides channels
        assert in_channels % 1 == 0
        assert out_channels % group_conv_groups == 0

        self.block = nn.Sequential(
            # depthwise conv
            nn.Conv2d(
                in_channels=in_channels, out_channels=in_channels, kernel_size=3,
                padding=1, groups=in_channels, bias=True),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),

            # pointwise conv
            nn.Conv2d(
                in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),

            # group conv
            nn.Conv2d(
                in_channels=out_channels, out_channels=out_channels, kernel_size=3,
                padding=1, groups=group_conv_groups, bias=True),
            nn.BatchNorm2d(out_channels),
        )

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.block(x)
        out = out + self.shortcut(x)
        return self.relu(out)

# Transformer block for muli-head self attention and feedforward network. 
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_hidden_dim, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # Self-attention
        attn_output = self.attention(self.norm1(x), self.norm1(x), self.norm1(x))[0] # [batch_size, seq_len, embed_dim]
        x = x + attn_output  # Add & Norm

        # Feedforward network
        ffn_output = self.ffn(self.norm2(x))
        x = x + ffn_output  # Add & Norm

        return x


class SelfAttentionPooling(nn.Module):
    """Learn token weights and compute a weighted sum over sequence tokens."""
    def __init__(self, embed_dim):
        super().__init__()
        self.score = nn.Linear(embed_dim, 1)

    def forward(self, x):
        # x shape: [batch_size, seq_len, embed_dim]
        attn_weights = torch.softmax(self.score(x).squeeze(-1), dim=1)
        pooled = torch.sum(x * attn_weights.unsqueeze(-1), dim=1)
        return pooled



class StudentNet_DPGC_Residual(nn.Module):
    def __init__(self):
        super().__init__()

        self.block1 = DPGCResidualBlock(3, 32, group_conv_groups=8)
        self.block2 = DPGCResidualBlock(32, 32, group_conv_groups=8)
        self.block3 = DPGCResidualBlock(32, 64, group_conv_groups=8)
        self.block4 = DPGCResidualBlock(64, 128, group_conv_groups=8)
        self.block5 = DPGCResidualBlock(128, 128, group_conv_groups=8)

        self.pool = nn.MaxPool2d(2, 2)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, 11)
        '''
        self.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 11),
        )
        '''

    def forward(self, x):
        x = self.pool(self.block1(x))
        x = self.pool(self.block2(x))
        x = self.pool(self.block3(x))
        x = self.pool(self.block4(x))
        x = self.pool(self.block5(x))
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class StudentNet_DPGC_Residual_Transformer(nn.Module):
    def __init__(self):
        super().__init__()

        self.block1 = DPGCResidualBlock(3, 32, group_conv_groups=4)
        self.block2 = DPGCResidualBlock(32, 32, group_conv_groups=4)
        self.block3 = DPGCResidualBlock(32, 64, group_conv_groups=4)
        self.block4 = DPGCResidualBlock(64, 64, group_conv_groups=4)

        self.pool = nn.MaxPool2d(2, 2)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(96, 11)
        self.transformer = TransformerBlock(embed_dim=64, num_heads=1, ff_hidden_dim=128)
        self.attn_pool = SelfAttentionPooling(embed_dim=64)
        self.fc = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(64, 96),
        nn.ReLU(),
        nn.Linear(96, 11),
        )

    def forward(self, x):   
        x = self.pool(self.block1(x))
        x = self.pool(self.block2(x))
        x = self.pool(self.block3(x))
        x = self.pool(self.block4(x))
        B, C, H, W = x.size()
        x = x.view(B, C, H*W).transpose(1, 2)  # (B, H*W, C)
        x = self.transformer(x)  # (B, H*W, C)
        x = self.attn_pool(x)  # Self-attention pooling over sequence tokens
        # x = self.gap(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)