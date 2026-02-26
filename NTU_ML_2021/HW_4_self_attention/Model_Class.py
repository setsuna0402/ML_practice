'''
Define models used in this project.
Author: Dr. Ka Hou Leong
Date: 27/2/2026
'''
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class AMSoftmax(nn.Module):
    """
    Additive Margin Softmax (CosFace)
    logits_y = s*(cos_y - m), logits_j = s*cos_j (j != y)
    B: batch_size, C: num_classes, D: feature_dim
    """
    def __init__(self, in_features: int, num_classes: int, s: float = 30.0, m: float = 0.35):
        super().__init__()
        self.s = float(s)
        self.m = float(m)
        self.weight = nn.Parameter(torch.randn(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        # x: (B, D)
        x = F.normalize(x, p=2, dim=1)
        w = F.normalize(self.weight, p=2, dim=1)
        cosine = F.linear(x, w)  # (B, C)

        if labels is None:
            # inference / validation: no margin
            return cosine * self.s
        # view(-1): flatten the labels into 1D tensor. For example, if labels is of shape (B, 1), after view(-1), it will be of shape (B,).
        labels = labels.view(-1).long()  # (B,)
        one_hot = torch.zeros_like(cosine) # (B, C)
        # scatter_(dim, index, val)
        # Along dim, set the value of the index-th element as val
        # for example, label is 2, num_class is 4, batch_size is 1
        # one_hot : [[0, 0, 1, 0]]
        one_hot.scatter_(1, labels.unsqueeze(1), 1.0) # (B, C)

        logits = self.s * (cosine - one_hot * self.m)
        return logits


class Classifier(nn.Module):
    def __init__(self, d_model=80, n_spks=600, dropout=0.1):
        super().__init__()
        # Project the dimension of features from that of input into d_model.
        self.prenet = nn.Linear(40, d_model)
        # TODO:
        #   Change Transformer to Conformer.
        #   https://arxiv.org/abs/2005.08100
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, dim_feedforward=256, nhead=1)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)

        # Project the the dimension of features from d_model into speaker nums.
        self.pred_layer = nn.Sequential(
            # nn.Linear(d_model, n_spks),
            # nn.ReLU(),
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
        # out = self.encoder_layer(out)
        out = self.encoder(out)
        # out: (batch size, length, d_model)
        out = out.transpose(0, 1)
        # mean pooling
        stats = out.mean(dim=1)

        # out: (batch, n_spks)
        out = self.pred_layer(stats)
        return out
    
class Classifier_AMS(nn.Module):
    def __init__(self, d_model=80, n_spks=600, dropout=0.1, am_s=30.0, am_m=0.35):
        super().__init__()
        # Project the dimension of features from that of input into d_model.
        self.prenet = nn.Linear(40, d_model)
        # TODO:
        #   Change Transformer to Conformer.
        #   https://arxiv.org/abs/2005.08100
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, dim_feedforward=256, nhead=1)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)

        # Project the the dimension of features from d_model into speaker nums.
        self.embed_layer = nn.Sequential(
            # nn.Linear(d_model, n_spks),
            # nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.am_head = AMSoftmax(in_features=d_model, num_classes=n_spks, s=am_s, m=am_m)

    def forward(self, mels, labels=None):
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
        # out = self.encoder_layer(out)
        out = self.encoder(out)
        # out: (batch size, length, d_model)
        out = out.transpose(0, 1)
        # mean pooling
        # stats: (batch size, d_model)
        stats = out.mean(dim=1)

        # emb: (batch, d_model)
        emb = self.embed_layer(stats)
        # out: (batch, n_spks)
        out = self.am_head(emb, labels)
        return out