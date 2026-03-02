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
import torchaudio
from torchaudio.models import Conformer

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
        # x: (B, in_features)
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

class SelfAttentivePooling(nn.Module):
    def __init__(self, d_model: int, hidden: int = 256):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.Tanh(), # keep the value of attention score between -1 and 1 to stabilize training.
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        # x: [B, L, D]
        e = self.attn(x).squeeze(-1)  # [B, L]
        a = torch.softmax(e, dim=1).unsqueeze(-1)  # [B, L, 1]
        pooled = torch.sum(a * x, dim=1)           # [B, D]
        return pooled

class SelfAttentivePooling_MultiHead(nn.Module):
    def __init__(self, d_model: int, hidden: int = 64, num_head: int = 4, eps: float  = 1e-5):
        super().__init__()
        self.num_head = num_head
        self.eps = eps
        self.attn = nn.Sequential(
            nn.Linear(d_model, hidden),
            # nn.Tanh(), # keep the value of attention score between -1 and 1 to stabilize training.
            nn.ReLU(),
            nn.Linear(hidden, num_head),
        )

    def forward(self, x):
        # x: [B, L, D]
        e = self.attn(x)  # [B, L, H]
        a = torch.softmax(e, dim=1)  # [B, L, H]
        a_t = a.permute(0, 2, 1) # [B, H, L]
        # if H=1, pooled is the weighted sum of x, which is the same as the single head self attentive pooling.
        mean = torch.matmul(a_t, x) # [B, H, D]

        # calculate std
        # expand for broadcasting (this magic is suggested by ChatGPT, it makes sense to me but it is something I can't do by myself.)
        # if H = 1, it is just a weighted std vector of x
        mean_expanded = mean.unsqueeze(2)     # [B, H, 1, D]
        x_expanded = x.unsqueeze(1)           # [B, 1, L, D]
        a_expanded = a_t.unsqueeze(-1)        # [B, H, L, 1]

        var = torch.sum(a_expanded * (x_expanded - mean_expanded) ** 2, dim=2)  # [B, H, D]
        std = torch.sqrt(var.clamp_min(self.eps))  # [B, H, D] # clamp_min: ensure the value is larger than eps for numerical stability.

        # ===== concatenate =====
        pooled = torch.cat([mean, std], dim=-1)   # [B, H, 2D]

        a_t_a = torch.matmul(a_t, a) # [B, H, H]
        # Add a regularization term to encourage the attention heads to be different from each other.
        # The regularization term is the Frobenius norm of (a_t_a - I), where I is the identity matrix.
        # Follow the paper "Self-Attentive Speaker Embeddings for Text-Independent Speaker Verification"
        I_mat = torch.eye(self.num_head, device=a_t_a.device).unsqueeze(0)  # [1,H,H] # identity matrix
        diff = a_t_a - I_mat                                                # [B,H,H]
        reg = torch.norm(diff, dim=(1,2)).mean()                            # scalar
        # pooled = torch.sum(a * x, dim=1)           # [B, D]
        return pooled, reg

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
    def __init__(self, d_model: int = 192, n_spks: int = 600, dropout: float = 0.2, am_s: float = 30.0, am_m: float = 0.35, n_transformer_layer: int = 2):
        super().__init__()
        # Project the dimension of features from that of input into d_model.
        self.prenet = nn.Linear(40, d_model)
        # TODO:
        #   Change Transformer to Conformer.
        #   https://arxiv.org/abs/2005.08100
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, dim_feedforward=768, nhead=1, dropout=dropout)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=n_transformer_layer)

        # Project the the dimension of features from d_model into speaker nums.
        self.embed_layer = nn.Sequential(
            # nn.Linear(d_model, n_spks),
            # nn.ReLU(),
            nn.Linear(d_model, d_model),
            # nn.ReLU(),
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
    
class Classifier_AMS_SAP(nn.Module):
    def __init__(self, d_model: int = 192, n_spks: int = 600, dropout: float = 0.2, am_s: float = 30.0, am_m: float = 0.35, n_transformer_layer: int = 2):
        super().__init__()
        # Project the dimension of features from that of input into d_model.
        self.prenet = nn.Linear(40, d_model)
        # TODO:
        #   Change Transformer to Conformer.
        #   https://arxiv.org/abs/2005.08100
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, dim_feedforward=768, nhead=1, dropout=dropout)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=n_transformer_layer)

        # Project the the dimension of features from d_model into speaker nums.
        self.embed_layer = nn.Sequential(
            # nn.Linear(d_model, n_spks),
            # nn.ReLU(),
            nn.Linear(d_model, d_model),
            # nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.am_head = AMSoftmax(in_features=d_model, num_classes=n_spks, s=am_s, m=am_m)
        self.sap_layer = SelfAttentivePooling(d_model=d_model)

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
        # stats = out.mean(dim=1)
        stats = self.sap_layer(out)
        # emb: (batch, d_model)
        emb = self.embed_layer(stats)
        # out: (batch, n_spks)
        out = self.am_head(emb, labels)
        return out
    
class Classifier_AMS_SAP_MultiHead(nn.Module):
    def __init__(self, d_model: int = 128, n_spks: int = 600, dropout: float = 0.2, am_s: float = 30.0, am_m: float = 0.35, n_transformer_layer: int = 2, n_head: int = 4):
        super().__init__()
        # Project the dimension of features from that of input into d_model.
        self.prenet = nn.Linear(40, d_model)
        # TODO:
        #   Change Transformer to Conformer.
        #   https://arxiv.org/abs/2005.08100
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, dim_feedforward=512, nhead=1, dropout=dropout)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=n_transformer_layer)

        # Project the the dimension of features from d_model into speaker nums.
        self.embed_layer = nn.Sequential(
            # nn.Linear(d_model, n_spks),
            # nn.ReLU(),
            nn.Linear(2 * n_head * d_model, 4096), # factor of two comes from std
            nn.ReLU(),
            # nn.BatchNorm1d(2048),
            nn.Linear(4096, 2048), # factor of two comes from std
            # nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.am_head = AMSoftmax(in_features=2048, num_classes=n_spks, s=am_s, m=am_m)
        self.sap_layer = SelfAttentivePooling_MultiHead(d_model=d_model, num_head=n_head)

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
        # stats = out.mean(dim=1)
        stats, reg_scalar = self.sap_layer(out) # [B, H, D], scalar regularization term is also returned but we do not use it here.
        stats = stats.reshape(stats.size(0), -1) # [B, H*D*2] flatten the last two dimensions (head and mean/std).
        # emb: (batch, d_model)
        emb = self.embed_layer(stats)
        # out: (batch, n_spks)
        out = self.am_head(emb, labels)
        return out, reg_scalar


class Classifier_Conformer_AMS_SAP_MultiHead(nn.Module):
    def __init__(self, d_model: int = 128, n_spks: int = 600, dropout: float = 0.2, am_s: float = 30.0, am_m: float = 0.35, n_transformer_layer: int = 2, n_head: int = 4,
                 kernel_size: int = 31):
        super().__init__()
        # Project the dimension of features from that of input into d_model.
        self.prenet = nn.Linear(40, d_model)
        # TODO:
        #   Change Transformer to Conformer.
        #   https://arxiv.org/abs/2005.08100
        # Define comformer encoder layer and encoder here.
        self.encoder = Conformer(input_dim=d_model, num_heads=1, num_layers=n_transformer_layer, ffn_dim=512, depthwise_conv_kernel_size=kernel_size, dropout=dropout)
        # self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, dim_feedforward=512, nhead=1, dropout=dropout)
        # self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=n_transformer_layer)

        # Project the the dimension of features from d_model into speaker nums.
        self.embed_layer = nn.Sequential(
            # nn.Linear(d_model, n_spks),
            # nn.ReLU(),
            nn.Linear(2 * n_head * d_model, 4096), # factor of two comes from std
            nn.ReLU(),
            # nn.BatchNorm1d(2048),
            nn.Linear(4096, 2048), # factor of two comes from std
            # nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.am_head = AMSoftmax(in_features=2048, num_classes=n_spks, s=am_s, m=am_m)
        self.sap_layer = SelfAttentivePooling_MultiHead(d_model=d_model, num_head=n_head)

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
        # out = out.permute(1, 0, 2) # For transformer. No need of conformer. Conformer expect input of shape (batch size, length, d_model).
        # The encoder layer expect features in the shape of (length, batch size, d_model).
        # out = self.encoder_layer(out)
        lengths = torch.full((out.shape[0],), out.shape[1], dtype=torch.long, device=out.device)
        out, _ = self.encoder(out, lengths=lengths) # conformer encoder expect input of shape (batch size, length, d_model).
        # out: (batch size, length, d_model)
        # out = out.transpose(0, 1) # For transformer. No need of conformer. Conformer expect input of shape (batch size, length, d_model).
        # mean pooling
        # stats: (batch size, d_model)
        # stats = out.mean(dim=1)
        stats, reg_scalar = self.sap_layer(out) # [B, H, D], scalar regularization term is also returned but we do not use it here.
        stats = stats.reshape(stats.size(0), -1) # [B, H*D*2] flatten the last two dimensions (head and mean/std).
        # emb: (batch, d_model)
        emb = self.embed_layer(stats)
        # out: (batch, n_spks)
        out = self.am_head(emb, labels)
        return out, reg_scalar