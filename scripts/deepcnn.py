# -*- coding: utf-8 -*-
"""
DeepCNN main model (~66k parameters, with inc_angle bypass)
Architecture: 4 groups of Conv+BN+ReLU (+MaxPool in the first 3 groups) -> GAP -> Dropout -> cat(angle) -> FC
Per-layer parameter counts are written to the log by the caller.
"""
import torch
import torch.nn as nn


class DeepCNN(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 72, 3, padding=1), nn.BatchNorm2d(72), nn.ReLU(),
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(72 + 1, 1)   # 72-dim GAP output + 1-dim angle bypass

    def forward(self, x_img, x_angle):
        h = self.features(x_img)                 # (B, 72, 9, 9)
        h = h.mean(dim=(2, 3))                   # GlobalAvgPool -> (B, 72)
        h = self.dropout(h)
        h = torch.cat([h, x_angle], dim=1)       # (B, 73)
        return self.head(h)                       # (B, 1) logit


def count_params(model):
    per_layer = [(name, p.numel()) for name, p in model.named_parameters()]
    total = sum(n for _, n in per_layer)
    return per_layer, total
