# -*- coding: utf-8 -*-
"""
Widened DeepCNN (~400k parameters), with inc_angle bypass
Architecture: 5 groups of Conv+BN+ReLU (MaxPool in the first 4 groups) -> GAP(128) -> Dropout -> cat(angle) -> FC
Channels: 2 -> 32 -> 64 -> 128 -> 128 -> 128
"""
import torch
import torch.nn as nn


class DeepCNNWide(nn.Module):
    def __init__(self, dropout=0.4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),   # 75->37
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),  # 37->18
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),  # 18->9
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),  # 9->4
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),                  # 4->4
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(128 + 1, 1)   # 128-dim GAP output + 1-dim angle bypass

    def forward(self, x_img, x_angle):
        h = self.features(x_img)                 # (B, 128, 4, 4)
        h = h.mean(dim=(2, 3))                   # GlobalAvgPool -> (B, 128)
        h = self.dropout(h)
        h = torch.cat([h, x_angle], dim=1)       # (B, 129)
        return self.head(h)                       # (B, 1) logit


def count_params(model):
    per_layer = [(name, p.numel()) for name, p in model.named_parameters()]
    total = sum(n for _, n in per_layer)
    return per_layer, total
