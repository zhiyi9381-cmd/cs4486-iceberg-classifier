# -*- coding: utf-8 -*-
"""
VGG9-style large model (~1.76M parameters), with inc_angle bypass
Architecture: 5 groups of Conv+BN+ReLU (2 layers per group, MaxPool between groups) -> GAP(256) -> Dropout -> cat(angle) -> FC
Channels: 2 -> 32 -> 64 -> 128 -> 256 -> 256
Used to train stronger nn_pred features in a GPU environment
"""
import torch
import torch.nn as nn


class VGG9(nn.Module):
    def __init__(self, dropout=0.4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),   # 75->37
            nn.Dropout(dropout),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),   # 37->18
            nn.Dropout(dropout),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),  # 18->9
            nn.Dropout(dropout),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),  # 9->4
            nn.Dropout(dropout),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(256 + 1, 1)

    def forward(self, x_img, x_angle):
        h = self.features(x_img)                 # (B, 256, 4, 4)
        h = h.mean(dim=(2, 3))                   # GAP -> (B, 256)
        h = self.dropout(h)
        h = torch.cat([h, x_angle], dim=1)       # (B, 257)
        return self.head(h)


def count_params(model):
    per_layer = [(name, p.numel()) for name, p in model.named_parameters()]
    total = sum(n for _, n in per_layer)
    return per_layer, total
