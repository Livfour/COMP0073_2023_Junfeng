import torch
import torch.nn as nn
from .BaseHead import BaseHead

class SimpleHead(BaseHead):
    def __init__(self, embed_dim=768):
        super().__init__(is_diffusion=False)
        self.up = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(embed_dim, 17, 3, padding=1)

    def forward(self, x):
        x = self.up(x)
        x = self.relu(x)
        x = self.conv(x)
        return x
    
    def predict(self, x):
        return self.forward(x)