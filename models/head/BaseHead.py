import torch
import torch.nn as nn

class BaseHead(nn.Module):
    def __init__(self, is_diffusion=False) -> None:
        super().__init__()
        self.is_diffusion = is_diffusion
