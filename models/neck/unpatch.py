import torch
import torch.nn as nn


class UnPatch(nn.Module):
    def __init__(self,
                 img_size=(256, 192),
                 patch_size=16,
                 num_img_channel=3,
                 embed_dim=768,
                 ) -> None:
        super().__init__()
        self.H, self.W = img_size
        self.Hp, self.Wp = int(self.H // patch_size), int(self.W // patch_size)

    def forward(self, f):
        """
        f : (B, Num_Patches, Embed_Dim)
        """
        B, N, C = f.shape
        H, W = self.Hp, self.Wp
        f = f.permute(0, 2, 1).reshape(B, C, H, W)
        return f
