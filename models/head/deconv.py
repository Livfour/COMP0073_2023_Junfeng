import torch
import torch.nn as nn
from models.head.BaseHead import BaseHead

class DeconvHead(BaseHead):
    def __init__(self, embed_dim=768, pretrained_path=None):
        super().__init__(is_diffusion=False)
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256, eps=1e-5, momentum=0.1,
                           affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256, eps=1e-5, momentum=0.1,
                           affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
        )
        self.final_layer = nn.Conv2d(256, 17, 1)

        if pretrained_path is not None:
            pretrained_vit_dict = torch.load(pretrained_path)["state_dict"]
            updated_vit_dict = {
                k.replace("keypoint_head.", ''): v for k, v in pretrained_vit_dict.items()}
            self.load_state_dict(updated_vit_dict, strict=False)

    def forward(self, x):
        x = self.deconv_layers(x)
        x = self.final_layer(x)
        return x
    
    def predict(self, x):
        return self.forward(x)
