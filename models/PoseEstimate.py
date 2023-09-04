import torch.nn as nn

class PoseEstimate(nn.Module):
    def __init__(self,
                 encoder: nn.Module,
                 neck: nn.Module,
                 head: nn.Module
                 ) -> None:
        super().__init__()
        self.encoder = encoder
        self.neck = neck
        self.head = head
        self.is_diffusion = self.head.is_diffusion

        if self.is_diffusion:
            self.forward = self.forward_diffusion
        else:
            self.forward = self.forward_normal
    
    def forward_normal(self, x):
        x = self.encoder(x)
        x = self.neck(x)
        x = self.head(x)
        return x
    
    def forward_diffusion(self, x, h):
        x = self.encoder(x)
        x = self.neck(x)
        x = self.head(x, h)
        return x

    def predict(self, x):
        x = self.encoder(x)
        x = self.neck(x)
        x = self.head.predict(x)
        return x
    
    def denoise(self, x, h):
        x = self.encoder(x)
        x = self.neck(x)
        x = self.head.denoise(x, h)
        return x


