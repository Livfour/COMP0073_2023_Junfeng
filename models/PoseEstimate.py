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
            self.predict = self.predict_diffusion
        else:
            self.forward = self.forward_normal
            self.predict = self.predict_normal
    
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

    def predict_normal(self, x):
        x = self.encoder(x)
        x = self.neck(x)
        x = self.head.predict(x)
        return x

    def predict_diffusion(self, x, n=1):
        x = self.encoder(x)
        x = self.neck(x)
        x = self.head.predict(x, n=n)
        return x
    
    def denoise(self, x, h, n=1, t=100):
        x = self.encoder(x)
        x = self.neck(x)
        x = self.head.denoise(x, h, n=n, t=t)
        return x


