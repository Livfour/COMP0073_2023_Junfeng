import torch
import torch.nn as nn
from models.head.BaseHead import BaseHead
from models.head.deconv import DeconvHead


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
    alphas_cumprod = torch.cos(
        ((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def ddpm_schedules(time_steps):
    betas = cosine_beta_schedule(timesteps=time_steps)
    alphas = 1 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    return {
        "betas": betas,
        "alphas": alphas,
        "alphas_cumprod": alphas_cumprod,
        "sqrt_alphas_cumprod": sqrt_alphas_cumprod,
        "sqrt_one_minus_alphas_cumprod": sqrt_one_minus_alphas_cumprod,
    }


class MaskHead(BaseHead):
    def __init__(self,
                 heatmap_size=(64, 48),
                 emb_dim=768,
                 num_keypoints=17,
                 num_timesteps=1000,
                 pretrained_path=None,
                 ) -> None:
        super().__init__(is_diffusion=True)
        self.H_heatmap, self.W_heatmap = heatmap_size
        self.emb_dim = emb_dim
        self.num_keypoints = num_keypoints
        self.num_timesteps = num_timesteps

        for k, v in ddpm_schedules(self.num_timesteps).items():
            self.register_buffer(k, v)

        self.time_emb = nn.Linear(1, self.num_keypoints)
        self.down = nn.MaxPool2d(kernel_size=4, stride=4)
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=self.emb_dim*2,
                out_channels=self.emb_dim,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(self.emb_dim, eps=1e-5, momentum=0.1,
                           affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
        )
        self.final_conv = DeconvHead(pretrained_path=pretrained_path)

        self.sigmod = nn.Sigmoid()

    def _add_noise(self, h, _ts):
        eps = torch.randn_like(h)
        h_t = self.sqrt_alphas_cumprod[_ts, None, None] * h + \
            self.sqrt_one_minus_alphas_cumprod[_ts, None, None] * eps
        return h_t

    @staticmethod
    def _squeeze(h):
        return torch.max(h, dim=1, keepdim=True)[0]

    def _mask_fusion(self, f_t, h_t):
        h_t_bar = self._squeeze(h_t)
        h_t_bar = self.sigmod(h_t_bar)
        f_t_masked = f_t * h_t_bar
        f_t_out = torch.cat([f_t_masked, f_t], dim=1)
        return f_t_out

    def forward(self, f, h, _ts=None):
        """
        f: (B, embed_dim, num_patches_height, num_patches_width)
        h: (B, num_keypoints, heatmap_height, heatmap_width)
        """
        B = f.shape[0]
        if _ts is None:
            _ts = torch.randint(0, self.num_timesteps, (B,),
                                device=f.device).reshape(-1, 1)
        h_t = self._add_noise(h, _ts)
        _ts = _ts / self.num_timesteps
        _ts = self.time_emb(_ts).reshape(B, -1, 1, 1)
        h_t = h_t * _ts
        h_t = self.down(h_t)
        f_t = self._mask_fusion(f, h_t)
        out = self.conv(f_t)
        out = self.final_conv(out)
        return out

    def predict(self, f):
        """
        f: (B, embed_dim, num_patches_height, num_patches_width)
        """

        B = f.shape[0]
        h = torch.randn(B, self.num_keypoints, self.H_heatmap, self.W_heatmap).to(
            f.device)
        _ts = torch.tensor(self.num_timesteps-1,
                           device=f.device).repeat(B, 1)
        return self.forward(f, h, _ts)

    def denoise(self, f, heatmaps):
        B = f.shape[0]
        _ts = torch.tensor(1, device=f.device).repeat(B, 1)
        return self.forward(f, heatmaps, _ts)
