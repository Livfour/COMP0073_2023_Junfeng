import torch
import torch.nn as nn
from models.head.BaseHead import BaseHead

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


class SelfRefinement(nn.Module):
    def __init__(self, num_channels, num_iterations):
        super(SelfRefinement, self).__init__()
        self.num_iterations = num_iterations
        self.conv = nn.Conv2d(num_channels, num_channels,
                              kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        for _ in range(self.num_iterations):
            x = self.conv(x)
        return x


class ResMaskHead2(BaseHead):
    def __init__(self,
                 heatmap_size=(64, 48),
                 emb_dim=768,
                 num_keypoints=17,
                 num_timesteps=1000,
                 ) -> None:
        super().__init__(is_diffusion=True)
        self.H_heatmap, self.W_heatmap = heatmap_size
        self.emb_dim = emb_dim
        self.num_keypoints = num_keypoints
        self.num_timesteps = num_timesteps

        for k, v in ddpm_schedules(self.num_timesteps).items():
            self.register_buffer(k, v)

        self.time_emb = nn.Linear(1, self.num_keypoints)

        self.self_refine1 = SelfRefinement(self.num_keypoints, 1)
        self.self_refine2 = SelfRefinement(self.num_keypoints, 1)
        self.self_refine3 = SelfRefinement(self.num_keypoints, 1)

        self.down1 = nn.Identity()
        self.down2 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.num_keypoints,
                out_channels=self.num_keypoints,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(self.num_keypoints, eps=1e-5, momentum=0.1,
                           affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
        )

        self.down4 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.num_keypoints,
                out_channels=self.num_keypoints,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(self.num_keypoints, eps=1e-5, momentum=0.1,
                           affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=self.num_keypoints,
                out_channels=self.num_keypoints,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(self.num_keypoints, eps=1e-5, momentum=0.1,
                           affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
        )

        self.up1 = nn.Identity()
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(self.emb_dim, self.emb_dim,
                               4, 2, 1, bias=False),
            nn.BatchNorm2d(self.emb_dim, eps=1e-5, momentum=0.1,
                           affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
        )
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(self.emb_dim, self.emb_dim,
                               4, 2, 1, bias=False),
            nn.BatchNorm2d(self.emb_dim, eps=1e-5, momentum=0.1,
                           affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.emb_dim, self.emb_dim,
                               4, 2, 1, bias=False),
            nn.BatchNorm2d(self.emb_dim, eps=1e-5, momentum=0.1,
                           affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
        )

        self.up1_conv = nn.Identity()
        self.up2_conv = nn.Sequential(
           nn.ConvTranspose2d(self.emb_dim*2, self.emb_dim*2,
                               4, 2, 1, bias=False),
            nn.BatchNorm2d(self.emb_dim*2, eps=1e-5, momentum=0.1,
                            affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
        )
        self.up4_conv = nn.Sequential(
            nn.ConvTranspose2d(self.emb_dim*2, self.emb_dim*2,
                               4, 2, 1, bias=False),
            nn.BatchNorm2d(self.emb_dim*2, eps=1e-5, momentum=0.1,
                            affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.emb_dim*2, self.emb_dim*2,
                               4, 2, 1, bias=False),
            nn.BatchNorm2d(self.emb_dim*2, eps=1e-5, momentum=0.1,
                            affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
        )

        self.sigmod = nn.Sigmoid()

        self.dect_head = nn.Sequential(
            nn.Conv2d(self.emb_dim*2, self.emb_dim, 3, 1, 1),
            nn.BatchNorm2d(self.emb_dim, eps=1e-5, momentum=0.1,
                            affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.emb_dim, 256, 3, 1, 1),
            nn.BatchNorm2d(256, eps=1e-5, momentum=0.1,
                            affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
                        nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256, eps=1e-5, momentum=0.1,
                            affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, self.num_keypoints, 1),
        )


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
        
        h_t_1 = self.down4(h_t)
        h_t_2 = self.down2(h_t)
        h_t_3 = self.down1(h_t)

        h_t_1 = self.self_refine1(h_t_1)
        h_t_2 = self.self_refine2(h_t_2)
        h_t_3 = self.self_refine3(h_t_3)

        f_t_1 = self.up1(f)
        f_t_2 = self.up2(f)
        f_t_3 = self.up4(f)

        f_t_1 = self._mask_fusion(f_t_1, h_t_1)
        f_t_2 = self._mask_fusion(f_t_2, h_t_2)
        f_t_3 = self._mask_fusion(f_t_3, h_t_3)
        
        f_t_1 = self.up4_conv(f_t_1)
        f_t_2 = self.up2_conv(f_t_2)
        f_t_3 = self.up1_conv(f_t_3)
        del h_t_1, h_t_2, h_t_3
        return self.dect_head(f_t_1 + f_t_2 + f_t_3)

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
        _ts = torch.tensor(200, device=f.device).repeat(B, 1)
        return self.forward(f, heatmaps, _ts)