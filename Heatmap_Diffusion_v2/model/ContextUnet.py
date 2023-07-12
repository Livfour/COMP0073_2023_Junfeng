"""
Simple Unet Structure.
"""
import torch
import torch.nn as nn
from torchvision import models


class Conv3(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, is_res: bool = False
    ) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(),
        )

        self.is_res = is_res

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.main(x)
        if self.is_res:
            x = x + self.conv(x)
            return x / 1.414
        else:
            return self.conv(x)


class UnetDown(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(UnetDown, self).__init__()
        layers = [Conv3(in_channels, out_channels), nn.MaxPool2d(2)]
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return self.model(x)


class UnetUp(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(UnetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            Conv3(out_channels, out_channels),
            Conv3(out_channels, out_channels),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = torch.cat((x, skip), 1)
        x = self.model(x)
        return x


class TimeSiren(nn.Module):
    def __init__(self, emb_dim: int) -> None:
        super(TimeSiren, self).__init__()

        self.lin1 = nn.Linear(1, emb_dim, bias=False)
        self.lin2 = nn.Linear(emb_dim, emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, 1)
        x = torch.sin(self.lin1(x))
        x = self.lin2(x)
        return x


class EmbedFC(nn.Module):
    def __init__(self, in_dim: int, emb_dim: int) -> None:
        super(EmbedFC, self).__init__()

        self.fc = nn.Linear(in_dim, emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class ContextUnet(nn.Module):
    def __init__(self, n_channel, n_feat=256, n_cfeat=10):
        super(ContextUnet, self).__init__()

        self.n_channel = n_channel
        self.n_feat = n_feat
        self.n_cfeat = n_cfeat

        self.init_conv = Conv3(self.n_channel, n_feat, is_res=True)

        self.down1 = UnetDown(n_feat, n_feat)
        self.down2 = UnetDown(n_feat, 2 * n_feat)
        self.down3 = UnetDown(2 * n_feat, 2 * n_feat)

        self.to_vec = nn.Sequential(nn.AvgPool2d((4)), nn.GELU())

        self.timeembed0 = EmbedFC(1, 2*n_feat)
        self.timeembed1 = EmbedFC(1, 2*n_feat)
        self.timeembed2 = EmbedFC(1, 1*n_feat)
        self.contextembed0 = EmbedFC(n_cfeat, 2*n_feat)
        self.contextembed1 = EmbedFC(n_cfeat, 2*n_feat)
        self.contextembed2 = EmbedFC(n_cfeat, 1*n_feat)

        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, 4, 4),
            nn.GroupNorm(8, 2 * n_feat),
            nn.ReLU(),
        )

        self.up1 = UnetUp(4 * n_feat, 2 * n_feat)
        self.up2 = UnetUp(4 * n_feat, n_feat)
        self.up3 = UnetUp(2 * n_feat, n_feat)

        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat, self.n_channel, 3, 1, 1),
        )

        self.image_encoder = models.resnet50(
            weights="ResNet50_Weights.DEFAULT")

    def forward(self, x, t, c=None):
        t = t.view(-1, 1)
        # print(f"t shape: {t.shape}")
        # print(f"c shape: {c.shape}")
        c = self.image_encoder(c)
        # print(f"encoded c shape: {c.shape}")

        # print(f"input shape: {x.shape}")
        x = self.init_conv(x)
        # print(f"init conv shape: {x.shape}")
        down1_out = self.down1(x)
        # print(f"down1 shape: {down1_out.shape}")
        down2_out = self.down2(down1_out)
        # print(f"down2 shape: {down2_out.shape}")
        down3_out = self.down3(down2_out)
        # print(f"down3 shape: {down3_out.shape}")

        hiddenvec = self.to_vec(down3_out)
        # print(f"hiddenvec shape: {hiddenvec.shape}")

        # print(f"t shape: {t.shape}")
        temb0 = self.timeembed0(t).view(-1, self.n_feat * 2, 1, 1)
        # print(f"temb0 shape: {temb0.shape}")
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1)
        # print(f"temb1 shape: {temb1.shape}")
        temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)
        # print(f"temb2 shape: {temb2.shape}")

        # print(f"c shape: {c.shape}")
        cemb0 = self.contextembed0(c).view(-1, self.n_feat * 2, 1, 1)
        # print(f"cemb0 shape: {cemb0.shape}")
        cemb1 = self.contextembed1(c).view(-1, self.n_feat * 2, 1, 1)
        # print(f"cemb1 shape: {cemb1.shape}")
        cemb2 = self.contextembed2(c).view(-1, self.n_feat, 1, 1)
        # print(f"cemb2 shape: {cemb2.shape}")

        up0_out = self.up0(hiddenvec)
        # print(f"up0 shape: {up0_out.shape}")
        up1_out = self.up1(cemb0 * up0_out + temb0, down3_out)
        # print(f"up1 shape: {up1_out.shape}")
        up2_out = self.up2(cemb1 * up1_out + temb1, down2_out)
        # print(f"up2 shape: {up2_out.shape}")
        up3_out = self.up3(cemb2 * up2_out + temb2, down1_out)
        # print(f"up3 shape: {up3_out.shape}")
        out = self.out(torch.cat((up3_out, x), 1))
        # print(f"out shape: {out.shape}")
        return out
