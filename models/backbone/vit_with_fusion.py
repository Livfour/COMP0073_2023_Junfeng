import torch
import torch.nn as nn
from timm.models.vision_transformer import Block
from utilities.utilities import get_3d_sincos_pos_embed
from models.backbone.vit import ViTEncoder


class FusionVit(nn.Module):
    def __init__(self, pretrained_path=None):
        super().__init__()
        self.encoder = ViTEncoder(pretrained_path=pretrained_path)
        self.fusion = TransformerFusion()

    def forward(self, video):
        f1 = self.encoder(video[:, 0])
        f2 = self.encoder(video[:, 1])
        f3 = self.encoder(video[:, 2])
        return self.fusion(f1, f2, f3)


class TransformerFusion(nn.Module):
    def __init__(self,
                 img_size=(256, 192),
                 patch_size=16,
                 num_img_channels=3,
                 num_frames=3,
                 embed_dim=768,
                 depth=4,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 drop_path_rate=0.3,
                 ) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_img_channels = num_img_channels
        self.num_frames = num_frames
        self.embed_dim = embed_dim
        self.Hp = int(img_size[0] // patch_size)
        self.Wp = int(img_size[1] // patch_size)
        self.num_patches = self.Hp * self.Wp

        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_frames * self.num_patches, embed_dim))

        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads,
                  mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop_path=drop_path_rate)
            for i in range(depth)])

        self.last_norm = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self):
        self.pos_embed = nn.Parameter(get_3d_sincos_pos_embed(
            embed_dim=self.embed_dim, grid_size=(self.num_frames, self.Hp, self.Wp)).reshape(1, -1, self.embed_dim))

    def forward(self, f1, f2, f3):
        """
        f: (B, num_frame * num_patches, embed_dim)
        """
        B = f1.shape[0]
        f = torch.cat([f1, f2, f3], dim=1)
        f = f + self.pos_embed
        for blk in self.blocks:
            f = blk(f)
        f = self.last_norm(f)
        f = f[:, self.num_patches:self.num_patches * 2, :]
        f = f.permute(0, 2, 1).contiguous().view(
            B, self.embed_dim, self.Hp, self.Wp)
        return f
