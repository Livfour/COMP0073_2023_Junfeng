import torch
import torch.nn as nn
from timm.models.vision_transformer import Block, PatchEmbed


class ViTEncoder(nn.Module):
    def __init__(self,
                 img_size=(256, 192),
                 patch_size=16,
                 num_img_channel=3,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 drop_path_rate=0.3,
                 pretrained_path=None,
                 train_backbone=False,
                 ) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_img_channel = num_img_channel
        self.embed_dim = embed_dim
        self.Hp = int(img_size[0] // patch_size)
        self.Wp = int(img_size[1] // patch_size)
        self.num_patches = self.Hp * self.Wp
        self.pretrained_path = pretrained_path
        self.train_backbone = train_backbone

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=num_img_channel,
            embed_dim=embed_dim,
        )

        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, embed_dim))

        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads,
                  mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop_path=drop_path_rate)
            for i in range(depth)])

        self.last_norm = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self):
        if self.pretrained_path is not None:
            pretrained_vit_dict = torch.load(
                self.pretrained_path)["state_dict"]
            updated_vit_dict = {
                k.replace("backbone.", ''): v for k, v in pretrained_vit_dict.items()}
            self.load_state_dict(updated_vit_dict, strict=False)

        if not self.train_backbone:
            for p in self.parameters():
                p.requires_grad = False

    def forward(self, x):
        """
        x : (B, C, H, W)
        B : batch size
        C : number of channels
        H : height
        W : width
        """
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]
        for blk in self.blocks:
            x = blk(x)
        x = self.last_norm(x)
        return x
