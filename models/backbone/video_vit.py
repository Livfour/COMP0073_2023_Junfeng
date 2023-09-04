import torch
import torch.nn as nn
from timm.models.vision_transformer import Block, PatchEmbed
from utilities.utilities import get_3d_sincos_pos_embed

class VideoViTEncoder(nn.Module):
    def __init__(self,
                 img_size=(256, 192),
                 patch_size=16,
                 num_img_channel=3,
                 num_frame=3,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 drop_path_rate=0.3,
                 pretrained_path=None,
                 ) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_img_channel = num_img_channel
        self.num_frame = num_frame
        self.embed_dim = embed_dim
        self.Hp = int(img_size[0] // patch_size)
        self.Wp = int(img_size[1] // patch_size)
        self.num_patches = self.Hp * self.Wp
        self.pretrained_path = pretrained_path

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=num_img_channel,
            embed_dim=embed_dim,
        )

        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_frame * self.num_patches, embed_dim))

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
            del updated_vit_dict["pos_embed"]
            self.load_state_dict(updated_vit_dict, strict=False)
        
        self.pos_embed = nn.Parameter(get_3d_sincos_pos_embed(
            embed_dim=self.embed_dim, grid_size=(self.num_frame, self.Hp, self.Wp)).reshape(1, -1, self.embed_dim))

    def forward(self, x):
        """
        x : (B, F, C, H, W)
        B : batch size
        F : number of frames
        C : number of channels
        H : height
        W : width
        """
        B, F, C, H, W = x.shape
        x = x.permute(1, 0, 2, 3, 4).contiguous()
        y = torch.empty(
            (F, B, self.num_patches, self.embed_dim), device=x.device)
        for i in range(F):
            y[i] = self.patch_embed(x[i])
        y = y.permute(1, 0, 2, 3).contiguous().view(B, -1, self.embed_dim)
        y = y + self.pos_embed
        for blk in self.blocks:
            y = blk(y)
        y = self.last_norm(y)
        y = y[:, self.num_patches:self.num_patches*2, :].permute(0, 2, 1).contiguous().view(B, self.embed_dim, self.Hp, self.Wp)
        return y
