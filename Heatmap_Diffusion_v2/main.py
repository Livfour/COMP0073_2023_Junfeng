import torch
from torch.utils.data import DataLoader

from dataloader import dataloader_train, dataloader_val

from model.ContextUnet import ContextUnet
from model.ddpm import DDPM
from train import train
from loss import JointsMSELoss

"""
    This script is used to train a diifusion model for generating reasonable heatmap with image condition.
"""
if __name__ == "__main__":
    model_path = "./checkpoint/heatmap_diffusion_v2.pth"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device.")
    diffusion_model = DDPM(
        eps_model=ContextUnet(
            n_channel=17, n_feat=256, n_cfeat=1000),
        betas=(1e-4, 0.02),
        num_timesteps=1000,
        criterion=JointsMSELoss())

    train(
        dataloader_train=dataloader_train,
        dataloader_val=dataloader_val,
        diffusion_model=diffusion_model,
        n_epoch=100,
        device=device,
        model_path=model_path,
    )
