import torch
from torch.utils.data import DataLoader

from dataloader import dataset_train

from model.unet import NaiveUnet
from model.ddpm import DDPM
from train import train
from loss import JointsMSELoss

"""
    This script is used to train a diifusion model for generating reasonable heatmap without any condition.
"""
if __name__ == "__main__":
    model_path = "./checkpoint/heatmap_diffusion_v1.pth"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataloader_train = DataLoader(dataset_train, batch_size=16, shuffle=True, num_workers=16)
    diffusion_model = DDPM(NaiveUnet(in_channels=17,
                                     out_channels=17,
                                     n_feat=256),
                           betas=(1e-4, 0.02),
                           num_timesteps=1000,
                           criterion=JointsMSELoss())
    train(dataloader_train, diffusion_model, n_epoch=100,
          device=device, model_path=model_path)
