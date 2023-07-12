from tqdm import tqdm
from typing import Optional
import logging

import torch
import torch.nn as nn

logging.basicConfig(filename="./logs/loss.log",level=logging.INFO)
def train(
        dataset_loader,
        diffusion_model: nn.Module,
        n_epoch: int = 10,
        device: str = "cuda",
        model_path: str = "./checkpoint/heatmap_diffusion_v1.pth",
) -> None:
    
    try:
        diffusion_model.load_state_dict(torch.load(model_path)["model_state_dict"])
        trained_epoch = torch.load(model_path)["epoch"]
        print("Checkpoint loaded.")
    except FileNotFoundError:
        trained_epoch = 0
        print("No checkpoint found. Training from scratch.")

    diffusion_model.to(device)
    diffusion_model.train()
    
    optim = torch.optim.Adam(diffusion_model.parameters(), lr=1e-6)

    for i in range(trained_epoch + 1, trained_epoch + 1 + n_epoch):
        print(f"Epoch {i}: ")
        pbar = tqdm(dataset_loader)
        for _, heatmaps, _ in pbar:
            optim.zero_grad()
            heatmaps = heatmaps.to(device)
            loss = diffusion_model(heatmaps)
            loss.backward()
            optim.step()
            pbar.set_description(f"loss: {loss.item():.6f}")

        # save model
        torch.save({
            "epoch": i,
            "model_state_dict": diffusion_model.state_dict()
        }, model_path)

        # record loss
        logging.info(f"Epoch {i} loss: {loss.item():.6f}")