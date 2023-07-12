from tqdm import tqdm
from typing import Optional
import logging

import torch
import torch.nn as nn

logging.basicConfig(filename="./logs/loss.log",level=logging.INFO)

def train(
        dataloader_train,
        dataloader_val,
        diffusion_model: nn.Module,
        n_epoch: int = 100,
        device: str = "cuda",
        model_path: str = "./checkpoint/heatmap_diffusion_v2.pth",
) -> None:
    
    try:
        diffusion_model.load_state_dict(torch.load(model_path)["model_state_dict"])
        trained_epoch = torch.load(model_path)["epoch"]
        lr = 1e-7
        last_loss = torch.load(model_path)["loss"]
        print("Checkpoint loaded.")
    except FileNotFoundError:
        trained_epoch = 0
        lr = 1e-5
        last_loss = None
        print("No checkpoint found. Training from scratch.")

    diffusion_model.to(device)
    diffusion_model.train()
    
    optim = torch.optim.Adam(diffusion_model.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode="min", patience=2)
    lr_scheduler.last_epoch = trained_epoch
    if last_loss is not None:
        lr_scheduler.step(last_loss)

    for i in range(trained_epoch + 1, trained_epoch + 1 + n_epoch):
        print(f"Epoch {i}: lr: {optim.param_groups[0]['lr']}")
        pbar = tqdm(dataloader_train)
        loss_train_ema = 0
        for images, heatmaps, _ in pbar:
            optim.zero_grad()
            images = images.to(device)
            heatmaps = heatmaps.to(device)
            loss = diffusion_model(heatmaps, images)
            loss.backward()
            if loss_train_ema is None:
                loss_train_ema = loss.item()
            else:
                loss_train_ema = loss_train_ema * 0.9 + loss.item() * 0.1
            optim.step()
            pbar.set_description(f"loss: {loss_train_ema:.6f}")

        val_loss = 0
        with torch.no_grad():
            for images, heatmaps, _ in dataloader_val:
                images = images.to(device)
                heatmaps = heatmaps.to(device)
                loss = diffusion_model(heatmaps, images)
                val_loss += loss.item()
        val_loss /= len(dataloader_val)

        # save model per epoch
        torch.save({
            "epoch": i,
            "lr": optim.param_groups[0]["lr"],
            "loss": val_loss,
            "model_state_dict": diffusion_model.state_dict()
        }, model_path)

        # save model per 10 epoch
        if i % 10 == 0:
            torch.save({
                "epoch": i,
                "lr": optim.param_groups[0]["lr"],
                "loss": val_loss,
                "model_state_dict": diffusion_model.state_dict()
            }, f"./checkpoint/heatmap_diffusion_v2_{i}.pth")

        # record loss
        logging.info(f"Epoch {i},lr: {optim.param_groups[0]['lr']}, loss_train: {loss_train_ema:.6f}, loss_val: {val_loss:.6f}")