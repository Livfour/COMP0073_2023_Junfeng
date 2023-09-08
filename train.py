import os
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

from datasets.posetrack21 import PoseTrack21
from models.PoseEstimate import PoseEstimate
from models.backbone import ResNet50
from models.backbone.vit import ViTEncoder
from models.backbone.video_vit import VideoViTEncoder
from models.backbone.vit_with_fusion import FusionVit
from models.head.LMSFI import LMSFI
from models.head import DeconvHead, MaskHead, ResMaskHead, SimpleHead
from models.head.ResMaskHeadv2 import ResMaskHead2
from models.head.MaskHead2 import MaskHead2
from loss import JointsMSELoss
from utilities.utilities import keypoints_to_mask
from mmpose.evaluation import pose_pck_accuracy


def warmup_linear(x):
    warmup_iters = 500
    warmup_ratio = 0.001
    if x < warmup_iters:
        return warmup_ratio + (1-warmup_ratio) * x / warmup_iters
    return 1


def save_model(save_path, model, optim, lr_scheduler, epoch, loss_train, loss_val, AP, best_AP):
    torch.save({
        "model_state_dict": model.state_dict(),
        "optim": optim.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "epoch": epoch,
        "loss_train": loss_train,
        "loss_val": loss_val,
        "AP": AP,
        "best_AP": best_AP,
    }, save_path)


def train(
        dataloader_train,
        dataset_val,
        model,
        model_path,
        best_model_path,
        total_epochs,
        device,
        optim,
        criterion,
):
    lr_scheduler = LambdaLR(optimizer=optim, lr_lambda=warmup_linear)
    if not os.path.exists(model_path):
        print("No checkpoint found. Load pretrained model.")
        trained_epoch = 0
        last_loss = None
        best_AP = 0
    else:
        model_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(model_dict["model_state_dict"])
        optim.load_state_dict(model_dict["optim"])
        lr_scheduler.load_state_dict(model_dict["lr_scheduler"])
        trained_epoch = model_dict["epoch"]
        last_loss = model_dict["loss_train"]
        best_AP = model_dict["best_AP"]

    assert trained_epoch < total_epochs, "Already trained!"

    # init
    model.to(device)
    iter = len(dataloader_train)

    # train
    for epoch in range(trained_epoch + 1, total_epochs + 1):
        # train on train set
        model.train()
        print(f"Epoch: {epoch}")
        pbar = tqdm(dataloader_train)
        loss_train_ema = last_loss
        for i, (videos, heatmaps) in enumerate(pbar):
            optim.zero_grad()
            videos = videos.to(device)
            heatmaps = heatmaps.to(device)
            pred = model(videos, heatmaps)
            loss = criterion(pred, heatmaps)
            if loss_train_ema is None:
                loss_train_ema = loss.item()
            else:
                loss_train_ema = loss_train_ema * 0.9 + loss.item() * 0.1
            loss.backward()
            optim.step()
            lr_scheduler.step()
            pbar.set_description(
                f"lr:{lr_scheduler.get_last_lr()}, loss: {loss_train_ema:.6f}")
            step = (epoch-1) * iter + i
            writer.add_scalars("loss by steps", {
                               "train": loss_train_ema}, step)

        last_loss = loss_train_ema

        writer.add_scalars("loss by epochs", {
            "train": loss_train_ema,
        }, epoch)

        # eval
        if epoch % 20 != 0:
            continue
        model.eval()
        loss_val = 0
        acc = 0
        count = 0
        max_count = 250
        pbar = tqdm(range(max_count))
        for i in pbar:
            i = np.random.randint(0, len(dataset_val))
            _, video_transformed, keypoints, _, heatmaps = dataset_val[i]
            mask = keypoints_to_mask(keypoints)
            video_transformed = video_transformed.to(device).unsqueeze(0)
            with torch.no_grad():
                pred_heatmaps = model.predict(video_transformed)
                heatmaps = heatmaps.to(device).unsqueeze(0)
                loss_val += criterion(pred_heatmaps, heatmaps).item()
                pred_heatmaps = pred_heatmaps.cpu().numpy()
                heatmaps = heatmaps.cpu().numpy()
            _, avg_acc, _ = pose_pck_accuracy(
                pred_heatmaps, heatmaps, mask, thr=0.05)
            acc += avg_acc
            count += 1

        AP = acc / count
        if AP > best_AP:
            best_AP = AP
            save_model(best_model_path, model, optim, lr_scheduler, epoch,
                       last_loss, loss_val, AP, best_AP)

        save_model(model_path, model, optim, lr_scheduler, epoch,
                   last_loss, loss_val, AP, best_AP)

        average_loss_val = loss_val / count
        writer.add_scalars("AP by epochs", {
            "val": AP,
        }, epoch)
        writer.add_scalars("loss by steps", {
            "val": average_loss_val,
        }, step)
        writer.add_scalars("loss by epochs", {
            "val": average_loss_val,
        }, epoch)


if __name__ == "__main__":
    writer = SummaryWriter("./runs/Fusion_ResMask_02")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = torch.cuda.get_device_name(0)
    model_path = "./checkpoints/Fusion_ResMask_02.pth"
    best_model_path = "./checkpoints/Fusion_ResMask_02_Best.pth"
    pretrained_path = "./checkpoints/vitpose_base_coco_aic_mpii.pth"
    dataset_root_dir = "/home/junfeng/datasets/PoseTrack21"

    dataset_train = PoseTrack21(
        root_dir=dataset_root_dir,
        set="train",
    )

    dataset_val = PoseTrack21(
        root_dir=dataset_root_dir,
        set="test",
    )

    dataloader_train = DataLoader(
        dataset_train,
        batch_size=64,
        shuffle=True,
        drop_last=True,
        num_workers=16,
        collate_fn=dataset_train.collate_fn,
    )

    encoder = FusionVit(pretrained_path=pretrained_path)
    # encoder = ResNet50()

    neck = nn.Identity()
    # neck = nn.Conv2d(1024, 768, 1)

    # head = MaskHead(pretrained_path=pretrained_path)
    head = ResMaskHead(pretrained_path=pretrained_path)
    # head = DeconvHead(pretrained_path=pretrained_path)
    # head = MaskHead2(pretrained_path=pretrained_path)
    # head = DeconvHead(pretrained_path=pretrained_path)

    model = PoseEstimate(encoder=encoder, neck=neck, head=head)
    model.to(device)

    total_epochs = 1500
    lr = 5e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = JointsMSELoss(use_target_weight=True, device=device)
    print(f"Start training on {device_name}")

    train(
        dataloader_train=dataloader_train,
        dataset_val=dataset_val,
        model=model,
        model_path=model_path,
        best_model_path=best_model_path,
        total_epochs=total_epochs,
        device=device,
        optim=optimizer,
        criterion=criterion,
    )
