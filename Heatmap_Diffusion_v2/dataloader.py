import os
import json
import numpy as np
import torch
from torchvision import transforms
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
from utilities import natural_keys, heatmaps2keypoints, show_image

root_dir = "/home/junfeng/datasets/COCO2017_v0"
images_dir = os.path.join(root_dir, "images")
annotations_dir = os.path.join(root_dir, "annotations")
train_images_dir = os.path.join(images_dir, "train")
val_images_dir = os.path.join(images_dir, "val")
train_person_keypoints_file = os.path.join(
    annotations_dir, "person_keypoints_train2017.json")
val_person_keypoints_file = os.path.join(
    annotations_dir, "person_keypoints_val2017.json")


class COCOPose(Dataset):
    def __init__(self, root_dir, set) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.sigma = np.array([
            2., 2., 2., 2., 2., 4., 4., 3., 3., 3., 3., 4., 4., 4., 4., 4., 4.
        ], dtype=np.float32)
        self.size = (256, 192)
        self.heatmap_size = (128, 96)
        self.transform = transforms.Compose([
            transforms.Resize(self.size, antialias=True),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
        ])
        match set:
            case "train":
                self.images_dir = train_images_dir
                self.file_list = sorted(os.listdir(
                    self.images_dir), key=natural_keys)
                self.annotations_file = train_person_keypoints_file
                with open(self.annotations_file, 'r') as f:
                    self.annotations = json.load(f)
            case "val":
                self.images_dir = val_images_dir
                self.file_list = sorted(os.listdir(
                    self.images_dir), key=natural_keys)
                self.annotations_file = val_person_keypoints_file
                with open(self.annotations_file, 'r') as f:
                    self.annotations = json.load(f)
            case _:
                raise ValueError("Invalid set name")

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, index: int):
        # read and transform image
        image = read_image(os.path.join(
            self.images_dir, self.file_list[index])) / 255.0
        image_out = self.transform(image)

        # read and transform keypoints
        keypoints = torch.tensor(
            self.annotations[index]['keypoints']).reshape((-1, 3))
        keypoints_for_heatmap = self.center_scale_keypoints(
            keypoints, image.shape[1:])
        # generate target: heatmaps and visibilities
        heatmaps, visibilities = self.generate_target(keypoints_for_heatmap)

        return image_out, heatmaps, visibilities

    def show_sample(self, index: int):
        image, heatmaps, visibilities = self.__getitem__(index)
        keypoints = heatmaps2keypoints(heatmaps, image.shape, visibilities)
        show_image(image, keypoints)

    def center_scale_keypoints(self, keypoints, size):
        h, w = size
        keypoints = keypoints.float()
        keypoints[:, 0] = 2. * keypoints[:, 0] / w - 1.
        keypoints[:, 1] = 2. * keypoints[:, 1] / h - 1.
        return keypoints

    def generate_target(self, keypoints):
        heatmaps = torch.zeros(
            (17, self.heatmap_size[0], self.heatmap_size[1]))
        visibilities = torch.zeros((17, 1))
        for i, keypoint in enumerate(keypoints):
            heatmaps[i], visibilities[i] = self.generate_gaussian(
                keypoint, self.sigma[i])
        return heatmaps, visibilities

    def generate_gaussian(self, keypoint, sigma):
        h, w = self.heatmap_size
        x, y, v = keypoint
        heatmap = torch.zeros(self.heatmap_size)

        # if this keypoint is not labeled, return a heatmap of zeros
        if v < 0.5:
            return heatmap, v

        # Heatmap pixel per output pixel
        mu_x = int(0.5 * (x + 1.) * w)
        mu_y = int(0.5 * (y + 1.) * h)

        tmp_size = sigma * 3

        # Top-left
        x1, y1 = int(mu_x - tmp_size), int(mu_y - tmp_size)

        # Bottom right
        x2, y2 = int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)

        # Check that any part of the gaussian is in-bounds, clip otherwise
        if x1 >= w or y1 >= h or x2 < 0 or y2 < 0:
            return heatmap, 0

        # Generate gaussian
        size = 2 * tmp_size + 1
        tx = np.arange(0, size, 1, np.float32)
        ty = tx[:, np.newaxis]
        x0 = y0 = size // 2

        # The gaussian is not normalized, we want the center value to equal 1
        g = torch.tensor(
            np.exp(- ((tx - x0) ** 2 + (ty - y0) ** 2) / (2 * sigma ** 2)))

        # Determine the bounds of the source gaussian
        g_x_min, g_x_max = max(0, -x1), min(x2, w) - x1
        g_y_min, g_y_max = max(0, -y1), min(y2, h) - y1

        # Image range
        img_x_min, img_x_max = max(0, x1), min(x2, w)
        img_y_min, img_y_max = max(0, y1), min(y2, h)

        heatmap[img_y_min:img_y_max,
                img_x_min:img_x_max] = g[g_y_min:g_y_max, g_x_min:g_x_max]

        return heatmap, v


dataset_train = COCOPose(root_dir=root_dir, set="train")
dataset_val = COCOPose(root_dir=root_dir, set="val")

dataloader_train = DataLoader(dataset_train, batch_size=16, shuffle=True)
dataloader_val = DataLoader(dataset_val, batch_size=16, shuffle=True)
