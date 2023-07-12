import re
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches

keypoints_dict = {
    0: "nose",
    1: "left_eye",
    2: "right_eye",
    3: "left_ear",
    4: "right_ear",
    5: "left_shoulder",
    6: "right_shoulder",
    7: "left_elbow",
    8: "right_elbow",
    9: "left_wrist",
    10: "right_wrist",
    11: "left_hip",
    12: "right_hip",
    13: "left_knee",
    14: "right_knee",
    15: "left_ankle",
    16: "right_ankle"
}

keypoint_pairs = [(0, 1), (0, 2), (0, 5), (0, 6), (1, 3), (2, 4), (5, 6), (5, 7), (5, 11),
                  (6, 8), (6, 12), (7, 9), (8, 10), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split('(\d+)', text)]

def show_image(image, keypoints=None):
    fig, ax = plt.subplots(1)
    fig.set_size_inches(10, 10)
    plt.imshow(image.permute(1, 2, 0))
    if keypoints is not None:
        for keypoint in keypoints:
            x, y, v = keypoint
            if v != 0:
                point = patches.Circle((x, y), radius=2, color='red')
                ax.add_patch(point)
        for keypoint_pair in keypoint_pairs:
            x1, y1, v1 = keypoints[keypoint_pair[0]]
            x2, y2, v2 = keypoints[keypoint_pair[1]]
            if v1 != 0 and v2 != 0:
                ax.plot([x1, x2], [y1, y2], linewidth=2, color='c')
    plt.show()


def heatmaps2keypoints(heatmaps, image_size, visibilities):
    scale = torch.tensor(image_size[1:]) / torch.tensor(heatmaps.shape[1:])
    keypoints = torch.zeros((len(heatmaps), 3))
    for i, (heatmap, visibility) in enumerate(zip(heatmaps, visibilities)):
        if visibility == 0:
            keypoints[i] = torch.tensor([0, 0, 0])
        else:
            flat_index = torch.argmax(heatmap)
            y, x = np.unravel_index(flat_index, heatmap.shape)
            keypoints[i] = torch.tensor(
                [x * scale[1], y * scale[0], visibility])
    return keypoints
