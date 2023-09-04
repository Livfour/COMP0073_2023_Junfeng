import re
from typing import List, Tuple
from tqdm import tqdm
import numpy as np
import json
import os
import sys
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import posetrack21.api as api
from mmpose.codecs import UDPHeatmap


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._original_stdout


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

skeletons = [(0, 1), (0, 2), (0, 5), (0, 6), (1, 3), (2, 4), (5, 6), (5, 7), (5, 11),
             (6, 8), (6, 12), (7, 9), (8, 10), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split('(\d+)', text)]


def show_image(image,
               keypoints=None,
               heatmaps=None,
               showlabel=False,
               keypoints_labels=[
                   "nose",
                   "left_eye",
                   "right_eye",
                   "left_ear",
                   "right_ear",
                   "left_shoulder",
                   "right_shoulder",
                   "left_elbow",
                   "right_elbow",
                   "left_wrist",
                   "right_wrist",
                   "left_hip",
                   "right_hip",
                   "left_knee",
                   "right_knee",
                   "left_ankle",
                   "right_ankle"
               ],
               skeletons=[
                   (0, 1),
                   (0, 2),
                   (0, 5),
                   (0, 6),
                   (1, 3),
                   (2, 4),
                   (5, 6),
                   (5, 7),
                   (5, 11),
                   (6, 8),
                   (6, 12),
                   (7, 9),
                   (8, 10),
                   (11, 12),
                   (11, 13),
                   (13, 15),
                   (12, 14),
                   (14, 16)]
               ):
    fig, ax = plt.subplots(1)
    fig.set_size_inches(8, 8)
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).numpy()
    plt.imshow(image)
    H, W, C = image.shape
    if keypoints is not None:
        for keypoint in keypoints:
            x, y, v = keypoint
            if x != 0 and y != 0:
                point = patches.Circle((x, y), radius=2, color='red')
                if showlabel:
                    ax.text(x, y, keypoints_labels[keypoints.index(keypoint)])
                ax.add_patch(point)
        for skeleton in skeletons:
            x1, y1, v1 = keypoints[skeleton[0]]
            x2, y2, v2 = keypoints[skeleton[1]]
            if (x1 != 0 and y1 != 0) and (x2 != 0 and y2 != 0):
                ax.plot([x1, x2], [y1, y2], linewidth=2, color='c')
    if heatmaps is not None:
        keypoints = heatmaps_to_keypoints(heatmaps, (H, W))
        for keypoint in keypoints:
            x, y, v = keypoint
            if v != 0:
                point = patches.Circle((x, y), radius=2, color='red')
                ax.add_patch(point)
        for skeleton in skeletons:
            x1, y1, v1 = keypoints[skeleton[0]]
            x2, y2, v2 = keypoints[skeleton[1]]
            if v1 != 0 and v2 != 0:
                ax.plot([x1, x2], [y1, y2], linewidth=2, color='c')
    plt.show()


def show_image2(image,
                keypoints=None,
                keypoints_visibilities=None,
                heatmaps=None,
                bbox=None,
                showlabel=False,
                keypoints_labels=[
                    "nose",
                    "left_eye",
                    "right_eye",
                    "left_ear",
                    "right_ear",
                    "left_shoulder",
                    "right_shoulder",
                    "left_elbow",
                    "right_elbow",
                    "left_wrist",
                    "right_wrist",
                    "left_hip",
                    "right_hip",
                    "left_knee",
                    "right_knee",
                    "left_ankle",
                    "right_ankle"
                ],
                skeletons=[
                    (0, 1),
                    (0, 2),
                    (0, 5),
                    (0, 6),
                    (1, 3),
                    (2, 4),
                    (5, 6),
                    (5, 7),
                    (5, 11),
                    (6, 8),
                    (6, 12),
                    (7, 9),
                    (8, 10),
                    (11, 12),
                    (11, 13),
                    (13, 15),
                    (12, 14),
                    (14, 16)]
                ):
    fig, ax = plt.subplots(1)
    fig.set_size_inches(8, 8)
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).numpy()
    plt.imshow(image)
    H, W, C = image.shape
    for i, v in enumerate(keypoints_visibilities):
        if v:
            x, y = keypoints[i]
            point = patches.Circle((x, y), radius=2, color='red')
            ax.add_patch(point)
            if showlabel:
                ax.text(x, y, keypoints_labels[i])
    for skeleton in skeletons:
        if keypoints_visibilities[skeleton[0]] and keypoints_visibilities[skeleton[1]]:
            x1, y1 = keypoints[skeleton[0]]
            x2, y2 = keypoints[skeleton[1]]
            ax.plot([x1, x2], [y1, y2], linewidth=2, color='c')
    if bbox is not None:
        x, y, w, h = bbox
        rect = patches.Rectangle((x, y), w, h,
                                 linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    plt.show()


def show_heatmaps(heatmaps):
    if not isinstance(heatmaps, torch.Tensor):
        heatmaps = torch.tensor(heatmaps)
    fig, axes = plt.subplots(1, 18, figsize=(20, 2))
    heatmaps = torch.cat(
        [heatmaps, heatmaps.sum(dim=0, keepdim=True)], dim=0)
    for i, (heatmap, ax) in enumerate(zip(heatmaps, axes)):
        ax.imshow(heatmap.squeeze(), cmap="hot")
        ax.axis("off")
    plt.show()


def get_scores(heatmaps):
    scores = []
    for heatmap in heatmaps:
        if torch.max(heatmap) > 0.2:
            scores.append(torch.max(heatmap))
        else:
            scores.append(torch.tensor(0, dtype=torch.float32))
    if len(scores) == 0:
        return torch.tensor(0, dtype=torch.float32)
    return torch.mean(torch.tensor(scores))


def heatmaps_to_keypoints(heatmaps, image_size, threshold=0.2):
    if isinstance(heatmaps, torch.Tensor):
        heatmaps = heatmaps.numpy()
    visibilities = np.zeros(len(heatmaps))
    for i, heatmap in enumerate(heatmaps):
        if np.max(heatmap) > threshold:
            visibilities[i] = 1
    scale = np.array(image_size) / np.array(heatmaps.shape[1:])
    keypoints = np.zeros((len(heatmaps), 3))
    for i, (heatmap, visibility) in enumerate(zip(heatmaps, visibilities)):
        if visibility == 0:
            keypoints[i] = np.array([0, 0, 0])
        else:
            flat_index = np.argmax(heatmap)
            y, x = np.unravel_index(flat_index, heatmap.shape)
            keypoints[i] = np.array(
                [x * scale[1], y * scale[0], visibility])
    return keypoints


def get_1d_sincos_pos_embed(embed_dim, pos_len):
    """
    embed_dim: output dimension for each position
    pos_len: length of positions to be encoded
    out: (pos_len, embed_dim)
    """
    position = torch.arange(0, pos_len).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, embed_dim, 2).float()
                         * -(np.log(10000.0) / embed_dim))

    pe = torch.zeros(pos_len, embed_dim)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    return pe


def get_2d_sincos_pos_embed(embed_dim, grid_size):
    """
    Generate 2D positional encoding.
    """
    h, w = grid_size
    pe_row = get_1d_sincos_pos_embed(embed_dim=embed_dim // 2, pos_len=h)
    pe_col = get_1d_sincos_pos_embed(embed_dim=embed_dim // 2, pos_len=w)

    # Stack and expand to get full 2D positional encoding
    pe = torch.cat((pe_row.unsqueeze(1).expand(-1, w, -1),
                    pe_col.unsqueeze(0).expand(h, -1, -1)), dim=2)

    return pe


def get_3d_sincos_pos_embed(embed_dim, grid_size):
    """
    Generate 3D positional encoding.
    """
    d, h, w = grid_size
    pe_depth = get_1d_sincos_pos_embed(embed_dim=embed_dim // 3, pos_len=d)
    pe_row = get_1d_sincos_pos_embed(embed_dim=embed_dim // 3, pos_len=h)
    pe_col = get_1d_sincos_pos_embed(embed_dim=embed_dim // 3, pos_len=w)

    pe_depth = pe_depth.unsqueeze(1).unsqueeze(2).expand(-1, h, w, -1)
    pe_row = pe_row.unsqueeze(0).unsqueeze(2).expand(d, -1, w, -1)
    pe_col = pe_col.unsqueeze(0).unsqueeze(1).expand(d, h, -1, -1)

    pe = torch.cat((pe_depth, pe_row, pe_col), dim=3)

    return pe


def eval(dataset, model, target_dt_path="/home/junfeng/Documents/WorkSpace/SummerProject/Code_New/Basic_Encoder&Decoder/results/val", loss_fn=None, threshold=0.2):
    device = model.device
    loss_eval = np.empty(0)
    pbar = tqdm(range(len(dataset)))
    with torch.no_grad():
        for index in pbar:
            pbar.set_description("Evaluating: ")
            anns_all = dataset.get_anns_all(index)
            target_dt_file_path = os.path.join(
                target_dt_path, dataset.ann_file_list[index])
            anns_all["annotations"] = []
            for i, (_, video_transformed, bbox, _, heatmaps, image_id, track_id) in enumerate(dataset.get_for_eval(index)):
                x, y, w, h = bbox
                with torch.no_grad():
                    image = video_transformed[1].unsqueeze(0)
                    image = image.to(device)
                    heatmaps = heatmaps.unsqueeze(0).to(device)
                    pred_heatmaps = model(image)
                    if loss_fn is not None:
                        loss_eval = np.append(
                            loss_eval, loss_fn(pred_heatmaps, heatmaps).item())
                    pred_heatmaps = pred_heatmaps.squeeze().cpu()
                    pred_keypoints = heatmaps_to_keypoints(
                        pred_heatmaps, (h, w), threshold=threshold)
                    for j in range(len(pred_keypoints)):
                        if pred_keypoints[j][0] != 0 and pred_keypoints[j][1] != 0:
                            pred_keypoints[j][0] += x
                            pred_keypoints[j][1] += y
                    pred_keypoints = pred_keypoints.flatten().tolist()
                    anns_all["annotations"].append({
                        "image_id": image_id,
                        "category_id": 1,
                        "keypoints": pred_keypoints,
                        "track_id": track_id
                    })
            json.dump(anns_all, open(target_dt_file_path, "w"))
    print(np.mean(loss_eval))
    with HiddenPrints():
        evaluator = api.get_api(trackers_folder=target_dt_path,
                                gt_folder=dataset.anns_dir,
                                eval_type='pose_estim',
                                num_parallel_cores=8,
                                use_parallel=True)
        results = evaluator.eval()
    if loss_fn is not None:
        return results, np.mean(loss_eval)
    else:
        return results


def keypoints_to_mask(keypoints):
    """
    only labeled visible keypoints are set to True
    """
    mask = np.zeros((1, 17)).astype(bool)
    for i in range(17):
        if keypoints[i, 2] > 0.5:
            mask[0, i] = True
    return mask


def keypoints_to_label_flag(keypoints):
    """
    labeled keypoints are set to True
    """
    is_label = np.zeros((1, 17)).astype(bool)
    for i in range(17):
        if keypoints[i, 0] != 0 or keypoints[i, 1] != 0:
            is_label[0, i] = True
    return is_label


def keypoints_to_visible(keypoints):
    """
    labeled keypoints are set to 1.
    """
    visible = np.zeros((17,))
    for i in range(17):
        if keypoints[i, 0] != 0 or keypoints[i, 1] != 0:
            visible[i] = 1
    return visible


def get_max_preds(batch_heatmaps):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    '''
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals


def calc_dists(preds, target, normalize):
    preds = preds.astype(np.float32)
    target = target.astype(np.float32)
    dists = np.zeros((preds.shape[1], preds.shape[0]))
    for n in range(preds.shape[0]):  # batch num
        for c in range(preds.shape[1]):  # keypoint type
            if target[n, c, 0] > 1 and target[n, c, 1] > 1:
                normed_preds = preds[n, c, :] / normalize[n]
                normed_targets = target[n, c, :] / normalize[n]
                dists[c, n] = np.linalg.norm(
                    normed_preds - normed_targets)  # Euclidean distance
            else:
                dists[c, n] = -1
    return dists


def dist_acc(dists, thr=0.5, percentage=True):
    ''' Return percentage below threshold while ignoring values with a -1 '''
    dist_cal = np.not_equal(dists, -1)
    num_dist_cal = dist_cal.sum()
    if num_dist_cal > 0:
        less_thr_count = np.less(dists[dist_cal], thr).sum() * 1.0
        if percentage:
            return less_thr_count / num_dist_cal
        else:
            # less_thr_count = match  / num_dist_cal （val）
            return less_thr_count, num_dist_cal
    else:
        if percentage:
            return -1
        else:
            return -1, -1


def accuracy(pred_heatmaps, target_keypoints, hm_type='udp_heatmaps', thr=0.5, image_size=(256, 192)):
    '''
    Calculate accuracy according to PCK (),
    but uses ground truth heatmap rather than x,y locations
    First value to be returned is average accuracy across 'idxs',
    followed by individual accuracies
    '''
    idx = list(range(17))
    image_height, image_width = image_size
    heatmaps_height = 64
    heatmaps_width = 48
    norm = 1.0
    if hm_type == 'gaussian':
        pred, _ = get_max_preds(pred_heatmaps)
        target = target_keypoints
    elif hm_type == 'udp_heatmaps':
        decoder = UDPHeatmap(input_size=image_size, heatmap_size=(
            heatmaps_width, heatmaps_height))
        pred = decoder.decode(pred_heatmaps)[0]
        pred[np.isnan(pred)] = .0
        target = target_keypoints[:, :2].reshape(-1, 17, 2)
    print(pred)
    print(target)
    norm = np.ones((pred.shape[0], 2)) * np.array([heatmaps_height, heatmaps_width]) / 10
    # use a fixed length as a measure rather than the length of body parts
    dists = calc_dists(pred, target, norm)

    acc = np.zeros((len(idx) + 1))
    avg_acc = 0
    cnt = 0

    for i in range(len(idx)):
        acc[i + 1] = dist_acc(dists[idx[i]], thr)
        if acc[i + 1] >= 0:
            avg_acc = avg_acc + acc[i + 1]
            cnt += 1

    avg_acc = avg_acc / cnt if cnt != 0 else 0
    if cnt != 0:
        acc[0] = avg_acc

    return acc, avg_acc, cnt
