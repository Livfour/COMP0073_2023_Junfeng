import os
import sys
import io
import json
import numpy as np
import torch
import cv2
from pycocotools.coco import COCO
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from mmpose.codecs import UDPHeatmap
from utilities.utilities import keypoints_to_label_flag


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._original_stdout


dataset_info = dict(
    dataset_name='posetrack21',
    paper_info=dict(
        author='Andreas Doering and Di Chen and Shanshan Zhang and Bernt Schiele and Juergen Gall',
        title='PoseTrack21: A Dataset for Person Search, Multi-Object Tracking and Multi-Person Pose Tracking',
        booktitle='CVPR',
        year='2022',
        githubpage='https://github.com/anDoer/PoseTrack21',
    ),
    keypoint_info={
        0:
        dict(name='nose', id=0, color=[51, 153, 255], type='upper', swap=''),
        1:
        dict(
            name='head_bottom',
            id=1,
            color=[51, 153, 255],
            type='upper',
            swap=''),
        2:
        dict(
            name='head_top', id=2, color=[51, 153, 255], type='upper',
            swap=''),
        3:
        dict(
            name='left_ear',
            id=3,
            color=[51, 153, 255],
            type='upper',
            swap='right_ear'),
        4:
        dict(
            name='right_ear',
            id=4,
            color=[51, 153, 255],
            type='upper',
            swap='left_ear'),
        5:
        dict(
            name='left_shoulder',
            id=5,
            color=[0, 255, 0],
            type='upper',
            swap='right_shoulder'),
        6:
        dict(
            name='right_shoulder',
            id=6,
            color=[255, 128, 0],
            type='upper',
            swap='left_shoulder'),
        7:
        dict(
            name='left_elbow',
            id=7,
            color=[0, 255, 0],
            type='upper',
            swap='right_elbow'),
        8:
        dict(
            name='right_elbow',
            id=8,
            color=[255, 128, 0],
            type='upper',
            swap='left_elbow'),
        9:
        dict(
            name='left_wrist',
            id=9,
            color=[0, 255, 0],
            type='upper',
            swap='right_wrist'),
        10:
        dict(
            name='right_wrist',
            id=10,
            color=[255, 128, 0],
            type='upper',
            swap='left_wrist'),
        11:
        dict(
            name='left_hip',
            id=11,
            color=[0, 255, 0],
            type='lower',
            swap='right_hip'),
        12:
        dict(
            name='right_hip',
            id=12,
            color=[255, 128, 0],
            type='lower',
            swap='left_hip'),
        13:
        dict(
            name='left_knee',
            id=13,
            color=[0, 255, 0],
            type='lower',
            swap='right_knee'),
        14:
        dict(
            name='right_knee',
            id=14,
            color=[255, 128, 0],
            type='lower',
            swap='left_knee'),
        15:
        dict(
            name='left_ankle',
            id=15,
            color=[0, 255, 0],
            type='lower',
            swap='right_ankle'),
        16:
        dict(
            name='right_ankle',
            id=16,
            color=[255, 128, 0],
            type='lower',
            swap='left_ankle')
    },
    skeleton_info={
        0:
        dict(link=('left_ankle', 'left_knee'), id=0, color=[0, 255, 0]),
        1:
        dict(link=('left_knee', 'left_hip'), id=1, color=[0, 255, 0]),
        2:
        dict(link=('right_ankle', 'right_knee'), id=2, color=[255, 128, 0]),
        3:
        dict(link=('right_knee', 'right_hip'), id=3, color=[255, 128, 0]),
        4:
        dict(link=('left_hip', 'right_hip'), id=4, color=[51, 153, 255]),
        5:
        dict(link=('left_shoulder', 'left_hip'), id=5, color=[51, 153, 255]),
        6:
        dict(link=('right_shoulder', 'right_hip'), id=6, color=[51, 153, 255]),
        7:
        dict(
            link=('left_shoulder', 'right_shoulder'),
            id=7,
            color=[51, 153, 255]),
        8:
        dict(link=('left_shoulder', 'left_elbow'), id=8, color=[0, 255, 0]),
        9:
        dict(
            link=('right_shoulder', 'right_elbow'), id=9, color=[255, 128, 0]),
        10:
        dict(link=('left_elbow', 'left_wrist'), id=10, color=[0, 255, 0]),
        11:
        dict(link=('right_elbow', 'right_wrist'), id=11, color=[255, 128, 0]),
        12:
        dict(link=('nose', 'head_bottom'), id=12, color=[51, 153, 255]),
        13:
        dict(link=('nose', 'head_top'), id=13, color=[51, 153, 255]),
        14:
        dict(
            link=('head_bottom', 'left_shoulder'), id=14, color=[51, 153,
                                                                 255]),
        15:
        dict(
            link=('head_bottom', 'right_shoulder'),
            id=15,
            color=[51, 153, 255])
    },
    joint_weights=[
        1., 1., 1., 1., 1., 1., 1., 1.2, 1.2, 1.5, 1.5, 1., 1., 1.2, 1.2, 1.5,
        1.5
    ],
    sigmas=[
        0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072, 0.062,
        0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089
    ])


class PoseTrack21(Dataset):
    def __init__(self, root_dir, set):
        super().__init__()
        self.root_dir = root_dir
        self.set = set
        self.sigma = np.array([
            2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.
        ], dtype=np.float32)
        self.img_size = (256, 192)
        self.img_H, self.img_W = self.img_size
        self.heatmap_size = (64, 48)
        self.heatmap_H, self.heatmap_W = self.heatmap_size
        self.sigmas = np.ones(17, dtype=np.float32) * 2.0
        self.heatmap_H, self.heatmap_W = self.heatmap_size
        self.udp_heatmap_generator = UDPHeatmap(input_size=(
            self.img_W, self.img_H), heatmap_size=(self.heatmap_W, self.heatmap_H))
        self.keypoints_labels = [
            "nose",
            "head_bottom",
            "head_top",
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
            "right_ankle",
        ]
        match set:
            case "train":
                self.anns_dir = os.path.join(
                    self.root_dir, "posetrack_data", "train")
                self.transform = A.Compose([
                    A.Affine(
                        scale=(0.8, 1.2),
                        rotate=(-40, 40),
                        p=0.5,
                    ),
                    A.Resize(
                        height=self.img_H,
                        width=self.img_W,
                        interpolation=cv2.INTER_CUBIC,
                    ),
                    A.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                    ToTensorV2(),
                ],
                    keypoint_params=A.KeypointParams(
                    format="xy",
                    label_fields=['class_labels'],
                ),
                    additional_targets={
                        "prev_frame": "image", "next_frame": "image"},
                )
            case "val":
                self.anns_dir = os.path.join(
                    self.root_dir, "posetrack_data", "val")
                self.transform = A.Compose([
                    A.Resize(
                        height=self.img_H,
                        width=self.img_W,
                        interpolation=cv2.INTER_CUBIC,
                    ),
                    A.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                    ToTensorV2(),
                ],
                    keypoint_params=A.KeypointParams(
                    format="xy",
                    label_fields=['class_labels'],
                ),
                    additional_targets={
                        "prev_frame": "image", "next_frame": "image"},
                )
            case "test":
                self.anns_dir = os.path.join(
                    self.root_dir, "posetrack_data", "test")
                self.transform = A.Compose([
                    A.Resize(
                        height=self.img_H,
                        width=self.img_W,
                        interpolation=cv2.INTER_CUBIC,
                    ),
                    A.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                    ToTensorV2(),
                ],
                    keypoint_params=A.KeypointParams(
                    format="xy",
                    label_fields=['class_labels'],
                ),
                    additional_targets={
                        "prev_frame": "image", "next_frame": "image"},
                )
            case "train_val":
                self.anns_dir = os.path.join(
                    self.root_dir, "posetrack_data", "train_val")
                self.transform = A.Compose([
                    A.Resize(
                        height=self.img_H,
                        width=self.img_W,
                        interpolation=cv2.INTER_CUBIC,
                    ),
                    A.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                    ToTensorV2(),
                ],
                    keypoint_params=A.KeypointParams(
                    format="xy",
                    label_fields=['class_labels'],
                ),
                    additional_targets={
                        "prev_frame": "image", "next_frame": "image"},
                )
            case _:
                raise ValueError("Invalid set name")
        self.ann_file_list = sorted(os.listdir(self.anns_dir))

    def __len__(self):
        return len(self.ann_file_list)

    def get_anns_all(self, index: int):
        ann_file = os.path.join(self.anns_dir, self.ann_file_list[index])
        anns = json.load(open(ann_file, "r"))
        return anns

    def get_for_eval(self, index: int):
        ann_file = os.path.join(self.anns_dir, self.ann_file_list[index])
        with HiddenPrints():
            coco_anns = COCO(os.path.join(self.anns_dir, ann_file))
        image_ids = coco_anns.getImgIds()
        images_infos = coco_anns.loadImgs(image_ids)
        labled_image_ids = []
        for image_info in images_infos:
            if image_info['is_labeled']:
                labled_image_ids.append(int(image_info['id']))
        for i, image_id in enumerate(labled_image_ids):
            if i == 0:
                prev_frame_id = image_id
                key_frame_id = image_id
                next_frame_id = labled_image_ids[i+1]
            elif i == len(labled_image_ids) - 1:
                prev_frame_id = labled_image_ids[i-1]
                key_frame_id = image_id
                next_frame_id = image_id
            else:
                prev_frame_id = labled_image_ids[i-1]
                key_frame_id = image_id
                next_frame_id = labled_image_ids[i+1]
            prev_frame_anns = coco_anns.loadAnns(
                coco_anns.getAnnIds(imgIds=prev_frame_id))
            key_frame_anns = coco_anns.loadAnns(
                coco_anns.getAnnIds(imgIds=key_frame_id))
            next_frame_anns = coco_anns.loadAnns(
                coco_anns.getAnnIds(imgIds=next_frame_id))
            for key_frame_ann in key_frame_anns:
                bbox = key_frame_ann['bbox']
                keypoints = key_frame_ann['keypoints']
                track_id = key_frame_ann["track_id"]
                prev_frame_ann = self._search_track_id(
                    prev_frame_anns, track_id)
                next_frame_ann = self._search_track_id(
                    next_frame_anns, track_id)
                if prev_frame_ann is None:
                    prev_frame_ann = key_frame_ann
                if next_frame_ann is None:
                    next_frame_ann = key_frame_ann

                prev_frame_path = os.path.join(
                    self.root_dir, coco_anns.loadImgs(prev_frame_id)[0]['file_name'])
                key_frame_path = os.path.join(
                    self.root_dir, coco_anns.loadImgs(key_frame_id)[0]['file_name'])
                next_frame_path = os.path.join(
                    self.root_dir, coco_anns.loadImgs(next_frame_id)[0]['file_name'])

                prev_frame = cv2.cvtColor(cv2.imread(
                    prev_frame_path), cv2.COLOR_BGR2RGB)
                key_frame = cv2.cvtColor(cv2.imread(
                    key_frame_path), cv2.COLOR_BGR2RGB)
                next_frame = cv2.cvtColor(cv2.imread(
                    next_frame_path), cv2.COLOR_BGR2RGB)

                video, video_transformed, expanded_bbox, keypoints, keypoints_transformed, heatmaps = self._prepare_targets(
                    key_frame=key_frame,
                    prev_frame=prev_frame,
                    next_frame=next_frame,
                    bbox=bbox,
                    keypoints=keypoints,
                )

                yield video, video_transformed, torch.tensor(expanded_bbox), keypoints, keypoints_transformed, heatmaps, image_id, track_id

    def get_for_train(self, size: int):
        count = 0
        video_transformed_list = []
        heatmaps_list = []
        for _, video_transformed, _, _, _, heatmaps, _, _ in self.get_for_eval():
            video_transformed_list.append(video_transformed)
            heatmaps_list.append(heatmaps)
            count += 1
            if count == size:
                yield torch.stack(video_transformed_list), torch.stack(heatmaps_list)
                count = 0

    def __getitem__(self, index: int):
        ann_file = os.path.join(self.anns_dir, self.ann_file_list[index])
        with HiddenPrints():
            coco_anns = COCO(os.path.join(self.anns_dir, ann_file))
        image_ids = coco_anns.getImgIds()
        images_infos = coco_anns.loadImgs(image_ids)
        labled_image_ids = []
        for image_info in images_infos:
            if image_info['is_labeled']:
                labled_image_ids.append(int(image_info['id']))

        index = np.random.choice(len(labled_image_ids))
        image_id = labled_image_ids[index]
        if index == 0:
            prev_frame_id = image_id
            key_frame_id = image_id
            next_frame_id = labled_image_ids[index+1]
        elif index == len(labled_image_ids) - 1:
            prev_frame_id = labled_image_ids[index-1]
            key_frame_id = image_id
            next_frame_id = image_id
        else:
            prev_frame_id = labled_image_ids[index-1]
            key_frame_id = image_id
            next_frame_id = labled_image_ids[index+1]
        prev_frame_anns = coco_anns.loadAnns(
            coco_anns.getAnnIds(imgIds=prev_frame_id))
        key_frame_anns = coco_anns.loadAnns(
            coco_anns.getAnnIds(imgIds=key_frame_id))
        next_frame_anns = coco_anns.loadAnns(
            coco_anns.getAnnIds(imgIds=next_frame_id))
        key_frame_ann = np.random.choice(key_frame_anns)

        bbox = key_frame_ann['bbox']
        keypoints = key_frame_ann['keypoints']
        track_id = key_frame_ann["track_id"]
        prev_frame_ann = self._search_track_id(
            prev_frame_anns, track_id)
        next_frame_ann = self._search_track_id(
            next_frame_anns, track_id)
        if prev_frame_ann is None:
            prev_frame_ann = key_frame_ann
        if next_frame_ann is None:
            next_frame_ann = key_frame_ann

        prev_frame_path = os.path.join(
            self.root_dir, coco_anns.loadImgs(prev_frame_id)[0]['file_name'])
        key_frame_path = os.path.join(
            self.root_dir, coco_anns.loadImgs(key_frame_id)[0]['file_name'])
        next_frame_path = os.path.join(
            self.root_dir, coco_anns.loadImgs(next_frame_id)[0]['file_name'])

        prev_frame = cv2.cvtColor(cv2.imread(
            prev_frame_path), cv2.COLOR_BGR2RGB)
        key_frame = cv2.cvtColor(cv2.imread(
            key_frame_path), cv2.COLOR_BGR2RGB)
        next_frame = cv2.cvtColor(cv2.imread(
            next_frame_path), cv2.COLOR_BGR2RGB)

        video, video_transformed, _, keypoints, keypoints_transformed, heatmaps = self._prepare_targets(
            key_frame=key_frame,
            prev_frame=prev_frame,
            next_frame=next_frame,
            bbox=bbox,
            keypoints=keypoints,
        )

        return video, video_transformed, keypoints, keypoints_transformed, heatmaps
    
    def get_multi_person(self, index: int):
        ann_file = os.path.join(self.anns_dir, self.ann_file_list[index])
        with HiddenPrints():
            coco_anns = COCO(os.path.join(self.anns_dir, ann_file))
        image_ids = coco_anns.getImgIds()
        images_infos = coco_anns.loadImgs(image_ids)
        labled_image_ids = []
        for image_info in images_infos:
            if image_info['is_labeled']:
                labled_image_ids.append(int(image_info['id']))

        index = np.random.choice(len(labled_image_ids))
        image_id = labled_image_ids[index]
        if index == 0:
            prev_frame_id = image_id
            key_frame_id = image_id
            next_frame_id = labled_image_ids[index+1]
        elif index == len(labled_image_ids) - 1:
            prev_frame_id = labled_image_ids[index-1]
            key_frame_id = image_id
            next_frame_id = image_id
        else:
            prev_frame_id = labled_image_ids[index-1]
            key_frame_id = image_id
            next_frame_id = labled_image_ids[index+1]
        prev_frame_anns = coco_anns.loadAnns(
            coco_anns.getAnnIds(imgIds=prev_frame_id))
        key_frame_anns = coco_anns.loadAnns(
            coco_anns.getAnnIds(imgIds=key_frame_id))
        next_frame_anns = coco_anns.loadAnns(
            coco_anns.getAnnIds(imgIds=next_frame_id))
        key_frame_ann = np.random.choice(key_frame_anns)

        prev_frame_path = os.path.join(
            self.root_dir, coco_anns.loadImgs(prev_frame_id)[0]['file_name'])
        key_frame_path = os.path.join(
            self.root_dir, coco_anns.loadImgs(key_frame_id)[0]['file_name'])
        next_frame_path = os.path.join(
            self.root_dir, coco_anns.loadImgs(next_frame_id)[0]['file_name'])

        prev_frame = cv2.cvtColor(cv2.imread(
            prev_frame_path), cv2.COLOR_BGR2RGB)
        key_frame = cv2.cvtColor(cv2.imread(
            key_frame_path), cv2.COLOR_BGR2RGB)
        next_frame = cv2.cvtColor(cv2.imread(
            next_frame_path), cv2.COLOR_BGR2RGB)
        
        video_list = []
        video_transformed_list = []
        expanded_bbox_list = []
        keypoints_list = []
        keypoints_transformed_list = []
        heatmaps_list = []

        for key_frame_ann in key_frame_anns:
    
            bbox = key_frame_ann['bbox']
            keypoints = key_frame_ann['keypoints']
            track_id = key_frame_ann["track_id"]
            prev_frame_ann = self._search_track_id(
                prev_frame_anns, track_id)
            next_frame_ann = self._search_track_id(
                next_frame_anns, track_id)
            if prev_frame_ann is None:
                prev_frame_ann = key_frame_ann
            if next_frame_ann is None:
                next_frame_ann = key_frame_ann



            video, video_transformed, expanded_bbox, keypoints, keypoints_transformed, heatmaps = self._prepare_targets(
                key_frame=key_frame,
                prev_frame=prev_frame,
                next_frame=next_frame,
                bbox=bbox,
                keypoints=keypoints,
            )
            video_list.append(video)
            video_transformed_list.append(video_transformed)
            expanded_bbox_list.append(expanded_bbox)
            keypoints_list.append(keypoints)
            keypoints_transformed_list.append(keypoints_transformed)
            heatmaps_list.append(heatmaps)
        
        return key_frame, video_list, video_transformed_list, expanded_bbox_list, keypoints_list, keypoints_transformed_list, heatmaps_list

    def _prepare_targets(self, key_frame, prev_frame, next_frame, bbox, keypoints):
        # crop image according to bounding box
        prev_frame = self._crop_image(
            image=prev_frame, bbox=bbox)
        key_frame, expanded_bbox, keypoints = self._crop_image(
            image=key_frame, bbox=bbox, keypoints=keypoints)
        next_frame = self._crop_image(
            image=next_frame, bbox=bbox)
        # replace visibility flag in keypoints by keypoints labels
        keypoints_transformed, keypoints_labels_transformed = self._prepare_keypoints_and_labels(
            keypoints=keypoints, labels=self.keypoints_labels, img_size=key_frame.shape[:2])

        # transform image and keypoints
        transformed = self.transform(
            image=key_frame, prev_frame=prev_frame, next_frame=next_frame, keypoints=keypoints_transformed, class_labels=keypoints_labels_transformed)
        key_frame_transformed = transformed['image']
        prev_frame_transformed = transformed['prev_frame']
        next_frame_transformed = transformed['next_frame']
        keypoints_transformed = transformed['keypoints']
        keypoints_labels_transformed = transformed['class_labels']

        # restore visibility flag in keypoints
        keypoints_transformed = self._append_invisible_keypoints(
            keypoints_transformed, keypoints_labels_transformed)
        keypoints_transformed = self._append_visibility_flags(
            keypoints_transformed=keypoints_transformed, original_keypoints=keypoints)
        keypoints_transformed = np.array(
            keypoints_transformed, dtype=np.float32)
        label_flag = keypoints_to_label_flag(keypoints_transformed)
        heatmaps = self.udp_heatmap_generator.encode(
            keypoints_transformed[:, :2].reshape(1, 17, 2), label_flag)["heatmaps"]
        video = [prev_frame, key_frame, next_frame]
        video_transformed = torch.stack(
            [prev_frame_transformed, key_frame_transformed, next_frame_transformed])

        return video, video_transformed, expanded_bbox, keypoints, torch.tensor(keypoints_transformed), torch.tensor(heatmaps)

    def collate_fn(self, batch):
        _, videos_transformed, _, _, heatmaps = list(zip(*batch))
        videos_transformed = torch.stack(videos_transformed)
        heatmaps = torch.stack(heatmaps)
        return videos_transformed, heatmaps

    @staticmethod
    def _search_track_id(anns, track_id):
        for ann in anns:
            if ann["track_id"] == track_id:
                return ann
        return None

    @staticmethod
    def _crop_image(image, bbox, keypoints=None):
        x, y, w, h = bbox
        bbox = [x - w / 8, y - h / 8, w * 1.25, h * 1.25]
        bbox = [int(max(x, 0)) for x in bbox]
        x, y, w, h = bbox
        if keypoints is not None:
            keypoints = np.copy(keypoints).reshape(-1, 3)
            for i, keypoint in enumerate(keypoints):
                if keypoint[0] != 0 and keypoint[1] != 0:
                    keypoints[i, 0] = max(keypoint[0] - x, 0)
                    keypoints[i, 1] = max(keypoint[1] - y, 0)
                else:
                    keypoints[i, 0] = 0
                    keypoints[i, 1] = 0
            return image[y:y+h, x:x+w], bbox, keypoints
        else:
            return image[y:y+h, x:x+w]

    @staticmethod
    def _prepare_keypoints_and_labels(keypoints, labels, img_size):
        h, w = img_size
        processed_keypoints = []
        processed_labels = []
        for i, keypoint in enumerate(keypoints):
            x, y, _ = keypoint
            if x != 0 and y != 0 and x < w and y < h:
                processed_keypoints.append((x, y))
                processed_labels.append(labels[i])
        return processed_keypoints, processed_labels

    @staticmethod
    def _center_scale_keypoints(keypoints, size):
        h, w = size
        if isinstance(keypoints, list):
            keypoints = np.array(keypoints, dtype=np.float32)
        keypoints_scale = np.copy(keypoints)
        keypoints_scale[:, 0] = 2. * keypoints[:, 0] / w - 1.
        keypoints_scale[:, 1] = 2. * keypoints[:, 1] / h - 1.
        return keypoints_scale

    def _append_invisible_keypoints(self, keypoints, labels):
        processed_keypoints = []
        for label in self.keypoints_labels:
            if label in labels:
                index = labels.index(label)
                processed_keypoints.append(keypoints[index])
            else:
                processed_keypoints.append((0, 0))

        return processed_keypoints

    @staticmethod
    def _append_visibility_flags(keypoints_transformed, original_keypoints):
        processed_keypoints = []
        for i, keypoint in enumerate(keypoints_transformed):
            x, y = keypoint
            if x == 0 and y == 0:
                processed_keypoints.append((x, y, 0))
            else:
                processed_keypoints.append((x, y, original_keypoints[i][2]))
        return processed_keypoints


dataset_root_dir = "/home/junfeng/datasets/PoseTrack21"
dataset_train = PoseTrack21(
    root_dir=dataset_root_dir,
    set="train",
)
dataset_val = PoseTrack21(
    root_dir=dataset_root_dir,
    set="val",
)
