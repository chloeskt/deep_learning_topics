"""Data utility functions."""
import os

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# pylint: disable=C0326
SEG_LABELS_LIST = [
    {"id": -1, "name": "void", "rgb_values": [0, 0, 0]},
    {"id": 0, "name": "building", "rgb_values": [128, 0, 0]},
    {"id": 1, "name": "grass", "rgb_values": [0, 128, 0]},
    {"id": 2, "name": "tree", "rgb_values": [128, 128, 0]},
    {"id": 3, "name": "cow", "rgb_values": [0, 0, 128]},
    {"id": 4, "name": "horse", "rgb_values": [128, 0, 128]},
    {"id": 5, "name": "sheep", "rgb_values": [0, 128, 128]},
    {"id": 6, "name": "sky", "rgb_values": [128, 128, 128]},
    {"id": 7, "name": "mountain", "rgb_values": [64, 0, 0]},
    {"id": 8, "name": "airplane", "rgb_values": [192, 0, 0]},
    {"id": 9, "name": "water", "rgb_values": [64, 128, 0]},
    {"id": 10, "name": "face", "rgb_values": [192, 128, 0]},
    {"id": 11, "name": "car", "rgb_values": [64, 0, 128]},
    {"id": 12, "name": "bicycle", "rgb_values": [192, 0, 128]},
    {"id": 13, "name": "flower", "rgb_values": [64, 128, 128]},
    {"id": 14, "name": "sign", "rgb_values": [192, 128, 128]},
    {"id": 15, "name": "bird", "rgb_values": [0, 64, 0]},
    {"id": 16, "name": "book", "rgb_values": [128, 64, 0]},
    {"id": 17, "name": "chair", "rgb_values": [0, 192, 0]},
    {"id": 18, "name": "road", "rgb_values": [128, 64, 128]},
    {"id": 19, "name": "cat", "rgb_values": [0, 192, 128]},
    {"id": 20, "name": "dog", "rgb_values": [128, 192, 128]},
    {"id": 21, "name": "body", "rgb_values": [64, 64, 0]},
    {"id": 22, "name": "boat", "rgb_values": [192, 64, 0]},
]

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def label_img_to_rgb(label_img):
    label_img = np.squeeze(label_img)
    labels = np.unique(label_img)
    label_infos = [l for l in SEG_LABELS_LIST if l["id"] in labels]

    label_img_rgb = np.array([label_img, label_img, label_img]).transpose(1, 2, 0)
    for l in label_infos:
        mask = label_img == l["id"]
        label_img_rgb[mask] = l["rgb_values"]

    return label_img_rgb.astype(np.uint8)


class SegmentationData(data.Dataset):
    def __init__(self, image_paths_file):
        self.root_dir_name = os.path.dirname(image_paths_file)

        self.to_tensor = transforms.ToTensor()

        with open(image_paths_file) as f:
            self.image_names = f.read().splitlines()

        self.images = []
        self.targets = []
        for name in tqdm(self.image_names):
            img_id = name.replace(".bmp", "")

            img = Image.open(
                os.path.join(self.root_dir_name, "images", img_id + ".bmp")
            ).convert("RGB")
            img = self.to_tensor(img)

            target = Image.open(
                os.path.join(self.root_dir_name, "targets", img_id + "_GT.bmp")
            )

            target = np.array(target, dtype=np.int64)
            target_labels = target[..., 0]
            for label in SEG_LABELS_LIST:
                mask = np.all(target == label["rgb_values"], axis=2)
                target_labels[mask] = label["id"]
            target_labels = torch.from_numpy(target_labels.copy())

            self.images.append(img)
            self.targets.append(target_labels)

    def __getitem__(self, key):
        if isinstance(key, slice):
            # get the start, stop, and step from the slice
            return [self[ii] for ii in range(*key.indices(len(self)))]
        elif isinstance(key, int):
            # handle negative indices
            if key < 0:
                key += len(self)
            if key < 0 or key >= len(self):
                raise IndexError("The index (%d) is out of range." % key)
            # get the data from direct index
            return self.get_item_from_index(key)
        else:
            raise TypeError("Invalid argument type.")

    def __len__(self):
        return len(self.image_names)

    def get_item_from_index(self, index):
        img = self.images[index]
        target_labels = self.targets[index]

        W = img.shape[-1]
        H = img.shape[-2]
        padding = [
            (240 - W + 1) // 2 if W < 240 else 0,
            (240 - H + 1) // 2 if H < 240 else 0,
        ]
        img = transforms.functional.pad(img, padding, fill=0)
        img = transforms.functional.center_crop(img, 240)
        normalize = transforms.Normalize(mean=MEAN, std=STD)
        img = normalize(img)

        W = target_labels.shape[-1]
        H = target_labels.shape[-2]
        padding = [
            (240 - W + 1) // 2 if W < 240 else 0,
            (240 - H + 1) // 2 if H < 240 else 0,
        ]
        target_labels = transforms.functional.pad(target_labels, padding, fill=-1)
        target_labels = transforms.functional.center_crop(target_labels, 240)

        return img, target_labels
