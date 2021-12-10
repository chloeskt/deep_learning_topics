import numpy as np
import torch
from torchvision.transforms import transforms


class Normalize:
    """Normalize input images"""

    def __call__(self, sample):
        image, keypoints = sample["image"], sample["keypoints"]
        return {"image": image / 255.0, "keypoints": keypoints}  # scale to [0, 1]


class ToTensor:
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        to_tensor = transforms.ToTensor()
        image, keypoints = sample["image"], sample["keypoints"]
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.reshape(1, 96, 96)
        image = torch.from_numpy(image)
        if keypoints is not None:
            # keypoints = torch.from_numpy(keypoints)
            return {"image": image, "keypoints": keypoints}
        else:
            return {"image": image}


class RandomHorizontalFlip:
    """
    Horizontally flip image randomly with given probability
    Args:
        p (float): probability of the image being flipped.
                   Defalut value = 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):

        flip_indices = [
            (0, 2),
            (1, 3),
            (4, 8),
            (5, 9),
            (6, 10),
            (7, 11),
            (12, 16),
            (13, 17),
            (14, 18),
            (15, 19),
            (22, 24),
            (23, 25),
        ]
        image, keypoints = sample["image"], sample["keypoints"]
        print(keypoints)
        if np.random.random() < self.p:
            image = image[:, ::-1]
            if keypoints is not None:
                for a, b in flip_indices:
                    print(a)
                    print(keypoints[a])
                    keypoints[a], keypoints[b] = keypoints[b], keypoints[a]
                keypoints[::2] = 96.0 - keypoints[::2]
        return {"image": image, "keypoints": keypoints}
