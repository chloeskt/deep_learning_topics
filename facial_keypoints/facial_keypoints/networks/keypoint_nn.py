"""Models for facial keypoint detection"""
import os

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.transforms import transforms

from ..data.facial_keypoints_dataset import (
    FacialKeypointsDataset,
)


class KeypointModel(pl.LightningModule):
    """Facial keypoint detection model"""

    def __init__(self, hparams):
        super(KeypointModel, self).__init__()
        self.hparams = hparams

        # Shape of a Convolutional Layer
        # K - out_channels : the number of filters in the convolutional layer
        # F - kernel_size
        # S - the stride of the convolution
        # P - the padding
        # D - the dilation of the convolution
        # W - the width/height (square) of the previous layer

        # 96x96 pixels
        # kernel_size_conv = 3
        # padding_conv = 1
        # stride_conv = 1

        ## self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)
        # output size = (W + 2*P - D*(F-1) -1)/S + 1 = (96 + 2*1 - 1*(3-1) - 1)/1 + 1 = 96
        # the output Tensor for one image, will have the dimensions: (16, 96, 96)
        # after max pooling: (W + 2*P - D*(F-1) -1)/S + 1 = (96 + 2*0 - 1*(2-1) - 1)/2 + 1 = 48
        # hence (16, 48, 48)
        self.conv1 = nn.Conv2d(
            1,
            16,
            kernel_size=self.hparams["kernel_size_conv"],
            padding=self.hparams["padding_conv"],
            stride=self.hparams["stride_conv"],
        )

        # output size = (W + 2*P - D*(F-1) -1)/S + 1 = (48 + 2*1 - 1*(3-1) - 1)/1 + 1 = 48
        # the output Tensor for one image, will have the dimensions: (32, 45, 45)
        # after max pooling: (W + 2*P - D*(F-1) -1)/S + 1 = (48 + 2*0 - 1*(2-1) - 1)/2 + 1 = 24
        # hence (32, 24, 24)
        self.conv2 = nn.Conv2d(
            16,
            32,
            kernel_size=self.hparams["kernel_size_conv"],
            padding=self.hparams["padding_conv"],
            stride=self.hparams["stride_conv"],
        )

        # output size = (W + 2*P - D*(F-1) -1)/S + 1 = (24 + 2*1 - 1*(3-1) - 1)/1 + 1 = 24
        # the output Tensor for one image, will have the dimensions: (64, 24, 24)
        # after max pooling: (W + 2*P - D*(F-1) -1)/S + 1 = (24 + 2*0 - 1*(2-1) - 1)/2 + 1 = 12
        # hence (64, 12, 12)
        self.conv3 = nn.Conv2d(
            32,
            64,
            kernel_size=self.hparams["kernel_size_conv"],
            padding=self.hparams["padding_conv"],
            stride=self.hparams["stride_conv"],
        )

        self.pool = nn.MaxPool2d(
            2,
            padding=self.hparams["padding_maxpool"],
            stride=self.hparams["stride_maxpool"],
        )

        self.fc1 = nn.Linear(64 * 12 * 12, 530)
        # self.fc1 = nn.Linear(32 * 24 * 24, 1024)
        self.fc2 = nn.Linear(530, 30)
        # self.fc3 = nn.Linear(520, 30)

        self.dropout = nn.Dropout(self.hparams["dropout_proba"])

    def forward(self, x):
        if x.dim() == 3:
            x = torch.unsqueeze(x, 0)

        x = self.pool(torch.nn.functional.leaky_relu(self.conv1(x)))
        x = self.pool(torch.nn.functional.leaky_relu(self.conv2(x)))
        x = self.pool(torch.nn.functional.leaky_relu(self.conv3(x)))
        # flattening x tensor
        x = x.view(x.size(0), -1)
        x = torch.nn.functional.leaky_relu(self.fc1(self.dropout(x)))
        # x = torch.nn.functional.leaky_relu(self.fc2(self.dropout(x)))
        x = self.fc2(self.dropout(x))

        return x

    def general_step(self, batch, batch_idx, mode):

        if mode != "test":
            image, keypoints = batch["image"], batch["keypoints"]
            # forward pass
            predicted_keypoints = self.forward(image).view(-1, 15, 2)

            # loss
            criterion = torch.nn.MSELoss()
            loss = criterion(
                torch.squeeze(keypoints), torch.squeeze(predicted_keypoints)
            )

            return loss

        raise NotImplementedError

    def general_end(self, outputs, mode):
        # average over all batches aggregated during one epoch
        total_loss = torch.stack([x[mode + "_loss"] for x in outputs]).sum()
        score = 1.0 / (2 * (total_loss / len(self.dataset[mode])))
        return total_loss, score

    def training_step(self, batch, batch_idx):
        loss = self.general_step(batch, batch_idx, "train")
        tensorboard_logs = {"loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        loss = self.general_step(batch, batch_idx, "val")
        return {"val_loss": loss}

    def validation_end(self, outputs):
        total_loss, score = self.general_end(outputs, "val")
        tensorboard_logs = {"val_loss": total_loss, "score_val": score}
        return {"val_loss": total_loss, "score_val": score, "log": tensorboard_logs}

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset["train"],
            shuffle=True,
            batch_size=self.hparams["batch_size"],
            # num_workers=8,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset["val"],
            batch_size=self.hparams["batch_size"],
            # num_workers=8,
        )

    def configure_optimizers(self):

        optim = torch.optim.Adam(
            self.parameters(recurse=True),
            lr=self.hparams["learning_rate"],
            weight_decay=self.hparams["weight_decay"],
        )

        scheduler = ReduceLROnPlateau(
            optim,
            patience=self.hparams["scheduler_patience"],
            factor=self.hparams["scheduler_factor"],
            mode="min",
        )

        dict_to_return = {
            "optimizer": optim,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }

        return dict_to_return

    def prepare_data(self):
        path = os.path.dirname(os.path.abspath(os.getcwd()))
        data_root = os.path.join(path, "datasets", "facial_keypoints")
        train_dataset = FacialKeypointsDataset(
            train=True,
            transform=transforms.ToTensor(),
            root=data_root,
        )
        val_dataset = FacialKeypointsDataset(
            train=False,
            transform=transforms.ToTensor(),
            root=data_root,
        )
        self.dataset = {}
        self.dataset["train"], self.dataset["val"] = (
            train_dataset,
            val_dataset,
        )


class DummyKeypointModel(pl.LightningModule):
    """Dummy model always predicting the keypoints of the first train sample"""

    def __init__(self):
        super().__init__()
        self.prediction = torch.tensor(
            [
                [
                    0.4685,
                    -0.2319,
                    -0.4253,
                    -0.1953,
                    0.2908,
                    -0.2214,
                    0.5992,
                    -0.2214,
                    -0.2685,
                    -0.2109,
                    -0.5873,
                    -0.1900,
                    0.1967,
                    -0.3827,
                    0.7656,
                    -0.4295,
                    -0.2035,
                    -0.3758,
                    -0.7389,
                    -0.3573,
                    0.0086,
                    0.2333,
                    0.4163,
                    0.6620,
                    -0.3521,
                    0.6985,
                    0.0138,
                    0.6045,
                    0.0190,
                    0.9076,
                ]
            ]
        )

    def forward(self, x):
        return self.prediction.repeat(x.size()[0], 1, 1, 1)
