"""SegmentationNN"""
import os

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchvision import models

from ..data.segmentation_dataset import SegmentationData


class Encoder(pl.LightningModule):
    def __init__(self, name="alexnet"):
        super().__init__()
        self.name = name
        if self.name == "fcn":
            self.encoder = models.segmentation.fcn_resnet101(pretrained=True).eval()
        elif self.name == "deeplabv3":
            self.encoder = models.segmentation.deeplabv3_resnet101(
                pretrained=True
            ).eval()
        elif self.name == "alexnet":
            self.encoder = models.alexnet(pretrained=True).features
            self.set_parameter_requires_grad(feature_extracting=True)
            self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
            self.input_size = 224

    def forward(self, x):
        x = self.encoder(x)
        if self.name == "alexnet":
            x = self.avgpool(x)
            # output alexnet size torch.Size([8, 256, 6, 6])
            # (6-1)*2 - 2*0 + 1*(7-1) + 1 + 1 = 18
            # output (8, 23, 240, 240)
            # Final output torch.Size([1, 23, 240, 240])
            # Hout=(Hin−1)×stride[0]−2×padding[0]+dilation[0]×(kernel_size[0]−1)+output_padding[0]+1
        return x

    def set_parameter_requires_grad(self, feature_extracting):
        if feature_extracting:
            for param in self.encoder.parameters():
                param.requires_grad = False


class Decoder(pl.LightningModule):
    def __init__(self, name="alexnet", num_classes=23):
        super().__init__()
        self.name = name
        if self.name == "alexnet":
            self.decoder = nn.Sequential(
                # nn.ConvTranspose2d(256, 64, kernel_size=3, stride=1),
                nn.ConvTranspose2d(256, 128, kernel_size=7, stride=2, output_padding=1),
                # (6-1)*2 - 2*0 + 1*(7-1) + 1 + 1 = 18
                # output (8, 128, 18, 18)
                nn.ReLU(),
                nn.ConvTranspose2d(128, 64, kernel_size=7, stride=4, output_padding=1),
                # (18-1)*4 - 2*0 + 1*(7-1) + 1 + 1 = 76
                # output (8, 64, 46, 46)
                nn.ReLU(),
                nn.ConvTranspose2d(64, 32, kernel_size=7, stride=3, output_padding=1),
                # output (8, 32, 233, 233)
                # (76-1)*3 - 2*0 + 1*(7-1) + 1 + 1 = 233
                nn.ReLU(),
                nn.ConvTranspose2d(
                    32, num_classes, kernel_size=8, stride=1, output_padding=0
                ),
                # output (8, 23, 240, 240)
                # (233 - 1)*1 - 2*0 + 1*(8-1) + 0 + 1 = 240
                # Hout=(Hin−1)×stride[0]−2×padding[0]+dilation[0]×(kernel_size[0]−1)+output_padding[0]+1
            )

    def forward(self, x):
        return self.decoder(x)


class SegmentationNN(pl.LightningModule):
    def __init__(self, num_classes=23, hparams=None):
        super().__init__()
        self.hparams.update(hparams)

        self.hparams["feature_extract"] = True

        self.encoder = self.hparams["encoder"]
        self.decoder = self.hparams["decoder"]

    def forward(self, x):

        x = self.encoder(x)
        x = self.decoder(x)

        return x

    def set_parameter_requires_grad(self, feature_extracting):
        if feature_extracting:
            for param in self.model.parameters():
                param.requires_grad = False

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print("Saving model... %s" % path)
        torch.save(self, path)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset["train"],
            shuffle=True,
            batch_size=self.hparams["batch_size"],
            num_workers=8,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset["val"], batch_size=self.hparams["batch_size"], num_workers=8
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset["test"], batch_size=self.hparams["batch_size"], num_workers=8
        )

    def prepare_data(self) -> None:
        path = os.path.dirname(os.path.abspath(os.getcwd()))
        data_root = os.path.join(path, "datasets", "segmentation")
        # assign to use in dataloaders
        self.dataset = {}
        self.dataset["train"] = SegmentationData(
            image_paths_file=f"{data_root}/segmentation_data/train.txt",
        )
        self.dataset["val"] = SegmentationData(
            image_paths_file=f"{data_root}/segmentation_data/val.txt",
        )
        self.dataset["test"] = SegmentationData(
            image_paths_file=f"{data_root}/segmentation_data/test.txt",
        )

    def general_step(self, batch, batch_idx, mode):
        inputs, targets = batch
        outputs = self.forward(inputs)
        loss_func = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction="mean")
        loss = loss_func(outputs, targets)
        _, preds = torch.max(outputs, 1)
        targets_mask = targets >= 0
        n_correct = torch.tensor(
            np.mean((preds.cpu() == targets.cpu())[targets_mask].numpy())
        )
        return loss, n_correct

    def general_end(self, outputs, mode):
        # average over all batches aggregated during one epoch
        avg_loss = torch.stack([x[mode + "_loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x[mode + "_acc"] for x in outputs]).mean()
        # acc = total_correct / len(self.dataset[mode])
        # print("general end acc", acc)
        return avg_loss, avg_acc

    def training_step(self, batch, batch_idx):
        loss, n_correct = self.general_step(batch, batch_idx, "train")
        # tensorboard_logs = {"train_loss": loss, "train_acc": n_correct}
        self.log("train_loss", loss)
        self.log("train_acc", n_correct)
        return {"loss": loss, "train_acc": n_correct}

    def validation_step(self, batch, batch_idx):
        loss, n_correct = self.general_step(batch, batch_idx, "val")
        # tensorboard_logs = {"val_loss": loss, "val_acc": n_correct}
        self.log("val_loss", loss)
        self.log("val_acc", n_correct)
        return {"val_loss": loss, "val_acc": n_correct}

    def test_step(self, batch, batch_idx):
        loss, n_correct = self.general_step(batch, batch_idx, "test")
        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("test_acc", n_correct, on_step=True, on_epoch=True, prog_bar=True)
        return {"test_loss": loss, "test_acc": n_correct}

    def validation_end(self, outputs):
        avg_loss, avg_acc = self.general_end(outputs, "val")
        # print("Val-Acc={}".format(acc))
        tensorboard_logs = {"val_loss": avg_loss, "val_acc": avg_acc}
        return {"val_loss": avg_loss, "val_acc": avg_acc, "log": tensorboard_logs}

    def configure_optimizers(self):
        optim = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams["learning_rate"],
            weight_decay=self.hparams["weight_decay"],
        )

        return optim


class DummySegmentationModel(pl.LightningModule):
    def __init__(self, target_image):
        super().__init__()

        def _to_one_hot(y, num_classes):
            scatter_dim = len(y.size())
            y_tensor = y.view(*y.size(), -1)
            zeros = torch.zeros(*y.size(), num_classes, dtype=y.dtype)

            return zeros.scatter(scatter_dim, y_tensor, 1)

        target_image[target_image == -1] = 1

        self.prediction = _to_one_hot(target_image, 23).permute(2, 0, 1).unsqueeze(0)

    def forward(self, x):
        return self.prediction.float()
