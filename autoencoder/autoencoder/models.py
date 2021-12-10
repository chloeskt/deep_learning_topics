import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau


class Encoder(nn.Module):
    def __init__(self, hparams, input_size=28 * 28, latent_dim=20):
        super().__init__()

        # set hyperparams
        self.latent_dim = latent_dim
        self.input_size = input_size
        self.hparams = hparams
        self.encoder = None

        self.encoder = nn.Sequential(
            nn.Linear(input_size, self.hparams["n_hidden_encoder"]),
            nn.LeakyReLU(),
            # nn.Linear(self.hparams["n_hidden"], self.hparams["n_hidden"]),
            # nn.LeakyReLU(),
            nn.Linear(self.hparams["n_hidden_encoder"], self.latent_dim),
        )

        # # He init
        torch.nn.init.kaiming_normal_(self.encoder[0].weight)
        self.encoder[0].bias.data.fill_(0.05)

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, hparams, latent_dim=20, output_size=28 * 28):
        super().__init__()

        # set hyperparams
        self.hparams = hparams
        self.decoder = None

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, self.hparams["n_hidden_decoder"]),
            nn.LeakyReLU(),
            # nn.Linear(self.hparams["n_hidden"], self.hparams["n_hidden"]),
            # nn.LeakyReLU(),
            nn.Linear(self.hparams["n_hidden_decoder"], output_size),
        )

        # # He init
        torch.nn.init.kaiming_normal_(self.decoder[0].weight)
        self.decoder[0].bias.data.fill_(0.05)

    def forward(self, x):
        return self.decoder(x)


class Autoencoder(pl.LightningModule):
    def __init__(self, hparams, encoder, decoder, train_set, val_set, logger):
        super().__init__()
        self.hparams = hparams
        # set hyperparams
        self.encoder = encoder
        self.decoder = decoder
        self.train_set = train_set
        self.val_set = val_set
        self.logger = logger

    def forward(self, x):
        reconstruction = None

        reconstruction = self.encoder(x)
        reconstruction = self.decoder(reconstruction)

        return reconstruction

    def general_step(self, batch, batch_idx, mode):
        images = batch
        flattened_images = images.view(images.shape[0], -1)

        # forward pass
        reconstruction = self.forward(flattened_images)

        # loss
        loss = F.mse_loss(reconstruction, flattened_images)

        return loss, reconstruction

    def general_end(self, outputs, mode):
        # average over all batches aggregated during one epoch
        avg_loss = torch.stack([x[mode + "_loss"] for x in outputs]).mean()
        return avg_loss

    def training_step(self, batch, batch_idx):
        loss, _ = self.general_step(batch, batch_idx, "train")
        tensorboard_logs = {"loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        images = batch
        flattened_images = images.view(images.shape[0], -1)

        reconstruction = self.forward(flattened_images)
        loss = F.mse_loss(reconstruction, flattened_images)

        reconstruction = (
            reconstruction.view(reconstruction.shape[0], 28, 28).cpu().numpy()
        )
        images = np.zeros((len(reconstruction), 3, 28, 28))
        for i in range(len(reconstruction)):
            images[i, 0] = reconstruction[i]
            images[i, 2] = reconstruction[i]
            images[i, 1] = reconstruction[i]
        self.logger.experiment.add_images(
            "reconstructions", images, self.current_epoch, dataformats="NCHW"
        )
        return loss

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_set,
            shuffle=True,
            batch_size=self.hparams["batch_size"],
            # num_workers=multiprocessing.cpu_count(),
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_set,
            batch_size=self.hparams["batch_size"],
            # num_workers=multiprocessing.cpu_count(),
        )

    def configure_optimizers(self):

        optim = None

        optim = torch.optim.Adam(
            self.parameters(recurse=True),
            lr=self.hparams["learning_rate"],
            weight_decay=self.hparams["weight_decay"],
        )

        scheduler = StepLR(optim, step_size=1, gamma=0.001)

        return optim

    def getReconstructions(self, loader=None):
        self.eval()
        self = self.to(self.device)

        if not loader:
            loader = self.val_dataloader()

        reconstructions = []

        for batch in loader:
            X = batch
            X = X.to(self.device)
            flattened_X = X.view(X.shape[0], -1)
            reconstruction = self.forward(flattened_X)
            reconstructions.append(
                reconstruction.view(-1, 28, 28).cpu().detach().numpy()
            )

        return np.concatenate(reconstructions, axis=0)


class Classifier(pl.LightningModule):
    def __init__(
        self,
        hparams,
        encoder,
        train_set=None,
        val_set=None,
        test_set=None,
        num_classes=10,
    ):
        super().__init__()
        # set hyperparams
        self.hparams = hparams
        self.encoder = encoder
        self.model = nn.Identity()
        self.data = {"train": train_set, "val": val_set, "test": test_set}

        self.model = nn.Sequential(
            nn.Linear(self.encoder.latent_dim, self.hparams["n_hidden"]),
            nn.LeakyReLU(),
            # nn.Linear(self.hparams["n_hidden"], self.hparams["n_hidden"]),
            # nn.LeakyReLU(),
            nn.Linear(self.hparams["n_hidden"], num_classes),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.model(x)
        return x

    def general_step(self, batch, batch_idx, mode):
        images, targets = batch
        flattened_images = images.view(images.shape[0], -1)

        # forward pass
        out = self.forward(flattened_images)

        # loss
        loss = F.cross_entropy(out, targets)

        preds = out.argmax(axis=1)
        n_correct = (targets == preds).sum()
        return loss, n_correct

    def general_end(self, outputs, mode):
        # average over all batches aggregated during one epoch
        avg_loss = torch.stack([x[mode + "_loss"] for x in outputs]).mean()
        total_correct = (
            torch.stack([x[mode + "_n_correct"] for x in outputs]).sum().cpu().numpy()
        )
        acc = total_correct / len(self.data[mode])
        return avg_loss, acc

    def training_step(self, batch, batch_idx):
        loss, n_correct = self.general_step(batch, batch_idx, "train")
        tensorboard_logs = {"loss": loss}
        return {"loss": loss, "train_n_correct": n_correct, "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        loss, n_correct = self.general_step(batch, batch_idx, "val")
        return {"val_loss": loss, "val_n_correct": n_correct}

    def test_step(self, batch, batch_idx):
        loss, n_correct = self.general_step(batch, batch_idx, "test")
        return {"test_loss": loss, "test_n_correct": n_correct}

    def validation_end(self, outputs):
        avg_loss, acc = self.general_end(outputs, "val")
        # print("Val-Acc={}".format(acc))
        tensorboard_logs = {"val_loss": avg_loss, "val_acc": acc}
        return {"val_loss": avg_loss, "val_acc": acc, "log": tensorboard_logs}

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.data["train"],
            shuffle=True,
            batch_size=self.hparams["batch_size"],
            # num_workers=multiprocessing.cpu_count(),
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.data["val"],
            batch_size=self.hparams["batch_size"],
            # num_workers=multiprocessing.cpu_count(),
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.data["test"],
            batch_size=self.hparams["batch_size"],
            # num_workers=multiprocessing.cpu_count(),
        )

    def configure_optimizers(self):

        optim = None

        optim = torch.optim.Adam(
            self.model.parameters(),
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
                "frequency": 5,
            },
        }

        return dict_to_return

    def getAcc(self, loader=None):
        self.eval()
        self = self.to(self.device)

        if not loader:
            loader = self.test_dataloader()

        scores = []
        labels = []

        for batch in loader:
            X, y = batch
            X = X.to(self.device)
            flattened_X = X.view(X.shape[0], -1)
            score = self.forward(flattened_X)
            scores.append(score.detach().cpu().numpy())
            labels.append(y.detach().cpu().numpy())

        scores = np.concatenate(scores, axis=0)
        labels = np.concatenate(labels, axis=0)

        preds = scores.argmax(axis=1)
        acc = (labels == preds).mean()
        return preds, acc
