import os

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .rnn_nn import RNN
from .sentiment_dataset import (
    SentimentDataset,
    collate,
    download_data,
    load_vocab,
    load_sentiment_data,
)


class RNNClassifier(pl.LightningModule):
    @property
    def hparams(self):
        return self._hparams

    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        hidden_size,
        use_lstm=True,
        **additional_kwargs
    ):
        """
        Inputs:
            num_embeddings: size of the vocabulary
            embedding_dim: size of an embedding vector
            hidden_size: hidden_size of the rnn layer
            use_lstm: use LSTM if True, vanilla RNN if false, default=True
        """
        super().__init__()

        self.hparams = {
            "num_embeddings": num_embeddings,
            "embedding_dim": embedding_dim,
            "hidden_size": hidden_size,
            "use_lstm": use_lstm,
            **additional_kwargs,
        }

        self.embedding = nn.Embedding(
            self.hparams["num_embeddings"], self.hparams["embedding_dim"], 0
        )

        if self.hparams["use_lstm"]:
            # my own implementation is too slow, choose pytorch one instead
            # self.rnn = LSTM(self.hparams["embeddings_dim"], self.hparams["hidden_size"])
            self.rnn = nn.LSTM(
                self.hparams["embedding_dim"], self.hparams["hidden_size"]
            )
        else:
            self.rnn = RNN(self.hparams["embedding_dim"], self.hparams["hidden_size"])

        # dropout layer
        self.dropout = nn.Dropout(self.hparams["dropout_proba"])

        self.final_fc = nn.Linear(
            self.hparams["hidden_size"], self.hparams["num_classes"], bias=True
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, sequence, lenghts=None):
        """
        Inputs
            sequence: A long tensor of size (seq_len, batch_size)
            lengths: A long tensor of size batch_size, represents the actual
                sequence length of each element in the batch. If None, sequence
                lengths are identical.
        Outputs:
            output: A 1-D tensor of size (batch_size,) represents the probabilities of being
                positive, i.e. in range (0, 1)
        """
        embeddings = self.embedding(sequence)

        if lenghts is not None:
            packed_sequence = pack_padded_sequence(embeddings, lenghts)
            _, h_outputs = self.rnn.forward(packed_sequence)
        else:
            _, h_outputs = self.rnn.forward(embeddings)

        output = h_outputs[0].contiguous().view(-1, self.hparams["hidden_size"])

        # layer of dropout to help prevent overfitting
        if self.hparams["use_dropout"]:
            output = self.dropout(output)

        # Linear layer
        preds = self.final_fc(output)

        # sigmoid + reshape to get the output size (batch_size,)
        final_preds = self.sigmoid(preds).view(-1)

        # final_preds = self.final_fc.forward(h_outputs[0].squeeze(0))
        # final_preds = self.sigmoid(final_preds).squeeze(1)
        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

        return final_preds

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset["train"],
            batch_size=self.hparams["batch_size"],
            collate_fn=collate,
            drop_last=True,
            num_workers=4,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset["val"],
            batch_size=self.hparams["batch_size"],
            collate_fn=collate,
            drop_last=True,
            num_workers=4,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset["test"],
            batch_size=self.hparams["batch_size"],
            collate_fn=collate,
            drop_last=True,
            num_workers=0,
        )

    def prepare_data(self) -> None:
        i2dl_exercises_path = os.path.dirname(os.path.abspath(os.getcwd()))
        data_root = os.path.join(i2dl_exercises_path, "datasets", "SentimentData")
        base_dir = download_data(data_root)
        vocab = load_vocab(base_dir)
        train_data, val_data, test_data = load_sentiment_data(base_dir, vocab)

        self.dataset = {}
        self.dataset["train"] = SentimentDataset(train_data)
        self.dataset["val"] = SentimentDataset(val_data)
        self.dataset["test"] = SentimentDataset(test_data)

    def general_step(self, batch, batch_idx, mode):
        inputs, label, lengths = batch["data"], batch["label"], batch["lengths"]
        outputs = self.forward(inputs)

        loss_func = torch.nn.BCELoss()
        loss = loss_func(outputs, label)

        acc = ((outputs > 0.5) == label).sum().item() / label.size(0)

        return loss, acc

    def general_end(self, outputs, mode):
        # average over all batches aggregated during one epoch
        avg_loss = torch.stack([x[mode + "_loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x[mode + "_acc"] for x in outputs]).mean()
        return avg_loss, avg_acc

    def training_step(self, batch, batch_idx):
        loss, acc = self.general_step(batch, batch_idx, "train")
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "train_acc": acc}

    def validation_step(self, batch, batch_idx):
        loss, acc = self.general_step(batch, batch_idx, "val")
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return {"val_loss": loss, "val_acc": acc}

    def test_step(self, batch, batch_idx):
        loss, acc = self.general_step(batch, batch_idx, "test")
        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("test_acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        return {"test_loss": loss, "test_acc": acc}

    def validation_end(self, outputs):
        avg_loss, avg_acc = self.general_end(outputs, "val")
        tensorboard_logs = {"val_loss": avg_loss, "val_acc": avg_acc}
        return {"val_loss": avg_loss, "val_acc": avg_acc, "log": tensorboard_logs}

    def configure_optimizers(self):
        optim = torch.optim.Adam(
            self.parameters(),
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
                # "frequency": 5,
            },
        }
        return dict_to_return

    @hparams.setter
    def hparams(self, value):
        self._hparams = value
