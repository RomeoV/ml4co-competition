import threading
import re
import os
import random
import pickle
from typing import Tuple

import torch
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl
import torch_geometric as tg

from pytorch_lightning import Trainer
from models.baseline import ConfigPerformanceRegressor
from data_utils.dataset import MilpDataset, Folder, DataFormat
from torch_geometric.data import Data, DataLoader


class MilpGNNTrainable(pl.LightningModule):
    def __init__(self, config_dim, initial_lr=5e-4):
        super().__init__()
        self.save_hyperparameters()
        self.initial_lr = initial_lr

        self.model = ConfigPerformanceRegressor(config_dim=config_dim)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        instance_batch, config_batch, label_batch = batch
        pred = self.forward((instance_batch, config_batch))
        loss = F.gaussian_nll_loss(pred[:, 0:1], label_batch, pred[:, 1:2])
        # loss = F.mse_loss(pred[:, 0:1], label_batch)
        # self.log("my_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        instance_batch, config_batch, label_batch = batch
        pred = self.forward((instance_batch, config_batch))
        nll_loss = F.gaussian_nll_loss(pred[:, 0:1], label_batch, pred[:, 1:2])
        l1_loss = F.l1_loss(pred[:, 0:1], label_batch)
        l2_loss = F.mse_loss(pred[:, 0:1], label_batch)
        # self.log("val_loss", l1_loss, prog_bar=True)
        self.log_dict(
            {"val_nll_loss": nll_loss, "val_loss_l1": l1_loss, "val_loss_l2": l2_loss},
            prog_bar=True,
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.initial_lr)


def main():
    trainer = Trainer(max_epochs=3, gpus=1 if torch.cuda.is_available() else 0)
    model = MilpGNNTrainable(config_dim=6)
    data_train = DataLoader(
        MilpDataset(
            "data/max_train_data.csv",
            folder=Folder.TRAIN,
            data_format=DataFormat.MAX,
            dry=(not torch.cuda.is_available()),
        ),
        batch_size=128,
        drop_last=True,
    )
    data_valid = DataLoader(
        MilpDataset(
            "data/max_valid_data.csv",
            folder=Folder.TRAIN,
            data_format=DataFormat.MAX,
            dry=(not torch.cuda.is_available()),
        ),
        batch_size=128,
        drop_last=True,
    )
    trainer.fit(model, train_dataloaders=data_train, val_dataloaders=data_valid)


if __name__ == "__main__":
    main()
