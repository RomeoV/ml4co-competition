import threading
import re
import os
import random
import pickle
from typing import Tuple

import ecole
import torch
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl
import torch_geometric as tg

from pytorch_lightning import Trainer
from models.baseline import ConfigPerformanceRegressor
from data_utils.dataset import MilpDataset
from torch_geometric.data import Data, DataLoader


class MilpGNNTrainable(pl.LightningModule):
    def __init__(self, initial_lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.initial_lr = initial_lr

        self.model = ConfigPerformanceRegressor()

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
        l1_loss = F.l1_loss(pred[:, 0:1], label_batch)
        l2_loss = F.mse_loss(pred[:, 0:1], label_batch)
        # self.log("val_loss", l1_loss, prog_bar=True)
        self.log_dict({"val_loss_l1": l1_loss, "val_loss_l2": l2_loss}, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.initial_lr)


def main():
    trainer = Trainer(max_epochs=3, gpus=1 if torch.cuda.is_available() else 0)
    model = MilpGNNTrainable()
    data = DataLoader(
        MilpDataset("data/output.csv", samples_per_epoch=2048),
        batch_size=4,
        drop_last=True,
    )
    trainer.fit(model, data)


if __name__ == "__main__":
    main()
