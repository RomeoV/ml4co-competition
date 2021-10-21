import threading
import re
import os
import random
import pickle
import pathlib
import itertools
import scipy
import subprocess
from typing import Tuple

import torch
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl
import torch_geometric as tg

import pytorch_lightning
from pytorch_lightning import Trainer
from models.baseline import ConfigPerformanceRegressor
from models.sanity_check import SanityCheckGNNModel
from models.callbacks import EvaluatePredictedParametersCallback
from data_utils.dataset import MilpDataset, Folder, DataFormat, Problem, Mode
from data_utils.milp_data import MilpBipartiteData
from torch_geometric.data import Data, DataLoader, Batch


class MilpGNNTrainable(pl.LightningModule):
    def __init__(
        self,
        config_dim,
        optimizer,
        initial_lr,
        batch_size,
        git_hash,
        problem: Problem,
        n_gnn_layers,
        gnn_hidden_dim,
        ensemble_size,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model_ensemble = torch.nn.ModuleList(
            [
                ConfigPerformanceRegressor(
                    config_dim=config_dim,
                    n_gnn_layers=n_gnn_layers,
                    gnn_hidden_dim=gnn_hidden_dim,
                )
                for i in range(ensemble_size)
            ]
        )

    def forward(self, instance_batch):
        predictions = torch.stack(
            [
                model.forward(
                    instance_batch.clone(),
                )
                for model in self.model_ensemble
            ],
            axis=1,
        )
        mean_mu = predictions[:, :, :, 0].mean(axis=1)
        mean_var = predictions[:, :, :, 1].mean(axis=1)
        epi_var = (predictions[:, :, :, 0] - mean_mu.unsqueeze(1)).pow(2).mean(axis=1)
        return predictions, mean_mu, mean_var, epi_var

    def training_step(self, batch, batch_idx):
        instance_batch, label_batch, _instance_nums = batch
        instance_batch.cstr_feats.requires_grad_(True)
        instance_batch.var_feats.requires_grad_(True)
        instance_batch.edge_attr.requires_grad_(True)

        label_batch = label_batch.unsqueeze(axis=1)  # (B, 1, C)

        pred = self.forward(instance_batch)[0]
        pred_mu = pred[:, :, :, 0]
        pred_var = pred[:, :, :, 1]

        nll_loss = F.gaussian_nll_loss(pred_mu, label_batch, pred_var)
        l1_loss = F.l1_loss(pred_mu, label_batch)
        l2_loss = F.mse_loss(pred_mu, label_batch)
        sigs = pred_var.mean(axis=1).sqrt()
        self.log_dict(
            {
                "train_nll_loss": nll_loss,
                "train_sigmas": sigs,
                "train_l1": l1_loss,
                "train_l2": l2_loss,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return l2_loss

    def validation_step(self, batch, batch_idx):
        instance_batch, label_batch, _instance_nums = batch

        _pred, mean_mu, mean_var, epi_var = self.forward(instance_batch)
        nll_loss = F.gaussian_nll_loss(mean_mu, label_batch, mean_var)
        l1_loss = F.l1_loss(mean_mu, label_batch)
        l2_loss = F.mse_loss(mean_mu, label_batch)
        self.log_dict(
            {
                "val_sigmas": mean_var.sqrt(),
                "val_epi_sigmas": epi_var.sqrt(),
                "val_nll_loss": nll_loss,
                "val_loss_l1": l1_loss,
                "val_loss_l2": l2_loss,
            },
            prog_bar=True,
        )

    def configure_optimizers(self):
        if self.hparams.optimizer.lower() == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.initial_lr)
        if self.hparams.optimizer.lower() == "adamw":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.initial_lr, weight_decay=1e-4)
        elif self.hparams.optimizer.lower() == "sgd":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.initial_lr)
        elif self.hparams.optimizer.lower() == "rmsprop":
            optimizer = torch.optim.RMSprop(self.parameters(), lr=self.hparams.initial_lr)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, "min", verbose=True, min_lr=1e-6, factor=0.5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "monitor": "train_nll_loss",
        }


def _get_current_git_hash():
    retval = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, check=True)
    git_hash = retval.stdout.decode("utf-8").strip()
    return git_hash


def main():
    problem = Problem.TWO
    dry = subprocess.run(["hostname"], capture_output=True).stdout.decode()[:3] != "eu-"

    data_train = DataLoader(
        MilpDataset(
            "data/exhaustive_dataset_all_configs/2_load_balancing_results_9900.csv",
            folder=Folder.TRAIN,
            mode=Mode.TRAIN,
            data_format=DataFormat.MAX,
            problem=problem,
            dry=dry,
            instance_dir=f"{'../..' if dry else ''}/instances/{problem.value}/{Folder.TRAIN.value}",
        ),
        shuffle=True,
        batch_size=8 if not dry else 4,
        drop_last=False,
        num_workers=8 if not dry else 0,
        pin_memory=torch.cuda.is_available() and not dry,
    )
    config_dim = data_train.dataset.unique_configs_in_dataset.size

    data_valid = DataLoader(
        MilpDataset(
            "data/exhaustive_dataset_all_configs/2_load_balancing_results_validation_filtered.csv",
            folder=Folder.VALID,
            data_format=DataFormat.MAX,
            mode=Mode.VALID,
            problem=problem,
            dry=dry,
            instance_dir=f"{'../..' if dry else ''}/instances/{problem.value}/{Folder.VALID.value}",
        ),
        shuffle=False,
        batch_size=8 if not dry else 4,
        drop_last=False,
        num_workers=8 if not dry else 0,
        pin_memory=torch.cuda.is_available() and not dry,
    )
    assert (data_train.dataset.unique_configs_in_dataset == data_valid.dataset.unique_configs_in_dataset).all()

    model = MilpGNNTrainable(
        config_dim=config_dim,
        optimizer="AdamW",
        initial_lr=1e-4,
        batch_size=8 if not dry else 4,
        n_gnn_layers=4,
        gnn_hidden_dim=64,
        ensemble_size=3,
        git_hash=_get_current_git_hash(),
        problem=problem,
    )

    profiler = pl.profiler.AdvancedProfiler(filename="profiler_out")
    trainer = Trainer(
        max_epochs=1000,
        gpus=1 if torch.cuda.is_available() else 0,
        callbacks=[
            EvaluatePredictedParametersCallback(
                configs=data_valid.dataset.unique_configs_in_dataset,
                instance_dir=f"{'../..' if dry else ''}/instances/{problem.value}/{Folder.VALID.value}",
            ),
            pytorch_lightning.callbacks.LearningRateMonitor(logging_interval="epoch"),
        ],
        profiler=profiler,
    )
    trainer.fit(model, train_dataloaders=data_train, val_dataloaders=data_valid)


if __name__ == "__main__":
    main()
