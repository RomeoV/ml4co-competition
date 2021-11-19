import argparse
import threading
import re
import os
import random
import pickle
import pathlib
import itertools
import scipy
import datetime
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
from data_utils.dataset import MilpDataset, Folder, Problem, Mode
from data_utils.milp_data import MilpBipartiteData
from torch_geometric.data import Data, DataLoader, Batch


class MilpGNNTrainable(pl.LightningModule):
    def __init__(
        self,
        config_dim,
        optimizer,
        weight_decay,
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

    def forward(self, x, single_instance=False):
        instance_batch, config_batch = x  # we have to clone those
        predictions = torch.stack(
            [
                model.forward((instance_batch.clone(), config_batch.clone()), single_instance=single_instance)
                for model in self.model_ensemble
            ],
            axis=1,
        )
        mean_mu = predictions[:, :, 0:1].mean(axis=1)
        mean_var = predictions[:, :, 1:2].mean(axis=1)
        epi_var = (predictions[:, :, 0] - mean_mu).pow(2).mean(axis=1)
        return predictions, mean_mu, mean_var, epi_var

    def training_step(self, batch, batch_idx):
        instance_batch, config_batch, label_batch = batch
        instance_batch.cstr_feats.requires_grad_(True)
        instance_batch.var_feats.requires_grad_(True)
        instance_batch.edge_attr.requires_grad_(True)
        config_batch.requires_grad_(True)

        label_batch = label_batch.unsqueeze(axis=2)  # (B, 1, 1)

        pred = self.forward((instance_batch, config_batch))[0]
        pred_mu = pred[:, :, 0:1]
        pred_var = pred[:, :, 1:2]

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
        instance_batch, config_batch, label_batch = batch

        _pred, mean_mu, mean_var, epi_var = self.forward((instance_batch, config_batch))
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
            optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.initial_lr, weight_decay=self.hparams.weight_decay)
        if self.hparams.optimizer.lower() == "adamw":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.initial_lr, weight_decay=self.hparams.weight_decay)
        elif self.hparams.optimizer.lower() == "sgd":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.initial_lr, weight_decay=self.hparams.weight_decay)
        elif self.hparams.optimizer.lower() == "rmsprop":
            optimizer = torch.optim.RMSprop(self.parameters(), lr=self.hparams.initial_lr, weight_decay=self.hparams.weight_decay)
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--run_id",
        help="ID of current run (or experiment).",
        type=int,
        required=True,
    )
    parser.add_argument(
        "-t",
        "--max_time",
        help="Time after which training will be aborted in seconds.",
        default=None,
        type=float,
        required=False,
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    root_dir = os.path.join("runs", f"run{args.run_id:03d}")
    problem = Problem.ONE
    # dry = subprocess.run(["hostname"], capture_output=True).stdout.decode()[:3] != "eu-"
    dry = False

    model = MilpGNNTrainable(
        config_dim=4,
        optimizer="RMSprop",
        weight_decay=1e-3,
        initial_lr=5e-4,
        batch_size=64 if not dry else 4,
        n_gnn_layers=4,
        gnn_hidden_dim=64,
        ensemble_size=3,
        git_hash=_get_current_git_hash(),
        problem=problem,
    )
    data_train = DataLoader(
        MilpDataset(
            os.path.join(root_dir, "data"),
            folder=Folder.TRAIN,
            mode=Mode.TRAIN,
            problem=problem,
            dry=dry,
            instance_dir=f"{'../..' if dry else ''}/instances/{problem.value}/{Folder.TRAIN.value}",
        ),
        shuffle=True,
        batch_size=64 if not dry else 4,
        drop_last=False,
        num_workers=8 if not dry else 0,
        pin_memory=torch.cuda.is_available() and not dry,
    )
    configs_in_dataset = (
        data_train.dataset.csv_data.loc[
            :,
            [
                "presolve_config_encoding",
                "heuristic_config_encoding",
                "separating_config_encoding",
                "emphasis_config_encoding",
            ],
        ]
        .apply(tuple, axis=1)
        .unique()
    )

    data_valid = DataLoader(
        MilpDataset(
            os.path.join(root_dir, "data"),
            folder=Folder.TRAIN,
            mode=Mode.VALID,
            problem=problem,
            dry=dry,
            instance_dir=f"{'../..' if dry else ''}/instances/{problem.value}/{Folder.TRAIN.value}",
        ),
        shuffle=False,
        batch_size=64 if not dry else 4,
        drop_last=False,
        num_workers=8 if not dry else 0,
        pin_memory=torch.cuda.is_available() and not dry,
    )

    trainer = Trainer(
        gpus=1 if torch.cuda.is_available() else 0,
        callbacks=[
            EvaluatePredictedParametersCallback(
                configs=configs_in_dataset,
                instance_dir=f"{'../..' if dry else ''}/instances/{problem.value}/{Folder.TRAIN.value}",
            ),
            pytorch_lightning.callbacks.LearningRateMonitor(logging_interval="epoch"),
            pytorch_lightning.callbacks.ModelCheckpoint(save_last=True),
        ],
        default_root_dir=root_dir,
        max_time=datetime.timedelta(seconds=args.max_time),
        resume_from_checkpoint=_get_latest_checkpoint_path(args.run_id),
    )
    trainer.fit(model, train_dataloaders=data_train, val_dataloaders=data_valid)


def _get_latest_checkpoint_path(run_id):
    """Automatically get's the latest checkpoint called 'last.ckpt' from lightning logs. Returns 'None' if not available."""

    def sort_by_num(s):
        return re.search("[0-9]+", s).group(0)

    run_path = os.path.join("runs", f"run{run_id:03d}")
    if not os.path.isdir(os.path.join(run_path, "lightning_logs")):
        return None
    latest_version = sorted(os.listdir(os.path.join(run_path, "lightning_logs")), key=sort_by_num)[-1]
    checkpoint_path = os.path.join(run_path, "lightning_logs", latest_version, "checkpoints", "last.ckpt")
    return checkpoint_path


if __name__ == "__main__":
    main()
