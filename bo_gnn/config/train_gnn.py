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
from data_utils.dataset import MilpDataset, Folder, DataFormat, Problem
from data_utils.milp_data import MilpBipartiteData
from torch_geometric.data import Data, DataLoader, Batch


class MilpGNNTrainable(pl.LightningModule):
    def __init__(
        self,
        config_dim,
        optimizer,
        batch_size,
        git_hash,
        problem: Problem,
        initial_lr=5e-3,
        scale_labels=True,
        n_gnn_layers=1,
        gnn_hidden_layers=8,
        ensemble_size=3,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model_ensemble = torch.nn.ModuleList(
            [
                ConfigPerformanceRegressor(
                    config_dim=config_dim,
                    n_gnn_layers=n_gnn_layers,
                    gnn_hidden_layers=gnn_hidden_layers,
                )
                for i in range(ensemble_size)
            ]
        )

    def forward(self, x, single_instance=False):
        instance_batch, config_batch = x  # we have to clone those
        predictions = torch.stack(
            [
                model.forward(
                    (instance_batch.clone(), config_batch.clone()),
                    single_instance=single_instance,
                )
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
        if self.hparams.scale_labels:
            label_batch = (label_batch - self.mu) / self.sig
        pred = self.forward((instance_batch, config_batch))[0]
        loss = F.gaussian_nll_loss(pred[:, :, 0], label_batch, pred[:, :, 1])
        l1_loss = F.l1_loss(pred[:, :, 0], label_batch)
        l2_loss = F.mse_loss(pred[:, :, 0], label_batch)
        sigs = pred[:, :, 1].mean(axis=1).sqrt()
        self.log_dict(
            {
                "train_loss": loss,
                "train_sigmas": sigs,
                "train_l1": l1_loss,
                "train_l2": l2_loss,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        instance_batch, config_batch, label_batch = batch
        if self.hparams.scale_labels:
            label_batch = (label_batch - self.mu) / self.sig
        pred, mean_mu, mean_var, epi_var = self.forward((instance_batch, config_batch))
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
        elif self.hparams.optimizer.lower() == "sgd":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.initial_lr)
        elif self.hparams.optimizer.lower() == "rmsprop":
            optimizer = torch.optim.RMSprop(
                self.parameters(), lr=self.hparams.initial_lr
            )
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, "min", verbose=True, min_lr=1e-6
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "monitor": "train_loss",
        }


class EvaluatePredictedParametersCallback(pytorch_lightning.callbacks.Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        def find_best_configs(pl_module, instance, general_config):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            all_config_inputs = torch.stack(
                [
                    torch.tensor(
                        [a, b, c, *general_config],
                        dtype=torch.float32,
                    )
                    for a, b, c in itertools.product(range(4), range(4), range(4))
                ],
                axis=0,
            ).to(device)

            instance_batch = Batch.from_data_list([instance]).to(device)

            pl_module.eval()
            preds, mean_mu, mean_var, epi_var = pl_module.forward(
                (instance_batch, all_config_inputs), single_instance=True
            )
            pl_module.train()

            best_config_id = {}
            best_config_id["mean"] = mean_mu.argmin()
            best_config_id["optimistic"] = (mean_mu - mean_var.sqrt()).argmin()
            best_config_id["pessimistic"] = (mean_mu + mean_var.sqrt()).argmin()

            best_config = {
                k: all_config_inputs[v, 0:3].to(torch.int32)
                for k, v in best_config_id.items()
            }

            return best_config

        def percentile_of_config(config, instance, df):
            pred_with_config = df[
                (df.instance_file == instance)
                & (df.presolve_config_encoding == int(config[0]))
                & (df.heuristic_config_encoding == int(config[1]))
                & (df.separating_config_encoding == int(config[2]))
            ].time_limit_primal_dual_integral.median()

            percentile = scipy.stats.percentileofscore(
                df[df.instance_file == instance].time_limit_primal_dual_integral,
                pred_with_config,
            )

            return percentile

        def get_instance_data(instance_file):
            # TODO clean this up
            path = pathlib.Path(
                f"../../instances/{pl_module.hparams.problem.value}/train"
            )

            with open(
                os.path.join(path, instance_file.replace(".mps.gz", ".pkl")),
                "rb",
            ) as infile:
                instance_description_pkl = pickle.load(infile)
                instance_data = MilpBipartiteData(
                    var_feats=instance_description_pkl.variable_features,
                    cstr_feats=instance_description_pkl.constraint_features,
                    edge_indices=instance_description_pkl.edge_features.indices,
                    edge_values=instance_description_pkl.edge_features.values,
                )
            return instance_data

        percentiles = {"mean": [], "optimistic": [], "pessimistic": [], "default": []}
        val_data_df = trainer.val_dataloaders[0].dataset.csv_data_full

        for instance in val_data_df.instance_file.unique():
            general_config = val_data_df[val_data_df.instance_file == instance].iloc[
                0, 3:6
            ]  # timelimit, initial_primal, initial_dual
            best_configs = find_best_configs(
                pl_module,
                get_instance_data(instance),
                general_config,
            )

            for k, v in best_configs.items():
                percentiles[k].append(percentile_of_config(v, instance, val_data_df))

            percentile_of_default = percentile_of_config(
                np.array([1, 1, 1]), instance, val_data_df
            )
            percentiles["default"].append(percentile_of_default)

        percentile_means = {
            f"{k}_pred_percentile": torch.tensor(v).mean()
            for k, v in percentiles.items()
        }
        self.log_dict(percentile_means, prog_bar=True)


def _get_current_git_hash():
    retval = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, check=True
    )
    git_hash = retval.stdout.decode("utf-8").strip()
    return git_hash


def main():
    problem = Problem.ONE

    trainer = Trainer(
        max_epochs=1000,
        gpus=1 if torch.cuda.is_available() else 0,
        callbacks=[
            # EvaluatePredictedParametersCallback(),
            pytorch_lightning.callbacks.LearningRateMonitor(logging_interval="epoch"),
        ],
    )
    model = MilpGNNTrainable(
        config_dim=6,
        optimizer="Adam",
        batch_size=8,
        n_gnn_layers=4,
        gnn_hidden_layers=32,
        ensemble_size=1,
        git_hash=_get_current_git_hash(),
        problem=problem,
    )
    data_train = DataLoader(
        MilpDataset(
            "data/max_train_data.csv",
            folder=Folder.TRAIN,
            data_format=DataFormat.MAX,
            problem=problem,
            dry=(not torch.cuda.is_available()),
            only_default_config=True,
        ),
        shuffle=True,
        batch_size=8,
        drop_last=False,
        num_workers=3,
    )
    data_valid = DataLoader(
        MilpDataset(
            "data/max_valid_data.csv",
            folder=Folder.TRAIN,
            data_format=DataFormat.MAX,
            problem=problem,
            dry=(not torch.cuda.is_available()),
            only_default_config=True,
        ),
        shuffle=False,
        batch_size=8,
        drop_last=False,
        num_workers=3,
    )
    # TODO clean this up
    mu, sig = (
        data_train.dataset.csv_data_full.time_limit_primal_dual_integral.mean(),
        data_train.dataset.csv_data_full.time_limit_primal_dual_integral.std(),
    )
    model.mu, model.sig = mu, sig
    trainer.fit(model, train_dataloaders=data_train, val_dataloaders=data_valid)


if __name__ == "__main__":
    main()
