import threading
import re
import os
import random
import pickle
import pathlib
import itertools
import scipy
from typing import Tuple

import torch
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl
import torch_geometric as tg

import pytorch_lightning
from pytorch_lightning import Trainer
from models.baseline import ConfigPerformanceRegressor
from data_utils.dataset import MilpDataset, Folder, DataFormat
from data_utils.milp_data import MilpBipartiteData
from torch_geometric.data import Data, DataLoader, Batch


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


class EvaluatePredictedParametersCallback(pytorch_lightning.callbacks.Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        def find_best_configs(model, instance, initial_primal, initial_dual, timelimit):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            all_config_inputs = torch.stack(
                [
                    torch.tensor(
                        [a, b, c, initial_primal, initial_dual, timelimit],
                        dtype=torch.float32,
                    )
                    for a, b, c in itertools.product(range(4), range(4), range(4))
                ],
                axis=0,
            ).to(device)

            instance_batch = Batch.from_data_list(
                [instance] * len(all_config_inputs)
            ).to(device)

            model.eval()
            preds = model((instance_batch, all_config_inputs))
            model.train()

            best_config_id = {}
            best_config_id["mean"] = preds[:, 0].argmin()
            best_config_id["optimistic"] = (preds[:, 0] - preds[:, 1]).argmin()
            best_config_id["pessimistic"] = (preds[:, 0] + preds[:, 1]).argmin()

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
            path = pathlib.Path(f"../../instances/1_item_placement/train")

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

        percentiles = {"mean": [], "optimistic": [], "pessimistic": []}
        val_data_df = trainer.val_dataloaders[0].dataset.csv_data_full

        # for instance in np.random.choice(val_data_df.instance_file.unique(), 30):
        for instance in val_data_df.instance_file.unique():
            general_config = val_data_df[val_data_df.instance_file == instance].iloc[
                0, 3:6
            ]  # timelimit, initial_primal, initial_dual
            print("find best configs")
            best_configs = find_best_configs(
                pl_module.model,
                get_instance_data(instance),
                general_config[0],
                general_config[1],
                general_config[2],
            )

            print("compute percentiles")
            for k, v in best_configs.items():
                percentiles[k].append(percentile_of_config(v, instance, val_data_df))

        percentile_means = {
            f"{k}_pred_percentile": torch.tensor(v).mean()
            for k, v in percentiles.items()
        }
        self.log_dict(percentile_means, prog_bar=True)


def main():
    trainer = Trainer(
        max_epochs=100,
        gpus=1 if torch.cuda.is_available() else 0,
        callbacks=[EvaluatePredictedParametersCallback()],
    )
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
