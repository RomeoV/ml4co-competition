import itertools
import torch
import pytorch_lightning
import pathlib
import scipy
import os
import pickle

import numpy as np
from data_utils.milp_data import MilpBipartiteData
from torch_geometric.data import Batch


class EvaluatePredictedParametersCallback(pytorch_lightning.callbacks.Callback):
    def __init__(self, configs, instance_dir):
        self.configs = configs
        self.instance_dir = pathlib.Path(instance_dir)

    def on_train_epoch_start(self, trainer, pl_module):
        def find_best_config(pl_module, instance):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            instance_batch = Batch.from_data_list([instance]).to(device)

            pl_module.eval()
            _preds, mean_mu, mean_var, _epi_var = pl_module.forward(instance_batch)
            pl_module.train()

            best_config = {}
            best_config["mean"] = self.configs[mean_mu.argmin()]
            best_config["opti"] = self.configs[(mean_mu - mean_var.sqrt()).argmin()]
            best_config["pess"] = self.configs[(mean_mu + mean_var.sqrt()).argmin()]

            # best_config = {k: all_config_inputs[v, 0:4].to(torch.int32) for k, v in best_config_id.items()}

            return best_config

        def percentile_of_config(config, instance, dataset):
            df = dataset.csv_data_full
            pred_with_config = df[
                (df.instance_num == instance)
                & (df.presolve_config_encoding == int(config[0]))
                & (df.heuristic_config_encoding == int(config[1]))
                & (df.separating_config_encoding == int(config[2]))
                & (df.emphasis_config_encoding == int(config[3]))
            ].time_limit_primal_dual_integral.median()

            percentile = scipy.stats.percentileofscore(
                df[df.instance_num == instance].time_limit_primal_dual_integral,
                pred_with_config,
            )

            return percentile

        percentiles = {"mean": [], "opti": [], "pess": []}
        val_dataset = trainer.val_dataloaders[0].dataset

        for instance_num in val_dataset.unique_instance_nums:
            best_configs = find_best_config(pl_module, val_dataset.instance_graphs[instance_num])

            for k, v in best_configs.items():
                percentiles[k].append(percentile_of_config(v, instance_num, val_dataset))

        percentile_means = {f"{k}_perc": torch.tensor(v).mean() for k, v in percentiles.items()}
        self.log_dict(percentile_means, prog_bar=True)
