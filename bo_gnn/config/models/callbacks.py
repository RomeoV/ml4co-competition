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
        val_dataset = trainer.val_dataloaders[0].dataset
        config_id_to_tuple_map = val_dataset.unique_configs_in_dataset

        percentiles = {}
        percentiles["mean"] = []
        percentiles["opti"] = []
        percentiles["pess"] = []

        pl_module.eval()
        for instance_batch, _, instance_nums in trainer.val_dataloaders[0]:
            best_configs = self.__class__._find_best_configs(pl_module, instance_batch, config_id_to_tuple_map)

            for strat in ("mean", "opti", "pess"):
                for best_config, instance_num in zip(best_configs[strat], instance_nums):
                    percentiles[strat].append(
                        self.__class__._percentile_of_config(best_config, instance_num, val_dataset)
                    )
        pl_module.train()

        percentile_means = {f"{k}_perc": torch.tensor(v).mean() for k, v in percentiles.items()}
        self.log_dict(percentile_means, prog_bar=True)

    @staticmethod
    def _find_best_configs(pl_module, instance_batch, config_id_to_tuple_map):
        _preds, mean_mu, mean_var, _epi_var = pl_module.forward(instance_batch.to(pl_module.device))

        best_configs = {}
        best_configs["mean"] = mean_mu.argmin(dim=1)
        best_configs["opti"] = (mean_mu - mean_var.sqrt()).argmin(dim=1)
        best_configs["pess"] = (mean_mu + mean_var.sqrt()).argmin(dim=1)

        for strat, config_ids in best_configs.items():
            best_configs[strat] = [config_id_to_tuple_map[id_.item()] for id_ in config_ids]

        return best_configs

    @staticmethod
    def _percentile_of_config(config_tuple, instance_num, dataset):
        df = dataset.csv_data_fast.loc[instance_num]
        result_with_config = df[df.config_encoding == config_tuple].time_limit_primal_dual_integral.median()

        percentile = scipy.stats.percentileofscore(
            df.time_limit_primal_dual_integral,
            result_with_config,
        )

        return percentile
