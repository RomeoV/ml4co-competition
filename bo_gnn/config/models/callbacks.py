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
        def find_best_configs(pl_module, instance):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            all_config_inputs = torch.stack(
                [
                    torch.tensor(
                        [a, b, c],
                        dtype=torch.float32,
                    )
                    for (a, b, c) in self.configs
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
            with open(
                os.path.join(self.instance_dir, instance_file.replace(".mps.gz", ".pkl")),
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
            best_configs = find_best_configs(
                pl_module,
                get_instance_data(instance),
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
