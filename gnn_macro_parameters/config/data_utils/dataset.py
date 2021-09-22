import pandas as pd
import pathlib
import numpy as np
import os
import pickle
import torch_geometric as tg
import torch

from data_utils.milp_data import MilpBipartiteData


class MilpDataset(torch.utils.data.IterableDataset):
    def __init__(self, csv_file, instances_dir):
        super(MilpDataset).__init__()
        self.dataset_info = pd.read_csv(csv_file)
        self.instance_path = pathlib.Path(instances_dir)
        self.samples_per_epoch = len(self.dataset_info)

    def __iter__(self):
        ind = np.arange(len(self.dataset_info))
        np.random.shuffle(ind)


        for i in ind:
            instance_file, parameters, primal_dual_integral = self.dataset_info.loc[
                i, ["instance_file", "parameters", "primal_dual_integral"]
            ]
            primal_dual_integral_tensor = torch.tensor([primal_dual_integral], dtype=torch.float32)

            parameter_as_list = []
            for param_string in parameters:
                if param_string.isdigit():
                    parameter_as_list.append(int(param_string))

            parameters_tensor = torch.tensor(parameter_as_list, dtype=torch.float32)
            with open(
                os.path.join(
                    self.instance_path, instance_file.replace(".mps.gz", ".pkl")
                ),
                "rb",
            ) as infile:
                instance_description_pkl = pickle.load(infile)
                instance_data_tensor = MilpBipartiteData(
                    var_feats=instance_description_pkl.variable_features,
                    cstr_feats=instance_description_pkl.constraint_features,
                    edge_indices=instance_description_pkl.edge_features.indices,
                    edge_values=instance_description_pkl.edge_features.values,
                )
            yield (instance_data_tensor, parameters_tensor, primal_dual_integral_tensor)

    def __len__(self):
        return self.samples_per_epoch



