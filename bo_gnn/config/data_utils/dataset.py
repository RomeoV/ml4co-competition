import unittest
import torch
import pandas as pd
import random
import pathlib
import numpy as np
import os
import pickle
import enum

import torch
import torch_geometric as tg

from data_utils.milp_data import MilpBipartiteData


class DataFormat(enum.Enum):
    ROMEO = enum.auto()
    MAX = enum.auto()


class Folder(enum.Enum):
    TRAIN = "train"
    VALID = "valid"


class MilpDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        csv_file,
        folder: Folder,
        data_format: DataFormat,
        samples_per_epoch=1024,
        instances_dir=None,
    ):
        self.csv_data_full = pd.read_csv(csv_file)
        self.samples_per_epoch = samples_per_epoch

        if instances_dir:
            self.instance_path = pathlib.Path(instances_dir)
        else:
            self.instance_path = pathlib.Path(
                f"../../instances/1_item_placement/{folder.value}"
            )

        if data_format is DataFormat.ROMEO:
            self.cols = [
                "branching/clamp",
                "branching/lpgainnormalize",
                "branching/midpull",
                "branching/midpullreldomtrig",
                "branching/preferbinary",
                "branching/scorefac",
                "branching/scorefunc",
                "lp/colagelimit",
                "lp/pricing",
                "lp/rowagelimit",
                "nodeselection/childsel",
                "separating/cutagelimit",
                "separating/maxcuts",
                "separating/maxcutsroot",
                "separating/minortho",
                "separating/minorthoroot",
                "separating/poolfreq",
                "time_limit",
                "initial_primal_bound",
                "initial_dual_bound",
            ]
        elif data_format is DataFormat.MAX:
            self.cols = [
                "presolve_config_encoding",
                "heuristic_config_encoding",
                "separating_config_encoding",
                "time_limit",
                "initial_primal_bound",
                "initial_dual_bound",
            ]
        else:
            raise "Unsupported data format"

        self.csv_data = self.csv_data_full[self.cols]

    def __iter__(self):
        ind = np.arange(len(self.csv_data))
        np.random.shuffle(ind)

        ind = ind[: min(self.samples_per_epoch, len(ind))]

        for i in ind:
            instance_file, primal_dual_int = self.csv_data_full.loc[
                i, ["instance_file", "time_limit_primal_dual_integral"]
            ]
            primal_dual_int = torch.tensor([primal_dual_int], dtype=torch.float32)
            config_arr = torch.tensor(self.csv_data.loc[i], dtype=torch.float32)
            with open(
                os.path.join(
                    self.instance_path, instance_file.replace(".mps.gz", ".pkl")
                ),
                "rb",
            ) as infile:
                instance_description_pkl = pickle.load(infile)
                instance_data = MilpBipartiteData(
                    var_feats=instance_description_pkl.variable_features,
                    cstr_feats=instance_description_pkl.constraint_features,
                    edge_indices=instance_description_pkl.edge_features.indices,
                    edge_values=instance_description_pkl.edge_features.values,
                )
            yield instance_data, config_arr, primal_dual_int

    def __len__(self):
        return self.samples_per_epoch


class TestDataset(unittest.TestCase):
    def test_some_samples_romeo_format(self):
        ds = MilpDataset(
            csv_file="data/output.csv",
            folder=Folder.TRAIN,
            data_format=DataFormat.ROMEO,
        )
        ds_it = iter(ds)

        for i in range(5):
            inst, conf, labl = next(ds_it)

            self.assertEqual(inst.var_feats.ndim, 2)
            self.assertEqual(inst.cstr_feats.ndim, 2)
            self.assertEqual(inst.edge_index.ndim, 2)

            self.assertIsInstance(conf, torch.Tensor)
            self.assertEqual(conf.ndim, 1)

    def test_some_samples_max_format(self):
        ds = MilpDataset(
            csv_file="data/max_train_data.csv",
            folder=Folder.TRAIN,
            data_format=DataFormat.MAX,
        )
        ds_it = iter(ds)

        for i in range(5):
            inst, conf, labl = next(ds_it)

            self.assertEqual(inst.var_feats.ndim, 2)
            self.assertEqual(inst.cstr_feats.ndim, 2)
            self.assertEqual(inst.edge_index.ndim, 2)

            self.assertIsInstance(conf, torch.Tensor)
            self.assertEqual(conf.ndim, 1)
            self.assertEqual(conf.size()[0], 3 + 3)

    def test_data_loader(self):
        ds = MilpDataset(
            csv_file="data/output.csv",
            folder=Folder.TRAIN,
            data_format=DataFormat.ROMEO,
        )
        dl = tg.data.DataLoader(ds, batch_size=16)
        dl_it = iter(dl)
        for i in range(5):
            instance_batch, config_batch, label_batch = next(dl_it)

            self.assertEqual(instance_batch.var_feats.ndim, 2)
            self.assertEqual(instance_batch.var_feats.shape[1], 9)
            self.assertEqual(config_batch.shape, (16, 20))