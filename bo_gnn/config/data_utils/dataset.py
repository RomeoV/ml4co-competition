import unittest
import torch
import pandas as pd
import random
import pathlib
import numpy as np
import os
import pickle
import enum
from typing import Tuple, Union

import torch
import torch_geometric as tg
from tqdm import tqdm

from data_utils.milp_data import MilpBipartiteData


class Mode(enum.Enum):
    """ Whether to create train or validation split.

    This is necessary because so far we've only generated data for the train samples, so we have to split it ourselves.
    """
    TRAIN = enum.auto()
    VALID = enum.auto()

class Folder(enum.Enum):
    """ Where to look for the instances. """
    TRAIN = "train"
    VALID = "valid"


class Problem(enum.Enum):
    ONE = "1_item_placement"
    TWO = "2_load_balancing"
    THREE = "3_anonymoud"


class MilpDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir,
        folder: Folder,
        mode: Mode,
        problem: Problem,
        instance_dir=None,
        dry=False,
    ):
        self.cols = [
            "presolve_config_encoding",
            "heuristic_config_encoding",
            "separating_config_encoding",
            "emphasis_config_encoding",
        ]

        self.problem = problem
        self.mode = mode
        csv_files = list(pathlib.Path(data_dir).glob('*.csv'))
        self.csv_data_full = pd.concat([
            pd.read_csv(f) for f in csv_files
        ], ignore_index=True)

        if dry:
            self.csv_data_full = self.csv_data_full[
                self.csv_data_full.instance_file.str.match(".*_\d\d.mps")
            ].reset_index(drop=True)

        if mode == Mode.TRAIN:
            # Select instances ending/not ending in "0" for train/validation split
            self.csv_data_full = self.csv_data_full[
                ~self.csv_data_full.instance_file.str.match(".*0.mps.*")
            ].reset_index(drop=True)
        elif mode == Mode.VALID:
            self.csv_data_full = self.csv_data_full[
                self.csv_data_full.instance_file.str.match(".*0.mps.*")
            ].reset_index(drop=True)

        if instance_dir:
            self.instance_path = pathlib.Path(instance_dir)
        else:
            self.instance_path = pathlib.Path(
                f"../../instances/{problem.value}/{folder.value}"
            )

        # Preload instances into RAM for faster access.
        # Note: Problem 2 works with 64GB of total RAM.
        # Problem 1 should be much less.
        self.instance_graphs = {}
        print(f"Preloading dataset {mode.name}")
        for instance_file in tqdm(self.csv_data_full.instance_file.unique()):
            with open(
                    os.path.join(self.instance_path, instance_file.replace(".mps.gz", ".pkl")),
                    "rb",
            ) as infile:
                instance_description_pkl = pickle.load(infile)
                self.instance_graphs[instance_file] = MilpBipartiteData(
                    var_feats=instance_description_pkl.variable_features,
                    cstr_feats=instance_description_pkl.constraint_features,
                    edge_indices=instance_description_pkl.edge_features.indices,
                    edge_values=instance_description_pkl.edge_features.values,
                )

        self.csv_data = self.csv_data_full[self.cols]

        # compute per-instance mu and sigma and normalize label by that
        grouped_labels = self.csv_data_full.loc[:, ['instance_file', 'time_limit_primal_dual_integral']].groupby('instance_file')
        self.data_mu = grouped_labels.mean()
        self.data_sig = grouped_labels.std()

    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, idx):
        instance_file, primal_dual_int = self.csv_data_full.loc[
            idx, ["instance_file", "time_limit_primal_dual_integral"]
        ]
        mu, sig = self.data_mu.loc[instance_file][0], self.data_sig.loc[instance_file][0]
        primal_dual_int_standarized = torch.tensor([(primal_dual_int - mu) / 1.], dtype=torch.float32)
        config_arr = torch.tensor(self.csv_data.loc[idx], dtype=torch.float32)
        instance_data = self.instance_graphs[instance_file]
        return instance_data, config_arr, primal_dual_int_standarized


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
