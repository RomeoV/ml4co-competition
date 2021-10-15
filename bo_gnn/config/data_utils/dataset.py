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

from data_utils.milp_data import MilpBipartiteData


class DataFormat(enum.Enum):
    ROMEO = enum.auto()
    MAX = enum.auto()

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
        csv_file,
        folder: Folder,
        mode: Mode,
        data_format: DataFormat,
        problem: Problem,
        instances_dir=None,
        dry=False,
        only_one_config: Union[bool, Tuple[int, int, int]] = False,
    ):
        self.problem = problem
        self.csv_data_full = pd.read_csv(csv_file)

        if dry:
            instances = self.csv_data_full.instance_file.unique()
            chosen_instances = np.random.choice(
                instances, size=min(3, len(instances)), replace=False
            )
            self.csv_data_full = self.csv_data_full[
                self.csv_data_full.instance_file.isin(chosen_instances)
            ].reset_index(drop=True)

        if only_one_config:
            presolve_setting, heuristic_setting, separating_setting = only_one_config
            df = self.csv_data_full
            df = df[
                (df.presolve_config_encoding == presolve_setting)
                & (df.heuristic_config_encoding == heuristic_setting)
                & (df.separating_config_encoding == separating_setting)
            ].reset_index(drop=True)
            self.csv_data_full = df

        if mode == Mode.TRAIN:
            # Select instances ending/not ending in "0" for train/validation split
            self.csv_data_full = self.csv_data_full[
                ~self.csv_data_full.instance_file.str.match(".*0.mps.*")
            ].reset_index(drop=True)
        elif mode == Mode.VALID:
            self.csv_data_full = self.csv_data_full[
                self.csv_data_full.instance_file.str.match(".*0.mps.*")
            ].reset_index(drop=True)

        if instances_dir:
            self.instance_path = pathlib.Path(instances_dir)
        else:
            self.instance_path = pathlib.Path(
                f"../../instances/{problem.value}/{folder.value}"
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
            ]
        else:
            raise "Unsupported data format"

        self.csv_data = self.csv_data_full[self.cols]
        # breakpoint()
        # print("a")

    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, idx):
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()

        instance_file, primal_dual_int = self.csv_data_full.loc[
            idx, ["instance_file", "time_limit_primal_dual_integral"]
        ]
        primal_dual_int = torch.tensor([primal_dual_int], dtype=torch.float32)
        config_arr = torch.tensor(self.csv_data.loc[idx], dtype=torch.float32)
        with open(
            os.path.join(self.instance_path, instance_file.replace(".mps.gz", ".pkl")),
            "rb",
        ) as infile:
            instance_description_pkl = pickle.load(infile)
            instance_data = MilpBipartiteData(
                var_feats=instance_description_pkl.variable_features,
                cstr_feats=instance_description_pkl.constraint_features,
                edge_indices=instance_description_pkl.edge_features.indices,
                edge_values=instance_description_pkl.edge_features.values,
            )
        return instance_data, config_arr, primal_dual_int


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
