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


class DataFormat(enum.Enum):
    ROMEO = enum.auto()
    MAX = enum.auto()


class Mode(enum.Enum):
    """Whether to create train or validation split.

    This is necessary because so far we've only generated data for the train samples, so we have to split it ourselves.
    """

    TRAIN = enum.auto()
    VALID = enum.auto()


class Folder(enum.Enum):
    """Where to look for the instances."""

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
        instance_dir=None,
        dry=False,
    ):
        self.problem = problem
        self.mode = mode
        self.csv_data_full = pd.read_csv(csv_file)

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
            self.instance_path = pathlib.Path(f"../../instances/{problem.value}/{folder.value}")

        # Preload instances into RAM for faster access.
        # Note: Problem 2 works with 64GB of total RAM.
        # Problem 1 should be much less>
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
                "emphasis_config_encoding",
            ]
        else:
            raise "Unsupported data format"

        self.csv_data = self.csv_data_full[self.cols]
        self.csv_data_full["config_encoding"] = self.csv_data.loc[:, self.cols].apply(tuple, axis=1)

        # compute per-instance mu and sigma and normalize label by that
        grouped_labels = self.csv_data_full.loc[:, ["instance_file", "time_limit_primal_dual_integral"]].groupby(
            "instance_file"
        )
        self.data_mu = grouped_labels.mean()
        self.data_sig = grouped_labels.std()

        self.unique_configs_in_dataset = self.csv_data_full.config_encoding.unique()
        self.unique_instance_files = self.csv_data_full.instance_file.unique()
        self._last_index_dbg = None

    def __len__(self):
        return self.unique_instance_files.size

    def __getitem__(self, idx):
        instance_file = self.unique_instance_files[idx]

        primal_dual_int = (
            self.csv_data_full[self.csv_data_full.instance_file == instance_file]
            .groupby("config_encoding")
            .aggregate(np.random.choice)
            .time_limit_primal_dual_integral
        )

        # DBG
        if self._last_index_dbg is not None:
            assert (primal_dual_int.index == self._last_index_dbg).all()
        self._last_index_dbg = primal_dual_int.index
        # DBG END

        mu, sig = (
            self.data_mu.loc[instance_file][0],
            self.data_sig.loc[instance_file][0],
        )
        primal_dual_int_standarized = torch.tensor((primal_dual_int - mu) / sig, dtype=torch.float32)

        instance_data = self.instance_graphs[instance_file]

        return instance_data, primal_dual_int_standarized


class TestDataset(unittest.TestCase):
    def test_some_samples_max_format(self):
        ds = MilpDataset(
            csv_file="data/exhaustive_dataset_20_configs/1_item_placement_results_9898.csv",
            folder=Folder.TRAIN,
            mode=Mode.TRAIN,
            data_format=DataFormat.MAX,
            problem=Problem.ONE,
        )
        ds_it = iter(ds)

        for i in range(5):
            inst, labl = next(ds_it)

            self.assertEqual(inst.var_feats.ndim, 2)
            self.assertEqual(inst.cstr_feats.ndim, 2)
            self.assertEqual(inst.edge_index.ndim, 2)

    def test_data_loader(self):
        ds = MilpDataset(
            csv_file="data/exhaustive_dataset_20_configs/1_item_placement_results_9898.csv",
            folder=Folder.TRAIN,
            mode=Mode.TRAIN,
            data_format=DataFormat.MAX,
            problem=Problem.ONE,
        )
        dl = tg.data.DataLoader(ds, batch_size=16)
        dl_it = iter(dl)
        for i in range(5):
            instance_batch, label_batch = next(dl_it)

            self.assertEqual(instance_batch.var_feats.ndim, 2)
            self.assertEqual(instance_batch.var_feats.shape[1], 9)
