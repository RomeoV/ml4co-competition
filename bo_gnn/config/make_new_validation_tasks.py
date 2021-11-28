import os
import argparse
import random
import re
import pathlib
import torch
import operator
import time
import subprocess
import tqdm
import numpy as np
import pandas as pd
import itertools
from data_utils.milp_data import MilpBipartiteData
from data_utils.dataset import Problem
from train_gnn import MilpGNNTrainable
from torch_geometric.data import Batch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--run_id",
        help="ID of current run (or experiment).",
        type=int,
        required=True,
    )
    parser.add_argument(
        "-i",
        "--iter",
        help="Iter num of current run (or experiment).",
        type=int,
        required=True,
    )

    return parser.parse_args()


def _predict_instance_for_all_configs(model, instance_batch):
    config_list = [(*tpl, 0) for tpl in itertools.product(range(4), repeat=3)]
    config_batch= torch.tensor(
        config_list, dtype=torch.float32
    )
    _, mean_mu, _, _ = model.forward((instance_batch, config_batch), single_instance=True)
    return mean_mu, config_list


def main():
    args = parse_args()

    # device = 'cuda' if torch.cuda.is_available else 'cpu'
    device = "cpu"
    latest_checkpoint_path = f"/runs/run{args.run_id:03d}/lightning_logs/version_{args.iter}/checkpoints/last.ckpt"
    assert os.path.isfile(latest_checkpoint_path)
    model = MilpGNNTrainable.load_from_checkpoint(latest_checkpoint_path).to(device)
    model.eval()

    df_list = []
    instance_path = pathlib.Path("/instances/1_item_placement/valid")
    instance_files = sorted(list(map(str, instance_path.glob("*.mps.gz"))))
    for f in tqdm.tqdm(instance_files):
        # "batch" with one element
        instance_file_path = os.path.join(instance_path, f.replace(".mps.gz", ".pkl"))
        instance_batch = Batch.from_data_list([MilpBipartiteData.load_from_picklefile(instance_file_path)])

        mean_mu, configs = _predict_instance_for_all_configs(model, instance_batch)

        best_config = configs[torch.argmin(mean_mu)]
        foo = torch.argmin(mean_mu)

        get = lambda i: operator.itemgetter(i)
        df = pd.DataFrame(
            {
                "instance_file": f,
                "presolve_config_encoding": [best_config[0]],
                "heuristic_config_encoding": [best_config[1]],
                "separating_config_encoding": [best_config[2]],
                "emphasis_config_encoding": [best_config[3]],
            }
        )
        df_list.append(df)

    df = pd.concat(df_list)

    out_dir = os.path.join("/runs", f"run{args.run_id:03d}", "tasks_valid", f"gen_input{args.iter:04d}")
    os.makedirs(out_dir, exist_ok=True)

    chunk_size = 20
    for i, start in enumerate(range(0, df.shape[0], chunk_size)):
        df_subset = df.iloc[start : start + chunk_size]
        df_subset.to_csv(os.path.join(out_dir, f"task{i:02d}.csv"), index=False)


if __name__ == "__main__":
    main()
