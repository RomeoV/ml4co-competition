import argparse
import os
import re
import json
import numpy as np
import itertools

import pandas as pd
import pyscipopt as pyopt
import ecole as ec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset_path",
        type=str,
        required=True,
        help="Dataset with statistics, i.e. dataset of MILP instances that have been run.",
    )
    parser.add_argument("-o", "--output_path", type=str, required=True)
    args = parser.parse_args()

    dataset_as_df = sweep_directories_to_dataframe(
        dataset_path=args.dataset_path,
        output_path=args.output_path,
    )

    dataset_as_df.to_csv(args.output_path)


def sweep_directories_to_dataframe(dataset_path: str, output_path: str):
    dataset_path = dataset_path

    instance_file_dataset = []
    config_vector_dataset = []
    primal_dual_integral_dataset = []

    config_encodings = _build_config_encodings()

    path_tree = os.walk(dataset_path, topdown=False)  # only 'leaf' directories
    path_regex = re.compile(
        ".*item_placement_(\d+)/config-(\d+)/seed-(\d+)/nsweep-(\d+).*"
    )

    for dir_path, _, files in path_tree:
        if match := path_regex.fullmatch(dir_path):
            instance_name, config_id, seed, nsweep = match.groups()

            with open(os.path.join(dir_path, "stats.json")) as json_file:
                stats = json.load(json_file)
                primal_dual_integral = stats["PrimalDualIntegral"]

            instance_file_dataset.append(f"{instance_name}.mps.gz")
            config_vector_dataset.append(config_encodings[int(config_id)])
            primal_dual_integral_dataset.append(primal_dual_integral)

    dataset_info_pd = pd.DataFrame(
        {
            "instance_file": instance_file_dataset,
            "parameters": config_vector_dataset,
            "primal_dual_integral": primal_dual_integral_dataset,
        }
    )

    return dataset_info_pd


def _build_config_encodings():
    param_to_one_hot = {
        "OFF": np.array([1, 0, 0, 0]),
        "DEFAULT": np.array([0, 1, 0, 0]),
        "FAST": np.array([0, 0, 1, 0]),
        "AGGRESSIVE": np.array([0, 0, 0, 1]),
    }
    parameter_combinations = itertools.product(
        param_to_one_hot.keys(), repeat=3
    )  # e.g. ["DEFAULT", "AGGRESSIVE", "OFF"] for presolve, heuristic, separating

    config_encodings = [
        np.concatenate([param_to_one_hot[p] for p in comb])
        for comb in parameter_combinations
    ]
    assert len(config_encodings) == 4 * 4 * 4
    assert config_encodings[0].size == 3 * 4

    return config_encodings


if __name__ == "__main__":
    main()
