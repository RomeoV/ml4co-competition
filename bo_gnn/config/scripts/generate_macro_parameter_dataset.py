import argparse
import os
import json
import numpy as np
import itertools

import pandas as pd
import pyscipopt as pyopt
import ecole as ec


class MILPDataset:
    def __init__(self, statistics_dataset_path: str, nsweeps: int, output_path: str):
        self.statistics_dataset_path = statistics_dataset_path
        self.nsweeps = nsweeps

        self.parameters_dataset = []
        self.primal_dual_integral_dataset = []
        self.instance_file_dataset = []

        parameters_and_one_hot_vector_dict = {
            "OFF": np.array([1, 0, 0, 0]),
            "DEFAULT": np.array([0, 1, 0, 0]),
            "FAST": np.array([0, 0, 1, 0]),
            "AGGRESSIVE": np.array([0, 0, 0, 1]),
        }
        parameter_combinations = list(
            itertools.product(parameters_and_one_hot_vector_dict.keys(), repeat=3)
        )
        self.parameters_to_one_hot_vector_mapping = {}
        for i, (presolve, heuristic, separating) in enumerate(parameter_combinations):
            parameters_as_one_hot_vector = []
            parameters_as_one_hot_vector.append(
                parameters_and_one_hot_vector_dict[presolve]
            )
            parameters_as_one_hot_vector.append(
                parameters_and_one_hot_vector_dict[heuristic]
            )
            parameters_as_one_hot_vector.append(
                parameters_and_one_hot_vector_dict[separating]
            )
            self.parameters_to_one_hot_vector_mapping[i] = np.array(
                parameters_as_one_hot_vector
            ).flatten()

        for instance_name in os.listdir(self.statistics_dataset_path):
            if instance_name == ".DS_Store":
                continue
            instance_path = os.path.join(statistics_dataset_path, instance_name)
            assert os.path.isdir(instance_path), instance_path
            instances = os.listdir(instance_path)

            if ".DS_Store" in instances:
                instances.remove(".DS_Store")
            assert len(instances) == 64, len(instances)
            for configuration in instances:
                instance_configuration_path = os.path.join(instance_path, configuration)
                assert os.path.isdir(
                    instance_configuration_path
                ), instance_configuration_path
                primal_dual_integral_across_seeds = []
                for seed in os.listdir(instance_configuration_path):
                    if seed == ".DS_Store":
                        continue
                    instance_configuration_seed_sweep_path = os.path.join(
                        instance_configuration_path,
                        seed,
                        "nsweep-{}".format(self.nsweeps),
                    )
                    assert os.path.isdir(
                        instance_configuration_seed_sweep_path
                    ), instance_configuration_seed_sweep_path
                    with open(
                        os.path.join(
                            instance_configuration_seed_sweep_path, "stats.json"
                        )
                    ) as json_file:
                        stats = json.load(json_file)
                        primal_dual_integral_across_seeds.append(
                            stats["PrimalDualIntegral"]
                        )

                self.primal_dual_integral_dataset.append(
                    np.mean(primal_dual_integral_across_seeds)
                )
                parameter_setting = int(configuration.split("-")[1])
                self.parameters_dataset.append(
                    self.parameters_to_one_hot_vector_mapping[parameter_setting]
                )
                self.instance_file_dataset.append("{}.mps.gz".format(instance_name))

        dataset_info_pd = pd.DataFrame(
            {
                "instance_file": self.instance_file_dataset,
                "parameters": self.parameters_dataset,
                "primal_dual_integral": self.primal_dual_integral_dataset,
            }
        )
        dataset_info_pd.to_csv(output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--statistics_dataset_path",
        type=str,
        required=True,
        help="Dataset with statistics, i.e. dataset of MILP instances that have been run.",
    )
    parser.add_argument("-n", "--nsweeps", type=int, required=True)
    parser.add_argument("-o", "--output_path", type=str, required=True)
    arguments = parser.parse_args()

    dataset = MILPDataset(
        statistics_dataset_path=arguments.statistics_dataset_path,
        nsweeps=arguments.nsweeps,
        output_path=arguments.output_path,
    )


if __name__ == "__main__":
    main()
