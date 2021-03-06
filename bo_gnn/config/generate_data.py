import random
import sys
import pyscipopt as pyopt
import joblib as jl
import unittest
import csv
import os
import argparse
import tempfile
import subprocess
import itertools
import numpy as np

from filelock import FileLock
import pathlib
import json

from typing import List, Dict, Optional

sys.path.append("../../common")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        "--task_name",
        help="Task name",
        choices=("1_item_placement", "2_load_balancing", "3_anonymous"),
    )
    parser.add_argument(
        "-p",
        "--path_to_instances",
        help="Give path to instances ",
        default="../../instances/"
    )

    parser.add_argument(
        "-j",
        "--num_jobs",
        help="Number of jobs at a time (e.g. 1 per thread).",
        type=int,
    )
    parser.add_argument(
        "-f",
        "--folder",
        help="Instance folder to evaluate.",
        type=str,
        choices=("train", "valid"),
    )
    parser.add_argument(
        "-o",
        "--output_file",
        help="Output csv file",
        default="data/output.csv",
        type=str,
    )
    parser.add_argument(
        "-t",
        "--time_limit",
        help="Solver time limit (in seconds).",
        default=900,
        type=int,
    )
    parser.add_argument(
        "-s",
        "--start_instance_number",
        help="Run from what starting instance number",
        required=True,
        type=int,
    )
    parser.add_argument(
        "-e",
        "--end_instance_number",
        help="Run up to end instance number",
        required=True,
        type=int,
    )
    parser.add_argument(
        "-r",
        "--number_of_random_seeds",
        help="How many random seeds to use",
        default=1,
        type=int,
    )
    parser.add_argument(
        "-k",
        "--k_best_config_ids_from",
        help="Indicate the beginning of the range of the k best configs to run. Note this parameter can be set on its own.",
        type=int,
    )

    parser.add_argument(
        "-l",
        "--k_best_config_ids_to",
        help="Indicate the end of the range of the k best configs to run. Note this parameter can only be set in combination with k_best_config_ids_from",
        type=int,
    )

    parser.add_argument(
        "--run_selected_instances",
        help="If true then run specific instances that are defined in data_utils/{task}_instance_and_id_specification.json",
        dest="run_selected_instances",
        action='store_true'
    )

    parser.add_argument("-d", "--dry_run", help="Dry run.", action="store_true")
    parser.add_argument("--fixed_seed", help="Fix the random seeds.", action="store_true")
    args = parser.parse_args()
    return args

total_number_of_configs = 352

def main():
    args = parse_args()
    solve_instances_and_periodically_write_to_file(
        path_to_instances=args.path_to_instances,
        n_jobs=args.num_jobs,
        folder=args.folder,
        output_file=args.output_file,
        time_limit=args.time_limit,
        dry_run=args.dry_run,
        start_instance_number=args.start_instance_number,
        end_instance_number=args.end_instance_number,
        task_name=args.task_name,
        number_of_random_seeds=args.number_of_random_seeds,
        k_best_config_ids_from=args.k_best_config_ids_from,
        k_best_config_ids_to=args.k_best_config_ids_to,
        run_selected_instances=args.run_selected_instances,
	fixed_seed=args.fixed_seed
    )


def solve_instances_and_periodically_write_to_file(
    path_to_instances: str,
    output_file,
    folder,
    start_instance_number: int,
    end_instance_number: int,
    n_jobs=-1,
    time_limit=900,
    dry_run=True,
    task_name: str = "1_item_placement",
    number_of_random_seeds: int = 1,
    k_best_config_ids_from: Optional[int] = None,
    k_best_config_ids_to: Optional[int] = None,
    run_selected_instances: bool=False,
    fixed_seed: bool=False
):
    instance_path = pathlib.Path(f"{path_to_instances}/{task_name}/{folder}")

    if run_selected_instances:
        with open("data_utils/{}_instance_and_id_specification.json".format(task_name), "r") as file:
            instances_to_run = json.load(file)["selected_instances"]
        assert len(instances_to_run) > 0, len(instances_to_run)
        instance_paths = [os.path.join(instance_path, f"{task_name[2:]}_{instance_id}.mps.gz") for instance_id in
                         instances_to_run]
    else:
        instance_paths = [os.path.join(instance_path, f"{task_name[2:]}_{instance_id}.mps.gz") for instance_id in range(start_instance_number, end_instance_number)]


    number_of_instances = end_instance_number - start_instance_number

    all_instances = []
    all_actions = []
    if k_best_config_ids_from is not None:
        with open("data_utils/{}_instance_and_id_specification.json".format(task_name), "r") as file:
            selected_config_ids_from = json.load(file)["selected_config_ids"][str(k_best_config_ids_from)]
            assert len(selected_config_ids_from) > 0

    if k_best_config_ids_to is not None:
        assert k_best_config_ids_from is not None
        assert k_best_config_ids_from < k_best_config_ids_to
        with open("data_utils/{}_instance_and_id_specification.json".format(task_name), "r") as file:
            selected_config_ids_to = json.load(file)["selected_config_ids"][str(k_best_config_ids_to)]
            assert len(selected_config_ids_to) > 0

        selected_config_ids = [config_id for config_id in selected_config_ids_to if config_id not in selected_config_ids_from]
    else:
        selected_config_ids = selected_config_ids_from

    if k_best_config_ids_from is not None:
        for instance in instance_paths:
            number_of_configs = len(selected_config_ids)
            all_instances += [instance] * number_of_configs * number_of_random_seeds
            all_actions += [{'config_id': config_id, 'random_seed': random_seed if fixed_seed else np.random.randint(100000)} for config_id, random_seed in
                            itertools.product(selected_config_ids, range(number_of_random_seeds))]
    else:
        for instance in instance_paths:
            number_of_configs = total_number_of_configs
            all_instances += [instance] * number_of_configs * number_of_random_seeds
            all_actions += [{'config_id': config_id, 'random_seed': random_seed if fixed_seed else np.random.randint(100000)} for config_id, random_seed in
                            itertools.product(range(number_of_configs), range(number_of_random_seeds))]

    print("Running every instance with {} configs".format(number_of_configs))
    assert len(all_instances) == len(all_actions)

    # Batch problems so that we can write to file after each batch.
    # This is useful when we run a very large number of instances which might fail
    # late into the problem solving (e.g. running out of RAM).
    all_instances_batched = _to_batches(all_instances, n_instances=number_of_instances, n_jobs=n_jobs)
    all_actions_batched = _to_batches(all_actions, n_instances=number_of_instances, n_jobs=n_jobs)


    for instances, actions in zip(all_instances_batched, all_actions_batched):
        results = jl.Parallel(n_jobs=n_jobs, verbose=100)(
            jl.delayed(solve_a_problem)(
                instance,
                parameters=action,
                time_limit=time_limit if not dry_run else 5,
                dry_run=dry_run,
            )
            for instance, action in zip(instances, actions)
        )

        _write_results_to_csv(output_file, results)


def solve_a_problem(
    instance_path: str, parameters: Dict[str,int], time_limit: int = 900, dry_run: bool = False,
):
    config_id = parameters["config_id"]

    with open(instance_path.replace(".mps.gz", ".json")) as f:
        instance_info = json.load(f)
    print("Running instance {}".format(instance_path))

    model = pyopt.Model()
    model.readProblem(instance_path)
    model.readParams(f"meta_configs/config-{config_id}.set")
    model.setParam("limits/time", time_limit)
    model.setParam("limits/memory", 12*1024)
    model.setParam("randomization/randomseedshift", parameters["random_seed"])

    info = {}

    assert os.path.isfile("meta_configs/config_id_to_parameters.json")

    with open("meta_configs/config_id_to_parameters.json", "r") as file:
        config_ids_to_parameters = json.load(file)


    if not dry_run:
        model.hideOutput()
        model.optimize()

        info.update(
            {
                "presolve_config_encoding": config_ids_to_parameters[str(config_id)][0],
                "heuristic_config_encoding": config_ids_to_parameters[str(config_id)][1],
                "separating_config_encoding": config_ids_to_parameters[str(config_id)][2],
                "emphasis_config_encoding": config_ids_to_parameters[str(config_id)][3],
                "config_id": config_id,
                "instance_file": pathlib.PosixPath(instance_path).name,
                "time_limit": time_limit,
                "initial_primal_bound": instance_info["primal_bound"],
                "initial_dual_bound": instance_info["dual_bound"],
                "time_limit_primal_dual_integral": model.getPrimalDualIntegral(),
            }
        )
    info.update(
        {
            "presolve_config_encoding": config_ids_to_parameters[str(config_id)][0],
            "heuristic_config_encoding": config_ids_to_parameters[str(config_id)][1],
            "separating_config_encoding": config_ids_to_parameters[str(config_id)][2],
            "emphasis_config_encoding": config_ids_to_parameters[str(config_id)][3],
            "config_id": config_id,
            "instance_file": pathlib.PosixPath(instance_path).name,
            "time_limit": time_limit,
        }
    )
    return info


def _write_results_to_csv(
    output_file, results: List[Dict]
):
    # We use a lockfile so we can write to the file from multiple processes.
    lock = FileLock(f"{output_file}.lck")
    with lock:
        with open(output_file, "a") as ofile:
            writer = csv.DictWriter(ofile, fieldnames=results[0].keys())

            if ofile.tell() == 0:  # file.tell() -> "cursor position"
                writer.writeheader()
            for r in results:
                writer.writerow(r)

def _to_batches(full_list, n_instances, n_jobs):
    if n_instances > jl.effective_n_jobs(n_jobs=n_jobs):

        def chunks(lst, n):
            return [lst[i : i + n] for i in range(0, len(lst), n)]

        batched_list = chunks(full_list, jl.effective_n_jobs(n_jobs=n_jobs))
    else:
        batched_list = [full_list]

    return batched_list


if __name__ == "__main__":
    main()


class TestSolveProblem(unittest.TestCase):
    def setUp(self):
        self.instance_path = pathlib.Path("../../instances/1_item_placement/train")
        self.instance_files = list(map(str, self.instance_path.glob("*.mps.gz")))

        self.paramfile = "parameters.pcs"

        self.time_limit = 2
        self.keywords = [
            "instance_file",
            "time_limit",
            "time_limit_primal_dual_integral",
        ]

    def test_parallel_solve(self):
        retval = jl.Parallel(n_jobs=2)(
            jl.delayed(solve_a_problem)(
                s, parameters={"config_id": 0, "random_seed": 0}, time_limit=self.time_limit, dry_run=False
            )
            for s in random.sample(self.instance_files, 2)
        )
        self.assertIsInstance(retval, list)

        for d in retval:
            for k in self.keywords:
                self.assertIn(k, d.keys())
            self.assertIsInstance(d["time_limit"], int)
            self.assertIsInstance(d["time_limit_primal_dual_integral"], float)
            self.assertIsInstance(d["instance_file"], str)

    def test_parallel_solve_with_sampled_configs(self):
        actions = [{"config_id": 0, "random_seed": 0}, {"config_id": 1, "random_seed": 1}]

        retval = jl.Parallel(n_jobs=2)(
            jl.delayed(solve_a_problem)(
                s, parameters=a, time_limit=self.time_limit, dry_run=False
            )
            for s, a in zip(random.sample(self.instance_files, 2), actions)
        )
        self.assertIsInstance(retval, list)

        for d in retval:
            for k in self.keywords:
                self.assertIn(k, d.keys())
            self.assertIsInstance(d["time_limit"], int)
            self.assertIsInstance(d["time_limit_primal_dual_integral"], float)
            self.assertIsInstance(d["instance_file"], str)

    def test_solve_failure_with_invalid_action(self):
        actions = [{"config_id": 0}]# invalid parameter
        with self.assertRaises(KeyError):
            solve_a_problem(
                self.instance_files[0],
                actions[0],
                time_limit=self.time_limit,
                dry_run=True,
            )

    def test_solve_random_instances_and_periodically_write_to_file(self):
        # Note that the context manager already creates the file, i.e. no header will be written
        with tempfile.NamedTemporaryFile() as tmpfile:
            solve_instances_and_periodically_write_to_file(
                path_to_instances="../../instances/",
                output_file=tmpfile.name,
                n_jobs=1,
                folder="train",
                start_instance_number=0,
                end_instance_number=2,
                time_limit=5,
                dry_run=True,
            )
            loc = int(subprocess.check_output(["wc", "-l", tmpfile.name]).split()[0])
            self.assertEqual(loc, 129, msg=subprocess.check_output(["cat", tmpfile.name]))

        with tempfile.NamedTemporaryFile() as tmpfile:
            solve_instances_and_periodically_write_to_file(
                path_to_instances="../../instances/",
                output_file=tmpfile.name,
                n_jobs=1,
                folder="train",
                start_instance_number=0,
                end_instance_number=2,
                time_limit=5,
                dry_run=True,
            )
            loc = int(subprocess.check_output(["wc", "-l", tmpfile.name]).split()[0])
            self.assertEqual(loc, 129)


class TestUtilityFunctions(unittest.TestCase):
    def setUp(self):
        self.paramfile = "parameters.pcs"
