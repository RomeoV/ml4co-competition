import random
import time
import sys
import joblib as jl
import unittest
import csv
import os
import argparse
import tempfile
import subprocess

from filelock import FileLock
import ecole
import pathlib
import json
import ecole as ec
import numpy as np
from typing import List, Dict

sys.path.append("../../common")
from environments import Configuring

from environments import Configuring as Environment
from rewards import TimeLimitPrimalDualIntegral
from config_utils import sampleActions


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        "--num_instances",
        help="Number of instances to solve (consider putting multiple of cpus)",
        type=int,
    )
    parser.add_argument(
        "-j",
        "--num_jobs",
        help="Number of jobs at a time (e.g. 1 per thread).",
        type=int,
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
        default=5 * 60,
        type=int,
    )
    parser.add_argument("-d", "--dry_run", help="Dry run.", action="store_true")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    solve_random_instances(
        n_instances=args.num_instances,
        n_jobs=args.num_jobs,
        output_file=args.output_file,
        time_limit=args.time_limit,
        dry_run=args.dry_run,
    )


def solve_random_instances(
    n_instances, output_file, n_jobs=-1, time_limit=5 * 60, dry_run=True
):
    instance_path = pathlib.Path("../../instances/1_item_placement/train")
    instance_files = list(map(str, instance_path.glob("*.mps.gz")))

    paramfile = "parameters.pcs"

    instances = random.choices(instance_files, k=n_instances)
    actions = sampleActions(paramfile, n_samples=n_instances)

    if n_instances > jl.effective_n_jobs(n_jobs=n_jobs):

        def chunks(lst, n):
            return [lst[i : i + n] for i in range(0, len(lst), n)]

        instance_batches = chunks(instances, jl.effective_n_jobs(n_jobs=n_jobs))
        action_batches = chunks(actions, jl.effective_n_jobs(n_jobs=n_jobs))
    else:
        instance_batches = [instances]
        action_batches = [actions]

    for instance_batch, action_batch in zip(instance_batches, action_batches):
        results = jl.Parallel(n_jobs=n_jobs, verbose=100)(
            jl.delayed(solve_a_problem)(
                instance,
                config=action,
                time_limit=time_limit if not dry_run else 5,
                dry_run=dry_run,
            )
            for instance, action in zip(instance_batch, action_batch)
        )

        lock = FileLock(f"{output_file}.lck")
        with lock:
            output_file_new = not os.path.isfile(output_file)
            with open(output_file, "a") as ofile:
                writer = csv.DictWriter(ofile, fieldnames=results[0].keys())

                if output_file_new:
                    writer.writeheader()
                for r in results:
                    writer.writerow(r)


def solve_a_problem(
    instance_path, config: Dict = {}, time_limit: int = 20, dry_run: bool = False
):
    integral_function = TimeLimitPrimalDualIntegral()
    with open(instance_path.replace(".mps.gz", ".json")) as f:
        instance_info = json.load(f)

    integral_function.set_parameters(
        initial_primal_bound=instance_info["primal_bound"],
        initial_dual_bound=instance_info["dual_bound"],
        objective_offset=0,
    )

    env = Environment(
        time_limit=time_limit,
        observation_function=ec.observation.MilpBipartite(),
        reward_function=integral_function,
        scip_params={"limits/memory": 19 * 1024 if not dry_run else 2 * 1024},
    )
    obs, _, _, _, _ = env.reset(instance_path)
    _, _, reward, done, info = env.step(config)
    info.update(config)
    info.update(
        {
            "instance_file": pathlib.PosixPath(instance_path).name,
            "time_limit": time_limit,
            "initial_primal_bound": instance_info["primal_bound"],
            "initial_dual_bound": instance_info["dual_bound"],
            "time_limit_primal_dual_integral": reward,
        }
    )
    return info


if __name__ == "__main__":
    main()


class TestSolveProblem(unittest.TestCase):
    def setUp(self):
        self.instance_path = pathlib.Path("../../instances/1_item_placement/train")
        self.instance_files = list(map(str, self.instance_path.glob("*.mps.gz")))

        self.paramfile = "parameters.pcs"

        self.time_limit = 5
        self.keywords = [
            "instance_file",
            "time_limit",
            "time_limit_primal_dual_integral",
            "status",
            "solvingtime",
            "nnodes",
        ]

    def test_parallel_solve(self):
        retval = jl.Parallel(n_jobs=2)(
            jl.delayed(solve_a_problem)(
                s, config={}, time_limit=self.time_limit, dry_run=True
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
        N = 2
        actions = sampleActions(self.paramfile, n_samples=N)

        retval = jl.Parallel(n_jobs=2)(
            jl.delayed(solve_a_problem)(
                s, config=a, time_limit=self.time_limit, dry_run=True
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
        actions = sampleActions(self.paramfile, n_samples=1)
        actions[0]["foo"] = 1.0  # invalid parameter
        with self.assertRaises(ec.core.scip.Exception):
            solve_a_problem(
                self.instance_files[0],
                actions[0],
                time_limit=self.time_limit,
                dry_run=True,
            )

    def test_solve_random_instances(self):
        # Note that the context manager already creates the file, i.e. no header will be written
        with tempfile.NamedTemporaryFile() as tmpfile:
            solve_random_instances(
                n_instances=2,
                n_jobs=1,
                output_file=tmpfile.name,
                time_limit=5,
                dry_run=True,
            )
            loc = int(subprocess.check_output(["wc", "-l", tmpfile.name]).split()[0])
            self.assertEqual(loc, 2, msg=subprocess.check_output(["cat", tmpfile.name]))

        with tempfile.NamedTemporaryFile() as tmpfile:
            solve_random_instances(
                n_instances=1,
                n_jobs=2,
                output_file=tmpfile.name,
                time_limit=5,
                dry_run=True,
            )
            loc = int(subprocess.check_output(["wc", "-l", tmpfile.name]).split()[0])
            self.assertEqual(loc, 1)
