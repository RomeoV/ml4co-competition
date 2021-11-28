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
import pandas as pd
import pyscipopt
from collections import OrderedDict

from filelock import FileLock
import ecole
import pathlib
import json
import ecole as ec
import numpy as np
from typing import List, Dict, Callable, Tuple

sys.path.append("../../common")
from environments import Configuring

from environments import Configuring as Environment
from rewards import TimeLimitPrimalDualIntegral
from config_utils import sampleActions, getParamsFromFile
from data_utils.dataset import Problem


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
        required=True
    )
    parser.add_argument(
        "-t",
        "--task_num",
        help="CSV file with tasks to solve",
        type=int,
        required=True,
    )
    parser.add_argument(
        "-j",
        "--num_jobs",
        help="Number of jobs at a time (e.g. 1 per thread).",
        type=int,
        default=-1,
        required=False,
    )
    parser.add_argument(
        "-f",
        "--folder",
        help="Instance folder to evaluate.",
        type=str,
        choices=("train", "valid"),
        default="train",
        required=False,
    )
    parser.add_argument(
        "-p",
        "--problem",
        help="Problem folder to evaluate.",
        type=str,
        choices=("one", "two"),
        required=True,
    )
    parser.add_argument(
        "-T",
        "--time_limit",
        help="Solver time limit (in seconds).",
        type=float,
        required=True,
    )
    parser.add_argument("-d", "--dry_run", help="Dry run.", action="store_true")

    args = parser.parse_args()
    args.problem = Problem.ONE if args.problem == "one" else Problem.TWO
    return args


def main():
    args = parse_args()

    task_file = os.path.join("/runs", f"run{args.run_id:03d}", "tasks", f"gen_input{args.iter:04d}", f"task{args.task_num:02d}.csv")
    task_df = pd.read_csv(task_file)
    os.makedirs(os.path.join("/runs", f"run{args.run_id:03d}", "data"), exist_ok=True)
    output_file = os.path.join("/runs", f"run{args.run_id:03d}", "data", f"dataset_iter{args.iter:04d}.csv")

    solve_random_instances_and_periodically_write_to_file(
        task_df=task_df,
        n_jobs=args.num_jobs,
        problem=args.problem,
        folder=args.folder,
        output_file=output_file,
        time_limit=args.time_limit,
        dry_run=args.dry_run,
    )


def solve_random_instances_and_periodically_write_to_file(
        task_df, output_file, problem, folder, n_jobs=-1, time_limit=5 * 60, dry_run=True
):
    instance_path = pathlib.Path(f"/instances/{problem.value}/{folder}")
    instance_files = list(map(str, instance_path.glob("*.mps.gz")))

    instances = [os.path.join(instance_path, f) for f in task_df.instance_file]
    actions = list(
        task_df.loc[
            :,
            [
                "presolve_config_encoding",
                "heuristic_config_encoding",
                "separating_config_encoding",
                "emphasis_config_encoding",
            ],
        ].apply(tuple, axis=1)
    )

    # Batch problems so that we can write to file after each batch.
    # This is useful when we run a very large number of instances which might fail
    # late into the problem solving (e.g. running out of RAM).
    instance_batches = _to_batches(instances, n_jobs=n_jobs)
    action_batches = _to_batches(actions, n_jobs=n_jobs)

    for instance_batch, action_batch in zip(instance_batches, action_batches):
        results = jl.Parallel(n_jobs=n_jobs, verbose=100)(
            jl.delayed(solve_a_problem)(
                instance,
                config_encoding=action,
                time_limit=time_limit if not dry_run else 5,
                dry_run=dry_run,
            )
            for instance, action in zip(instance_batch, action_batch)
        )

        _write_results_to_csv(output_file, results)


def solve_a_problem(instance_path, config_encoding: Tuple[int, int, int, int], time_limit: int, dry_run: bool = False):
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
        scip_params={"limits/memory": 11 * 1024 if not dry_run else 2 * 1024},
    )
    obs, _, _, _, _ = env.reset(instance_path)

    config = _config_encoding_to_dict(config_encoding)
    _, _, reward, done, info = env.step(config)
    result = {
        "instance_file": pathlib.Path(instance_path).name,
        "presolve_config_encoding": config_encoding[0],
        "heuristic_config_encoding": config_encoding[1],
        "separating_config_encoding": config_encoding[2],
        "emphasis_config_encoding": config_encoding[3],
        "time_limit_primal_dual_integral": reward,
    }
    return result


def _write_results_to_csv(output_file, results: List[Dict], fmt_fcts: List[Callable] = []):
    # We use a lockfile so we can write to the file from multiple processes.
    lock = FileLock(f"{output_file}.lck")
    with lock:
        with open(output_file, "a") as ofile:
            writer = csv.DictWriter(ofile, fieldnames=results[0].keys())

            if ofile.tell() == 0:  # file.tell() -> "cursor position"
                writer.writeheader()
            for r in results:
                writer.writerow(r)


def _config_encoding_to_dict(config_encoding):
    presolve, heuristic, separating, emphasis = config_encoding

    SETTINGS = OrderedDict(
        {
            0: pyscipopt.SCIP_PARAMSETTING.OFF,
            1: pyscipopt.SCIP_PARAMSETTING.DEFAULT,
            2: pyscipopt.SCIP_PARAMSETTING.FAST,
            3: pyscipopt.SCIP_PARAMSETTING.AGGRESSIVE,
        }
    )
    EMPHASIS_SETTINGS = OrderedDict(
        {
            0: pyscipopt.SCIP_PARAMEMPHASIS.DEFAULT,
            1: pyscipopt.SCIP_PARAMEMPHASIS.CPSOLVER,
            2: pyscipopt.SCIP_PARAMEMPHASIS.EASYCIP,
            3: pyscipopt.SCIP_PARAMEMPHASIS.FEASIBILITY,
            4: pyscipopt.SCIP_PARAMEMPHASIS.HARDLP,
            5: pyscipopt.SCIP_PARAMEMPHASIS.OPTIMALITY,
            6: pyscipopt.SCIP_PARAMEMPHASIS.COUNTER,
            7: pyscipopt.SCIP_PARAMEMPHASIS.PHASEFEAS,
            8: pyscipopt.SCIP_PARAMEMPHASIS.PHASEIMPROVE,
            9: pyscipopt.SCIP_PARAMEMPHASIS.PHASEPROOF,
            # 10: pyscipopt.SCIP_PARAMEMPHASIS.NUMERICS,
        }
    )

    model = pyscipopt.Model()
    # we pick 'emphasis' first as it overrides a lot of other options
    if emphasis != 0:  # don't do anything in default case
        model.setEmphasis(EMPHASIS_SETTINGS[emphasis])
    if presolve != 1:
        model.setPresolve(SETTINGS[presolve])
    if heuristic != 1:
        model.setHeuristics(SETTINGS[heuristic])
    if separating != 1:
        model.setSeparating(SETTINGS[separating])

    return _clean_config(model.getParams())


def _clean_config(config_dict):
    config_dict.pop("limits/time", None)
    config_dict.pop("limits/memory", None)
    config_dict.pop("timing/enabled", None)
    config_dict.pop("timing/reading", None)
    config_dict.pop("timing/rareclockcheck", None)
    config_dict.pop("timing/statistictiming", None)
    config_dict.pop("timing/clocktype", None)
    return config_dict


def _makeMapCategoricalsToNumericalLevels(paramfile):
    params = getParamsFromFile(paramfile)

    def mapCategoricalsToNumericalLevels(results: Dict):
        for p in filter(lambda p: isinstance(p, CS.CategoricalHyperparameter), params):
            if p.name in results and isinstance(results[p.name], str):
                results[f"{p.name}_cat"] = results[p.name]
                results[p.name] = p.choices.index(results[p.name])
        return results

    return mapCategoricalsToNumericalLevels


def _to_batches(full_list, n_jobs):
    n_instances = len(full_list)
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
    def test_two_tasks(self):
        tasks = pd.DataFrame(
            {
                "instance_file": ["item_placement_1.mps.gz", "item_placement_1.mps.gz"],
                "presolve_config_encoding": [0, 1],
                "heuristic_config_encoding": [0, 1],
                "separating_config_encoding": [0, 1],
                "emphasis_config_encoding": [0, 0],
            }
        )

        with tempfile.NamedTemporaryFile() as out_file:
            solve_random_instances_and_periodically_write_to_file(
                task_df=tasks,
                n_jobs=2,
                folder="train",
                output_file=out_file.name,
                time_limit=10,
                dry_run=False,
            )

            output_df = pd.read_csv(out_file.name)
            print(output_df.time_limit_primal_dual_integral)

            assert (tasks.loc[:, tasks.columns] == output_df.loc[:, tasks.columns]).all().all()


class TestSolveProblem_old(unittest.TestCase):
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
            jl.delayed(solve_a_problem)(s, config={}, time_limit=self.time_limit, dry_run=True)
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
            jl.delayed(solve_a_problem)(s, config=a, time_limit=self.time_limit, dry_run=True)
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

    def test_solve_random_instances_and_periodically_write_to_file(self):
        # Note that the context manager already creates the file, i.e. no header will be written
        with tempfile.NamedTemporaryFile() as tmpfile:
            solve_random_instances_and_periodically_write_to_file(
                n_instances=2,
                n_jobs=1,
                folder="train",
                output_file=tmpfile.name,
                time_limit=5,
                dry_run=True,
            )
            loc = int(subprocess.check_output(["wc", "-l", tmpfile.name]).split()[0])
            self.assertEqual(loc, 3, msg=subprocess.check_output(["cat", tmpfile.name]))

        with tempfile.NamedTemporaryFile() as tmpfile:
            solve_random_instances_and_periodically_write_to_file(
                n_instances=1,
                n_jobs=2,
                folder="train",
                output_file=tmpfile.name,
                time_limit=5,
                dry_run=True,
            )
            loc = int(subprocess.check_output(["wc", "-l", tmpfile.name]).split()[0])
            self.assertEqual(loc, 2)

    def test_categorical_to_numerical(self):
        import pandas as pd

        with tempfile.NamedTemporaryFile(mode="w") as tmpfile:
            solve_random_instances_and_periodically_write_to_file(
                n_instances=2,
                n_jobs=2,
                folder="train",
                output_file=tmpfile.name,
                time_limit=5,
                dry_run=True,
            )

            df = pd.read_csv(tmpfile.name)  # we use pandas to avoid type parsing problems
            self.assertIn("branching/scorefunc", df.columns)
            self.assertIn("branching/scorefunc_cat", df.columns)
            self.assertEqual(df["branching/scorefunc"].dtype, int)
            self.assertEqual(df["branching/scorefunc_cat"].dtype, object)  # aka string


class TestUtilityFunctions(unittest.TestCase):
    def setUp(self):
        self.paramfile = "parameters.pcs"

    def test_categorical_to_numerical_formatter(self):
        d = {"branching/scorefunc": "q"}
        fmt_fnc = _makeMapCategoricalsToNumericalLevels(self.paramfile)
        d = fmt_fnc(d)

        self.assertIn("branching/scorefunc", d)
        self.assertIn("branching/scorefunc_cat", d)
        self.assertIsInstance(d["branching/scorefunc"], int, msg=d)
        self.assertIsInstance(d["branching/scorefunc_cat"], str, msg=d)
