import os
import argparse
import random
import re
import pathlib
import torch
import operator
import time
import numpy as np
import pandas as pd
from data_utils.milp_data import MilpBipartiteData
from train_gnn import MilpGNNTrainable
from torch_geometric.data import Batch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--run_id",
        help="ID of current run (or experiment).",
        type=int,
    )
    parser.add_argument(
        "-t",
        "--num_tasks",
        help="Number of task files (one task per bsub).",
        type=int,
    )
    parser.add_argument(
        "-j",
        "--num_jobs",
        help="Number of jobs, i.e. problems per task.",
        type=int,
    )
    parser.add_argument(
        "-o",
        "--out_dir",
        help="Where to put the task files.",
        type=str,
    )

    return parser.parse_args()


def _get_latest_checkpoint_path(run_id):
    """Automatically get's the latest checkpoint called 'last.ckpt' from lightning logs. Returns 'None' if not available."""

    def sort_by_num(s):
        return re.search("[0-9]+", s).group(0)

    run_path = os.path.join("runs", f"run{run_id:03d}")
    if not os.path.isdir(os.path.join(run_path, "lightning_logs")):
        return None
    latest_version = sorted(os.listdir(os.path.join(run_path, "lightning_logs")), key=sort_by_num)[-1]
    checkpoint_path = os.path.join(run_path, "lightning_logs", latest_version, "checkpoints", "last.ckpt")
    return checkpoint_path


def _sample_random_instance_config_results(model, N):
    device = 'cuda' if torch.cuda.is_available else 'cpu'
    if N == 0:
        return (torch.tensor([]),)*4, ([], [])
    elif N > 64:
        (pred1, mu1, sig_ale1, sig_epi1), (random_inst1, random_conf1) = _sample_random_instance_config_results(model, 64)
        (pred2, mu2, sig_ale2, sig_epi2), (random_inst2, random_conf2) = _sample_random_instance_config_results(model, N-64)
        return (torch.cat((pred1, pred2), axis=0), torch.cat((mu1, mu2), axis=0), torch.cat((sig_ale1, sig_ale2), axis=0), torch.cat((sig_epi1, sig_epi2), axis=0)), (random_inst1 + random_inst2, random_conf1 + random_conf2)
    else:
        instance_files = list(map(str, pathlib.Path('/instances/1_item_placement/train').glob("*.pkl")))
        random_choice_of_instances = random.choices(instance_files, k=N)
        random_choice_of_configs = [(random.randint(0, 3), random.randint(0, 3), 1, 0) for _ in range(N)]

        instance_dir = "/instances/1_item_placement/train"
        instance_batch = Batch.from_data_list(
            [
                MilpBipartiteData.load_from_picklefile(
                    os.path.join(instance_dir, instance_file.replace(".mps.gz", ".pkl"))
                )
                for instance_file in random_choice_of_instances
            ]
        )

        config_batch = torch.stack(tuple(map(torch.tensor, random_choice_of_configs)), axis=0)

        model.eval()
        return [t.cpu().detach() for t in model.forward((instance_batch.to(device), config_batch.to(device)))], (random_choice_of_instances, random_choice_of_configs)


def _make_batches(full_data, batch_size):
    n_samples = full_data.shape[0]
    if n_samples > batch_size:
        def chunks(lst, n):
            return [lst[i : i + n] for i in range(0, len(lst), n)]

        batched_data = chunks(full_data, batch_size)
    else:
        batched_data = [full_list]

    return batched_data


def calibrate_epistemic_uncertainty(model):
    CALIBRATION_TARGET = 0.5

    (_predictions, _mean_mu, _mean_var, epi_var), _ = _sample_random_instance_config_results(model, 256)

    calibration_constant = CALIBRATION_TARGET / epi_var.mean()
    return calibration_constant


def main():
    t0 = time.time()
    args = parse_args()

    device = 'cuda' if torch.cuda.is_available else 'cpu'
    model = MilpGNNTrainable.load_from_checkpoint(_get_latest_checkpoint_path(args.run_id)).to(device)
    model.eval()

    print(f"At t={time.time()-t0} start to compute calibration.")
    lam = calibrate_epistemic_uncertainty(model)
    print(f"At t={time.time()-t0} end calibration.")

    for t in range(args.num_tasks):
        print(f"task {t}: At t={time.time()-t0} start predictions.")
        (_predictions, mean_mu, _mean_var, epi_var), (
            random_choice_of_instances,
            random_choice_of_configs,
        ) = _sample_random_instance_config_results(model, 512)
        print(f"task {t}: At t={time.time()-t0} end predictions.")

        print()
        best_indices = torch.argsort(mean_mu.flatten() - lam * epi_var)[: args.num_jobs]
        # we have to convert to np.array to be able to index with a tensor object
        chosen_instances = np.array(random_choice_of_instances)[best_indices]
        chosen_configs = np.array(random_choice_of_configs)[best_indices]

        df = pd.DataFrame(
            {
                "instance_file": chosen_instances,
                "presolve_config_encoding": map(operator.itemgetter(0), chosen_configs),
                "heuristic_config_encoding": map(operator.itemgetter(1), chosen_configs),
                "separating_config_encoding": map(operator.itemgetter(2), chosen_configs),
                "emphasis_config_encoding": map(operator.itemgetter(3), chosen_configs),
            }
        )
        df.to_csv(os.path.join(args.out_dir, f"task{t:02d}.csv"), index=False)


if __name__ == "__main__":
    main()
