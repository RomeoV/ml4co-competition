import argparse
import re
import os
import datetime
import subprocess
import torch
import pytorch_lightning

from pytorch_lightning import Trainer
from models.callbacks import EvaluatePredictedParametersCallback
from data_utils.dataset import MilpDataset, Folder, Problem, Mode
from torch_geometric.data import DataLoader

from models.trainable import MilpGNNTrainable


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
        "-t",
        "--max_time",
        help="Time after which training will be aborted in seconds.",
        default=None,
        type=float,
        required=False,
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    root_dir = os.path.join("/runs", f"run{args.run_id:03d}")
    problem = Problem.ONE
    # dry = subprocess.run(["hostname"], capture_output=True).stdout.decode()[:3] != "eu-"
    dry = False

    latest_checkpoint = _get_latest_checkpoint_path(args.run_id)
    if latest_checkpoint:
        model = MilpGNNTrainable.load_from_checkpoint(latest_checkpoint)
    else:
        model = MilpGNNTrainable(
            config_dim=4,
            optimizer="RMSprop",
            weight_decay=1e-3,
            initial_lr=5e-4,
            batch_size=64 if not dry else 4,
            n_gnn_layers=4,
            gnn_hidden_dim=64,
            ensemble_size=3,
            git_hash=_get_current_git_hash(),
            problem=problem,
        )
    data_train = DataLoader(
        MilpDataset(
            os.path.join(root_dir, "data_train"),
            folder=Folder.TRAIN,
            mode=Mode.TRAIN,
            problem=problem,
            dry=dry,
            instance_dir=f"{'../..' if dry else ''}/instances/{problem.value}/{Folder.TRAIN.value}",
        ),
        shuffle=True,
        batch_size=64 if not dry else 4,
        drop_last=False,
        num_workers=4 if not dry else 0,
        pin_memory=torch.cuda.is_available() and not dry,
    )
    configs_in_dataset = (
        data_train.dataset.csv_data.loc[
            :,
            [
                "presolve_config_encoding",
                "heuristic_config_encoding",
                "separating_config_encoding",
                "emphasis_config_encoding",
            ],
        ]
        .apply(tuple, axis=1)
        .unique()
    )

    data_valid = DataLoader(
        MilpDataset(
            os.path.join(root_dir, "data_train"),
            folder=Folder.TRAIN,
            mode=Mode.VALID,
            problem=problem,
            dry=dry,
            instance_dir=f"{'../..' if dry else ''}/instances/{problem.value}/{Folder.TRAIN.value}",
        ),
        shuffle=False,
        batch_size=64 if not dry else 4,
        drop_last=False,
        num_workers=4 if not dry else 0,
        pin_memory=torch.cuda.is_available() and not dry,
    )

    trainer = Trainer(
        gpus=1 if torch.cuda.is_available() else 0,
        callbacks=[
            EvaluatePredictedParametersCallback(
                configs=configs_in_dataset,
                instance_dir=f"{'../..' if dry else ''}/instances/{problem.value}/{Folder.TRAIN.value}",
            ),
            pytorch_lightning.callbacks.LearningRateMonitor(logging_interval="epoch"),
            pytorch_lightning.callbacks.ModelCheckpoint(save_last=True),
        ],
        default_root_dir=root_dir,
        max_time=datetime.timedelta(seconds=args.max_time) if args.max_time else None,
    )
    trainer.fit(model, train_dataloaders=data_train, val_dataloaders=data_valid)


def _get_latest_checkpoint_path(run_id):
    """Automatically get's the latest checkpoint called 'last.ckpt' from lightning logs. Returns 'None' if not available."""

    def sort_by_num(s):
        return re.search("[0-9]+", s).group(0)

    run_path = os.path.join("/runs", f"run{run_id:03d}")
    if not os.path.isdir(os.path.join(run_path, "lightning_logs")):
        return None
    latest_version = sorted(os.listdir(os.path.join(run_path, "lightning_logs")), key=sort_by_num)[-1]
    checkpoint_path = os.path.join(run_path, "lightning_logs", latest_version, "checkpoints", "last.ckpt")
    return checkpoint_path


def _get_current_git_hash():
    retval = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, check=True)
    git_hash = retval.stdout.decode("utf-8").strip()
    return git_hash


if __name__ == "__main__":
    main()
