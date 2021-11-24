import sys
import pathlib
import os
import pickle
import pandas
import numpy as np
from dataclasses import make_dataclass
from train_gnn import MilpGNNTrainable
from models.callbacks import EvaluatePredictedParametersCallback
from data_utils.dataset import MilpDataset, Folder, Problem, Mode
from data_utils.milp_data import MilpBipartiteData
from data_utils.dataset import Problem
from torch_geometric.data import Batch

def _prepare_df(df):
    cols = [
        "presolve_config_encoding",
        "heuristic_config_encoding",
        "separating_config_encoding",
        "emphasis_config_encoding",
    ]
    df["config_encoding"] = df.loc[:, cols].apply(tuple, axis=1)

    return df

def _load_instance(instance_file, problem):
    instance_path = f"/instances/{problem.value}/valid"
    with open(
        os.path.join(instance_path, instance_file.replace(".mps.gz", ".pkl")),
        "rb",
    ) as infile:
        instance_description_pkl = pickle.load(infile)
        instance_graph = MilpBipartiteData(
            var_feats=instance_description_pkl.variable_features,
            cstr_feats=instance_description_pkl.constraint_features,
            edge_indices=instance_description_pkl.edge_features.indices,
            edge_values=instance_description_pkl.edge_features.values,
        )
    return instance_graph

if sys.argv[1].lower() == "one":
    problem = Problem.ONE
elif sys.argv[1].lower() == "two":
    problem = Problem.TWO
else:
    "Please write 'one' or 'two'"

# load model
checkpoint_path = pathlib.Path(f"trained_model_checkpoints/{problem.value[2:]}/ckpt_problem_{1 if problem == Problem.ONE else 2}.ckpt")
model = MilpGNNTrainable.load_from_checkpoint(checkpoint_path).cpu()

# load chris' dataset
chris_df = pandas.read_csv(f"data/exhaustive_dataset_all_configs/{problem.value}_results_validation_filtered.csv")
chris_df = _prepare_df(chris_df)
unique_configs_in_dataset = np.sort(chris_df.config_encoding.unique())
# load default dataset
default_df = pandas.read_csv(f"data/{problem.value}_validation_with_default.csv")
default_df = _prepare_df(default_df)

Task = make_dataclass("Task", [("instance_file", str), ("presolve_config_encoding", int), ("heuristic_config_encoding", int), ("separating_config_encoding", int), ("emphasis_config_encoding", int)])
tasks = []

for instance_file in default_df.instance_file.unique():
    # get predicted config
    instance_batch = Batch.from_data_list([_load_instance(instance_file, problem)])
    best_config = EvaluatePredictedParametersCallback._find_best_configs(model, instance_batch, unique_configs_in_dataset)["mean"][0]
    tasks += [Task(instance_file, *best_config)]
    # get performance of predicted config
    best_config_performance = chris_df.groupby("config_encoding").aggregate(np.random.choice).time_limit_primal_dual_integral[best_config]
    # get performance of default config
    default_config_performance = default_df.groupby("config_encoding").aggregate(np.random.choice).time_limit_primal_dual_integral[(0,0,0,0)]
    # compute performance improvement
    rel_improvement =  (best_config_performance / default_config_performance) - 1
    print(rel_improvement)
    # store in dataframe

pandas.DataFrame(tasks).to_csv(f"{problem.value}_best_config_tasks.csv", index=False)
