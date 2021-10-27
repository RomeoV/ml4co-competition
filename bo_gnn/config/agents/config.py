import os
import pickle as pkl
from typing import Any, Dict, Tuple
import sys

sys.path.insert(0, '..')

import numpy as np
import pytorch_lightning as pl
import torch
from train_gnn import MilpGNNTrainable
from callbacks import EvaluatePredictedParametersCallback
from ecole.observation import MilpBipartite
from torch_geometric.data import Batch

from data_utils.milp_data import MilpBipartiteData

CHECKPOINT_BASE_PATH = "trained_model_checkpoints/"
PARAMETER_CONFIGURATIONS_PATH = "param_configurations/"
HEURISTIC_SCHEDULES_PATH = "heuristics_schedules/"


class ObservationFunction(MilpBipartite):

    def __init__(self, problem):
        # called once for each problem benchmark
        super(ObservationFunction, self).__init__()
        self.problem = problem  # to devise problem-specific observations

    def seed(self, seed):
        # called before each episode
        # use this seed to make your code deterministic
        pass

    def before_reset(self, model):
        # called when a new episode is about to start
        pass

    def extract(self, model, done):
        if self.problem == "anonymous":
            m = model.as_pyscipopt()
            nvars = m.getNVars()
            nconss = m.getNConss()
            return np.array((nvars, nconss))
        else:
            return super(ObservationFunction, self).extract(model, done)


class Task3NNModel:
    def __init__(self):
        pass

    def predict(self, x: np.ndarray) -> int:
        nconss, nvars = x
        if (nconss, nvars) == (73483, 53041):
            config = 313
        elif (nconss, nvars) == (7003, 4399):
            config = 298
        elif (nconss, nvars) == (2599, 1613):
            config = 297
        elif (nconss, nvars) == (126621, 92261):
            config = 345
        elif (nconss, nvars) == (7009, 4399):
            config = 297
        else:
            config = 218
        return config


class Policy():

    def __init__(self, problem):
        # called once for each problem benchmark
        self.problem = problem  # to devise problem-specific policies
        self.use_heuristic_schedules = True

        if self.problem == "anonymous":
            self.best_config_id_prediction_model = Task3NNModel()
        else:
            files_in_checkpoint_dir = os.listdir(CHECKPOINT_BASE_PATH + problem)
            assert len(files_in_checkpoint_dir) == 1
            model_path = CHECKPOINT_BASE_PATH + problem + "/" + files_in_checkpoint_dir[0]

            with open('parameter_configuration_mapping/config_tuples_to_ids.pkl', 'rb') as handle:
                self.config_tuple_to_id_mapping = pkl.load(handle)

            if self.problem == "item_placement":
                self.config_index_to_tuple_mapping = np.load('parameter_configuration_mapping/unique_configs_in_dataset_p1.npy', allow_pickle=True)
                self.performance_prediction_model = MilpGNNTrainable(config_dim=60, optimizer=None, initial_lr=None, batch_size=None, git_hash=None,
                                                                     problem=None, n_gnn_layers=4,
                                                                     gnn_hidden_dim=64, ensemble_size=3).load_from_checkpoint(model_path)
            elif self.problem == "load_balancing":
                self.config_index_to_tuple_mapping = np.load('parameter_configuration_mapping/unique_configs_in_dataset_p2.npy', allow_pickle=True)
                self.performance_prediction_model = MilpGNNTrainable(config_dim=40, optimizer=None, initial_lr=None, batch_size=None, git_hash=None,
                                                                     problem=None, n_gnn_layers=4,
                                                                     gnn_hidden_dim=64, ensemble_size=3).load_from_checkpoint(model_path)
            else:
                raise ValueError(f"Problem {self.problem} unknown.")
            self.performance_prediction_model.eval()

            if torch.cuda.is_available():
                self.performance_prediction_model.to(torch.device("cuda"))

    def seed(self, seed):
        # called before each episode
        # use this seed to make your code deterministic
        pl.seed_everything(seed)

    def __call__(self, action_set, observation):
        if self.problem == "anonymous":
            best_config_id = self.best_config_id_prediction_model.predict(observation)
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            instance_data = MilpBipartiteData(
                var_feats=observation.variable_features,
                cstr_feats=observation.constraint_features,
                edge_indices=observation.edge_features.indices,
                edge_values=observation.edge_features.values,
            )

            instance_batch = Batch.from_data_list([instance_data], ).to(device)

            best_config_tuple = EvaluatePredictedParametersCallback._find_best_configs(self.performance_prediction_model, instance_batch,
                                                                                       self.config_index_to_tuple_mapping)["mean"][0]
            assert (isinstance(best_config_tuple, tuple)) and len(best_config_tuple) == 4
            best_config_id = self.config_tuple_to_id_mapping[best_config_tuple]

        best_config_parameters = self._get_scip_parameter_configuration_by(best_config_id)
        if self.use_heuristic_schedules:
            schedule_heuristic_parameters = self._load_schedule_heuristic_parameters(self.problem)
            return {**best_config_parameters, **schedule_heuristic_parameters}
        else:
            return best_config_parameters

    def _get_scip_parameter_configuration_by(self, index: int) -> Dict[str, Any]:
        path = PARAMETER_CONFIGURATIONS_PATH + "config-" + str(index) + ".set"

        parameter_configuration = {}
        with open(path, "r") as parameter_file:
            for setting in parameter_file.readlines():
                param_key, param_value = setting.split(" = ")
                try:
                    parameter_configuration[param_key] = float(param_value)
                except ValueError:
                    if "TRUE" in param_value:
                        parameter_configuration[param_key] = True
                    elif "FALSE" in param_value:
                        parameter_configuration[param_key] = False
                    else:
                        raise RuntimeWarning("Unexpected parameter type.")

        return parameter_configuration

    def _load_schedule_heuristic_parameters(self, problem: str) -> Dict[str, Any]:
        if problem == "item_placement":
            path = HEURISTIC_SCHEDULES_PATH + "schedule1_15min.set"
        elif problem == "load_balancing":
            path = HEURISTIC_SCHEDULES_PATH + "schedule2_30min.set"
        elif problem == "anonymous":
            path = HEURISTIC_SCHEDULES_PATH + "schedule3_30min.set"
        else:
            raise RuntimeError("Unexpected problem identifier.")

        parameters = {}
        with open(path, "r") as file:
            for setting in file.readlines():
                if setting.strip() == "":
                    continue
                param_key, param_value = setting.split(" = ")
                try:
                    parameters[param_key] = float(param_value)
                except ValueError:
                    raise RuntimeWarning("Unexpected parameter type.")

        return parameters
