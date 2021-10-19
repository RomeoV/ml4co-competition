import os
from typing import Any, Dict
import sys

sys.path.insert(0,'..')

import pytorch_lightning as pl
import torch
from train_gnn import MilpGNNTrainable
from ecole.observation import MilpBipartite
from torch import Tensor
from torch_geometric.data import Batch

from data_utils.milp_data import MilpBipartiteData

CHECKPOINT_BASE_PATH = "trained_model_checkpoints/"
PARAMETER_CONFIGURATIONS_PATH = "param_configurations/"

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
        return super(ObservationFunction, self).extract(model, done)

class Policy():

    def __init__(self, problem):
        # called once for each problem benchmark
        self.problem = problem  # to devise problem-specific policies
        files_in_checkpoint_dir = os.listdir(CHECKPOINT_BASE_PATH + problem)
        assert len(files_in_checkpoint_dir) == 1
        model_path = CHECKPOINT_BASE_PATH + problem + "/" + files_in_checkpoint_dir[0]
        self.performance_prediction_model = MilpGNNTrainable.load_from_checkpoint(model_path)
        self.performance_prediction_model.eval()
        if torch.cuda.is_available():
            self.performance_prediction_model.to(torch.device("cuda"))

    def seed(self, seed):
        # called before each episode
        # use this seed to make your code deterministic
        pl.seed_everything(seed)

    def __call__(self, action_set, observation):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        graph = observation

        instance_data = MilpBipartiteData(
            var_feats=graph.variable_features,
            cstr_feats=graph.constraint_features,
            edge_indices=graph.edge_features.indices,
            edge_values=graph.edge_features.values,
        )

        config_batch = self._generate_config_data().to(device)
        instance_batch = Batch.from_data_list([instance_data],).to(device)

        predictions, mean_mu, mean_var, epi_var = self.performance_prediction_model((instance_batch, config_batch),
                                                                                    single_instance=True)

        lowest_mean_index = int(torch.argmin(mean_mu))
        return self._get_scip_parameter_configuration_by(lowest_mean_index)

    def _generate_config_data(self) -> Tensor:
        presolve_config_options = torch.tensor([0., 1., 2., 3.])
        heuristic_config_options = torch.tensor([0., 1., 2., 3.])
        separating_config_options = torch.tensor([0., 1., 2., 3.])
        return torch.cartesian_prod(presolve_config_options,
                                                   heuristic_config_options,
                                                   separating_config_options)

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