from typing import Union, List, Dict
import unittest

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH


def sampleActions(paramfile, n_samples, seed=1) -> List[Dict]:
    """Loads paramfile (in .pcs format) and returns list of samples as dict."""

    hparams = getParamsFromFile(paramfile)
    cs = CS.ConfigurationSpace(seed=seed)
    cs.add_hyperparameters(hparams)

    configs = cs.sample_configuration(n_samples)
    actions = cleanConfigs(configs)

    return actions


def getParamsFromFile(paramfile) -> List[CSH.Hyperparameter]:
    """ Parses .pcs format and returns list of hyperarameters.

    Adapted from https://github.com/ds4dm/ml4co-competition/blob/main/baseline/config/run_training.py.
    """

    with open(paramfile, "r") as pfile:
        params = []
        for line in pfile:
            line = line.strip().split(" ")

            paramname = line[0]
            paramtype = line[1]
            default = line[3][1:-1]

            if paramtype == "categorical":
                params += [
                    CSH.CategoricalHyperparameter(
                        paramname,
                        choices=line[2][1:-1].split(","),
                        default_value=default,
                    )
                ]
            else:
                bounds = line[2][1:-1].split(",")

                if paramtype == "real":
                    if float(default) > 1e100:
                        continue
                    # SMAC cannot handle too large limits, so change them to something smaller
                    if float(bounds[0]) <= -1e100:
                        bounds[0] = -1e100
                    if float(bounds[1]) >= 1e100:
                        bounds[1] = 1e100
                    params += [
                        CSH.UniformFloatHyperparameter(
                            paramname,
                            float(bounds[0]),
                            float(bounds[1]),
                            default_value=float(default),
                        )
                    ]

                elif paramtype == "integer":
                    params += [
                        CSH.UniformIntegerHyperparameter(
                            paramname,
                            lower=max(int(bounds[0]), 1),
                            upper=int(bounds[1]),
                            default_value=int(default),
                            log=True,
                        )
                    ]

    return params


def conf2tensor(conf: List):
    pass


def cleanConfigs(
    configs: Union[List, CS.configuration_space.Configuration]
) -> List[Dict]:
    """Converts Configuration to Dict and makes bools actually bool type."""

    if isinstance(configs, CS.configuration_space.Configuration):
        configs = [configs]
    configs = list(map(dict, configs))

    for conf in configs:
        for key in conf:  # replace 'TRUE' with True and vice versa
            if conf[key] in {"TRUE", "FALSE"}:
                conf[key] = conf[key] == "TRUE"

    return configs


class TestConfigLoader(unittest.TestCase):
    def setUp(self):
        self.some_config_parameters = [
            "branching/scorefac",
            "branching/preferbinary",
            "separating/maxcuts",
        ]

        self.param_file = "parameters.pcs"

    def test_one_sample(self):
        actions = sampleActions(self.param_file, n_samples=1)

        for p in self.some_config_parameters:
            self.assertIn(p, actions[0], msg=actions[0].keys())
        self.assertIsInstance(actions[0]["branching/preferbinary"], bool)
        self.assertIsInstance(actions[0]["branching/scorefac"], float)
        self.assertIsInstance(actions[0]["separating/maxcuts"], int)

    def test_multiple_samples(self):
        actions = sampleActions(self.param_file, n_samples=10)

        for a in actions:
            for p in self.some_config_parameters:
                self.assertIn(p, a, msg=a.keys())
            self.assertIsInstance(a["branching/preferbinary"], bool)
            self.assertIsInstance(a["branching/scorefac"], float)
            self.assertIsInstance(a["separating/maxcuts"], int)
