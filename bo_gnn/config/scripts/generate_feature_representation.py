import csv
import os
import pathlib
import sys

import tqdm
from typing import Sequence

sys.path.append("../../common")

from environments import Configuring as Environment

class FeatureExtractionObservationFunction():
    def __init__(self):
        pass

    def seed(self, seed):
        # called before each episode
        # use this seed to make your code deterministic
        pass

    def before_reset(self, model):
        # called when a new episode is about to start
        pass

    def extract(self, model, done):
        m = model.as_pyscipopt()
        nvars = m.getNVars()
        nconss = m.getNConss()
        nintvars = m.getNIntVars()
        nbinvars = m.getNBinVars()
        return [nvars, nconss, nintvars, nbinvars]

if __name__ == "__main__":

    def _extract_features(instance: str) -> Sequence[int]:
        observation_funciton = FeatureExtractionObservationFunction()
        env = Environment(
            time_limit=10,
            observation_function=observation_funciton,
            reward_function=None,
            scip_params={"limits/memory": 1024},
        )
        obs, _, _, _, _ = env.reset(instance)
        return obs


    base_path = "../../instances/3_anonymous/"
    for f in ["train", "valid"]:
        instance_path = pathlib.Path(f"../../instances/3_anonymous/{f}")
        instance_files = list(map(str, instance_path.glob("*.mps.gz")))
        n_instances = len(instance_files)
        instance_features = []

        for i, instance in tqdm.tqdm(enumerate(instance_files)):
            instance_name = os.path.normpath(instance).split(os.path.sep)[-1]
            instance_features.append([instance_name] + _extract_features(instance))

        file_name = base_path + "features_" + f + ".csv"
        with open(file_name, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(instance_features)

