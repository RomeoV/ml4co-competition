import pickle
import pathlib
import sys
import tqdm
from joblib import delayed, Parallel

sys.path.append("../../common")

from environments import Configuring as Environment
import ecole as ec

if __name__ == "__main__":

    def _process_instance(instance):

        env = Environment(
            time_limit=10,
            observation_function=ec.observation.MilpBipartite(),
            reward_function=None,
            scip_params={"limits/memory": 1024},
        )
        obs, _, _, _, _ = env.reset(instance)
        with open(instance.replace(".mps.gz", ".pkl"), "wb") as ofile:
            pickle.dump(obs, ofile)

    for f in ["train", "valid"]:
        instance_path = pathlib.Path(f"../../instances/2_load_balancing/{f}")
        instance_files = list(map(str, instance_path.glob("*.mps.gz")))

        Parallel(n_jobs=-2)(
            delayed(_process_instance)(instance)
            for instance in tqdm.tqdm(instance_files)
        )
