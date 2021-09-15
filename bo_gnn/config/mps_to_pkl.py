import pickle
import pathlib
import sys
import tqdm

sys.path.append("../../common")

from environments import Configuring as Environment
import ecole as ec

if __name__ == "__main__":
    for f in ["train", "valid"]:
        instance_path = pathlib.Path(f"../../instances/1_item_placement/{f}")
        instance_files = list(map(str, instance_path.glob("*.mps.gz")))

        env = Environment(
            time_limit=10,
            observation_function=ec.observation.MilpBipartite(),
            reward_function=None,
            scip_params={"limits/memory": 1024},
        )

        for instance in tqdm.tqdm(instance_files):
            obs, _, _, _, _ = env.reset(instance)
            with open(instance.replace(".mps.gz", ".pkl"), "wb") as ofile:
                pickle.dump(obs, ofile)
