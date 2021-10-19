import pickle as pkl

import pandas as pd

if __name__ == "__main__":
    base_path = "../data/3_anonymous/"
    performance_data = pd.read_csv(base_path + "3_anonymous_results_118.csv")
    save_path = base_path + "best_config_id_by_instance.pkl"

    best_config_id_per_instance = {}
    for instance in performance_data.instance_file.unique():
        performance_data_for_instance = performance_data[performance_data.instance_file == instance]
        row_index_of_best_performance = performance_data_for_instance.time_limit_primal_dual_integral.argmin()
        best_config_id = performance_data_for_instance.config_id.iloc[row_index_of_best_performance]

        best_config_id_per_instance[instance] = best_config_id

    with open(save_path, "wb") as f:
        pkl.dump(best_config_id_per_instance, f)