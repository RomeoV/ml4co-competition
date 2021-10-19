import pandas as pd
import argparse
import os
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--files_to_filter",
        help="List of files that still need to be filtered by selecting relevant configs",
        nargs='+'
    )
    parser.add_argument(
        "-r",
        "--ready_files",
        help="List of files that are already filtered and can just be appended",
        nargs='+'
    )
    parser.add_argument(
        "-t",
        "--task_name",
        help="Name of the task",
        choices=("1_item_placement", "2_load_balancing", "3_anonymous")
    )
    parser.add_argument(
        "-k",
        "--k_best_config_ids",
        help="How many of the k best configs should be used. This is only relevant if there are any filter_files.",
    )
    parser.add_argument(
        "-o",
        "--output_path",
        help="Path to output csv file",
        type=str,
        required=True
    )



    arguments = parser.parse_args()
    list_of_files_to_filter = arguments.files_to_filter
    list_of_ready_files = arguments.ready_files
    filtered_files_as_pd = []
    if list_of_files_to_filter is not None:
        for file_path in list_of_files_to_filter:
            assert os.path.isfile(file_path), file_path
            assert arguments.k_best_config_ids is not None
            filtered_files_as_pd.append(filter_file(file_path, arguments.task_name, arguments.k_best_config_ids))

    ready_files_as_pd = []
    if list_of_ready_files is not None:
        for file_path in list_of_ready_files:
            assert os.path.isfile(file_path), file_path
            ready_files_as_pd.append(pd.read_csv(file_path))

    all_frames = filtered_files_as_pd + ready_files_as_pd
    all_frames_as_pd = pd.concat(all_frames)
    all_frames_as_pd.to_csv(arguments.output_path)




def filter_file(file_path: str, task_name:str, k_best_config_ids: int)-> pd.DataFrame:
    data_frame = pd.read_csv(file_path)
    with open("data_utils/{}_instance_and_id_specification.json".format(task_name), "r") as file:
        selected_config_ids = json.load(file)["selected_config_ids"][str(k_best_config_ids)]
    return data_frame.loc[data_frame["config_id"].isin(selected_config_ids)]

if __name__ == '__main__':
    main()