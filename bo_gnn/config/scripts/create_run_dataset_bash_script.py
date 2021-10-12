import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n", "--user_name", type=str, help="Your user name on the cluster",
    )
    parser.add_argument(
        "-r", "--run_time", type=str, help="Run Time of jobs, indicate in same format as for submitting jobs i.e. 04:00",
    )
    parser.add_argument(
        "-c", "--number_of_cores", type=str, help="Number of cores of job",
    )
    parser.add_argument(
        "-d", "--dataset_path", type=str, help="Path to instances",
    )

    parser.add_argument(
        "-t",
        "--task_name",
        help="Task name",
        choices=("1_item_placement", "2_load_balancing", "3_anonymous"),
    )
    parser.add_argument(
        "-s",
        "--start_instance_number",
        help="Run from what starting instance number",
        required=True,
        type=int,
    )
    parser.add_argument(
        "-e",
        "--end_instance_number",
        help="Run up to end instance number",
        required=True,
        type=int,
    )
    parser.add_argument(
        "-j",
        "--number_of_jobs",
        help="Number of jobs to submit",
        required=True,
        type=int,
    )

    arguments = parser.parse_args()
    number_of_jobs = arguments.number_of_jobs
    start_instance_number = arguments.start_instance_number
    end_instance_number = arguments.end_instance_number

    assert (
        end_instance_number - start_instance_number
    ) % number_of_jobs == 0, (
        "Cant divide the instance range to your defined number of jobs"
    )
    range_per_job = (end_instance_number - start_instance_number) / number_of_jobs
    start_instance = start_instance_number
    end_instance = start_instance + range_per_job
    with open("scripts/run_dataset_generation_on_cluster.sh", "w+") as file:
        for _ in range(arguments.number_of_jobs):
            command = "bsub -o /cluster/scratch/{}/lsf/ -W {} -n 32 python3 generate_data.py -n {} -p " \
                      "{} -j {} -f train " \
                      "-o {}{}_results.csv " \
                      "-t 900 -s {} -e {} -r 1 \n".format(arguments.user_name, arguments.run_time, arguments.task_name,
                                                     arguments.dataset_path, arguments.number_of_cores, arguments.dataset_path, arguments.task_name,
                                                       int(start_instance), int(end_instance))
            file.write(command)
            start_instance = end_instance
            end_instance = end_instance + range_per_job


if __name__ == "__main__":
    main()
