#!/usr/bin/python3
import re
import argparse
import pandas as pd
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing


def extract_runtime_results(log_path):
    pattern = r"\[(\d+)\] finished, (\d+) cycles, exposed communication (\d+) cycles"

    sys_ids = []
    execution_cycles = []
    communication_cycles = []

    with open(log_path, "r") as f:
        log_lines = f.read()
        matches = re.findall(pattern, log_lines)

        if not matches:
            return pd.DataFrame(columns=[
                'sys_id', 'execution_cycles', 'exposed_communication_cycles'
            ])

        for sys_id, exec_cycles, comm_cycles in matches:
            sys_ids.append(sys_id)
            execution_cycles.append(int(exec_cycles))
            communication_cycles.append(int(comm_cycles))

    df = pd.DataFrame({
        'sys_id': sys_ids,
        'execution_cycles': execution_cycles,
        'exposed_communication_cycles': communication_cycles
    })

    return df


def extract_runtime_results_dir(log_dir, output_dir):
    df = extract_runtime_results(log_dir)
    file_name = os.path.join(output_dir, os.path.splitext(log_dir)[0].split('/')[-1] + "_res.csv")
    if not df.empty:
        df.to_csv(file_name, index=False)

def list_logs(root):
    files = os.listdir(root)
    filtered = list()
    for file in files:
        if file.endswith(".log"):
            filtered.append(os.path.join(root, file))
    return filtered

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sim_logfile", type=str, help="The output log file or directory", required=True
    )
    parser.add_argument(
        "--output_filename", type=str, help="The results file name (only if single file mode)"
    )
    args = parser.parse_args()

    if os.path.isfile(args.sim_logfile):
        runtimes_df = extract_runtime_results(args.sim_logfile)
        if not args.output_filename:
            raise ValueError("You must specify --output_filename when processing a single log file.")
        runtimes_df.to_csv(args.output_filename, index=False)
        #print(f"Results saved to {args.output_filename}")

    elif os.path.isdir(args.sim_logfile):
        os.makedirs(args.output_filename,  exist_ok=True)
        #runtimes_dfs, filenames = extract_runtime_results_dir(args.sim_logfile)
        logs = list_logs(args.sim_logfile)
        with multiprocessing.Pool() as pool:
            pool.starmap(extract_runtime_results_dir, [(log, args.output_filename) for log in logs])

    else:
        raise ValueError(f"{args.sim_logfile} is neither a file nor a directory.")
