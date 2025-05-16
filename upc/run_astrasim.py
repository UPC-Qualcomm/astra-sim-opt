#!/usr/bin/python3
import os
import subprocess
import multiprocessing
import argparse
import pandas as pd

def get_timings_df(csv_trace_file, output_file_name):
    df = pd.read_csv(csv_trace_file)
    # Filter issues and rename 'tick' to 'issue_tick'
    df_issues = df.query("action == 'issue'").drop(columns="action")

    # Filter callbacks and rename 'tick' to 'callback_tick'
    df_callbacks = (
        df.query("action == 'callback'")
        .drop(columns="action")
        .rename(columns={"issue_tick": "callback_tick"})
    )

    # Merge issues with callbacks on the identifying columns
    merged_df = df_issues.merge(
        df_callbacks[["sys_id", "node_id", "node_name", "node_type", "callback_tick"]],
        on=["sys_id", "node_id", "node_name", "node_type"],
        how="left",
        suffixes=("", ""),
    )

    # Add elapsed_time column
    merged_df["elapsed_time"] = merged_df["callback_tick"] - merged_df["issue_tick"]
    merged_df.fillna(0, inplace=True)
    merged_df.to_csv(output_file_name)

def run_command(command, cwd=None):
    result = subprocess.run(command, shell=True, cwd=cwd)
    return result.returncode == 0


def list_workloads(root):
    files = os.listdir(root)
    filtered = list()
    for file in files:
        if file.endswith(".0.et"):
            filtered.append(os.path.join(root, file[:-5]))
    return filtered


def run_astrasim(workload_path, system, network, memory, output_dir, network_log):
    astrasim_root = (
        "/home/tomas/repositories/upc/astra-sim"
        if os.getlogin() == "tomas"
        else "/home/mohammad/spain/experiments/astra-sim"
    )
    if astrasim_root is None:
        raise Exception(
            f"please specify astrasim folder path at variable astrasim_root at "
            f"{__file__}:run_astrasim()"
        )
    file_dir = os.path.split(os.path.abspath(__file__))[0]
    astrasim_bin = os.path.join(
        astrasim_root,
        "build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Unaware",
    )

    system = os.path.join(file_dir, system)
    network = os.path.join(file_dir, network)
    memory = os.path.join(file_dir, memory)
    os.makedirs(os.path.join(file_dir, output_dir), exist_ok=True)
    os.makedirs(os.path.join(file_dir, network_log), exist_ok=True)
    log = os.path.join(file_dir, output_dir, os.path.split(workload_path)[1])
    # with open(log, 'w') as outfile:
    #    pass
    network_log = os.path.join(
        file_dir, network_log, os.path.split(workload_path)[1] + ".csv"
    )
    cmd = (
        f"{astrasim_bin} "
        f"--system-configuration={system} "
        f"--workload-configuration={workload_path} "
        f"--network-configuration={network} "
        f"--remote-memory-configuration={memory} "
        f"--comm-group-configuration={workload_path}.json "
        f"--logging-configuration={log} "
        f"--network-log={network_log} "
    )
    print(cmd)
    success = run_command(cmd)
    if success:
        get_timings_df(f"{log}_trace.csv", f"{log}_trace_matched_timing.csv")
    if not success:
        return cmd
    return ""


from functools import partial

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workload_dir",
        type=str,
        help="The folder containing the workload",
        required=True,
    )
    parser.add_argument(
        "--system", type=str, help="The folder containing the workload", required=True
    )
    parser.add_argument(
        "--network", type=str, help="The folder containing the workload", required=True
    )
    parser.add_argument(
        "--memory", type=str, help="The folder containing the workload", required=True
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="The folder containing the workload",
        required=True,
    )
    parser.add_argument(
        "--network_log",
        type=str,
        help="The folder containing the network logs",
        required=True,
    )
    args = parser.parse_args()

    design_space = list_workloads(str(args.workload_dir))
    func = partial(
        run_astrasim,
        system=args.system,
        network=args.network,
        memory=args.memory,
        output_dir=args.output_dir,
        network_log=args.network_log,
    )

    with multiprocessing.Pool(int(multiprocessing.cpu_count() * 0.95)) as pool:
        failed_cmds = pool.map(func, design_space)
        print("\n\nrunfails:")
        for cmd in failed_cmds:
            if not cmd == "":
                print(cmd)
