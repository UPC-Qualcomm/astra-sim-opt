#!/usr/bin/python3
import os
import subprocess
import multiprocessing
import argparse


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


def run_astrasim(workload_path, system, network, memory, output_dir):
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
    log = os.path.join(file_dir, output_dir, os.path.split(workload_path)[1] + ".log")
    cmd = (
        f"{astrasim_bin} "
        f"--system-configuration={system} "
        f"--workload-configuration={workload_path} "
        f"--network-configuration={network} "
        f"--remote-memory-configuration={memory} "
        f"--comm-group-configuration={workload_path}.json > {log}"
    )
    print(cmd)
    success = run_command(cmd)
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
    args = parser.parse_args()

    design_space = list_workloads(str(args.workload_dir))
    func = partial(
        run_astrasim,
        system=args.system,
        network=args.network,
        memory=args.memory,
        output_dir=args.output_dir,
    )

    with multiprocessing.Pool(int(multiprocessing.cpu_count() * 0.95)) as pool:
        failed_cmds = pool.map(func, design_space)
        print("\n\nrunfails:")
        for cmd in failed_cmds:
            if not cmd == "":
                print(cmd)
