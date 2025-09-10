import os
import subprocess
from glob import glob

file_dir = os.path.dirname(os.path.abspath(__file__))

ASTRASIM_BINARIES = [
    ("G2", os.path.join(file_dir, "../../build/astra_g2/build/bin/AstraSim_G2_congestion")),
    ("Unaware", os.path.join(file_dir, "../../build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Unaware")),
    ("", os.path.join(file_dir, "../../build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware")),
]

TRACES_DIR = os.path.join(file_dir, "workload")
NETWORKS_DIR = os.path.join(file_dir, "configuration")
SYSTEMS_DIR = os.path.join(file_dir, "configuration")
RESULTS_DIR = os.path.join(file_dir, "output")

def list_workloads(root):
    filtered = []

    for dirpath, dirnames, filenames in os.walk(root):
        for file in filenames:
            if file.endswith(".0.et"):
                filtered.append(os.path.join(dirpath, file[:-5]))
                # Only need one match per trace folder, so break after first match in this directory
                break

    return filtered

def find_networks(directory):
    return glob(os.path.join(directory, "*.yml")) + glob(os.path.join(directory, "*.yaml"))

def get_system_path_from_network(network_path):
    base = os.path.splitext(os.path.basename(network_path))[0]
    return os.path.join(SYSTEMS_DIR, base + "_sys.json")

def main():
    # Remove results folder before running
    if os.path.exists(RESULTS_DIR):
        import subprocess
        import time
        subprocess.run(["rm", "-rf", RESULTS_DIR])
        for _ in range(50):
            if not os.path.exists(RESULTS_DIR):
                break
            time.sleep(0.1)
        else:
            print(f"WARNING: Could not remove {RESULTS_DIR} after several attempts.")

    traces = list_workloads(TRACES_DIR)
    networks = find_networks(NETWORKS_DIR)
    for trace in traces:
        trace_base = os.path.basename(trace)
        for network in networks:
            network_base = os.path.splitext(os.path.basename(network))[0]
            system = get_system_path_from_network(network)
            if not os.path.exists(system):
                print(f"WARNING: System config {system} does not exist for network {network}")
                continue
            for mode, binary in ASTRASIM_BINARIES:
                out_dir = os.path.join(
                    RESULTS_DIR, trace_base, f"{network_base}_{mode}"
                )
                os.makedirs(out_dir, exist_ok=True)
                # Save all output files inside out_dir
                log_file = os.path.join(out_dir, "astrasim.log")
                network_log_file = os.path.join(out_dir, "network.csv")
                # ...other output files can be added here if needed...

                # Check for .json file in the trace folder
                comm_group_json = None
                trace_dir = os.path.dirname(trace)
                for f in os.listdir(trace_dir):
                    if f.endswith(".json"):
                        comm_group_json = os.path.join(trace_dir, f)
                        break

                print(out_dir, system, network)
                cmd = (
                    f"{binary} "
                    f"--system-configuration={system} "
                    f"--workload-configuration={trace} "
                    f"--network-configuration={network} "
                    f"--remote-memory-configuration={os.path.join(NETWORKS_DIR, 'RemoteMemory.json')} "
                )
                if comm_group_json:
                    cmd += f"--comm-group-configuration={comm_group_json} "
                cmd += (
                    f"--logging-configuration={log_file} "
                    f"--network-log={network_log_file} "
                )
                ret = subprocess.run(cmd, shell=True, cwd=None)
                if ret.returncode != 0:
                    print(f"FAILED: {cmd}")

if __name__ == "__main__":
    main()
