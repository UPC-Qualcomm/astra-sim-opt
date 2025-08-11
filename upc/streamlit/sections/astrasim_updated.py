import streamlit as st
import os
from pathlib import Path
import subprocess
import json
import time
import simulation_res as sr
import trace_visualization as tv


def run_astrasim_unified(params):
    """
    Run AstraSim with the provided configuration parameters
    Uses updated configurations passed from generate_workload_and_run_simulation
    """
    if not params.get('sys_content') or not params.get('net_content'):
        st.error("Configuration files are missing. Please generate workload first.")
        return None
    
    config_dir, sim_dir = _setup_dirs(params["temp_dir"])
    paths = _compute_paths(params, config_dir, sim_dir)
    
    # Auto-run simulation without showing the header
    if st.session_state.get("submitted"):
        _clear_sim_dir(sim_dir)
        temp_sys_path, temp_net_path = _save_temp_configs(
            sim_dir, params['sys_content'], params['net_content']
        )
        
        returncode, elapsed_time = _run_astrasim_bin(
            paths, temp_sys_path, temp_net_path, params["temp_dir"]
        )
        
        # Load trace data into session state
        st.session_state.df_matched = tv.get_timings_df(
            paths['trace_file'], paths["timed_trace"]
        )
        
        _handle_sim_result(returncode, elapsed_time, paths, params['sys_content'])

    # Return outputs needed for next sections
    return {
        "sim_dir": sim_dir,
        "log": paths["log"],
        "res_log": paths["res_log"],
        "trace_file_name": os.path.basename(paths["trace_file"]),
        "timed_trace": paths["timed_trace"]
    }


@st.cache_data
def visualize_simulation_results(sim_outputs):
    if "show_npu_plots" not in st.session_state:
        st.session_state.show_npu_plots = False

    if st.session_state.show_npu_plots:
        plots = sr.plot_sim_results(sim_outputs["res_log"])
        for pl in plots:
            st.write(pl)


def _setup_dirs(temp_dir):
    config_dir = "../configuration"
    sim_dir = os.path.join(temp_dir, "sim/")
    os.makedirs(sim_dir, exist_ok=True)
    return config_dir, sim_dir


def _compute_paths(params, config_dir, sim_dir):
    as_bin = (
        "../../build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Unaware"
    )
    prefix = f"{params['dp']}_{params['tp']}_{params['sp']}_{params['pp']}_{params['sharding_val']}"
    temp_path = Path(params["temp_dir"])
    base_file_name = prefix
    for f in temp_path.iterdir():
        if f.is_file() and f.name.startswith(prefix):
            base_file_name = f.name.rsplit(".", 2)[0]  
            break
    workload = os.path.join(
        params["temp_dir"],
        base_file_name,
    )
    log = os.path.join(
        sim_dir,
        base_file_name,
    )
    # Find base file name in parent of temp_dir
    trace_file_name = f"{log}_trace.csv"
    timed_trace_file_name = f"{log}_trace_matched_timing.csv"
    memory = os.path.join(config_dir, "RemoteMemory.json")
    network_log = workload + ".csv"
    res_log = f"{log}_res.csv"
    return {
        "as_bin": as_bin,
        "workload": workload,
        "log": log,
        "memory": memory,
        "network_log": network_log,
        "res_log": res_log,
        "trace_file": trace_file_name,
        "timed_trace": timed_trace_file_name,
        "base_file_name": base_file_name
    }


def _clear_sim_dir(sim_dir):
    if os.path.isdir(sim_dir):
        subprocess.run(f"rm -rf {sim_dir}*", shell=True, cwd=None)


def _save_temp_configs(sim_dir, sys_content, net_content):
    temp_network_config = os.path.join(sim_dir, "network.yml")
    temp_sys_config = os.path.join(sim_dir, "system.json")
    with open(temp_network_config, "w") as f:
        f.write(net_content)
    with open(temp_sys_config, "w") as f:
        f.write(sys_content)
    return temp_sys_config, temp_network_config


def _run_astrasim_bin(paths, temp_sys_path, temp_net_path, temp_dir):
    cmd = (
        f"{paths['as_bin']} "
        f"--system-configuration={temp_sys_path} "
        f"--workload-configuration={paths['workload']} "
        f"--network-configuration={temp_net_path} "
        f"--remote-memory-configuration={paths['memory']} "
        f"--comm-group-configuration={paths['workload']}.json "
        f"--logging-configuration={paths['log']} "
        f"--network-log={paths['network_log']} > {temp_dir}test.txt"
    )
    with st.spinner("Running AstraSim..."):
        start_time = time.time()
        returncode = subprocess.run(cmd, shell=True, cwd=None).returncode
        elapsed_time = time.time() - start_time
    return returncode, elapsed_time


def _handle_sim_result(returncode, elapsed_time, paths, updated_sys_content):
    if returncode != 0:
        st.warning(f"Simulation Failed - code: {returncode}.")
    else:
        st.success(
            f"The Simulation has completed successfully within {elapsed_time:.2f} seconds."
        )
        collect_res_cmd = (
            f"python ../gather_all_NPUs_results.py "
            f"--sim_logfile {paths['timed_trace']}  --output_filename {paths['res_log']}"
        )
        subprocess.run(collect_res_cmd, shell=True, cwd=None)
        st.session_state.show_npu_plots = True
        parsed_sys_config = json.loads(updated_sys_content)
        st.session_state.peak_perf = parsed_sys_config.get("peak-perf", 300)
        st.session_state.peak_bw = parsed_sys_config.get("local-mem-bw", 2000)
