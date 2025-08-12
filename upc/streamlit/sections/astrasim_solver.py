import os
import streamlit as st
import subprocess
import json
import time
import trace_visualization as tv

def run_simulation_for_solver(params, sys_content, net_content):
    """
    Runs a single ASTRA-Sim simulation instance for the solver.
    This function is designed to be non-interactive and is safe to call from a loop.
    """
    sim_dir = os.path.join(params["temp_dir"])
    os.makedirs(sim_dir, exist_ok=True)

    # Save temporary system and network configuration files
    temp_sys_path = os.path.join(sim_dir, "system.json")
    temp_net_path = os.path.join(sim_dir, "network.yml")
    with open(temp_sys_path, "w") as f:
        f.write(sys_content)
    with open(temp_net_path, "w") as f:
        f.write(net_content)

    # Compute paths for this specific run
    paths = _compute_paths(params, sim_dir)

    # Run the ASTRA-Sim binary
    returncode, elapsed_time = _run_astrasim_bin(
        paths, temp_sys_path, temp_net_path, params["temp_dir"]
    )
    
    # Process the simulation results
    comm_cycles, comp_cycles = _handle_sim_result(returncode, elapsed_time, paths, sys_content)
    
    # Create df_matched for visualization after successful simulation
    if returncode == 0:
        st.session_state.df_matched = tv.get_timings_df(
            paths['trace_file'], paths["timed_trace"]
        )

    return {
        "elapsed_time": elapsed_time,
        "comm_cycles": comm_cycles,
        "comp_cycles": comp_cycles,
    }


def _compute_paths(params, sim_dir):
    as_bin = (
        "../../build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Unaware"
    )
    
    # Construct a unique name for the workload files
    workload_name = (
        f"{params['dp']}_{params['tp']}_{params['sp']}_{params['pp']}_{params['sharding']}.seq_{params['seq']}.batch_{params['batch']}"
    )


    workload_path = os.path.join(params["temp_dir"], workload_name)
    log_path = os.path.join(sim_dir, workload_name)
    
    # Path to the remote memory configuration
    remote_memory_config = os.path.join(
        os.path.dirname(__file__), '..', '..', 'configuration', 'RemoteMemory.json'
    )
    trace_file_name = f"{log_path}_trace.csv"
    timed_trace_file_name = f"{log_path}_trace_matched_timing.csv"
    return {
        "as_bin": as_bin,
        "workload": workload_path,
        "log": log_path,
        "memory": remote_memory_config,
        "network_log": f"{workload_path}.csv",
        "res_log": f"{log_path}_res.csv",
        "trace_file": trace_file_name,
        "timed_trace": timed_trace_file_name,
    }


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
        # Clear all session state safely, preserving important keys
        keys_to_preserve = {'temp_dir', 'session_id', 'df_matched', 'peak_perf', 'peak_bw', 'show_npu_plots'}
        preserved_values = {}
        
        # Save values we want to keep
        for key in keys_to_preserve:
            if key in st.session_state:
                preserved_values[key] = st.session_state[key]
        
        # Clear all session state safely using try-catch to handle missing keys
        keys_to_delete = [key for key in st.session_state.keys() if key not in keys_to_preserve]
        for key in keys_to_delete:
            try:
                del st.session_state[key]
            except KeyError:
                # Key was already deleted or doesn't exist, which is fine
                pass
        
        # Restore preserved values
        for key, value in preserved_values.items():
            st.session_state[key] = value
    return returncode, elapsed_time


def _handle_sim_result(returncode, elapsed_time, paths, updated_sys_content):
    if returncode != 0:
        st.warning(f"Simulation Failed - code: {returncode}.")
        return None, None
    else:
        collect_res_cmd = (
            f"python ../gather_all_NPUs_results.py "
            f"--sim_logfile {paths['timed_trace']}  --output_filename {paths['res_log']}"
        )
        subprocess.run(collect_res_cmd, shell=True, cwd=None)
        st.session_state.show_npu_plots = True
        parsed_sys_config = json.loads(updated_sys_content)
        st.session_state.peak_perf = parsed_sys_config.get("peak-perf", 300)
        st.session_state.peak_bw = parsed_sys_config.get("local-mem-bw", 2000)

        # Parse the results CSV to get cycles for the NPU with max exec_cycles
        import pandas as pd
        try:
            df = pd.read_csv(paths['res_log'])
            if not df.empty:
                max_idx = df['exec_cycles'].idxmax()
                row = df.loc[max_idx]
                comm_cycles = int(row['comm_cycles'])
                comp_cycles = int(row['comp_cycles'])
                # If memory_cycles column exists, otherwise set to None
                return comm_cycles, comp_cycles
        except Exception as e:
            st.warning(f"Could not parse results: {e}")
        return None, None
