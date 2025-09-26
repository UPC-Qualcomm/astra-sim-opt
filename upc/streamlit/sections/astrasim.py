import streamlit as st
import os
from pathlib import Path
import subprocess
import json
import time
import simulation_res as sr
import trace_visualization as tv


def run_astrasim(params):
    st.header(
        "Run AstraSim",
        help=(
            "This section allows you to run AstraSim on the generated workload trace.\n"
            "- Choose a network configuration from the available options.\n"
            "- Modify the system and network configurations if needed.\n"
            "- Click **Run AstraSim** to execute the simulation."
        )
    )
    config_dir, sim_dir = _setup_dirs(params["temp_dir"])
    configs = _get_config_names(config_dir)

    st.subheader("Simulation Configuration")
    config_display_map = {
        "2D_Torus": "2D Torus",
        "3D_Torus": "3D Torus", 
        "Dragonfly": "Dragonfly",
        "FoldedClos": "Folded-Clos"
    }
    
    display_options = [config_display_map.get(config, config) for config in configs]
    selected_display_name = st.selectbox("Select a Topology", display_options)
    
    reverse_map = {v: k for k, v in config_display_map.items()}
    selected_config_name = reverse_map.get(selected_display_name, selected_display_name)

    paths = _compute_paths(params, config_dir, sim_dir)
    col1, col2 = st.columns(2)
    with col1:
        if configs:
            net_content, sys_content = _load_config_files(
                config_dir, selected_config_name
            )
            updated_sys_content, updated_net_content = _show_config_editors(
                col1, col2, sys_content, net_content
            )
            if st.button("🚀 Run AstraSim"):
                _clear_sim_dir(sim_dir)
                temp_sys_path, temp_net_path = _save_temp_configs(
                    sim_dir, updated_sys_content, updated_net_content
                )
                st.success("Saved modified config")
                returncode, elapsed_time = _run_astrasim_bin(
                    paths, temp_sys_path, temp_net_path, params["temp_dir"]
                )
                #if "df_matched" not in st.session_state:
                st.session_state.df_matched = tv.get_timings_df(
                    paths['trace_file'], paths["timed_trace"]
                )
                _handle_sim_result(returncode, elapsed_time, paths, updated_sys_content)
        else:
            st.warning("No Configurations Found.")

    # Return outputs needed for next sections
    return {
        "sim_dir": sim_dir,
        "log": paths["log"],
        "res_log": paths["res_log"],
        "trace_file_name": os.path.basename(paths["trace_file"]),
        "timed_trace": paths["timed_trace"]
    }

@st.cache_data(show_spinner='Visualizing Simulation Results...')
def visualize_simulation_results(sim_outputs):
    if "show_npu_plots" not in st.session_state:
        st.session_state.show_npu_plots = False

    if st.session_state.show_npu_plots:
        plots = sr.plot_sim_results(sim_outputs["res_log"])
        for pl in plots:
            st.write(pl)
        st.markdown("<p style='text-align: center; font-size: 0.9em; color: #666;'>Per plot simulation results, showcasing the amount of overlapped compute and communication along with the exposed computation and communication.</p>", unsafe_allow_html=True)


def _setup_dirs(temp_dir):
    config_dir = "../configuration"
    sim_dir = os.path.join(temp_dir, "sim/")
    os.makedirs(sim_dir, exist_ok=True)
    return config_dir, sim_dir


def _get_config_names(config_dir):
    return [f.stem for f in Path(config_dir).glob("*.yml")]


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


def _load_config_files(config_dir, selected_config_name):
    selected_network_config = os.path.join(config_dir, f"{selected_config_name}.yml")
    selected_sys_config = os.path.join(config_dir, f"{selected_config_name}_sys.json")
    with open(selected_network_config, "r") as f:
        net_content = f.read()
    with open(selected_sys_config, "r") as f:
        sys_content = f.read()
    return net_content, sys_content


def _show_config_editors(col1, col2, sys_content, net_content):
    with col1:
        updated_sys_content = st.text_area(
            "Edit System Config", value=sys_content, height=400
        )
    with col2:
        updated_net_content = st.text_area(
            "Edit Network Config", value=net_content, height=400
        )
    return updated_sys_content, updated_net_content


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
        #for key in list(st.session_state.keys()):
        #    if key != 'submitted':
        #        del st.session_state[key]
    return returncode, elapsed_time


def _handle_sim_result(returncode, elapsed_time, paths, updated_sys_content):
    if returncode != 0:
        st.warning(f"Simulation Failed - code: {returncode}.")
    else:
        st.success(
            f"✅ The Simulation has completed successfully within {elapsed_time:.2f} seconds."
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
