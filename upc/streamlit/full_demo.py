import streamlit as st
import pandas as pd
import numpy as np
import os
import trace_visualization as tv
import roofline_visualization as rv
import time
import generate_single_workload as gen
from pathlib import Path
import subprocess


# Set Streamlit page config to wide mode
st.set_page_config(page_title="Roofline Viewer", layout="wide")

st.title("Generate a workload trace")


temp_dir = "temp/"

with st.form("model_config_form"):
    st.subheader("Model parameters")

    display_to_model = {v: k for k, v in gen.model_display_names.items()}
    model_names = list(gen.model_display_names.values())
    selected_model_name = st.selectbox("Choose a pre-defined model:", model_names)
    selected_model = display_to_model[selected_model_name]

    params = gen.Model.get_model_params(selected_model)
    param_names = ["din", "dout", "dmodel", "dff", "batch", "seq", "head", "num_stacks"]

    cols = st.columns(len(param_names))
    new_params = {}
    for i, col in enumerate(cols):
        with col:
            params[i] = st.number_input(
                param_names[i],
                value=params[i],
                key=f"{param_names[i]}_input"
            )

    st.subheader("Parallelism strategy")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        dp = st.text_input("Data Parallelism (DP)", 1)
    with col2:
        tp = st.text_input("Tensor Parallelism (TP)", 8)
    with col3:
        sp = st.text_input("Sequence Parallelism (SP)", 2)
    with col4:
        pp = st.text_input("Pipeline Parallelism (PP)", 4)
    with col5:
        sharding = st.checkbox("Sharding", value=False)

    # Submit button inside form
    submitted = st.form_submit_button("🚀 Run Model")

sharding_val = "1" if sharding else "0"

if submitted:
    subprocess.run(f"rm -rf {temp_dir}*", shell=True, cwd=None)
    with st.spinner(f"Generating a trace for `{selected_model_name}`..."):
        start_time = time.time()
        gen.generate_trace([dp, tp, sp, pp, sharding], params)
        elapsed_time = time.time() - start_time

    st.success(f"✅ Trace generation completed in {elapsed_time:.2f} seconds.")



st.markdown("---")
st.title("Run AstraSim")

config_dir = "../configuration"
sim_dir = temp_dir + "sim/"
os.makedirs(sim_dir, exist_ok=True)

# Fetch network config .yml files
configs = [f.stem for f in Path(config_dir).glob("*.yml")]

# Fetch system config files ending with _sys.json
json_files = [f.name for f in Path(config_dir).glob("*_sys.json")]


st.subheader("Configuration")
selected_config_name = st.selectbox("Select a config", configs)

# Create two columns
col1, col2 = st.columns(2)
with col1:

    if configs:

        selected_network_config = os.path.join(config_dir, f"{selected_config_name}.yml")
        selected_sys_config = os.path.join(config_dir, f"{selected_config_name}_sys.json")

        with open(selected_network_config, "r") as f:
            networ_file_content = f.read()

        with open(selected_sys_config, "r") as f:
            sys_file_content = f.read()

        with col1:
            updated_sys_content = st.text_area("Edit System Config", value=sys_file_content, height=400)
        with col2:
            updated_content = st.text_area("Edit Network Config", value=networ_file_content, height=400)


        if st.button("🚀 Run Network Simulator"):
            subprocess.run(f"rm {sim_dir}*", shell=True, cwd=None)
            temp_network_config = os.path.join(sim_dir, "network.yml")
            with open(temp_network_config, "w") as f:
                f.write(updated_content)

            temp_sys_config = os.path.join(sim_dir, "system.json")
            with open(temp_sys_config, "w") as f:
                f.write(updated_sys_content)
            st.success("Saved modified config")

            astrasim_bin = "../../build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Unaware"
            workload = os.path.join(temp_dir, f'{dp}_{tp}_{sp}_{pp}_{sharding_val}')
            log = os.path.join(sim_dir, f'{dp}_{tp}_{sp}_{pp}_{sharding_val}')
            memory = os.path.join(config_dir, "RemoteMemory.json")
            network_log = workload + '.csv'
            cmd = (
                f"{astrasim_bin} "
                f"--system-configuration={temp_sys_config} "
                f"--workload-configuration={workload} "
                f"--network-configuration={temp_network_config} "
                f"--remote-memory-configuration={memory} "
                f"--comm-group-configuration={workload}.json "
                f"--logging-configuration={log} "
                f"--network-log={network_log} "
            )
            print(cmd)
            returncode = -1
            with st.spinner(f"Generating a trace for `{selected_model_name}`..."):
                start_time = time.time()
                returncode = subprocess.run(cmd, shell=True, cwd=None).returncode
                elapsed_time = time.time() - start_time
                for key in st.session_state.keys():
                    del st.session_state[key]

            if returncode != 0:
                st.warning(f"Simulation Failed - code: {returncode}.")
            else:
                st.success("✅ The Simulation has completed successfully within {elapsed_time:.2f} seconds.")

    else:
        st.warning("No Configurations Found.")



st.markdown("---")

st.title("📊 Trace Visualization Viewer")

trace_file_name = f"{dp}_{tp}_{sp}_{pp}_{sharding_val}_trace.csv"
roofline_file_name = f"{dp}_{tp}_{sp}_{pp}_{sharding_val}_roofline.csv"
csv_trace_file = os.path.join(sim_dir, trace_file_name)
csv_roofline_file = os.path.join(sim_dir, roofline_file_name)

# Detect file change
if "last_roofline_file" not in st.session_state or st.session_state.last_roofline_file != csv_roofline_file:
    st.session_state.last_roofline_file = csv_roofline_file
    st.session_state.timestep_idx = 0
    st.session_state.selected_timestep = None

if os.path.isfile(csv_trace_file):
    st.success(f"✅ Found trace file: `{trace_file_name}`")

    df = pd.read_csv(csv_trace_file)
    df = tv.get_timings_df(df)
    max_npu = int(df["sys_id"].max())
    npu = st.number_input("NPU index", min_value=0, max_value=max_npu, value=0, step=1)

    # Set session state if not already
    if "active_plot" not in st.session_state:
        st.session_state.active_plot = None

    st.subheader("Visualize the compute and communication node over time.")
    st.altair_chart(tv.plot_one_npu(df, npu), use_container_width=True)

    st.markdown("---")

    col_plot1, col_plot2, col_plot3, col_plot4, col_plot5, col_plot6 = st.columns(6)

    with col_plot1:
        if st.button("Show Chakra Times"):
            st.session_state.active_plot = "chakra_times"
    with col_plot2:
        if st.button("Show All NPUs"):
            st.session_state.active_plot = "all_npus"
    with col_plot3:
        if st.button("3D Roofline"):
            st.session_state.active_plot = "roofline_3d"
    with col_plot4:
        if st.button("2D Roofline"):
            st.session_state.active_plot = "roofline_2d"
    with col_plot5:
        if st.button("2D Roofline over Time"):
            st.session_state.active_plot = "roofline_2d_over_time"
    with col_plot6:
        if st.button("2D Roofline at a timestep"):
            st.session_state.active_plot = "roofline_2d_timestep"


    # Time window input outside to persist value
    time_window = 0  # initialize

    if st.session_state.active_plot == "roofline_3d" or   st.session_state.active_plot == "roofline_2d_over_time":
        time_window = st.number_input("Averaging Time Window (cycles)", value=10000, step=1000)

    # Render plots based on session state value
    if st.session_state.active_plot == "chakra_times":
        st.altair_chart(tv.plot_one_npu(df, npu, plot_blocks=False, plot_times=True))
    elif st.session_state.active_plot == "all_npus":
        for n in range(max_npu):
            st.altair_chart(
                tv.plot_one_npu(df, n, plot_blocks=True, plot_times=False),
                use_container_width=True,
            )
    elif st.session_state.active_plot == "roofline_3d":
        st.subheader("Visualize 3D roofline model.")
        st.plotly_chart(
            rv.get_3d_roofline_plot(csv_roofline_file, npu, time_window=time_window)
        )
    elif st.session_state.active_plot == "roofline_2d":
        st.subheader("Visualize 2D roofline model.")
        st.altair_chart(rv.get_2d_roofline_plot_normal(csv_roofline_file, npu))
    elif st.session_state.active_plot == "roofline_2d_over_time":
        st.subheader("Visualize 2D roofline model overtime.")
        st.altair_chart(
            rv.get_2d_roofline_plot_with_time(
                csv_roofline_file, npu, time_window=time_window
            )
        )
    elif st.session_state.active_plot == "roofline_2d_timestep":
        st.subheader("Visualize 2D roofline model based on a timestep.")
        df, timesteps = rv.get_timesteps(csv_file=csv_roofline_file, npu=npu)

        # Session variables
        if "timestep_idx" not in st.session_state:
            st.session_state.timestep_idx = 0
        if "selected_timestep" not in st.session_state or st.session_state.selected_timestep not in timesteps:
            st.session_state.timestep_idx = 0
            st.session_state.selected_timestep = timesteps[st.session_state.timestep_idx ]

        # Control functions
        def next_step():
            idx = st.session_state.timestep_idx
            if idx < len(timesteps) - 1:
                st.session_state.timestep_idx = idx + 1
                st.session_state.selected_timestep = timesteps[
                    st.session_state.timestep_idx
                ]
            else:
                st.session_state.timestep_idx = 0
                st.session_state.selected_timestep = timesteps[
                    st.session_state.timestep_idx
                ]

        def prev_step():
            idx = st.session_state.timestep_idx
            if idx > 0:
                st.session_state.timestep_idx = idx - 1
                st.session_state.selected_timestep = timesteps[
                    st.session_state.timestep_idx
                ]
            else:
                st.session_state.timestep_idx = len(timesteps) - 1
                st.session_state.selected_timestep = timesteps[
                    st.session_state.timestep_idx
                ]


        if "is_playing" not in st.session_state:
            st.session_state.is_playing = False

        def play():
            st.session_state.is_playing = True

        def pause():
            st.session_state.is_playing = False
            
        # Organize the layout
        col1, col2, col3, col4, col5 = st.columns([8, 1, 1, 1, 1])
        with col2:
            if st.button("⬅️ Prev"):
                prev_step()
        with col3:
            if st.button("▶️ Play"):
                play()
        with col4:
            if st.button("⏸️ Pause"):
                pause()
        with col5:
            if st.button("Next ➡️"):
                next_step()
        with col1:
            selected_timestep = st.select_slider(
                "Select a timestep",
                options=timesteps,
                value=st.session_state.selected_timestep,
                label_visibility="collapsed",
            )
            # If user moved the slider, update the state
            if selected_timestep != st.session_state.selected_timestep:
                st.session_state.selected_timestep = selected_timestep
                st.session_state.timestep_idx = int(np.where(timesteps == selected_timestep)[0][0])

        df = df[df["issue_tick"] == st.session_state.selected_timestep]
        col1, col2 = st.columns([2, 1])
        with col1:
            st.altair_chart(rv.get_2d_roofline_plot_timestep(df))
        with col2:
            st.write("ℹ️ Points at this timestep:")
            st.dataframe(
                df.loc[
                    :,
                    [
                        "node_id",
                        "node_name",
                        "perf",
                        "operational_intensity",
                        "issue_tick",
                    ],
                ],
                use_container_width=True,
            )

        # Auto-advance logic
        if st.session_state.is_playing:
            next_step()  # move to next timestep
            time.sleep(0.2)  # 200 ms pause between steps
            st.rerun()

else:
    st.error(
        f"❌ The combination you entered does not correspond to an existing trace file: `{trace_file_name}`"
    )
