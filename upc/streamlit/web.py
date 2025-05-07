import streamlit as st
import pandas as pd
import numpy as np
import os
import trace_visualization as tv
import roofline_visualization as rv
import time


# Set Streamlit page config to wide mode
st.set_page_config(page_title="Roofline Viewer", layout="wide")

st.title("📊 Trace Visualization Viewer")

# Create columns for horizontal layout
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    dp = st.text_input("Data Parallelism (DP)", "1")
with col2:
    tp = st.text_input("Tensor Parallelism (TP)", "8")
with col3:
    sp = st.text_input("Pipeline Parallelism (SP)", "2")
with col4:
    pp = st.text_input("Placement Parallelism (PP)", "4")
with col5:
    sharding = st.checkbox("Sharding", value=False)

sharding_val = "1" if sharding else "0"

trace_file_name = f"{dp}_{tp}_{sp}_{pp}_{sharding_val}_trace.csv"
roofline_file_name = f"{dp}_{tp}_{sp}_{pp}_{sharding_val}_roofline.csv"
base_dir = os.path.abspath(os.path.join(os.getcwd(), "../output/GPT_3_1300M/2D_Torus/"))
csv_trace_file = os.path.join(base_dir, trace_file_name)
csv_roofline_file = os.path.join(base_dir, roofline_file_name)

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
