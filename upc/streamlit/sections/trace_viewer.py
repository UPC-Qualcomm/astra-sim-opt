import streamlit as st
import numpy as np
import trace_visualization as tv
import roofline_visualization as rv
import time
import io
from . import astrasim as astra

def trace_viewer(sim_outputs):
    st.title("📊 Trace Visualization Viewer")
    _init_session_state()
    if "df_matched" not in st.session_state:
        st.warning("The simulation trace is not available yet.")
    else:
        max_npu = int(st.session_state.df_matched["sys_id"].max())
        npu = _load_and_select_npu(max_npu)
        _show_basic_plots(st.session_state.df_matched, npu)
        _plot_selector(st.session_state.df_matched, npu, max_npu)


def _init_session_state():
    if "peak_perf" not in st.session_state:
        st.session_state.peak_perf = 300
    if "peak_bw" not in st.session_state:
        st.session_state.peak_bw = 2000


def _detect_file_change(csv_trace_file):
    if (
        "last_trace_file" not in st.session_state
        or st.session_state.last_trace_file != csv_trace_file
    ):
        st.session_state.last_trace_file = csv_trace_file
        st.session_state.timestep_idx = 0
        st.session_state.selected_timestep = None


def _load_and_select_npu(max_npu, idx = "default"):
    npu = st.number_input("NPU index", min_value=0, max_value=max_npu, value=0, step=1, key=f"npu_idx_{idx}")
    if "active_plot" not in st.session_state:
        st.session_state.active_plot = None
    return npu


def _show_basic_plots(df, npu, exp = ""):
    st.subheader("Visualize the compute and communication node over time.")
    st.altair_chart(tv.plot_one_npu(df, npu, exp=exp), use_container_width=True)
    mem, comp, idle = rv.get_info(
        df,
        npu=npu,
        perf=st.session_state.peak_perf,
        bw=st.session_state.peak_bw,
    )
    total = idle + comp + mem
    st.info(f"""
        **Percentage of time spent with memory bound operations**: {mem/total * 100:.2f}% \n
        **Percentage of time spent with compute bound operations**: {comp/total * 100:.2f}% \n
        **Percentage of time spent with idle**: {idle/total*100:.2f} \n
        **Total time spent**: {total*1e-6:.2f} ms \n
    """)
    st.markdown("---")


def _plot_selector(df, npu, max_npu):
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

    time_window = 0
    if st.session_state.active_plot in ["roofline_3d", "roofline_2d_over_time"]:
        time_window = st.number_input(
            "Averaging Time Window (cycles)", value=10000000, step=1000
        )

    if st.session_state.active_plot == "chakra_times":
        st.altair_chart(tv.plot_one_npu(df, npu, plot_blocks=False, plot_times=True))
    elif st.session_state.active_plot == "all_npus":
        for n in range(max_npu):
            st.altair_chart(
                tv.plot_one_npu(df, n, plot_blocks=True, plot_times=False),
                use_container_width=True,
            )
    elif st.session_state.active_plot == "roofline_3d":
        _show_3d_roofline(df, npu, time_window)
    elif st.session_state.active_plot == "roofline_2d":
        _show_2d_roofline(df, npu)
    elif st.session_state.active_plot == "roofline_2d_over_time":
        _show_2d_roofline_over_time(df, npu, time_window)
    elif st.session_state.active_plot == "roofline_2d_timestep":
        _show_2d_roofline_timestep(df, npu)

@st.cache_data(show_spinner='Loading 3D Roofline plot...')
def _show_3d_roofline(df, npu, time_window):
    st.subheader("Visualize 3D roofline model.")
    st.plotly_chart(
        rv.get_3d_roofline_plot(
            df,
            npu,
            time_window=time_window,
            perf=st.session_state.peak_perf,
            bw=st.session_state.peak_bw,
        )
    )

@st.cache_data(show_spinner='Loading 2D Roofline plot...')
def _show_2d_roofline(df, npu):
    st.subheader("Visualize 2D roofline model.")
    
    buf = io.BytesIO()
    rv.get_2d_roofline_plot_normal(
        df,
        npu,
        perf=st.session_state.peak_perf,
        bw=st.session_state.peak_bw,
    ).save(buf, format='png')
    buf.seek(0)
    png_bytes = buf.read()
    st.image(png_bytes)
    

@st.cache_data(show_spinner='Loading 2D Roofline plot over time...')
def _show_2d_roofline_over_time(df, npu, time_window):
    st.subheader("Visualize 2D roofline model overtime.")
    st.altair_chart(
        rv.get_2d_roofline_plot_with_time(
            df,
            npu,
            time_window=time_window,
            perf=st.session_state.peak_perf,
            bw=st.session_state.peak_bw,
        )
    )

def _show_2d_roofline_timestep(df, npu):
    st.subheader("Visualize 2D roofline model based on a timestep.")
    df, timesteps = rv.get_timesteps(df=df, npu=npu)
    _init_timestep_state()
    _timestep_controls(timesteps)
    _show_timestep_plot_and_table(df, timesteps)


def _init_timestep_state():
    if "timestep_idx" not in st.session_state:
        st.session_state.timestep_idx = 0


def play():
    st.session_state.is_playing = True


def pause():
    st.session_state.is_playing = False


def next_step(timesteps):
    idx = st.session_state.timestep_idx
    if idx < len(timesteps) - 1:
        st.session_state.timestep_idx = idx + 1
    else:
        st.session_state.timestep_idx = 0


def prev_step(timesteps):
    idx = st.session_state.timestep_idx
    if idx > 0:
        st.session_state.timestep_idx = idx - 1
    else:
        st.session_state.timestep_idx = len(timesteps) - 1


def _timestep_controls(timesteps):
    if "is_playing" not in st.session_state:
        st.session_state.is_playing = False

    col1, col2, col3, col4, col5 = st.columns([8, 1, 1, 1, 1])
    with col2:
        if st.button("⬅️ Prev"):
            prev_step(timesteps)
    with col3:
        if st.button("▶️ Play"):
            play()
    with col4:
        if st.button("⏸️ Pause"):
            pause()
    with col5:
        if st.button("Next ➡️"):
            next_step(timesteps)
    with col1:
        selected_timestep = st.select_slider(
            "Select a timestep",
            options=timesteps,
            value=timesteps[st.session_state.timestep_idx],
        )
        if selected_timestep != timesteps[st.session_state.timestep_idx]:
            st.session_state.timestep_idx = int(
                np.where(timesteps == selected_timestep)[0][0]
            )


def _show_timestep_plot_and_table(df, timesteps):
    df = df[df["issue_tick"] == timesteps[st.session_state.timestep_idx]]
    col1, col2 = st.columns([2, 1])
    with col1:
        buf = io.BytesIO()
        rv.get_2d_roofline_plot_timestep(
                df, perf=st.session_state.peak_perf, bw=st.session_state.peak_bw
            ).save(buf, format='png')  # Requires vl-convert or altair_saver
        buf.seek(0)
        png_bytes = buf.read()
        st.image(png_bytes)
       
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
    if st.session_state.is_playing:
        time.sleep(1)
        next_step(timesteps)
        st.rerun()


def render_sim_output_section(sim_outputs):
    st.header(
        "Simulation Visualizations",
        help=(
            "This section allows you to visualize the simulation results.\n"
        )
    )
    st.markdown("""
        <style>
        .stTabs [data-baseweb="tab"] {
            background-color: #e0e7ff !important;  /* Light blue */
            color: #222 !important;
            font-weight: bold;
            font-size: 1.2em;
            border-radius: 8px 8px 0 0;
            margin-right: 4px;
            padding: 10px 24px;
        }
        .stTabs [aria-selected="true"] {
            background-color: #6366f1 !important; /* Indigo */
            color: #fff !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    tabs = st.tabs([
        "Exposed Communication per NPU",
        "Chakra Traces",
        "Chakra Nodes Plot",
        "Roofline Model",
    ])

    with tabs[0]:
        st.subheader(
            "Simulation Time Breakdown per NPU",
            help=(
            "This section visualizes the simulation time breakdown for each NPU showing.\n"
            "- The amount of overlap between communication and computation cycles.\n"
            "- The exposed communication cycles indicating potential bottlenecks.\n"
            "- The exposed computation cycles indicating missed opportunities for optimization.\n"
            )
        )
        astra.visualize_simulation_results(sim_outputs)

    with tabs[1]:
        st.subheader(
            "Chakra Trace",
            help=(
            "This section visualizes the Chakra trace for the simulation.\n"
            "- It provides insights into the timing, overlapped execution and communication of each NPU.\n"
            )
        )
        _init_session_state()
        if "df_matched" not in st.session_state:
            st.warning("The simulation trace is not available yet.")
        else:
            option = st.radio("View Mode", ["Per NPU", "All NPUs"])
            max_npu = int(st.session_state.df_matched["sys_id"].max())
            if option == "Per NPU":
                npu = _load_and_select_npu(max_npu, "trace")
                _show_basic_plots(st.session_state.df_matched, npu, exp = "")#f"Parallelism Strategy: DP:{sim_outputs['dp']}, TP:{sim_outputs['tp']}, SP:{sim_outputs['sp']}, PP:{sim_outputs['pp']}, FSDP:{sim_outputs['sharding_val']}")
            else:
                st.write("Showing Chakra Trace for all NPUs")
                for n in range(max_npu):
                    st.altair_chart(
                        tv.plot_one_npu(st.session_state.df_matched, n, plot_blocks=True, plot_times=False),
                        use_container_width=True,
                    )

    with tabs[2]:
        st.subheader(
            "Chakra Nodes Timing Plot",
            help=(
            "This section visualizes the timing of each Chakra node for a selected NPU.\n"
            "- It helps in understanding the execution time of different nodes and their impact on overall performance.\n"
            )
        )
        max_npu = int(st.session_state.df_matched["sys_id"].max())
        npu = _load_and_select_npu(max_npu, "timing")
        st.altair_chart(tv.plot_one_npu(st.session_state.df_matched, npu, plot_blocks=False, plot_times=True, exp = ""))#)f" of Parallelism Strategy: DP:{sim_outputs['dp']}, TP:{sim_outputs['tp']}, SP:{sim_outputs['sp']}, PP:{sim_outputs['pp']}, FSDP:{sim_outputs['sharding_val']}."))

    with tabs[3]:
        st.subheader(
            "Roofline Model",
            help=(
            "This section visualizes the roofline model for the simulation.\n"
            "- It helps in understanding the performance limits of the system and identifying bottlenecks.\n"
            "- You can select the NPU and the mode of visualization.\n"
            )
        )

        max_npu = int(st.session_state.df_matched["sys_id"].max())
        col0, col1 = st.columns(2)
        with col0:
            npu = _load_and_select_npu(max_npu, "roofline")
        with col1:  
            plot_mode = st.radio(
                "Select Mode", 
                ["2D", "Averaged over time - 3D", "Averaged over time - 2D", "On individual time steps"],
                help=(
                    "Choose the roofline visualization mode:\n\n"
                    "• **2D**: Traditional roofline plot showing operational intensity vs performance\n"
                    "• **Averaged over time - 3D**: 3D roofline with time dimension, averaged over specified window\n" 
                    "• **Averaged over time - 2D**: 2D roofline plot with time-based averaging (Trajectory of the 3D plot)\n"
                    "• **On individual time steps**: Interactive roofline at specific timesteps with playback controls"
                )
            )

        if plot_mode == "2D":
            _show_2d_roofline(st.session_state.df_matched, npu)
        elif plot_mode == "Averaged over time - 3D":
            time_window = st.number_input(
                    "Averaging Time Window (cycles)", value=10000000, step=1000, key="time_window_3d"
                )
            _show_3d_roofline(st.session_state.df_matched, npu, time_window)
        elif plot_mode == "Averaged over time - 2D":
            time_window = st.number_input(
                "Averaging Time Window (cycles)", value=10000000, step=1000, key="time_window_2d"
            )
            _show_2d_roofline_over_time(st.session_state.df_matched, npu, time_window)
        else:
            _show_2d_roofline_timestep(st.session_state.df_matched, npu)