import streamlit as st
import os
import pandas as pd
import numpy as np
import trace_visualization as tv
import roofline_visualization as rv
import time
import streamlit as st
import os
import pandas as pd
import numpy as np
import time
import trace_visualization as tv
import roofline_visualization as rv


def trace_viewer(sim_outputs):
    st.title("📊 Trace Visualization Viewer")
    _init_session_state()
    if "df_matched" not in st.session_state:
        st.warning("The simulation trace is not available yet.")
    else:
        npu, max_npu = _load_and_select_npu(st.session_state.df_matched)
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


def _load_and_select_npu(df):
    max_npu = int(df["sys_id"].max())
    npu = st.number_input("NPU index", min_value=0, max_value=max_npu, value=0, step=1)
    if "active_plot" not in st.session_state:
        st.session_state.active_plot = None
    return npu, max_npu


def _show_basic_plots(df, npu):
    st.subheader("Visualize the compute and communication node over time.")
    st.altair_chart(tv.plot_one_npu(df, npu), use_container_width=True)
    mem, comp, idle = rv.get_info(
        df,
        npu=npu,
        perf=st.session_state.peak_perf,
        bw=st.session_state.peak_bw,
    )
    st.info(f"""
        **Percentage of time spent with memory bound operations**: {mem:.2f}% \n
        **Percentage of time spent with compute bound operations**: {comp:.2f}% \n
        **Percentage of time spent with idle**: {idle:.2f} \n
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
            "Averaging Time Window (cycles)", value=10000, step=1000
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


def _show_2d_roofline(df, npu):
    st.subheader("Visualize 2D roofline model.")
    st.altair_chart(
        rv.get_2d_roofline_plot_normal(
            df,
            npu,
            perf=st.session_state.peak_perf,
            bw=st.session_state.peak_bw,
        )
    )


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
            label_visibility="collapsed",
        )
        if selected_timestep != timesteps[st.session_state.timestep_idx]:
            st.session_state.timestep_idx = int(
                np.where(timesteps == selected_timestep)[0][0]
            )


def _show_timestep_plot_and_table(df, timesteps):
    df = df[df["issue_tick"] == timesteps[st.session_state.timestep_idx]]
    col1, col2 = st.columns([2, 1])
    with col1:
        st.altair_chart(
            rv.get_2d_roofline_plot_timestep(
                df, perf=st.session_state.peak_perf, bw=st.session_state.peak_bw
            )
        )
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
