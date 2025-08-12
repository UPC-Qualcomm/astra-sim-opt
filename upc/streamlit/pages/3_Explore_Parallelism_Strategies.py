import streamlit as st
import sections.trace_viewer as tv
import sections.trace_picker as picker
import helper.bulk_analysis_helper as helper
from tabs.bulk_analysis_tab_dim_red_clus_ml import render as render_tab1
from tabs.system_throughput import render_system_throughput as render_tab2
from scipy.stats import gmean
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(layout="wide")
st.markdown("""
    <style>
        .reportview-container {
            margin-top: -2em;
        }
        #MainMenu {visibility: hidden;}
        .stAppDeployButton {display:none;}
    </style>
""", unsafe_allow_html=True)
st.header("Explore Parallelism Strategies", help=(
    "Analyze how different parallelism strategies affect simulation time. \n"
    "- Select a model and configuration. \n"
    "- View simulation time breakdowns for each strategy."))
# Generate Workload
selected_model, selected_config = picker.get_model_and_config()
picker.set_session_peak_perf_bw(selected_config)

st.markdown("---")

col1, col2 = st.columns([12, 3])
with col2:
    option = st.radio(
        "Select Analysis Option:",
        options=["slowest", "average"],
        format_func=lambda x: {
            "slowest": "Slowest NPU",
            "average": "Average Across all NPUs",
        }[x],
        help=(
            "Choose analysis perspective:\n"
            "- **Slowest NPU**: Represent the experiment with the performance of the slowest NPU\n"
            "- **Average Across all NPUs**: Consider average performance across all NPUs"
        )
    )

with col1:
    #TODO: Switch variables rather than hardcoded values
    st.info(f"""
        **Peak performance for Single NPU**: {st.session_state.peak_perf} TFLOPs, \t
        **Peak memory bandwidth**: {st.session_state.peak_bw} GB/s, \n
        **Inter node Bandwidth**: {200} GB/s, \t
        **Intra node Bandwidth**: {900} GB/s, \t
        **Number of NPUs**: {32}
    """)

df = picker.get_all_parallelism_strategies_data(selected_model, selected_config, option)

df_sorted = df.sort_values(by="total", ascending=True)
figs = picker.plot_experiments_bound_breakdown(
    df.sort_values(by="total", ascending=True), chunk_size=33
)
for fig in figs:
    st.pyplot(fig)

st.markdown("<p style='text-align: center; font-size: 0.9em; color: #666; margin-top: 1em;'>Execution breakdown showing the percentage of time spent on memory-bound operations (blue), compute-bound operations (green), and exposed communication (red) for different parallelism strategies. Lower total execution time indicates better performance.</p>", unsafe_allow_html=True)

st.markdown("---")
# Custom CSS for colored tab backgrounds
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

tabs = st.tabs([f"**{name}**" for name in ["Visualization & ML Analysis", "Detailed Trace Visualization", "System Throughput", "ِCommunication Analysis"]])

with tabs[0]:
    st.subheader(
        "Study the relation between the different degrees of parallelism methods and the simulation time",
        help=(
            "A 3D plot to show the relationship between different parallelism strategies and simulation time.\n"
            "- Fix the first two dimensions (e.g., dp and tp).\n"
            "- Vary the third dimension (e.g., sp or pp).\n"
            "- Show neighbours grid for selected dimensions.\n"
            "- View local minima and their records.\n"
            "- Analyze the data slice based on selected dimensions."
        )
    )
    col0, col1 = st.columns([2, 4])
    with col0:
        all_dims = helper.get_dims()

        selected_fsdp = st.checkbox("Enable FSDP", value=True)
        local_min_df_all_axis = helper.find_local_minima_all_axis(
            df[df["fsdp"] == selected_fsdp], "dp", "tp", "sp", "pp", value_col="total"
        )        
        cols_to_show = ["dp", "tp", "sp", "pp", "fsdp", "total", "mem_percent", "comp_percent", "comm_percent"]
        col_rename_map = {
            "dp": "Data Parallelism",
            "tp": "Tensor Parallelism",
            "sp": "Sequence Parallelism",
            "pp": "Pipeline Parallelism",
            "fsdp": "Full Sharded",
            "total": "Time in cycles",
            "mem_percent": "Memory Bound Op (%)",
            "comp_percent": "Compute Bound Op (%)",
            "comm_percent": "Expose Comm (%)",
        }
        global_min_idx = local_min_df_all_axis["total"].idxmin()

        slider_dim1, slider_value1 = helper.select_dim(
            df,
            select_msg="Select slider dimension:",
            dims=all_dims,
            defualt_idx=3,
            key_prefix="first",
            global_min=global_min_idx,
        )
        remaining_dims = helper.get_remaining_dims(all_dims, excluded_dim=slider_dim1)
        slider_dim2, slider_value2 = helper.select_dim(
            df,
            select_msg="Select slider dimension:",
            dims=remaining_dims,
            defualt_idx=2,
            key_prefix="second",
            global_min=global_min_idx,
        )
        remaining_dims = helper.get_remaining_dims(
            remaining_dims, excluded_dim=slider_dim2
        )
        x_dim = st.selectbox("Select x-axis dimension:", remaining_dims, index=0)
        y_dim = st.selectbox("Select y-axis dimension:", remaining_dims, index=1)
        z_dim = "total"
        color_dim = "total"  # st.selectbox("Select color dimension:", remaining_dims, index=3 if len(remaining_dims) > 3 else 0)

        filtered_df = df[
            (df[slider_dim1] == slider_value1)
            & (df[slider_dim2] == slider_value2)
            & (df["fsdp"] == selected_fsdp)
        ]

        st.subheader("Data Slice:", help=("View the data slice based on selected dimensions."))
        st.dataframe(filtered_df.sort_values("total")[cols_to_show].rename(columns=col_rename_map))
        
        if not filtered_df.empty:
            local_min_df = helper.find_local_minima(
                filtered_df, x_dim, y_dim, z_dim, value_col="total"
            )
        if not local_min_df.empty:
            st.subheader(
                "Local Minima Records on the Data Slice:",
                help=(
                    "View the local minima records for the data slice.\n"
                    "- This helps in understanding the performance patterns at lower dimensions."
                )
            )
            st.dataframe(local_min_df.sort_values("total")[cols_to_show].rename(columns=col_rename_map))
        else:
            st.write("No local minima found for the selected configuration.")

    with col1:
        if filtered_df.empty:
            st.warning("No data for the selected combination.")
        else:
            fig = helper.selected_dims_3d_fig(
                filtered_df,
                local_min_df,
                x_dim,
                y_dim,
                z_dim,
                color_dim,
                global_min=global_min_idx,
            )

            show_neighbours_grid = st.checkbox(
                f"Show neighbours grid for {x_dim} vs {y_dim}", value=False
            )
            if show_neighbours_grid:
                fig_neighbours = helper.get_neighbours_map_fig(
                    filtered_df, x_dim, y_dim
                )
                st.plotly_chart(fig_neighbours, use_container_width=True)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.plotly_chart(fig, use_container_width=True)
            
            
    st.subheader(
        "Local Minima Records on the Entire Data (unsliced):",
        help=(
            "View the local minima records across all dimensions.\n"
            "- This helps in understanding the overall performance patterns.\n"
            "- Useful for identifying optimal configurations."
        )
    )
    st.dataframe(local_min_df_all_axis.sort_values("total")[cols_to_show].rename(columns=col_rename_map))

    st.markdown("---")

    st.subheader(
        "Detect the Parallelism Strategies Pattern at Lower Dimensions w.r.t Time",
        help=(
            "This section allows you to explore the parallelism strategies employed at lower dimensions.\n"
            "- Analyze the patterns of parallelism strategies at lower dimensions.\n"
            "- Understand how different strategies impact simulation time."
        )
    )

    render_tab1(df)

    
with tabs[1]:
    
    sim_outputs = picker.set_sim_input(selected_model, selected_config)
    st.markdown("---")
    if "df_matched" in st.session_state:
        tv.render_sim_output_section(sim_outputs)

with tabs[2]:
    render_tab2(df)

with tabs[3]:
    st.subheader(
        "Analyze the Exposed Communication Percentage",
        help=(
            "This section allows you to analyze the exposed communication percentage across different experiments.\n"
            "- The plot shows the sorted exposed communication percentage for each experiment.\n"
            "- It includes mean, standard deviation, and geometric mean."
        )
    )
    comm_mean, comm_std, comm_gmean = (
        df["comm_percent"].mean(),
        df["comm_percent"].std(),
        gmean(df["comm_percent"][df["comm_percent"] > 0]),
    )

    sorted_comm = np.sort(df["comm_percent"].values)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(sorted_comm, marker="o", linestyle="-", color="blue")
    ax.set_xlabel("Number of Experiments (Different Parallelism Strategies)", fontsize=18)
    ax.set_ylabel("Exposed\nComm. Time\n(%)", fontsize=18)
    ax.set_title(
        f"Sorted - Exposed Communication Percentage per Experiment\n(μ={comm_mean:.2f}%, σ={comm_std:.2f}%, gμ={comm_gmean:.2f}%)",
        fontsize=18,
    )
    ax.grid(True)

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.tick_params(axis="both", which="minor", labelsize=16)

    fig.subplots_adjust(left=0.13, right=0.98, top=0.80, bottom=0.22)
    #fig.savefig("comm_percentag.svg", format="svg", bbox_inches="tight")
    st.pyplot(fig)
    st.markdown("<p style='text-align: center; font-size: 0.9em; color: #666; margin-top: 1em;'>Exposed communication percentage across different parallelism strategies, sorted in ascending order. Lower values indicate less communication overhead. Statistical measures include arithmetic mean (μ), standard deviation (σ), and geometric mean (gμ).</p>", unsafe_allow_html=True)
