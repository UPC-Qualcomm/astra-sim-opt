import streamlit as st
import sections.trace_viewer as tv
import sections.trace_picker as picker
import helper.bulk_analysis_helper as helper
from tabs.bulk_analysis_tab_dim_red_clus_ml import render as render_tab1
from tabs.system_throughput import render_system_throughput as render_tab2

st.set_page_config(layout="wide")
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
    )

with col1:
    st.info(f"""
        **Peak performance**: {st.session_state.peak_perf} TFLOPs, \t
        **Peak bandwidth**: {st.session_state.peak_bw} GB/s
    """)

df = picker.get_all_parallelism_strategies_data(selected_model, selected_config, option)


df_sorted = df.sort_values(by="total", ascending=True)
figs = picker.plot_experiments_bound_breakdown(
    df.sort_values(by="total", ascending=True), chunk_size=32
)
for fig in figs:
    st.pyplot(fig)

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

tabs = st.tabs([f"**{name}**" for name in ["Visualization & ML Analysis", "Detailed Trace Visualization", "System Throughput"]])

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

        selected_fsdp = st.checkbox("Use FSDP?", value=True)
        local_min_df_all_axis = helper.find_local_minima_all_axis(
            df[df["fsdp"] == selected_fsdp], "dp", "tp", "sp", "pp", value_col="total"
        )
        st.write(local_min_df_all_axis)
        global_min_idx = local_min_df_all_axis["total"].idxmin()

        slider_dim1, slider_value1 = helper.select_dim(
            df,
            select_msg="Select dimension for the slider:",
            dims=all_dims,
            defualt_idx=3,
            key_prefix="first",
            global_min=global_min_idx,
        )
        remaining_dims = helper.get_remaining_dims(all_dims, excluded_dim=slider_dim1)
        slider_dim2, slider_value2 = helper.select_dim(
            df,
            select_msg="Select another dimension for the slider:",
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
        st.write(filtered_df.sort_values("total"))

    with col1:
        if filtered_df.empty:
            st.warning("No data for the selected combination.")
        else:
            local_min_df = helper.find_local_minima(
                filtered_df, x_dim, y_dim, z_dim, value_col="total"
            )

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
                "Local Minima Records on the data slice:",
                help=(
                    "View the local minima records for the data slice.\n"
                    "- This helps in understanding the performance patterns at lower dimensions."
                )
            )
            if not local_min_df.empty:
                st.dataframe(local_min_df.sort_values("total"))
            else:
                st.write("No local minima found for the selected configuration.")

    st.subheader(
        "Local Minima Records on the entire data (unsliced):",
        help=(
            "View the local minima records across all dimensions.\n"
            "- This helps in understanding the overall performance patterns.\n"
            "- Useful for identifying optimal configurations."
        )
    )
    st.dataframe(local_min_df_all_axis.sort_values("total"))

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
        tv.render_sim_ouput_section(sim_outputs)

with tabs[2]:
    render_tab2(df)
