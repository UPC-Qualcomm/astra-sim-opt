import streamlit as st
import sections.trace_picker as picker
import helper.bulk_analysis_helper as helper

# Custom Modules
from tabs.bulk_analysis_tab_dim_red_clus_ml import render as render_tab1
from tabs.bulk_analysis_tab_best_comb import render as render_tab2

st.set_page_config(page_title="Bulk analysis", layout="wide")

st.title("Parallelism Strategy - Execution time Breakdown")

selected_model, selected_config = picker.get_model_and_config()
picker.set_session_peak_perf_bw(selected_config)

st.markdown("---")

col1, col2 = st.columns([11, 2])
with col2:
    option = st.radio(
        "Select Bound Analysis Option:",
        options=["slowest", "fastest", "average"],
        format_func=lambda x: {
            "slowest": "Slowest NPU",
            "fastest": "Fastest NPU",
            "average": "Average Across all NPUs",
        }[x],
    )

with col1:
    st.info(f"""
        **Peak performance**: {st.session_state.peak_perf} TFLOPs, \t
        **Peak bandwidth**: {st.session_state.peak_bw} GB/s
    """)

df = picker.get_all_parallelism_strategies_data(selected_model, selected_config, option)

col0, col1 = st.columns([2, 4])
with col0:
    all_dims = helper.get_dims()

    selected_fsdp = st.checkbox("Use FSDP?", value=True)
    local_min_df_all_axis = helper.find_local_minima_all_axis(
        df[df["fsdp"] == selected_fsdp], "dp", "tp", "sp", "pp", value_col="total"
    )
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
    remaining_dims = helper.get_remaining_dims(remaining_dims, excluded_dim=slider_dim2)
    x_dim = st.selectbox("Select x-axis dimension:", remaining_dims, index=0)
    y_dim = st.selectbox("Select y-axis dimension:", remaining_dims, index=1)
    z_dim = "total"
    color_dim = "total"  # st.selectbox("Select color dimension:", remaining_dims, index=3 if len(remaining_dims) > 3 else 0)

    filtered_df = df[
        (df[slider_dim1] == slider_value1)
        & (df[slider_dim2] == slider_value2)
        & (df["fsdp"] == selected_fsdp)
    ]

    st.subheader("Data Slice:")
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
            fig_neighbours = helper.get_neighbours_map_fig(filtered_df, x_dim, y_dim)
            st.plotly_chart(fig_neighbours, use_container_width=True)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.plotly_chart(fig, use_container_width=True)
        st.subheader("Local Minima Records on the data slice:")
        if not local_min_df.empty:
            st.dataframe(local_min_df.sort_values("total"))
        else:
            st.write("No local minima found for the selected configuration.")

st.subheader("Local Minima Records on the entire data (unsliced):")
st.dataframe(local_min_df_all_axis.sort_values("total"))

st.markdown("---")

tabs = [
    "### Dimension Reduction & Clustering & ML Models",
    "### Best Performing Examples",
]

if "active_tab" not in st.session_state:
    st.session_state.active_tab = 0


col0, col1, _ = st.columns([2, 2, 8])
if col0.button(tabs[0]):
    st.session_state.active_tab = 0
if col1.button(tabs[1]):
    st.session_state.active_tab = 1


if st.session_state.active_tab == 0:
    render_tab1(df)
elif st.session_state.active_tab == 1:
    render_tab2(df, selected_model, selected_config)
