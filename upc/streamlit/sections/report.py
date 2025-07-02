import os
from pathlib import Path
from functools import reduce
import numpy as np
import pandas as pd
import streamlit as st
import math

# Custom Modules
import sections.trace_picker as picker


def load_and_prepare_file(file_path, selected_files):
    """Load result CSV, convert cycles to seconds, and rename columns."""
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    if file_name not in selected_files:
        return None

    df = pd.read_csv(file_path)
    df = df[["dp_mp_sp_pp_sharded", "exec_cycles", "comm_cycles", "exposed_comm_cycles", "comp_cycles", "exposed_comp_cycles"]]

    df["exec_cycles"] = df["exec_cycles"] / 1e9
    df["comm_cycles"] = df["comm_cycles"] / 1e9
    df["exposed_comm_cycles"] = df["exposed_comm_cycles"] / 1e9
    df["comp_cycles"] = df["comp_cycles"] / 1e9
    df["exposed_comp_cycles"] = df["exposed_comp_cycles"] / 1e9

    df = df.rename(columns={
        "exec_cycles": f"{file_name}_exec_cycles",
        "comm_cycles": f"{file_name}_comm_cycles",
        "exposed_comm_cycles": f"{file_name}_exposed_comm_cycles",
        "comp_cycles": f"{file_name}_comp_cycles",
        "exposed_comp_cycles": f"{file_name}_exposed_comp_cycles"
    })

    return df


def merge_dataframes(df_list):
    """Merge multiple DataFrames on 'dp_mp_sp_pp_sharded'."""
    return reduce(lambda left, right: pd.merge(left, right, on="dp_mp_sp_pp_sharded", how="inner"), df_list)


def compute_summary_stats(merged_df, selected_files):
    """Compute avg, std, and geomean for exec, comm, and comp cycles."""
    summary_stats = []
    for name in selected_files:
        exec_vals = merged_df[f"{name}_exec_cycles"]
        comm_vals = merged_df[f"{name}_comm_cycles"]
        exposed_comm_vals = merged_df[f"{name}_exposed_comm_cycles"]
        comp_vals = merged_df[f"{name}_comp_cycles"]
        exposed_comp_vals = merged_df[f"{name}_exposed_comp_cycles"]

        summary_stats.append({
            'topology': name,
            'avg_exec (s)': exec_vals.mean(),
            'std_exec (s)': exec_vals.std(),
            'geo_exec (s)': math.prod(exec_vals) ** (1 / len(exec_vals)),

            'avg_comm (s)': comm_vals.mean(),
            'std_comm (s)': comm_vals.std(),
            'geo_comm (s)': math.prod(comm_vals) ** (1 / len(comm_vals)),

            'avg_exposed_comm (s)': exposed_comm_vals.mean(),
            'std_exposed_comm (s)': exposed_comm_vals.std(),
            'geo_exposed_comm (s)': math.prod(exposed_comm_vals) ** (1 / len(exposed_comm_vals)),

            'avg_comp (s)': comp_vals.mean(),
            'std_comp (s)': comp_vals.std(),
            'geo_comp (s)': math.prod(comp_vals) ** (1 / len(comp_vals)),

            'avg_exposed_comp (s)': exposed_comp_vals.mean(),
            'std_exposed_comp (s)': exposed_comp_vals.std(),
            'geo_exposed_comp (s)': math.prod(exposed_comp_vals) ** (1 / len(exposed_comp_vals)),
        })

    return pd.DataFrame(summary_stats)


def add_topology_and_min_exec_cycles(merged_df):
    """Add columns for topology with lowest exec_cycles and its value per config."""
    exec_cols = [col for col in merged_df.columns if col.endswith('_exec_cycles')]
    merged_df['topology'] = merged_df[exec_cols].idxmin(axis=1).str.replace('_exec_cycles', '')
    merged_df['min_exec_cycles_value'] = merged_df[exec_cols].min(axis=1)
    return merged_df


def get_top_n_configs(merged_df, n_start, n_end):
    """Identify top N configs (by range) with lowest exec_cycles and their topology."""
    top_configs = merged_df.nsmallest(n_end, 'min_exec_cycles_value')[
        ['dp_mp_sp_pp_sharded', 'topology', 'min_exec_cycles_value']
    ].iloc[n_start:n_end].reset_index(drop=True)

    return top_configs


def count_best_topologies(merged_df):
    """Count how many times each topology had the best exec_cycles."""
    counts = merged_df['topology'].value_counts().reset_index()
    counts.columns = ['Topology', 'Number of Experiment']
    return counts


def analysis_across_topologies(selected_model):
    gathered_res_dir = Path(__file__).parent / "../../results" / selected_model
    gathered_res_files = picker.get_files_list(gathered_res_dir, ".csv")
    st.title("Experiment Results Report")

    # Extract experiment names
    file_names = sorted([os.path.splitext(os.path.basename(f))[0] for f in gathered_res_files])

    # Experiment selection checkboxes (in one row)
    st.subheader("Select The Topologies to Include")
    cols = st.columns(len(file_names))
    selected_files = []
    for i, name in enumerate(file_names):
        with cols[i]:
            if st.checkbox(name, value=True, key=f"chk_{name}"):
                selected_files.append(name)

    if not selected_files:
        st.warning("Please select at least one topology to display results.")
        st.stop()

    # Load and prepare selected files
    df_list = [load_and_prepare_file(res_file, selected_files) for res_file in gathered_res_files]
    df_list = [df for df in df_list if df is not None]

    if not df_list:
        st.warning("No matching data files found for selected experiments.")
        st.stop()

    # Merge DataFrames
    merged_df = merge_dataframes(df_list)

    # Compute Summary Statistics
    stats_df = compute_summary_stats(merged_df, selected_files)
    st.header("Summary Statistics per Topology - ordered by avg_exec (lower to higher)")
    st.dataframe(stats_df.sort_values('avg_exec (s)'))

    # Identify per-config best topology and min exec_cycles
    merged_df = add_topology_and_min_exec_cycles(merged_df)

    # Count Best-Performing Topologies
    counts_df = count_best_topologies(merged_df)
    st.header("Number of times each topology had the lowest exec_cycles")
    st.dataframe(counts_df)

    # Top-N Best Configurations with Slider
    st.header("Top Configurations with Lowest exec_cycles")

    n_range = st.slider(
        "Select range of configurations to display:",
        0, len(merged_df), (0, 5), step=1
    )

    n_start, n_end = n_range
    top_configs = get_top_n_configs(merged_df, n_start, n_end)

    st.subheader(f"Top configs from {n_start} to {n_end}")
    st.dataframe(top_configs)


