import os
from pathlib import Path
from functools import reduce
import numpy as np
import pandas as pd
import streamlit as st
import math
import matplotlib.pyplot as plt

# Custom Modules
import sections.trace_picker as picker


def load_and_prepare_file(file_path, selected_files):
    """Load result CSV, convert cycles to seconds, and rename columns."""
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    if file_name not in selected_files:
        return None

    df = pd.read_csv(file_path)
    df = df[["dp_mp_sp_pp_sharded", "exec_cycles", "comm_cycles", "exposed_comm_cycles", "comp_cycles", "exposed_comp_cycles", "seq", "batch"]]

    df["exec_cycles"] = df["exec_cycles"] / 1e9
    df["comm_cycles"] = df["comm_cycles"] / 1e9
    df["exposed_comm_cycles"] = df["exposed_comm_cycles"] / 1e9
    df["comp_cycles"] = df["comp_cycles"] / 1e9
    df["exposed_comp_cycles"] = df["exposed_comp_cycles"] / 1e9
    df["topology"] = file_name
    df["file_name"] = df["dp_mp_sp_pp_sharded"]+".seq_"+ df["seq"].astype(str) + ".batch_" + df["batch"].astype(str)

    return df


def merge_dataframes(df_list):
    """Merge multiple DataFrames on 'dp_mp_sp_pp_sharded'."""
    return pd.concat(df_list, ignore_index=True) #reduce(lambda left, right: pd.merge(left, right, on="dp_mp_sp_pp_sharded", how="inner"), df_list)


def compute_summary_stats(merged_df, selected_files):
    """Compute avg, std, and geomean for exec, comm, and comp cycles."""
    summary_stats = []
    for name in selected_files:
        topo_df = merged_df[merged_df["topology"] == name]
        exec_vals = topo_df[f"exec_cycles"]
        comm_vals = topo_df[f"comm_cycles"]
        exposed_comm_vals = topo_df[f"exposed_comm_cycles"]
        comp_vals = topo_df[f"comp_cycles"]
        exposed_comp_vals = topo_df[f"exposed_comp_cycles"]

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

def get_summary_plot(summary_df, figsize=(10, 3)):
    summary_df['avg_overlap (s)'] = summary_df['avg_exec (s)'] - summary_df['avg_exposed_comm (s)'] - summary_df['avg_exposed_comp (s)']
    summary_df = summary_df.sort_values('topology')

    components = ['avg_overlap (s)', 'avg_exposed_comm (s)', 'avg_exposed_comp (s)']
    labels = ['Overlap', 'Exposed Comm', 'Exposed Comp']
    colors = ['blue', 'lightcoral', 'lightgreen']

    x = summary_df['topology']
    bottom = [0] * len(summary_df)

    fig, ax = plt.subplots(figsize=figsize)

    for comp, color, label in zip(components, colors, labels):
        values = summary_df[comp]
        bars = ax.bar(x, values, bottom=bottom, label=label, color=color)

        for bar, val, base, topo in zip(bars, values, bottom, x):
            total = summary_df.loc[summary_df['topology'] == topo, 'avg_exec (s)'].values[0]
            percent = (val / total) * 100
            height = bar.get_height()
            if height > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    base + height / 2,
                    f"{percent:.2f}%",
                    ha='center',
                    va='center',
                    fontsize=9,
                    color='black'
                )

        bottom = [i + j for i, j in zip(bottom, values)]

    ax.set_xlabel('Topology')
    ax.set_ylabel('Average Time (s)')
    ax.set_title('Time Breakdown by Topology - Averaged Across Various Parallelsim Strategies')
    
    # Move legend outside right
    ax.legend(title='Breakdown Components', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    ax.tick_params(axis='x', labelrotation=45)
    #ax.grid(axis='y', linestyle='--', alpha=0.7)
    fig.tight_layout()

    return fig


def get_compare_topology_per_range(merged_df, start_idx, end_idx, configs_sorted, selected_files):
    labels = ['Overlap', 'Exposed Comm', 'Exposed Comp']
    colors = ['blue', 'lightcoral', 'lightgreen']

    selected_configs = configs_sorted[start_idx:end_idx]
    valid_configs = []
    for config in selected_configs:
        config_rows = merged_df[merged_df['dp_mp_sp_pp_sharded'] == config]
        topologies_present = config_rows['topology'].unique().tolist()
        if all(topo in topologies_present for topo in selected_files):
            valid_configs.append(config)

    if len(valid_configs) == 0:
        st.warning("No valid configs present in all selected topologies in this range.")
        return None

    num_configs = len(valid_configs)
    fig, axes = plt.subplots(1, num_configs, figsize=(6*num_configs, 8), squeeze=False)
    axes = axes[0]

    for i, config in enumerate(valid_configs):
        ax = axes[i]

        compare_df = merged_df[
            (merged_df['dp_mp_sp_pp_sharded'] == config) &
            (merged_df['topology'].isin(selected_files))
        ]

        compare_df = compare_df.set_index('topology').loc[selected_files].reset_index()
        compare_df['exec_cycles'] = compare_df['exec_cycles'] * 1e9
        compare_df['exposed_comm_cycles'] = compare_df['exposed_comm_cycles'] * 1e9
        compare_df['exposed_comp_cycles'] = compare_df['exposed_comp_cycles'] * 1e9
        overlap = compare_df['exec_cycles'] - (compare_df['exposed_comm_cycles'] + compare_df['exposed_comp_cycles'])
        exposed_comm = compare_df['exposed_comm_cycles']
        exposed_comp = compare_df['exposed_comp_cycles']

        bar_width = 0.5
        indices = np.arange(len(selected_files))

        p1 = ax.bar(indices, overlap, bar_width, label='Overlap', color=colors[0])
        p2 = ax.bar(indices, exposed_comm, bar_width, bottom=overlap, label='Exposed Comm', color=colors[1])
        p3 = ax.bar(indices, exposed_comp, bar_width, bottom=overlap+exposed_comm, label='Exposed Comp', color=colors[2])

        total = overlap + exposed_comm + exposed_comp

        for j in range(len(indices)):
            y_offset = 0
            for value in [overlap[j], exposed_comm[j], exposed_comp[j]]:
                pct = (value / total[j]) * 100
                ax.text(indices[j], y_offset + value / 2, f"{pct:.1f}%", 
                        ha='center', va='center', color='black', fontsize=12)
                y_offset += value

        ax.set_xticks(indices)
        ax.set_xticklabels(compare_df['topology'], fontsize=13)
        ax.tick_params(axis='x', labelrotation=45)
        ax.set_ylabel('Cycles', fontsize=16)
        ax.set_xlabel('Topology', fontsize=14)

        config_clean = config.replace('_res', '')
        dp, tp, sp, pp, fsdp = config_clean.split('_')
        title = f"DP={dp}, TP={tp}, SP={sp}, PP={pp}, FSDP={fsdp}, #NPUs={int(dp)*int(tp)*int(sp)*int(pp)}"
        ax.set_title(title, fontsize=16)

        ax.tick_params(axis='y', labelsize=13)  # y-axis tick values fontsize

        if i == 0:
            ax.legend(fontsize=12)

    fig.tight_layout()
    return fig


def add_topology_and_min_exec_cycles(merged_df):
    """Add columns for topology with lowest exec_cycles and its value per config."""

    TOPOLOGY_PRIORITY = ['FullyConnected', 'Switch', 'Ring', 'Dragonfly', 'FoldedClos', '2D_Torus', '3D_Torus']

    config_cols = [col for col in merged_df.columns if col not in ['topology', 'exec_cycles']]
    
    # Find min exec_cycles per config and corresponding topology with priority
    def pick_topology(group):
        min_value = group['exec_cycles'].min()
        topologies_with_min = group.loc[group['exec_cycles'] == min_value, 'topology'].tolist()
        # Pick topology with highest priority
        for topo in TOPOLOGY_PRIORITY:
            if topo in topologies_with_min:
                return pd.Series({'min_exec_cycles_value': min_value, 'topology': topo})
        return pd.Series({'min_exec_cycles_value': min_value, 'topology': topologies_with_min[0]})

    min_exec_cycles_df = (
        merged_df.groupby(config_cols)
        .apply(pick_topology)
        .reset_index()
    )

    return min_exec_cycles_df


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
    st.subheader("Results Report")

    # Extract experiment names
    file_names = sorted([os.path.splitext(os.path.basename(f))[0] for f in gathered_res_files])

    # Experiment selection checkboxes (in one row)
    st.text("Select The Topologies to Include")
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
    fig = get_summary_plot(stats_df)
    st.pyplot(fig)
    #**#st.header("Summary Statistics per Topology - ordered by avg_exec (lower to higher)")
    #**#st.dataframe(stats_df.sort_values('avg_exec (s)'))

    # Identify per-config best topology and min exec_cycles
    min_df = add_topology_and_min_exec_cycles(merged_df)

    # Count Best-Performing Topologies
    counts_df = count_best_topologies(min_df)
    st.subheader("Number of times each topology had the lowest simulation time")
    st.dataframe(counts_df)
    
    min_df = min_df[min_df["topology"] == counts_df["Topology"][0]]
    # Top-N Best Experiments with Slider
    st.subheader(f"Top Experiments (Parallelism Strategies) with Lowest simulation time in the best topology - {counts_df['Topology'][0]}")
    n_range = st.slider(
        "Select range of experiments to display:",
        0, min_df["dp_mp_sp_pp_sharded"].nunique(), (0, 4), step=1
    )

    n_start, n_end = n_range
    top_configs = get_top_n_configs(min_df, n_start, n_end)

    #st.subheader(f"Top experiments from {n_start} to {n_end}")
    st.dataframe(top_configs)

    configs_sorted = merged_df[merged_df['topology'] == counts_df["Topology"][0]].sort_values('exec_cycles')['dp_mp_sp_pp_sharded'].unique().tolist()

    fig = get_compare_topology_per_range(
        merged_df, n_start, n_end, configs_sorted, selected_files
    )

    return fig, selected_files, merged_df[merged_df["topology"] == counts_df["Topology"][0]].sort_values('exec_cycles')