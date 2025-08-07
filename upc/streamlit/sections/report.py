import os
from pathlib import Path
from functools import reduce
import numpy as np
import pandas as pd
import streamlit as st
import math
from matplotlib.patches import Patch
import matplotlib.pyplot as plt

# Custom Modules
import sections.trace_picker as picker
import helper.constants as constants

@st.cache_data
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

@st.cache_data
def merge_dataframes(df_list):
    """Merge multiple DataFrames on 'dp_mp_sp_pp_sharded'."""
    return pd.concat(df_list, ignore_index=True) #reduce(lambda left, right: pd.merge(left, right, on="dp_mp_sp_pp_sharded", how="inner"), df_list)

@st.cache_data
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

@st.cache_data
def get_summary_plot(summary_df, figsize=(10, 6)):
    font_increment = -10  # Adjust this value to increase font size
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
                    fontsize=constants.IN_PLOT_LABEL_SIZE+font_increment,
                    color='black'
                )

        bottom = [i + j for i, j in zip(bottom, values)]

    ax.set_xlabel('Topology', fontsize=constants.LABEL_SIZE+font_increment)
    ax.set_ylabel('Average Time (s)', fontsize=constants.LABEL_SIZE+font_increment)
    ax.set_title(
        'Time Breakdown by Topology - Averaged Across Various Parallelism Strategies',
        fontsize=constants.TITLE_SIZE+font_increment
    )

    # Move legend outside right
    ax.legend(
        title='Breakdown Components',
        bbox_to_anchor=(1., 1),
        loc='upper left',
        borderaxespad=0.,
        fontsize=constants.LEGEND_SIZE+font_increment,
        title_fontsize=constants.LEGEND_SIZE+font_increment
    )

    ax.tick_params(axis='x', labelrotation=45, labelsize=constants.XTICK_SIZE+font_increment)
    ax.tick_params(axis='y', labelsize=constants.YTICK_SIZE+font_increment)
    #ax.grid(axis='y', linestyle='--', alpha=0.7)
    fig.tight_layout()

    return fig

def get_compare_topology_per_range(merged_df, selected_configs, selected_files):
    labels = ['Overlap', 'Exposed Comp', 'Exposed Comm']
    colors = ['lightblue', 'lightgreen', 'lightcoral']
    
    valid_configs = []
    for config in selected_configs:
        config_rows = merged_df[merged_df['dp_mp_sp_pp_sharded'] == config]
        topologies_present = config_rows['topology'].unique().tolist()
        if all(topo in topologies_present for topo in selected_files):
            valid_configs.append(config)
    font_increment = 10
    if len(valid_configs) == 0:
        st.warning("No valid configs present in all selected topologies in this range.")
        return None

    num_configs = len(valid_configs)
    ncols = 4  
    nrows = math.ceil(num_configs / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(12*ncols, 11*nrows), squeeze=False)
    axes = axes.flatten()

    # Custom legend as a horizontal bar at the top
    legend_handles = [Patch(facecolor=color, label=label) for label, color in zip(labels, colors)]
    legend = fig.legend(
        handles=legend_handles,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.05),
        ncol=len(labels),
        fontsize=constants.LEGEND_SIZE+font_increment,
        frameon=False
    )

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
        exposed_comp = compare_df['exposed_comp_cycles']
        exposed_comm = compare_df['exposed_comm_cycles']

        bar_width = 0.7
        indices = np.arange(len(selected_files))

        p1 = ax.bar(indices, overlap, bar_width, label=labels[0], color=colors[0])
        p2 = ax.bar(indices, exposed_comp, bar_width, bottom=overlap, label=labels[1], color=colors[1])
        p3 = ax.bar(indices, exposed_comm, bar_width, bottom=overlap+exposed_comp, label=labels[2], color=colors[2])

        total = overlap + exposed_comp + exposed_comm

        for j in range(len(indices)):
            y_offset = 0
            for k, value in enumerate([overlap[j], exposed_comp[j], exposed_comm[j]]):
                pct = (value / total[j]) * 100
                text_color = 'black'
                ax.text(
                    indices[j],
                    y_offset + value / 2,
                    f"{pct:.1f}%",
                    ha='center',
                    va='center',
                    color=text_color,
                    fontsize=constants.IN_PLOT_LABEL_SIZE + font_increment
                )
                y_offset += value

        # Only show x ticks on bottom row
        row_idx = i // ncols
        col_idx = i % ncols
        if row_idx == nrows - 1:
            ax.set_xticks(indices)
            ax.set_xticklabels(compare_df['topology'], fontsize=constants.FONT_SIZE + font_increment)
            ax.tick_params(axis='x', labelrotation=25)
        else:
            ax.set_xticks([])
            ax.set_xticklabels([])

        # Only show y ticks/label on first column
        if col_idx == 0:
            ax.set_ylabel('Time (Cycles)', fontsize=constants.LABEL_SIZE + font_increment)
            ax.yaxis.get_offset_text().set_fontsize(constants.YTICK_SIZE + font_increment)
            ax.tick_params(axis='y', labelsize=constants.YTICK_SIZE + font_increment)
        else:
            ax.set_yticks([])
            ax.set_yticklabels([])

        config_clean = config.replace('_res', '')
        dp, tp, sp, pp, fsdp = config_clean.split('_')
        title = f"DP={dp}, TP={tp}, SP={sp}\nPP={pp}, FSDP={fsdp}"
        ax.set_title(title, fontsize=constants.TITLE_SIZE + font_increment)

    # Hide unused axes if any
    for j in range(len(valid_configs), len(axes)):
        fig.delaxes(axes[j])

    fig.tight_layout(rect=[0, 0, 1, 0.97])  # leave space for legend at top

    # Ensure legend is included in SVG by passing extra_artists
    #fig.savefig("topo_2.svg", format='svg', bbox_extra_artists=[legend], bbox_inches='tight')
    return fig

@st.cache_data
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

@st.cache_data
def get_top_n_configs(merged_df, selected_configs):
    """Get selected configs with their topology and exec_cycles values."""
    top_configs = merged_df[
        merged_df['dp_mp_sp_pp_sharded'].isin(selected_configs)
    ][['dp_mp_sp_pp_sharded', 'topology', 'min_exec_cycles_value']].drop_duplicates().reset_index(drop=True)

    return top_configs

@st.cache_data
def count_best_topologies(merged_df):
    """Count how many times each topology had the best exec_cycles."""
    counts = merged_df['topology'].value_counts().reset_index()
    counts.columns = ['Topology', 'Number of Experiment']
    return counts


def analysis_across_topologies(selected_model):
    gathered_res_dir = Path(__file__).parent / "../../results" / selected_model
    gathered_res_files = picker.get_files_list(gathered_res_dir, ".csv")
    st.subheader(
        "Results Report",
        help=(
            "This section provides a comprehensive report on simulation results across different interconnect network topologies.\n"
            "- Summary statistics for average execution time, exposed communication time, and overlapped time across available parallelism strategies."
        )
    )

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
    st.subheader(
        "Topologies score",
        help=(
            "This section shows how many times each topology had the lowest simulation time across various parallelism strategies.\n"
            "- Helps in identifying the most effective interconnect network designs."
        )
    )
    st.dataframe(counts_df)
    
    min_df = min_df[min_df["topology"] == counts_df["Topology"][0]]
    # Top-N Best Experiments with Multiselect
    st.subheader(
        f"Top Parallelism Strategies with Lowest Simulation Time in Best Topology: {counts_df['Topology'][0]}",
        help=(
            "This section highlights the parallelism strategies that achieved the lowest simulation time "
            "within the best-performing topology.\n"
            "- Use this to identify the most efficient parallelism configurations for your workload."
        )
    )
    
    configs_sorted = merged_df[merged_df['topology'] == counts_df["Topology"][0]].sort_values('exec_cycles')['dp_mp_sp_pp_sharded'].unique().tolist()
    max_selection = 4
    default_configs = configs_sorted[:min(max_selection, len(configs_sorted))]
    
    # Multiselect with maximum 8 options
    selected_configs = st.multiselect(
        "Select parallelism strategies to compare (max 8):",
        options=configs_sorted,
        default=default_configs,
        max_selections=max_selection
    )
    
    if not selected_configs:
        st.warning("Please select at least one parallelism strategy.")
        return None, selected_files, merged_df[merged_df["topology"] == counts_df["Topology"][0]].sort_values('exec_cycles')

    fig = get_compare_topology_per_range(
        merged_df, selected_configs, selected_files
    )

    return fig, selected_files, merged_df[merged_df["topology"] == counts_df["Topology"][0]].sort_values('exec_cycles')