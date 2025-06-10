import glob
import itertools
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import pandas as pd
import regex as re
import yaml
from intervaltree import IntervalTree
from scipy.stats import gmean
from tqdm import tqdm
import matplotlib.pyplot as plt
import plotly.colors as pc
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Custom Modules
import sections.trace_picker as picker
import trace_visualization as tv


parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(parent_dir)


from run_astrasim import run_astrasim


def render(df, selected_model, selected_config):
    options = list(range(1, len(df) + 1))
    num = st.select_slider(
        "Pick The Number of Best Experiments To Consider:", options=options, value=10
    )
    df_top = df.sort_values(by="total", ascending=True).head(num)
    st.subheader(f"Best {num} examples")
    st.dataframe(df_top)

    figs = picker.plot_experiments_bound_breakdown(df_top, chunk_size=32)
    for fig in figs:
        st.pyplot(fig)

    st.pyplot(get_exposed_comm_fig(df_top))

    st.markdown("---")

    # TODO: Move the calculation of the uncovered communication to the simulation scripts
    selected_file_name = st.selectbox(
        "Select an experiment:", df_top["file_name"].unique().tolist()
    )

    df_trace = load_and_prepare_trace(
        selected_file_name, selected_model, selected_config
    )
    sys_id_max_cb = get_sys_id_with_max_callback(df_trace)
    final_results_comm = get_exposed(df_trace, sys_id_max_cb, node_op_type="comm")
    final_results_comp = get_exposed(df_trace, sys_id_max_cb, node_op_type="comp")
    avg_bw_across_dims, _ = calculate_avg_bw_across_dims(selected_config)

    # TODO: Use more accurate version after identifying the links
    # Calculate the size of data bieng transfared during this unhidden communication.
    final_results_comm = calculate_data_size(final_results_comm, avg_bw_across_dims)
    max_idx = final_results_comm["exposed"].idxmax()
    recomended_avg_bw = calculate_optimal_avg_bw(
        final_results_comm.at[max_idx, "mean_data_size_nonoverlap"],
        100000,
    )
    final_results_comp["mean_data_size_nonoverlap"] = -1
    final_results_merged = pd.concat([final_results_comm, final_results_comp])
    final_results_merged["exposed_percent"] = (
        100 * final_results_merged["exposed"] / final_results_merged["elapsed_time"]
    )
    final_results_merged["overlap_percent"] = (
        100 * final_results_merged["overlap"] / final_results_merged["elapsed_time"]
    )
    final_results_merged = final_results_merged[
        [
            col
            for col in final_results_merged.columns
            if col != "mean_data_size_nonoverlap"
        ]
        + ["mean_data_size_nonoverlap"]
    ]
    st.dataframe(final_results_merged.sort_values("exposed", ascending=False))
    st.altair_chart(
        tv.plot_one_npu(
            df_trace,
            final_results_comm["sys_id"].values[0],
            plot_blocks=True,
            plot_times=False,
        )
    )

    st.title("AstraSim Bandwidth Sweep Runner")

    with st.form("bw_sweep_form"):
        st.subheader("Bandwidth Ranges Configuration")

        col1, col2, col3 = st.columns(3)
        with col1:
            bw1_min = st.number_input(
                "BW1 Min (GB/s)", min_value=100, max_value=5000, value=900, step=50
            )
        with col2:
            bw1_max = st.number_input(
                "BW1 Max (GB/s)", min_value=100, max_value=5000, value=1800, step=50
            )
        with col3:
            bw1_step = st.number_input(
                "BW1 Step (GB/s)", min_value=100, max_value=5000, value=100, step=50
            )

        col4, col5, col6 = st.columns(3)
        with col4:
            bw2_min = st.number_input(
                "BW2 Min (GB/s)", min_value=50, max_value=2000, value=400, step=50
            )
        with col5:
            bw2_max = st.number_input(
                "BW2 Max (GB/s)", min_value=50, max_value=2000, value=1200, step=50
            )
        with col6:
            bw2_step = st.number_input(
                "BW2 Step (GB/s)", min_value=50, max_value=2000, value=100, step=50
            )

        run_button = st.form_submit_button("Run Simulations")

    parallelism_strategies = df_top["file_name"].tolist()
    app_dir = Path(__file__).parent
    if run_button:
        st.info("Starting bandwidth sweep simulations...")

        base_yml_path = picker._get_configs_dir() + "/" + selected_config + ".yml"
        output_config_dir = picker._get_output_dir() + "/" + selected_config
        max_workers = os.cpu_count()

        run_bandwidth_sweep_parallel(
            base_yml_path=base_yml_path,
            output_config_dir=output_config_dir,
            parallelism_strategies=parallelism_strategies,
            bw1_values=list(range(int(bw1_min), int(bw1_max) + 1, int(bw1_step))),
            bw2_values=list(range(int(bw2_min), int(bw2_max) + 1, int(bw2_step))),
            app_dir=app_dir,
            max_workers=max_workers,
            selected_config=selected_config,
            selected_model=selected_model,
        )

        st.success("✅ All simulations completed successfully!")

    st.markdown("---")

    result = app_dir / "../../results" / selected_model

    st.title("Execution vs Communication Cycles Analysis")

    if not result.exists() or not any(result.iterdir()):
        st.warning("Results directory is empty. Please run simulations first.")
    else:
        all_data = []

        for prefix in parallelism_strategies:
            st.subheader(f"Prefix: {prefix}")

            df_filtered = get_filtered_df(result, prefix)

            if not df_filtered.empty:

                df_filtered["total_cycles"] = df_filtered["execution_cycles"]
                df_filtered["prefix"] = prefix
                all_data.append(df_filtered)

                figures = plot_overlapped(df_filtered, prefix, chunk_size=30)
                for fig in figures:
                    st.pyplot(fig)

            else:
                st.write(f"No matching files found for prefix {prefix}.")

        if all_data:
            combined_df = pd.concat(all_data).reset_index(drop=True)
            st.subheader("Total Cycles Comparison Across Bandwidth Settings")

            summary_fig = plot_total_cycles_summary(
                combined_df.sort_values(["intra_bw", "inter_bw"])
            )
            st.plotly_chart(summary_fig, use_container_width=True)

            st.subheader("Raw Combined Data")
            st.dataframe(combined_df)

            csv = combined_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Combined Data as CSV",
                data=csv,
                file_name="combined_cycles_analysis.csv",
            )

def get_exposed_comm_fig(df):
    comm_mean, comm_std, comm_gmean = (
        df["comm_percent"].mean(),
        df["comm_percent"].std(),
        gmean(df["comm_percent"][df["comm_percent"] > 0]),
    )
    sorted_comm = np.sort(df["comm_percent"].values)
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(sorted_comm, marker="o", linestyle="-", color="blue")

    ax.set_xticks(range(len(df)))
    ax.set_xlabel("Experiment Number", fontsize=25)
    ax.set_ylabel("Exposed\nComm. Time\n(%)", fontsize=25)
    ax.set_title(
        f"Sorted - Exposed Communication Percentage per Experiment\n(μ={comm_mean:.2f}%, σ={comm_std:.2f}%, gμ={comm_gmean:.2f}%)",
        fontsize=25,
    )
    ax.grid(True)

    ax.tick_params(axis="both", which="major", labelsize=24)
    ax.tick_params(axis="both", which="minor", labelsize=24)

    return fig


def load_and_prepare_trace(file_name, selected_model, selected_config):
    output_dir = picker._get_output_dir()
    suffix = "_trace_matched_timing.csv"
    file_path = os.path.join(
        output_dir, selected_model, selected_config, file_name + suffix
    )
    df = pd.read_csv(file_path)
    df.columns = [
        "index",
        "timestamp",
        "sys_id",
        "node_id",
        "node_name",
        "node_type",
        "num_ops",
        "tensor_size",
        "perf",
        "operational_intensity",
        "issue_tick",
        "callback_tick",
        "elapsed_time",
    ]

    df["issue_tick"] = df["issue_tick"].astype(int)
    df["callback_tick"] = df["callback_tick"].astype(int)
    return df


def get_sys_id_with_max_callback(df):
    return df.loc[df["callback_tick"].idxmax(), "sys_id"]


def build_compute_interval_tree(nodes):
    tree = IntervalTree()
    for _, row in nodes.iterrows():
        tree.addi(row["issue_tick"], row["callback_tick"])
    return tree


def compute_exposed_overlap_cycles(tree, node):
    start, end = node["issue_tick"], node["callback_tick"]
    duration = node["elapsed_time"]

    overlap = 0
    for interval in sorted(tree.overlap(start, end)):
        overlap_start = max(start, interval.begin)
        overlap_end = min(end, interval.end)
        overlap += overlap_end - overlap_start

    exposed = duration - overlap
    return exposed, overlap


def get_exposed(df, sys_id, node_op_type="comm"):
    group = df[df["sys_id"] == sys_id]

    if node_op_type == "comm":
        main_nodes = group[group["node_type"].isin([5, 6, 7])].copy()
        other_nodes = group[group["node_type"] == 4][["issue_tick", "callback_tick"]]
    elif node_op_type == "comp":
        main_nodes = group[group["node_type"] == 4].copy()
        other_nodes = group[group["node_type"].isin([5, 6, 7])][
            ["issue_tick", "callback_tick"]
        ]
    else:
        raise ValueError("main_type must be either 'comm' or 'comp'")

    results = []

    if other_nodes.empty:
        main_nodes["exposed"] = main_nodes["callback_tick"] - main_nodes["issue_tick"]
        main_nodes = main_nodes[main_nodes["exposed"] > 0]
        if not main_nodes.empty:
            results.append(
                main_nodes[
                    [
                        "sys_id",
                        "node_id",
                        "node_name",
                        "node_type",
                        "exposed",
                        "overlap",
                        "elapsed_time",
                    ]
                ]
            )
    else:
        tree = build_compute_interval_tree(other_nodes)
        for _, node in main_nodes.iterrows():
            exposed, overlap = compute_exposed_overlap_cycles(tree, node)
            if exposed > 0 or overlap > 0:
                results.append(
                    pd.DataFrame(
                        {
                            "sys_id": [node["sys_id"]],
                            "node_id": [node["node_id"]],
                            "node_name": [node["node_name"]],
                            "node_type": [node["node_type"]],
                            "elapsed_time": [node["elapsed_time"]],
                            "exposed": [exposed],
                            "overlap": [overlap],
                        }
                    )
                )

    if results:
        return pd.concat(results, ignore_index=True)
    else:
        return pd.DataFrame(
            columns=[
                "sys_id",
                "node_id",
                "node_name",
                "node_type",
                "exposed",
                "elapsed_time",
                "overlap",
            ]
        )


def calculate_avg_bw_across_dims(selected_config):
    config_dir = picker._get_configs_dir()
    suffix = ".yml"
    file_path = os.path.join(config_dir, selected_config + suffix)
    with open(file_path, "r") as f:
        config = yaml.safe_load(f)

    bandwidths = config.get("bandwidth", [])

    average_bandwidth = sum(bandwidths) / len(bandwidths)

    return average_bandwidth, bandwidths

def calculate_data_size(df, avg_bw_across_dims):
    df["mean_data_size_nonoverlap"] = (
        df[df["node_type"].isin([5, 6, 7])]["exposed"] * avg_bw_across_dims
    )
    return df


def calculate_optimal_avg_bw(max_data_size_nonoverlap, desired_latency=1000):
    return max_data_size_nonoverlap / desired_latency


def run_single_simulation(
    base_yml_path,
    output_config_dir,
    model_name,
    bw1,
    bw2,
    app_dir,
    selected_config,
    selected_model,
):
    workload_configuration = app_dir / "../../workload" / selected_model / model_name
    memory_config = app_dir / "../../configuration" / "RemoteMemory.json"
    network_log = app_dir / "../../network_log" / selected_model
    output = app_dir / "../../output" / selected_model
    result = app_dir / "../../results" / selected_model
    suffix = f"_bw_{bw1}_{bw2}"

    for path in [output, result, network_log]:
        new_path = path.with_name(path.name + suffix)
        os.system(f"rm -rf {new_path}")

    os.makedirs(output, exist_ok=True)
    os.makedirs(network_log, exist_ok=True)
    os.makedirs(result, exist_ok=True)

    with open(base_yml_path, "r") as f:
        config = yaml.safe_load(f)

    config["bandwidth"] = [bw1, bw2]

    output_config_path = (
        app_dir
        / "../../output"
        / selected_model
        / f"{model_name}_bw_{bw1}_{bw2}_network.yml"
    )
    output_config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_config_path, "w") as f:
        yaml.dump(
            config,
            f,
            default_flow_style=None,  
            sort_keys=False,  
        )

    failed_cmd = run_astrasim(
        workload_path=str(workload_configuration),
        system=str(app_dir / "../../configuration" / f"{selected_config}_sys.json"),
        network=str(output_config_path),
        memory=str(memory_config),
        output_dir=str(output),
        network_log=str(network_log),
        suffix=suffix,
    )

    if failed_cmd != "":
        raise RuntimeError(f"Simulation failed: {failed_cmd}")


@st.cache_data
def run_bandwidth_sweep_parallel(
    base_yml_path,
    output_config_dir,
    parallelism_strategies,
    bw1_values,
    bw2_values,
    app_dir,
    max_workers,
    selected_config,
    selected_model,
):
    combinations = list(
        itertools.product(parallelism_strategies, bw1_values, bw2_values)
    )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                run_single_simulation,
                base_yml_path,
                output_config_dir,
                parallelism_strategy,
                bw1,
                bw2,
                app_dir,
                selected_config,
                selected_model,
            )
            for parallelism_strategy, bw1, bw2 in combinations
        ]

        for f in tqdm(
            as_completed(futures), total=len(futures), desc="Running simulations"
        ):
            try:
                f.result()
            except Exception as e:
                st.error(f"Simulation failed: {e}")

    output = app_dir / "../../output" / selected_model
    result = app_dir / "../../results" / selected_model
    subprocess.run(
        f"python ../gather_all_NPUs_results.py --sim_logfile {output}  --output_filename {result}",
        shell=True,
        cwd=None,
    )


def get_filtered_df(result_path, prefix):
    files = glob.glob(str(result_path / f"{prefix}_*.csv"))
    records = []
    for file in files:
        df = pd.read_csv(file)
        if (
            "execution_cycles" in df.columns
            and "exposed_communication_cycles" in df.columns
        ):
            min_row = df.loc[df["execution_cycles"].idxmin()]

            match = re.search(r"bw_(\d+)_(\d+)", file)
            intra_bw, inter_bw = (
                (int(match.group(1)), int(match.group(2))) if match else (None, None)
            )
            bw_label = f"{intra_bw}_{inter_bw}"

            records.append(
                {
                    "file_name": os.path.basename(file).replace(".csv", ""),
                    "bw_label": bw_label,
                    "execution_cycles": min_row["execution_cycles"],
                    "comm_cycles": min_row["exposed_communication_cycles"],
                    "intra_bw": intra_bw,
                    "inter_bw": inter_bw,
                }
            )
    return pd.DataFrame(records)


def plot_overlapped(df, prefix, chunk_size=30):
    df = df.sort_values(by="execution_cycles").reset_index(drop=True)
    num_chunks = (len(df) + chunk_size - 1) // chunk_size
    figures = []

    for chunk_idx in range(num_chunks):
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, len(df))
        df_chunk = df.iloc[start:end].reset_index(drop=True)

        fig, ax = plt.subplots(figsize=(12, 4))
        index = range(len(df_chunk))

        ax.bar(
            index,
            df_chunk["execution_cycles"],
            0.4,
            color="skyblue",
            label="Exec Cycles",
        )
        ax.bar(
            index,
            df_chunk["comm_cycles"],
            0.4,
            color="red",
            alpha=0.6,
            label="Comm Cycles",
        )

        for i, (exec_c, comm_c) in enumerate(
            zip(df_chunk["execution_cycles"], df_chunk["comm_cycles"])
        ):
            perc = (comm_c / exec_c) * 100 if exec_c != 0 else 0
            ax.text(i, comm_c / 2, f"{perc:.1f}%", ha="center", fontsize=8)

        ax.set_xticks(index)
        ax.set_xticklabels(df_chunk["bw_label"], rotation=45, ha="right")
        ax.set_ylabel("Cycles")
        ax.set_title(
            f"{prefix} - Exec vs Comm Cycles (examples: {start} to {end} from {len(df)})"
        )
        ax.legend()
        plt.tight_layout()
        figures.append(fig)

    return figures


def plot_total_cycles_summary(all_data):
    unique_prefixes = all_data["prefix"].unique()
    color_sequence = pc.qualitative.Set2 
    color_map = {
        prefix: color_sequence[i % len(color_sequence)]
        for i, prefix in enumerate(unique_prefixes)
    }

    fig = px.line(
        all_data,
        x="bw_label",
        y="total_cycles",
        color="prefix",
        markers=True,
        title="Total Cycles vs Bandwidth for Different Parallelism Strategies",
        labels={
            "bw_label": "Intra_Inter Bandwidth (GB/s)",
            "total_cycles": "Total Cycles",
            "prefix": "Parallelism Config",
        },
        color_discrete_map=color_map,
    )

    min_points = all_data.loc[
        all_data.groupby("prefix")["total_cycles"].idxmin()
    ].reset_index(drop=True)

    for _, row in min_points.iterrows():
        prefix = row["prefix"]
        color = color_map[prefix]  

        fig.add_trace(
            go.Scatter(
                x=[row["bw_label"]],
                y=[row["total_cycles"]],
                mode="markers+text",
                marker=dict(size=12, color=color, symbol="diamond"),
                name=f"Best: {prefix}",
                text=[f"{row['total_cycles']:.0f}"],
                textposition="top center",
                showlegend=False,
            )
        )

    fig.update_layout(xaxis_tickangle=-45, height=600, hovermode="x unified")

    return fig
