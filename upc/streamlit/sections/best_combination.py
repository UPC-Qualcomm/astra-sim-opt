import os
import pandas as pd
import yaml
from intervaltree import IntervalTree
import streamlit as st

import sections.trace_picker as picker
import trace_visualization as tv


def best_combinations(df, selected_config, selected_model):
    selected_file_name = st.selectbox(
        "Select an experiment:", df["file_name"].unique().tolist()
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
