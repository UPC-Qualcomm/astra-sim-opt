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
    #avg_bw_across_dims, _ = calculate_avg_bw_across_dims(selected_config)

    # TODO: Use more accurate version after identifying the links
    # Calculate the size of data bieng transfared during this unhidden communication.
    #final_results_comm = calculate_data_size(final_results_comm, avg_bw_across_dims)
    #max_idx = final_results_comm["exposed"].idxmax()
    #recomended_avg_bw = calculate_optimal_avg_bw(
    #    final_results_comm.at[max_idx, "mean_data_size_nonoverlap"],
    #    100000,
    #)
    #final_results_comp["mean_data_size_nonoverlap"] = -1
    final_results_merged = pd.concat([final_results_comm, final_results_comp])
    final_results_merged["exposed_percent"] = (
        100 * final_results_merged["exposed"] / final_results_merged["elapsed_time"]
    )
    final_results_merged["overlap_percent"] = (
        100 * final_results_merged["overlap"] / final_results_merged["elapsed_time"]
    )
    #final_results_merged = final_results_merged[
    #    [
    #        col
    #        for col in final_results_merged.columns
    #        if col != "mean_data_size_nonoverlap"
    #    ]
    #    + ["mean_data_size_nonoverlap"]
    #]
    st.info(collect_summary(final_results_merged, sys_id_max_cb))
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
    #df.columns = [
    #    "index",
    #    "timestamp",
    #    "sys_id",
    #    "node_id",
    #    "node_name",
    #    "node_type",
    #    "num_ops",
    #    "tensor_size",
    #    "perf",
    #    "operational_intensity",
    #    "issue_tick",
    #    "callback_tick",
    #    "elapsed_time",
    #]

    df["issue_tick"] = df["issue_tick"].astype(int)
    df["callback_tick"] = df["callback_tick"].astype(int)
    return df


def get_sys_id_with_max_callback(df):
    return df.loc[df["callback_tick"].idxmax(), "sys_id"]


def build_interval_tree(nodes):
    tree = IntervalTree()
    for row in nodes.itertuples(index=False):
        tree.addi(row.issue_tick, row.callback_tick)
    tree.merge_overlaps()  # OPTIMIZED: merge once here
    return tree

def total_active_time(tree):
    return sum(interval.end - interval.begin for interval in tree)

def exposed_time(primary_tree, excluding_tree):
    total_exposed = 0
    for interval in primary_tree:
        overlaps = excluding_tree.overlap(interval.begin, interval.end)
        if not overlaps:
            total_exposed += interval.end - interval.begin
        else:
            sub_intervals = [(interval.begin, interval.end)]
            for o in overlaps:
                new_sub_intervals = []
                for s_start, s_end in sub_intervals:
                    if o.begin >= s_end or o.end <= s_start:
                        new_sub_intervals.append((s_start, s_end))
                    else:
                        if s_start < o.begin:
                            new_sub_intervals.append((s_start, o.begin))
                        if o.end < s_end:
                            new_sub_intervals.append((o.end, s_end))
                sub_intervals = new_sub_intervals
            total_exposed += sum(e - s for s, e in sub_intervals)
    return total_exposed

def collect_summary(df, sys_id):
    group = df[df["sys_id"] == sys_id]

    comm_nodes = group[group["node_type"].isin([5, 6, 7])]
    comp_nodes = group[group["node_type"] == 4]

    comm_tree = build_interval_tree(comm_nodes)
    comp_tree = build_interval_tree(comp_nodes)

    total_comm_time = total_active_time(comm_tree)
    total_comp_time = total_active_time(comp_tree)

    comm_exposed_to_comp = exposed_time(comm_tree, comp_tree)
    comp_exposed_to_comm = exposed_time(comp_tree, comm_tree)

    return {
        "sys_id": sys_id,
        "total_comm_time": total_comm_time,
        "total_comp_time": total_comp_time,
        "comm_exposed_to_comp": comm_exposed_to_comp,
        "comp_exposed_to_comm": comp_exposed_to_comm,
    }

def build_compute_interval_tree(nodes):
    tree = IntervalTree()
    for row in nodes.itertuples(index=False):
        tree.addi(row.issue_tick, row.callback_tick, row.node_id)
    tree.merge_overlaps()  # OPTIMIZED: merge once here
    return tree

def compute_overlap(tree, node_id, start, end):
    overlaps = tree.overlap(start, end)
    overlap_duration = 0
    for o in overlaps:
        if o.data != node_id:
            overlap_start = max(start, o.begin)
            overlap_end = min(end, o.end)
            overlap_duration += max(0, overlap_end - overlap_start)
    return overlap_duration

def get_exposed(df, sys_id, node_op_type="comm"):
    group = df[df["sys_id"] == sys_id]

    if node_op_type == "comm":
        main_nodes = group[group["node_type"].isin([5, 6, 7])]
    elif node_op_type == "comp":
        main_nodes = group[group["node_type"] == 4]
    else:
        raise ValueError("node_op_type must be either 'comm' or 'comp'")

    comp_nodes = group[group["node_type"] == 4]
    comm_nodes = group[group["node_type"].isin([5, 6, 7])]

    comp_tree = build_compute_interval_tree(comp_nodes)
    comm_tree = build_compute_interval_tree(comm_nodes)

    results = []

    for row in main_nodes.itertuples(index=False):
        duration = row.elapsed_time

        overlap_comp = compute_overlap(comp_tree, row.node_id, row.issue_tick, row.callback_tick)
        overlap_comm = compute_overlap(comm_tree, row.node_id, row.issue_tick, row.callback_tick)

        total_overlap = min(duration, overlap_comp + overlap_comm)

        result = {
            "sys_id": row.sys_id,
            "node_id": row.node_id,
            "node_name": row.node_name,
            "node_type": row.node_type,
            "elapsed_time": duration,
            "exposed": duration - total_overlap,
            "overlap": total_overlap,
            "overlap_with_comp": overlap_comp,
            "overlap_with_comm": overlap_comm,
            "exposed_with_comp": duration - overlap_comp,
            "exposed_with_comm": duration - overlap_comm,
            "num_ops": row.num_ops,
            "tensor_size": row.tensor_size,
            "perf": row.perf,
            "operational_intensity": row.operational_intensity,
            "issue_tick": row.issue_tick,
            "callback_tick": row.callback_tick,
        }

        results.append(result)

    if results:
        return pd.DataFrame.from_records(results)
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
                "overlap_with_comp",
                "overlap_with_comm",
                "exposed_with_comp",
                "exposed_with_comm",
                "num_ops",
                "tensor_size",
                "perf",
                "operational_intensity",
                "issue_tick",
                "callback_tick",
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
