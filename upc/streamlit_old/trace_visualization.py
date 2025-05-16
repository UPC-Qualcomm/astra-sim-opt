import streamlit as st
import os
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt
import seaborn as sns
from IPython.display import display
import numpy as np


NodeType = {
    "INVALID_NODE": 0,
    "METADATA_NODE": 1,
    "MEM_LOAD_NODE": 2,
    "MEM_STORE_NODE": 3,
    "COMP_NODE": 4,
    "COMM_SEND_NODE": 5,
    "COMM_RECV_NODE": 6,
    "COMM_COLL_NODE": 7,
}
node_types = list(NodeType.keys())


def get_timings_df(df: pd.DataFrame) -> pd.DataFrame:
    df_issues = df.query("action == 'issue'").drop(columns="action")
    df_callbacks = df.query("action == 'callback'").drop(columns="action")
    return df_issues.merge(
        df_callbacks,
        on=["sys_id", "node_id", "node_name", "node_type"],
        suffixes=("_issue", "_callback"),
    ).assign(elapsed_time=lambda d: d["tick_callback"] - d["tick_issue"])


def plot_elapsed_times(
    df: pd.DataFrame, npu, sys_id: int = 0, max_height: int = 600
) -> alt.Chart:
    df = df.query(f"sys_id == {sys_id}")
    unique_nodes = df["node_name"].nunique()
    chart_height =  min(unique_nodes * 40, max_height)  # min/max to keep reasonable bounds

    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("elapsed_time:Q", title="Elapsed Time"),
            y=alt.Y(
                "node_name:N",
                sort=alt.SortField(field="tick_issue", order="ascending"),
                title="Node Name",
            ),
            tooltip=["node_name", "elapsed_time", "tick_issue"],
        )
        .properties(
            width=600,
            height=chart_height,
            title=f"Elapsed Time by Node Name - NPU {npu}",
        )
        .configure_axis(
            labelFontSize=15,
            labelLimit=250,
            titleFontSize=18,
            titlePadding=50,  # Increase this number as needed for your label lengths
        )
        .configure_view(stroke=None)
        .interactive()
    )


def get_overlapped_blocks(df: pd.DataFrame) -> dict[str, list[int]]:
    """df should be filtered by sys_id and node_type."""
    df_sorted = df.sort_values("tick_issue")
    blocks = {
        "start": [],
        "end": [],
    }
    prev_start, prev_end = 0, 0
    for _, row in df_sorted.iterrows():
        if row["tick_issue"] <= prev_end:
            # overlap
            # merge the two blocks with proper start and end times
            # update the prev variables
            # do not append the block until we know it is not overlapping with any other subsequent block
            prev_end = max(prev_end, row["tick_callback"])
        else:
            # there is no overlap, append the block and update the prev variables
            blocks["start"].append(prev_start)
            blocks["end"].append(prev_end)
            prev_start = row["tick_issue"]
            prev_end = row["tick_callback"]

    blocks["start"].append(prev_start)
    blocks["end"].append(prev_end)
    return blocks


def plot_overlapped_blocks(df: pd.DataFrame, npu) -> alt.Chart:
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("start:Q", title="Timestamp"),
            x2="end:Q",
            y=alt.Y("node_type:N", title="Node Type"),
            color="node_type:N",  # Different color for each node_type
        )
        .configure_axis(
            grid=False,  # Remove the grid lines
            ticks=False,
        )
        .properties(
            height=400, width=800, title=f"Duration of Blocks by Node Type - NPU {npu}"
        )
    )

    return chart.interactive()


def plot_one_npu(df, npu=0, plot_blocks=True, plot_times=False):
    df_0 = df.query(f"sys_id == {npu}")
    df_0_comp = pd.DataFrame.from_dict(
        get_overlapped_blocks(df_0.query("node_type == 4"))
    ).assign(node_type="COMPUTATION")
    df_0_comm = pd.DataFrame.from_dict(
        get_overlapped_blocks(df_0.query("node_type in (5, 6, 7)"))
    ).assign(node_type="COMMUNICATION")
    df_0_blocks = pd.concat([df_0_comp, df_0_comm])
    if plot_blocks:
        return plot_overlapped_blocks(df_0_blocks, npu)
    if plot_times:
        return plot_elapsed_times(df, npu=npu, max_height=1500)


def plot_all_npus(df):
    for npu in range(64):
        df_0 = df.query(f"sys_id == {npu}")
        plot_one_npu(df, npu, plot_blocks=True, plot_times=False)
