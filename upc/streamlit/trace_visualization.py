import streamlit as st
import os
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt
import seaborn as sns
from IPython.display import display
import numpy as np
import helper.constants as constants

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


@st.cache_data
def get_timings_df(csv_trace_file, output_file_name) -> pd.DataFrame:
    df = pd.read_csv(csv_trace_file)
    # Filter issues and rename 'tick' to 'issue_tick'
    df_issues = df.query("action == 'issue'").drop(columns="action")

    # Filter callbacks and rename 'tick' to 'callback_tick'
    df_callbacks = (
        df.query("action == 'callback'")
        .drop(columns="action")
        .rename(columns={"issue_tick": "callback_tick"})
    )

    # Merge issues with callbacks on the identifying columns
    merged_df = df_issues.merge(
        df_callbacks[["sys_id", "node_id", "node_name", "node_type", "callback_tick"]],
        on=["sys_id", "node_id", "node_name", "node_type"],
        how="left",
        suffixes=("", ""),
    )

    # Add elapsed_time column
    merged_df["elapsed_time"] = merged_df["callback_tick"] - merged_df["issue_tick"]

    merged_df.to_csv(output_file_name)

    return merged_df

@st.cache_data
def plot_elapsed_times(
    df: pd.DataFrame, npu, sys_id: int = 0, max_height: int = 600, exp = ""
) -> alt.Chart:
    df = df.query(f"sys_id == {sys_id}")
    #df = df.sort_values("elapsed_time", ascending=False).reset_index(drop=True)
    #df_sorted = df.sort_values("elapsed_time", ascending=False).reset_index(drop=True)

    # Create 10 quantile-based bins
    #df_sorted['bin'] = pd.qcut(df_sorted['elapsed_time'], q=100, duplicates='drop')

    # Pick one row per bin (e.g. first occurrence)
    #df_diverse = df_sorted.groupby('bin').first().reset_index(drop=True)

    # Drop the helper column
    # Drop the record with max elapsed time
    #df_diverse = df_diverse[df_diverse["elapsed_time"] != df_diverse["elapsed_time"].max()]
    #df = df_diverse
    df = df[:100]
    unique_nodes = df["node_name"].nunique()
    chart_height = max(
        unique_nodes * constants.FONT_SIZE * 1.5, max_height
    )  # min/max to keep reasonable bounds
    return (
        alt.Chart(df)
        #.transform_calculate(
        #    operation_type="datum.node_type == 4 ? 'COMP' : datum.node_type == 7 ? 'COMM' : 'OTHER'"
        #)
        .mark_bar()
        .encode(
            x=alt.X(
                "elapsed_time:Q",
                title="Time (M-Cycles)",
                axis=alt.Axis(
                    format=",d",
                    labelExpr="datum.value / 1000000 + 'M'"
                )
            ),
            y=alt.Y(
                "node_name:N",
                sort=alt.SortField(field="elapsed_time", order="descending"),
                title="",
            ),
            #color=alt.Color(
                #"operation_type:N",
                #scale=alt.Scale(
                #    domain=["COMM", "COMP"]
                #),
                #legend=alt.Legend(title=None)
            #),
            tooltip=["node_name", "elapsed_time", "issue_tick"],
        )
        .properties(
            width=600,
            height=chart_height,
            title=["Elapsed Time Of Compute and Communication Operations", f"NPU {npu}{exp}"]
        )
        .configure_axis(
            labelFontSize=constants.XTICK_SIZE+2,
            labelColor='black',
            titleFontSize=constants.LABEL_SIZE+2,
            titleFont='Arial',
            titleColor='black',
            labelFont='Arial',
            labelBaseline="middle",
            titlePadding=30,
            labelLimit=300
        )
        .configure_title(
            fontSize=constants.TITLE_SIZE+2,
            font='Arial',
            anchor='start',
            color='black',
            fontWeight="normal"
        )
        .configure_legend(
            labelFontSize=constants.LEGEND_SIZE+2,
            titleFontSize=constants.LEGEND_SIZE+2,
            labelColor='black',
            titleColor='black'
        )
        .configure_view(stroke=None)
        .interactive()
    )

@st.cache_data
def get_overlapped_blocks(df: pd.DataFrame) -> dict[str, list[int]]:
    """df should be filtered by sys_id and node_type."""
    df_sorted = df.sort_values("issue_tick")
    blocks = {
        "start": [],
        "end": [],
    }
    prev_start, prev_end = 0, 0
    for _, row in df_sorted.iterrows():
        if row["issue_tick"] <= prev_end:
            # overlap
            # merge the two blocks with proper start and end times
            # update the prev variables
            # do not append the block until we know it is not overlapping with any other subsequent block
            prev_end = max(prev_end, row["callback_tick"])
        else:
            # there is no overlap, append the block and update the prev variables
            blocks["start"].append(prev_start)
            blocks["end"].append(prev_end)
            prev_start = row["issue_tick"]
            prev_end = row["callback_tick"]

    blocks["start"].append(prev_start)
    blocks["end"].append(prev_end)
    return blocks

@st.cache_data
def plot_overlapped_blocks(df: pd.DataFrame, npu, exp = "") -> alt.Chart:
    type_labels = {"COMMUNICATION": "COMM", "COMPUTATION": "COMP"}
    df = df.copy()
    df["type_label"] = df["node_type"].map(type_labels)
    df = df.sort_values("start").reset_index(drop=True)
    chart = (
        alt.Chart(df)
        .mark_bar(size=70)
        .encode(
            x=alt.X(
                "start:Q",
                title="Time (Cycles)",
                axis=alt.Axis(
                    format=",.1f",
                    tickMinStep=1000000,
                    labelExpr="datum.value / 1e6 + 'M'",
                    ticks=True, 
                    labels=True   
                )
            ),
            x2="end:Q",
            y=alt.Y("type_label:N", title="Node Type", axis=alt.Axis(labels=False, ticks=False)),
            color=alt.Color("type_label:N", legend=alt.Legend(title=None)),
        )
        .configure_axis(
            grid=False,
            ticks=False,
            labelFontSize=constants.XTICK_SIZE,
            titleFontSize=constants.LABEL_SIZE,
            labelColor="black",
            titleColor="black",
        )
        .configure_title(
            fontSize=constants.TITLE_SIZE,
            color="black",
            fontWeight="normal"
        )
        .configure_legend(
            labelFontSize=constants.LEGEND_SIZE,
            labelColor="black"
        )
        .properties(
            height=400,
            width=1200,
            title=["Operation Blocks Duration Through Time", exp],
        )
    )

    return chart.interactive()

@st.cache_data
def plot_one_npu(df, npu=0, plot_blocks=True, plot_times=False, exp = ""):
    df_0 = df.query(f"sys_id == {npu}")
    df_0_comp = pd.DataFrame.from_dict(
        get_overlapped_blocks(df_0.query("node_type == 4"))
    ).assign(node_type="COMPUTATION")
    df_0_comm = pd.DataFrame.from_dict(
        get_overlapped_blocks(df_0.query("node_type in (5, 6, 7)"))
    ).assign(node_type="COMMUNICATION")
    df_0_blocks = pd.concat([df_0_comp, df_0_comm])
    if plot_blocks:
        return plot_overlapped_blocks(df_0_blocks, npu, exp=exp)
    if plot_times:
        return plot_elapsed_times(df, npu=npu, max_height=1500, exp=exp)


def plot_all_npus(df):
    for npu in range(64):
        df_0 = df.query(f"sys_id == {npu}")
        plot_one_npu(df, npu, plot_blocks=True, plot_times=False, exp = "")
