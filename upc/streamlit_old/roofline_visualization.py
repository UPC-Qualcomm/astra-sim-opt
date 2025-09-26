import streamlit as st
import os
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt
import seaborn as sns
from IPython.display import display
import numpy as np
import plotly.graph_objects as go


def add_comm_points(df, npu=0):
    df_single_npu = df.query(f"sys_id == {npu}")
    df_single_npu = df_single_npu.drop(df.columns[0], axis=1)
    df_single_npu["perf"] = df_single_npu["perf"] / 1e12
    df_single_npu["operational_intensity"] = df_single_npu["operational_intensity"]
    new_rows = []

    df_sorted = df_single_npu.sort_values("issue_tick").reset_index(drop=True)
    # Iterate over the group, skipping the first and last row
    for i in range(1, len(df_sorted)):
        prev_row = df_sorted.iloc[i - 1]
        current_row = df_sorted.iloc[i]

        # Prepare the new row as specified
        new_row = {
            "sys_id": current_row["sys_id"],
            "node_id": 0,
            "node_name": prev_row["node_name"] + "_comm_" + current_row["node_name"],
            "num_ops": 0,
            "tensor_size": 0,
            "perf": 0,
            "operational_intensity": 0,
            "elapsed_time": current_row["issue_tick"] - prev_row["callback_tick"],
            "issue_tick": prev_row["callback_tick"],
            "callback_tick": current_row["issue_tick"],
        }
        new_rows.append(new_row)

    # Append the new rows to the original dataframe
    df_new = pd.DataFrame(new_rows)
    df_combined = pd.concat([df_single_npu, df_new], ignore_index=True)

    # Sort the final dataframe by sys_id and issue_tick
    df_combined_sorted = df_combined.sort_values(["sys_id", "issue_tick"]).reset_index(
        drop=True
    )

    df_combined_sorted = df_combined_sorted[df_combined_sorted["elapsed_time"] != 0]

    return df_combined_sorted


def expand_df_and_average(df, time_window=50000):
    new_rows = []
    buffer_row = None
    remaining_time = 0

    for i in range(len(df)):
        row = df.iloc[i].copy()

        # If we have a buffer row from previous iteration
        if buffer_row is not None:
            # Calculate how much time we need to complete the window
            time_needed = time_window - remaining_time

            if row["elapsed_time"] >= time_needed:
                # We can complete the window
                # Create a new row that completes the window
                new_row = buffer_row.copy()
                new_row["elapsed_time"] = time_window
                new_row["callback_tick"] = new_row["issue_tick"] + time_window

                # Calculate weighted averages for performance metrics
                weight_prev = remaining_time / time_window
                weight_curr = time_needed / time_window
                new_row["perf"] = (buffer_row["perf"] * weight_prev) + (
                    row["perf"] * weight_curr
                )
                # if (buffer_row['operational_intensity'] == 0 and row['operational_intensity'] != 0):
                #    new_row['operational_intensity'] = row['operational_intensity']
                # elif (row['operational_intensity'] == 0 and buffer_row['operational_intensity'] != 0):
                #    new_row['operational_intensity'] = buffer_row['operational_intensity']
                # elif (row['operational_intensity'] != 0 and buffer_row['operational_intensity'] != 0):
                #    new_row['operational_intensity'] = (buffer_row['operational_intensity'] + row['operational_intensity'] ) / 2
                # new_row['operational_intensity'] = max(buffer_row['operational_intensity'] , row['operational_intensity'])
                new_row["operational_intensity"] = (
                    buffer_row["operational_intensity"] * weight_prev
                ) + (row["operational_intensity"] * weight_curr)

                new_rows.append(new_row)

                # Update the current row
                row["elapsed_time"] -= time_needed
                row["issue_tick"] += time_needed

                # Process the remaining time in the current row
                remaining_full_windows = int(row["elapsed_time"] // time_window)

                # Add full windows
                for j in range(remaining_full_windows):
                    window_row = row.copy()
                    window_row["elapsed_time"] = time_window
                    window_row["issue_tick"] = row["issue_tick"] + (j * time_window)
                    window_row["callback_tick"] = window_row["issue_tick"] + time_window
                    new_rows.append(window_row)

                # Calculate the remaining time after full windows
                remaining_time = row["elapsed_time"] % time_window
                if remaining_time > 0:
                    # Store the remaining part for the next iteration
                    buffer_row = row.copy()
                    buffer_row["elapsed_time"] = remaining_time
                    buffer_row["issue_tick"] = row["issue_tick"] + (
                        remaining_full_windows * time_window
                    )
                    buffer_row["callback_tick"] = (
                        buffer_row["issue_tick"] + remaining_time
                    )
                else:
                    buffer_row = None
                    remaining_time = 0
            else:
                # We can't complete the window yet
                remaining_time += row["elapsed_time"]

                # Update buffer row with weighted averages
                total_time = buffer_row["elapsed_time"] + row["elapsed_time"]
                weight_buffer = buffer_row["elapsed_time"] / total_time
                weight_row = row["elapsed_time"] / total_time

                buffer_row["perf"] = (buffer_row["perf"] * weight_buffer) + (
                    row["perf"] * weight_row
                )
                # if (buffer_row['operational_intensity'] == 0 and  row['operational_intensity'] != 0):
                #    buffer_row['operational_intensity'] = row['operational_intensity']
                # elif (row['operational_intensity'] == 0 and buffer_row['operational_intensity'] != 0):
                #    buffer_row['operational_intensity'] = buffer_row['operational_intensity']
                # elif (row['operational_intensity'] != 0 and buffer_row['operational_intensity'] != 0):
                #    buffer_row['operational_intensity'] = (buffer_row['operational_intensity'] + row['operational_intensity'] ) / 2
                # buffer_row['operational_intensity'] = max(buffer_row['operational_intensity'] , row['operational_intensity'] )
                buffer_row["operational_intensity"] = (
                    buffer_row["operational_intensity"] * weight_buffer
                ) + (row["operational_intensity"] * weight_row)
                buffer_row["elapsed_time"] = total_time
                buffer_row["callback_tick"] = buffer_row["issue_tick"] + total_time
        else:
            # No buffer row, process the current row directly
            full_windows = int(row["elapsed_time"] // time_window)

            # Add full windows
            for j in range(full_windows):
                window_row = row.copy()
                window_row["elapsed_time"] = time_window
                window_row["issue_tick"] = row["issue_tick"] + (j * time_window)
                window_row["callback_tick"] = window_row["issue_tick"] + time_window
                new_rows.append(window_row)

            # Calculate the remaining time
            remaining_time = row["elapsed_time"] % time_window
            if remaining_time > 0:
                # Store the remaining part for the next iteration
                buffer_row = row.copy()
                buffer_row["elapsed_time"] = remaining_time
                buffer_row["issue_tick"] = row["issue_tick"] + (
                    full_windows * time_window
                )
                buffer_row["callback_tick"] = buffer_row["issue_tick"] + remaining_time
            else:
                buffer_row = None

    # Don't forget to add the last buffer row if it exists
    if buffer_row is not None:
        new_rows.append(buffer_row)

    # Create the new dataframe
    if new_rows:
        df_new = pd.DataFrame(new_rows)
        # Sort the final dataframe by issue_tick
        df_new = df_new.sort_values("issue_tick").reset_index(drop=True)
        return df_new
    else:
        # Return an empty dataframe with the same columns as the input
        return pd.DataFrame(columns=df.columns)


def plot_roofline(df, beta=2000, pi=300):
    # Compute intersection point
    I_c = (pi * 1e12) / (beta * 1e9)
    P_c = pi

    x_min = max(0, df["operational_intensity"].min())
    x_max = df["operational_intensity"].max() * 1.1  # Add 10% headroom
    x_vals = np.linspace(x_min, x_max, 200)

    # Create dataframes for the roofline model lines
    roofline_data = pd.DataFrame(
        {
            "operational_intensity": x_vals,
            "beta_line": (beta * x_vals) * 1e-3,  # To preserve same scale for the lines
            "pi_line": [pi] * len(x_vals),
        }
    )

    # Base scatter plot
    chart = (
        alt.Chart(df)
        .mark_circle()
        .encode(
            x=alt.X(
                "operational_intensity:Q", title="Operational Intensity (FLOPs/byte)"
            ),
            y=alt.Y("perf:Q", title="Performance (TFLOPs/sec)"),
            size=alt.value(100),
            tooltip=list(df.columns),
        )
        .properties(
            title="Roofline Model: Performance vs Operational Intensity",
            width=600,
            height=400,
        )
    )

    # Bandwidth line (sloped)
    beta_line = (
        alt.Chart(roofline_data)
        .mark_line(color="red")
        .encode(x="operational_intensity:Q", y="beta_line:Q")
    )

    # Peak performance line (horizontal)
    pi_line = (
        alt.Chart(roofline_data)
        .mark_line(color="green")
        .encode(x="operational_intensity:Q", y="pi_line:Q")
    )

    # Intersection point marker
    intersect_point = pd.DataFrame({"operational_intensity": [I_c], "perf": [P_c]})
    intersection = (
        alt.Chart(intersect_point)
        .mark_point(color="black", shape="cross", size=200)
        .encode(x="operational_intensity:Q", y="perf:Q")
    )

    final_chart = (chart + beta_line + pi_line).interactive()
    return final_chart


def plot_roofline_timestep(df, beta=2000, pi=300):
    # Compute intersection point
    I_c = (pi * 1e12) / (beta * 1e9)
    P_c = pi

    x_min = min(0, df["operational_intensity"].min())
    x_max = max(df["operational_intensity"].max(), I_c) * 1.5  # Add 10% headroom
    y_min = 0
    y_max = max(df["perf"].max(), pi) * 1.2

    x_vals = np.linspace(x_min, x_max, 200)

    # Create dataframes for the roofline model lines
    roofline_data = pd.DataFrame(
        {
            "operational_intensity": x_vals,
            "beta_line": (beta * x_vals) * 1e-3,  # To preserve same scale for the lines
            "pi_line": [pi] * len(x_vals),
        }
    )

    # Base scatter plot
    chart = (
        alt.Chart(df)
        .mark_circle()
        .encode(
            x=alt.X(
                "operational_intensity:Q",
                title="Operational Intensity (FLOPs/byte)",
                scale=alt.Scale(domain=[x_min, x_max]),
            ),
            y=alt.Y(
                "perf:Q",
                title="Performance (TFLOPs/sec)",
                scale=alt.Scale(domain=[y_min, y_max]),
            ),
            size=alt.value(100),
            tooltip=list(df.columns),
        )
        .properties(
            title="Roofline Model: Performance vs Operational Intensity",
            width=1000,
            height=400,
        )
    )

    # Bandwidth line (sloped)
    beta_line = (
        alt.Chart(roofline_data)
        .mark_line(color="red", strokeDash=[5, 5])
        .encode(x="operational_intensity:Q", y="beta_line:Q")
    )

    # Peak performance line (horizontal)
    pi_line = (
        alt.Chart(roofline_data)
        .mark_line(color="green", strokeDash=[3, 3])
        .encode(x="operational_intensity:Q", y="pi_line:Q")
    )

    # Intersection point marker
    intersect_point = pd.DataFrame({"operational_intensity": [I_c], "perf": [P_c]})
    intersection = (
        alt.Chart(intersect_point)
        .mark_point(color="black", shape="cross", size=200)
        .encode(x="operational_intensity:Q", y="perf:Q")
    )

    final_chart = (chart + beta_line + pi_line).interactive()
    return final_chart


def plot_roofline_time(df, beta=2000, pi=300):
    # Compute intersection point
    I_c = (pi * 1e12) / (beta * 1e9)
    P_c = pi

    # Create a new column labeling points based on operational intensity threshold
    df["oi_category"] = np.select(
        [df["operational_intensity"] == 0, df["operational_intensity"] < I_c],
        ["Idle", "Memory bound"],
        default="Compute bound",
    )

    # Base scatter plot: x = issue_tick, y = perf, color = operational_intensity

    chart = (
        alt.Chart(df)
        .mark_circle(size=100)
        .encode(
            x=alt.X("issue_tick:Q", title="Issue Time (cycles)"),
            y=alt.Y(
                "perf:Q",
                title="Performance (TFLOPs/sec)",
                scale=alt.Scale(domain=[-1, df["perf"].max() * 1.05]),
            ),
            color=alt.Color(
                "oi_category:N",
                title="Operational Intensity Category",
                scale=alt.Scale(
                    domain=["Idle", "Memory bound", "Compute bound"],
                    range=["rgba(0,0,0,0.3)", "red", "steelblue"],
                ),
            ),
            tooltip=[
                alt.Tooltip("node_id:N", title="Node ID"),
                alt.Tooltip("node_name:N", title="Node Name"),
                alt.Tooltip("perf:Q", title="Performance (TFLOPs/sec)", format=".2f"),
                alt.Tooltip("issue_tick:Q", title="Issue Time (cycles)"),
                alt.Tooltip(
                    "operational_intensity:Q",
                    title="Operational Intensity",
                    format=".2f",
                ),
            ],
        )
        .properties(
            title="Performance vs Issue Time (colored by Operational Intensity threshold)",
            width=800,
            height=400,
        )
    )

    line = (
        alt.Chart(df)
        .mark_line()
        .encode(x=alt.X("issue_tick:Q"), y=alt.Y("perf:Q"), tooltip=list(df.columns))
    )

    pi_line = (
        alt.Chart(
            pd.DataFrame(
                {
                    "perf": [P_c, P_c],
                    "issue_tick": [df["issue_tick"].min(), df["issue_tick"].max()],
                }
            )
        )
        .mark_line(color="green", strokeDash=[5, 5])
        .encode(x="issue_tick:Q", y="perf:Q")
    )

    final_chart = (chart + pi_line + line).interactive()
    return final_chart


def plot_3d_roofline(df, beta=2000, pi=300):
    # Convert to SI units
    beta = beta * 1e9  # GB/s to B/s
    pi = pi * 1e12  # TFLOP/s

    # Sort by issue_tick for orderly processing
    df_sorted = df.sort_values(by="issue_tick")
    df_sorted["perf"] = df_sorted["perf"] * 1e12
    # Set grid ranges based on data
    op_intensity_min = df_sorted["operational_intensity"].min()
    op_intensity_max = df_sorted["operational_intensity"].max()
    issue_tick_min = df_sorted["issue_tick"].min()
    issue_tick_max = df_sorted["issue_tick"].max()

    # Create grid for operational intensity (OI) and issue_tick (time)
    op_intensity = np.linspace(op_intensity_min, op_intensity_max, 50)
    issue_tick = np.linspace(issue_tick_min, issue_tick_max, 50)
    OI_grid, T_grid = np.meshgrid(op_intensity, issue_tick)

    # Compute Bandwidth bound surface (perf = beta * I)
    perf_bandwidth = beta * OI_grid

    # Compute Compute limit surface (perf = pi)
    perf_compute = np.full_like(OI_grid, pi)

    fig = go.Figure()

    # Scatter original data points
    fig.add_trace(
        go.Scatter3d(
            x=df_sorted["operational_intensity"],
            y=df_sorted["issue_tick"],
            z=df_sorted["perf"],
            mode="lines+markers",
            marker=dict(
                size=3,
                color=df_sorted["issue_tick"],
                colorscale="Viridis",
                opacity=0.4,
                showscale=False,
            ),
            customdata=np.stack((df_sorted["node_id"],), axis=-1),
            hovertemplate="Node ID: %{customdata[0]}<br>"
            + "Op Intensity: %{x:.2f}<br>"
            + "Perf: %{z:.2e}<br>"
            + "Issue Tick: %{y}<extra></extra>",
            name="Original Points",
        )
    )

    # Add Bandwidth bound surface
    fig.add_trace(
        go.Surface(
            x=OI_grid,
            y=T_grid,
            z=perf_bandwidth,
            colorscale=[[0, "red"], [1, "red"]],
            opacity=0.3,
            showscale=False,
            name="Bandwidth Bound",
        )
    )

    # Add Compute limit surface
    fig.add_trace(
        go.Surface(
            x=OI_grid,
            y=T_grid,
            z=perf_compute,
            colorscale=[[0, "green"], [1, "green"]],
            opacity=0.3,
            showscale=False,
            name="Compute Limit",
        )
    )

    # Configure 3D axes and optionally set axis ranges to match your data
    fig.update_layout(
        scene=dict(
            xaxis=dict(
                title="Operational Intensity (FLOPs/byte)",
                range=[op_intensity_min, op_intensity_max],
            ),
            yaxis=dict(
                title="Issue Time (cycles)", range=[issue_tick_min, issue_tick_max]
            ),
            zaxis=dict(title="Performance (FLOPs/sec)", range=[0, pi]),
        ),
        title="3D Roofline Model",
        width=900,
        height=800,
    )

    return fig


def get_3d_roofline_plot(df, npu=0, bw=2000, perf=300, time_window=0):
    df_single_npu = df[df["sys_id"] == npu].copy()
    df_single_npu["perf"] = df_single_npu["perf"] / 1e12
    if time_window != 0:
        df_single_npu = expand_df_and_average(df_single_npu, time_window=time_window)
    return plot_3d_roofline(
        df_single_npu.loc[
            :, ["perf", "operational_intensity", "issue_tick", "node_id"]
        ].drop_duplicates(),
        bw,
        perf,
    )


def get_2d_roofline_plot_normal(df, npu=0, bw=2000, perf=300):
    df_single_npu = df[df["sys_id"] == npu].copy()
    df_single_npu["perf"] = df_single_npu["perf"] / 1e12
    return plot_roofline(
        df_single_npu.loc[
            :, ["perf", "operational_intensity", "issue_tick", "node_id", "node_name"]
        ].drop_duplicates(),
        bw,
        perf,
    )


def get_2d_roofline_plot_with_time(df, npu=0, bw=2000, perf=300, time_window=0):
    df_single_npu = df[df["sys_id"] == npu].copy()
    df_single_npu["perf"] = df_single_npu["perf"] / 1e12
    if time_window != 0:
        df_single_npu = expand_df_and_average(df_single_npu, time_window=time_window)
    return plot_roofline_time(
        df_single_npu.loc[
            :, ["perf", "operational_intensity", "issue_tick", "node_id", "node_name"]
        ].drop_duplicates(),
        bw,
        perf,
    )


def get_timesteps(df, npu=0, time_window=0):
    df_single_npu = df[df["sys_id"] == npu].copy()
    df_single_npu["perf"] = df_single_npu["perf"] / 1e12
    if time_window != 0:
        df_single_npu = expand_df_and_average(df_single_npu, time_window=time_window)
    timesteps = df_single_npu["issue_tick"].unique()
    return df_single_npu, timesteps


def get_2d_roofline_plot_timestep(df, bw=2000, perf=300):
    return plot_roofline_timestep(
        df.loc[
            :, ["perf", "operational_intensity", "issue_tick", "node_id", "node_name"]
        ].drop_duplicates(),
        bw,
        perf,
    )


def get_info(df, npu=0, perf=300, bw=2000):
    df_local = df[df["sys_id"] == npu].copy()
    compute_memory_boundary = (perf / bw) * 1e3  # FLOPS/Byte
    total_time = df_local["callback_tick"].max()
    df_memory = df_local[
        (df_local["operational_intensity"] < compute_memory_boundary)
        & (df_local["operational_intensity"] != 0)
    ]
    df_compute = df_local[df_local["operational_intensity"] >= compute_memory_boundary]
    df_idle = df_local[df_local["operational_intensity"] == 0]

    mem_time = df_memory["elapsed_time"].sum()
    comp_time = df_compute["elapsed_time"].sum()
    idle_time = df_idle["elapsed_time"].sum()

    

    return mem_time, comp_time, total_time - (mem_time + comp_time)


##TODO: Check if needed - get info while considering the overlapping within MEM and COMP
#def get_info(df, npu=0, perf=300, bw=2000):
#    df_local = df[df["sys_id"] == npu].copy()
#    compute_memory_boundary = (perf / bw) * 1e3  # FLOPS/Byte
#
#    # Add start and end times
#    df_local['start'] = df_local['issue_tick']
#    df_local['end'] = df_local['callback_tick']
#
#    # Classify intervals
#    mem_intervals = df_local[
#        (df_local["operational_intensity"] < compute_memory_boundary) &
#        (df_local["operational_intensity"] != 0)
#    ][['start', 'end']].values.tolist()
#
#    comp_intervals = df_local[
#        (df_local["operational_intensity"] >= compute_memory_boundary)
#    ][['start', 'end']].values.tolist()
#
#    idle_intervals = df_local[
#        (df_local["operational_intensity"] == 0)
#    ][['start', 'end']].values.tolist()
#
#    # Merge intervals to avoid double-counting
#    mem_time = merge_intervals(mem_intervals)
#    comp_time = merge_intervals(comp_intervals)
#    idle_time = merge_intervals(idle_intervals)
#
#    # Total time is the span from min start to max end
#    total_time = df_local['end'].max() - df_local['start'].min()
#
#    # Optionally, you can check that mem_time + comp_time + idle_time <= total_time
#    return mem_time, comp_time, idle_time
#
#
#def merge_intervals(intervals):
#    """Merge overlapping intervals and return total covered time."""
#    if not intervals:
#        return 0
#    sorted_intervals = sorted(intervals, key=lambda x: x[0])
#    merged = [sorted_intervals[0]]
#    for start, end in sorted_intervals[1:]:
#        last_end = merged[-1][1]
#        if start <= last_end:
#            merged[-1][1] = max(last_end, end)
#        else:
#            merged.append([start, end])
#    total = sum(end - start for start, end in merged)
#    return total

# TODO:
# Plot with slider
