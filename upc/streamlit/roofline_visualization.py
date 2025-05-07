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
    for i in range(1, len(df_sorted) - 1):
        prev_row = df_sorted.iloc[i - 1]
        next_row = df_sorted.iloc[i + 1]
        current_row = df_sorted.iloc[i]

        # Prepare the new row as specified
        new_row = {
            "sys_id": current_row["sys_id"],
            "node_id": 0,
            'node_name': prev_row['node_name'] + '_comm_' + next_row['node_name'],
            "num_ops": 0,
            "tensor_size": 0,
            "perf": 0,
            "operational_intensity": 0,
            "elapsed_time": next_row["issue_tick"] - prev_row["callback_tick"],
            "issue_tick": prev_row["callback_tick"],
            "callback_tick": next_row["issue_tick"],
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

    # Save to a new CSV, or overwrite as needed
    # df_combined_sorted.to_csv("1_1_16_4_0_roofline_with_comm.csv", index=False)

    return df_combined_sorted


def expand_df_by_cycles(df, time_window=50000):
    new_rows = []

    for i in range(len(df)):
        row = df.iloc[i]
        elapsed_time = row["elapsed_time"]

        rep = elapsed_time // time_window
        remaining_time = elapsed_time % time_window

        for j in range(int(rep)):
            # Prepare the new row as specified
            new_row = row.copy()
            new_row["elapsed_time"] = time_window
            new_row["issue_tick"] = row["issue_tick"] + (j * time_window)
            new_row["callback_tick"] = new_row["issue_tick"] + time_window
            new_rows.append(new_row)

        if remaining_time != 0:
            new_row = row.copy()
            new_row["elapsed_time"] = remaining_time
            new_row["issue_tick"] = row["issue_tick"] + (rep * time_window)
            new_row["callback_tick"] = new_row["issue_tick"] + remaining_time
            new_row["perf"] = new_row["perf"] * remaining_time / time_window
            new_row['operational_intensity'] = new_row['operational_intensity'] * remaining_time / time_window
            new_rows.append(new_row)

    df_new = pd.DataFrame(new_rows)
    # Sort the final dataframe by sys_id and issue_tick
    df_new = df_new.sort_values(["issue_tick"]).reset_index(drop=True)

    return df_new


def expand_df_and_avarage(df, time_window = 50000):
    new_rows = []

    remaining_time_prev = 0
    perf_prev = 0
    issue_tick_perv = 0
    operational_intensity_prev = 0
    for i in range(len(df)):
        row = df.iloc[i]
        elapsed_time = remaining_time_prev + row['elapsed_time']

        rep = elapsed_time//time_window
        remaining_time = elapsed_time % time_window
        
        if (rep > 0):
            new_row = row.copy()
            new_row['elapsed_time'] = time_window
            new_row['issue_tick'] = row['issue_tick'] if issue_tick_perv == 0 else issue_tick_perv
            new_row['callback_tick'] = new_row['issue_tick'] + time_window
            new_row['perf'] = (perf_prev * remaining_time_prev / time_window) + (new_row['perf'] * (time_window - remaining_time_prev) / time_window) 
            new_row['operational_intensity'] = (operational_intensity_prev * remaining_time_prev / time_window) + (new_row['operational_intensity'] * (time_window - remaining_time_prev) / time_window) 
            new_rows.append(new_row)

            for j in range(1, int(rep)):
                # Prepare the new row as specified
                new_row = row.copy()
                new_row['elapsed_time'] = time_window
                new_row['issue_tick'] = row['issue_tick'] + (j * time_window)
                new_row['callback_tick'] = new_row['issue_tick'] + time_window
                new_rows.append(new_row)
        
        if (remaining_time == 0):
            remaining_time_prev = 0
            perf_prev = 0
            issue_tick_perv = 0
            operational_intensity_prev = 0
        else:
            remaining_time_prev = remaining_time
            perf_prev = row['perf']
            operational_intensity_prev = row['operational_intensity']
            issue_tick_perv = row['issue_tick'] + (rep * time_window)

    df_new = pd.DataFrame(new_rows)
    # Sort the final dataframe by sys_id and issue_tick
    df_new = df_new.sort_values(['issue_tick']).reset_index(drop=True)

    return df_new

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
    x_max = max(df["operational_intensity"].max(), 100) * 1.1  # Add 10% headroom
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


def plot_roofline_time(df, beta=2000, pi=300):
    # Compute intersection point
    I_c = (pi * 1e12) / (beta * 1e9)
    P_c = pi

    # Create a new column labeling points based on operational intensity threshold
    df["oi_category"] = np.select(
        [df["operational_intensity"] == 0, df["operational_intensity"] < I_c],
        ["Zero OI", "Memory bound"],
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
                legend=None,
                scale=alt.Scale(
                    domain=["Zero OI", "Memory bound", "Compute bound"],
                    range=["rgba(0,0,0,0.1)", "red", "steelblue"],
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
            yaxis=dict(title="Issue Tick", range=[issue_tick_min, issue_tick_max]),
            zaxis=dict(title="Performance (FLOPs/sec)", range=[0, pi]),
        ),
        title="3D Roofline Model",
        width=900,
        height=800,
    )

    return fig


def get_3d_roofline_plot(csv_file, npu=0, bw=2000, perf=300, time_window=0):
    df = pd.read_csv(csv_file)
    df_single_npu = add_comm_points(df, npu=npu)
    if time_window != 0:
        df_single_npu = expand_df_and_avarage(df_single_npu, time_window=time_window)
    return plot_3d_roofline(
        df_single_npu.loc[
            :, ["perf", "operational_intensity", "issue_tick", "node_id"]
        ].drop_duplicates(),
        bw,
        perf,
    )


def get_2d_roofline_plot_normal(csv_file, npu=0, bw=2000, perf=300):
    df = pd.read_csv(csv_file)
    df_single_npu = add_comm_points(df, npu=npu)
    return plot_roofline(
        df_single_npu.loc[
            :, ["perf", "operational_intensity", "issue_tick", "node_id", "node_name"]
        ].drop_duplicates(),
        bw,
        perf,
    )


def get_2d_roofline_plot_with_time(csv_file, npu=0, bw=2000, perf=300, time_window=0):
    df = pd.read_csv(csv_file)
    df_single_npu = add_comm_points(df, npu=npu)
    if time_window != 0:
        df_single_npu = expand_df_and_avarage(df_single_npu, time_window=time_window)
    return plot_roofline_time(
        df_single_npu.loc[
            :, ["perf", "operational_intensity", "issue_tick", "node_id", "node_name"]
        ].drop_duplicates(),
        bw,
        perf,
    )


def get_timesteps(csv_file, npu=0, time_window=0):
    df = pd.read_csv(csv_file)
    df_single_npu = add_comm_points(df, npu=npu)
    if time_window != 0:
        df_single_npu = expand_df_and_avarage(df_single_npu, time_window=time_window)
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


# TODO:
# Plot with slider
