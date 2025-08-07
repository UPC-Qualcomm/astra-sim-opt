import glob
import os
import pandas as pd
import regex as re
import matplotlib.pyplot as plt
import plotly.colors as pc
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import numpy as np
import helper.constants as constants
from plotly.subplots import make_subplots

def show_inter_intra_sim_res(parallelism_strategies, result_dir, selected_config, is_3d=False):
    ###st.title("Execution vs Communication Cycles Analysis")

    if not result_dir.exists() or not any(result_dir.iterdir()):
        st.warning(f"Results directory {result_dir.parts[-2]} is empty. Please run simulations first.")
    else:
        all_data = []

        for strategy in parallelism_strategies:
            # st.subheader(f"strategy: {strategy}")

            df_filtered = get_filtered_df(result_dir, strategy)

            if not df_filtered.empty:
                all_data.append(df_filtered)

                # figures = plot_overlapped(df_filtered, strategy, chunk_size=30)
                # for fig in figures:
                #    st.pyplot(fig)

        if all_data:
            combined_df = get_combined_df(all_data, group_by="strategy")
            ###st.subheader(f"Total Cycles Comparison Across Bandwidth Settings - Network config {selected_config}")
            if is_3d:
                summary_fig = get_total_cycles_plot_3d(
                    combined_df,
                    title=f"Total Cycles vs Bandwidth for Different Parallelism Strategies - Network Config: {selected_config}, #NPU=DP*TP*SP*PP",
                    group_by="strategy",
                    legend_title='DP, TP, SP, PP, FSDP',
                )
            else:
                summary_fig = get_total_cycles_split_by_intra(
                    combined_df,
                    title=(
                        "Total Cycles vs Bandwidth for Different Parallelism Strategies\n"
                        f"Network Config: {selected_config}, #NPU=DP*TP*SP*PP"
                    ),
                    group_by="strategy",
                    legend_title='DP, TP, SP, PP, FSDP',
                )
            if isinstance(summary_fig, list):
                for i, fig in enumerate(summary_fig):
                    st.plotly_chart(fig, use_container_width=True, key=f"plot_{selected_config}_{i}")
            else:
                st.plotly_chart(summary_fig, use_container_width=True, key=f"plot_{selected_config}")

            # st.subheader("Raw Combined Data")
            # st.dataframe(combined_df_sorted, key=f"df_{selected_config}")

            # csv = combined_df_sorted.to_csv(index=False).encode("utf-8")
            # st.download_button(
            #    "Download Combined Data as CSV",
            #    data=csv,
            #    file_name="combined_cycles_analysis.csv",
            # )


def plot_overlapped(df, strategy, chunk_size=30):
    df = df.sort_values(by="exec_cycles").reset_index(drop=True)
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
            df_chunk["exec_cycles"],
            0.4,
            color="skyblue",
            label="Exec Cycles",
        )
        ax.bar(
            index,
            df_chunk["exposed_comm_cycles"],
            0.4,
            color="red",
            alpha=0.6,
            label="Comm Cycles",
        )

        for i, (exec_c, comm_c) in enumerate(
            zip(df_chunk["exec_cycles"], df_chunk["exposed_comm_cycles"])
        ):
            perc = (comm_c / exec_c) * 100 if exec_c != 0 else 0
            ax.text(i, comm_c / 2, f"{perc:.1f}%", ha="center", fontsize=8)

        ax.set_xticks(index)
        ax.set_xticklabels(df_chunk["bw_label"], rotation=45, ha="right")
        ax.set_ylabel("Cycles")
        ax.set_title(
            f"{strategy} - Exec vs Comm Cycles (examples: {start} to {end} from {len(df)})"
        )
        ax.legend()
        plt.tight_layout()
        figures.append(fig)

    return figures

@st.cache_data
def get_filtered_df(result_path, strategy):
    files = glob.glob(str(result_path / f"{strategy}_*.csv"))
    records = []
    if not files:
        st.warning(f"No files found for strategy: {strategy.split('.')[0]}")
        return pd.DataFrame()
    for file in files:
        df = pd.read_csv(file)
        if (
            "exec_cycles" in df.columns
            and "exposed_comm_cycles" in df.columns
        ):
            min_row = df.loc[df["exec_cycles"].idxmin()]

            match = re.search(r"bw_(\d+)_(\d+)", file)
            intra_bw, inter_bw = (
                (int(match.group(1)), int(match.group(2))) if match else (None, None)
            )
            bw_label = f"{intra_bw}_{inter_bw}"

            records.append(
                {
                    "file_name": os.path.basename(file).replace(".csv", ""),
                    "bw_label": bw_label,
                    "exec_cycles": min_row["exec_cycles"],
                    "exposed_comm_cycles": min_row["exposed_comm_cycles"],
                    "intra_bw": intra_bw,
                    "inter_bw": inter_bw,
                }
            )
    df_filtered = pd.DataFrame(records)
    df_filtered["total_cycles"] = df_filtered["exec_cycles"]
    df_filtered["strategy"] = strategy.split(".")[0] 
    return df_filtered

@st.cache_data
def get_total_cycles_plot(
    df,
    title,
    group_by="strategy",
    legend_title='',
    y_label="Total Cycles",
    x_label="Intra, Inter - Bandwidth (GB/s)",
):
    # Copy and transform the DataFrame
    all_data = df.copy()
    all_data["bw_label"] = all_data["bw_label"].str.split("_").apply(
        lambda parts: f"Intra: {parts[0]}, Inter: {parts[1]}"
    )

    # Define color mapping
    unique_groups = all_data[group_by].unique()
    color_sequence = pc.qualitative.Set2
    color_map = {
        group: color_sequence[i % len(color_sequence)]
        for i, group in enumerate(unique_groups)
    }

    # Create the main line plot
    fig = px.line(
        all_data,
        x="bw_label",
        y="total_cycles",
        color=group_by,
        markers=True,
        title=title,
        labels={
            "bw_label": x_label,
            "total_cycles": y_label,
            group_by: legend_title,
        },
        color_discrete_map=color_map,
    )

    # Add markers for minimum total_cycles per group
    min_points = all_data.loc[
        all_data.groupby(group_by)["total_cycles"].idxmin()
    ].reset_index(drop=True)

    for _, row in min_points.iterrows():
        group = row[group_by]
        color = color_map[group]

        fig.add_trace(
            go.Scatter(
                x=[row["bw_label"]],
                y=[row["total_cycles"]],
                mode="markers+text",
                marker=dict(size=12, color=color, symbol="diamond"),
                name=f"Best: {group}",
                textposition="top center",
                textfont=dict(color='black', size=constants.TITLE_SIZE),
                showlegend=False,
            )
        )

    # Update layout with global and specific font settings
    fig.update_layout(
        font=dict(
            size=constants.LABEL_SIZE,
            color='black'
        ),
        xaxis_tickangle=-25,
        height=600,
        width=1000,
        margin=dict(l=20, r=20),
        hovermode="x unified",
        title_font=dict(size=constants.TITLE_SIZE, color="black"),
        legend_font=dict(size=constants.LEGEND_SIZE, color="black"),
        legend_title=dict(font=dict(size=constants.LEGEND_SIZE, color="black")),
        xaxis=dict(
            title_font=dict(size=constants.LABEL_SIZE, color="black"),
            tickfont=dict(size=constants.LABEL_SIZE, color="black"),
            showgrid=True,
            gridcolor='lightgray',
            gridwidth=3,
            griddash='dot'
        ),
        yaxis=dict(
            title_font=dict(size=constants.LABEL_SIZE, color="black"),
            tickfont=dict(size=constants.LABEL_SIZE, color="black"),
        ),
    )

    fig.update_traces(line=dict(width=4), textfont=dict(color='black'))     

    return fig

def get_total_cycles_split_by_intra(
    df,
    title,
    group_by="strategy",
    legend_title='',
    y_label="Time (Cycles)",
):
    # Font size adjustment
    font_size_delta = -8

    # Copy and parse bandwidths
    all_data = df.copy()
    all_data[["intra", "inter"]] = all_data["bw_label"].str.split("_", expand=True).astype(int)

    # Find the minimum value across all data for highlighting
    min_value = all_data["total_cycles"].min()
    all_data['is_min'] = all_data["total_cycles"] == min_value

    # Define color mapping
    unique_groups = all_data[group_by].unique()
    color_sequence = pc.qualitative.Set2
    color_map = {
        group: color_sequence[i % len(color_sequence)]
        for i, group in enumerate(unique_groups)
    }

    # Get sorted unique intra values
    intra_values = sorted(all_data["intra"].unique())

    # Filter out intra values that have no data
    valid_intra_values = []
    subplot_widths = []

    for intra_val in intra_values:
        subset_data = all_data[all_data["intra"] == intra_val]
        if not subset_data.empty:
            valid_intra_values.append(intra_val)
            subplot_widths.append(len(subset_data["inter"].unique()))

    # Normalize subplot widths
    total_width = sum(subplot_widths)
    subplot_widths = [w / total_width for w in subplot_widths]

    # Create subplots with custom widths
    fig = make_subplots(
        rows=1, 
        cols=len(valid_intra_values),
        column_widths=subplot_widths,
        subplot_titles=None
    )

    # Add traces
    for group in unique_groups:
        group_data = all_data[all_data[group_by] == group]
        for i, intra_val in enumerate(valid_intra_values):
            subset_data = group_data[group_data["intra"] == intra_val].sort_values("inter")
            if not subset_data.empty:
                group_min = group_data["total_cycles"].min()
                symbols = ['diamond' if y == group_min else 'circle' for y in subset_data["total_cycles"]]
                sizes = [14 if y == group_min else 10 for y in subset_data["total_cycles"]]
                
                fig.add_trace(
                    go.Scatter(
                        x=subset_data["inter"],
                        y=subset_data["total_cycles"],
                        mode="lines+markers",
                        name=str(group),
                        line=dict(color=color_map[group], width=6),
                        marker=dict(
                            color=color_map[group],
                            symbol=symbols,
                            size=sizes
                        ),
                        showlegend=(i == 0),
                    ),
                    row=1, col=i+1
                )

    # Add layout and global styles
    html_title = title.replace('\n', '<br>')
    fig.update_layout(
        title={
            "text": f"<span style='font-weight:normal'>{html_title}</span>",
            "x": 0.5,
            "xanchor": "center"
        },
        font=dict(size=constants.LABEL_SIZE + font_size_delta, color="black"),
        height=600,
        width=1000,
        margin=dict(l=300, r=40, t=120, b=160),
        title_font=dict(size=constants.TITLE_SIZE + font_size_delta, color="black", family="Arial"),
        legend_font=dict(size=constants.LEGEND_SIZE + font_size_delta, color="black"),
        legend_title=dict(font=dict(size=constants.LEGEND_SIZE + font_size_delta, color="black")),
        showlegend=True
    )

    # Set consistent y-axis
    overall_min_cycles = all_data["total_cycles"].min()
    overall_max_cycles = all_data["total_cycles"].max()
    y_padding = 0.05 * (overall_max_cycles - overall_min_cycles)

    for i, intra_val in enumerate(valid_intra_values):
        subset_data = all_data[all_data["intra"] == intra_val]
        unique_inter_values = sorted(subset_data["inter"].unique())

        fig.update_xaxes(
            showgrid=True,
            gridcolor='lightgray',
            griddash='dot',
            tickfont=dict(size=constants.XTICK_SIZE + font_size_delta, color='black'),
            tickmode='array',
            tickvals=unique_inter_values,
            ticktext=[str(val) for val in unique_inter_values],
            row=1, col=i+1,
            tickangle=-45  # Change from -25 to 25 for correct rotation direction
        )

        fig.update_yaxes(
            showgrid=True,
            gridcolor='lightgray',
            griddash='dot',
            tickfont=dict(size=constants.XTICK_SIZE + font_size_delta, color='black'),
            showticklabels=(i == 0),
            title_font=dict(size=constants.LABEL_SIZE + font_size_delta, color='black'),
            range=[overall_min_cycles - y_padding, overall_max_cycles + y_padding],
            row=1, col=i+1
        )

    # Intra BW annotations (below each subplot)
    for i, intra_val in enumerate(valid_intra_values):
        label_text = f"Intra BW (GB/s):          {intra_val}" if i == 0 else f"{intra_val}"
        fig.add_annotation(
            text=label_text,
            xref="x domain" if i == 0 else f"x{i+1} domain",
            yref="paper",
            x=-0.45 if i == 0 else 0.5,
            y=-0.33,
            showarrow=False,
            font=dict(size=constants.LABEL_SIZE + font_size_delta, color="black"),
            xanchor="center"
        )

    # Inter BW label (only once under first plot)
    fig.add_annotation(
        text="Inter BW (GB/s):",
        xref="x domain",
        yref="paper",
        x=-0.8,
        y=-0.2,
        showarrow=False,
        font=dict(size=constants.LABEL_SIZE + font_size_delta, color="black"),
        xanchor="center"
    )

    # Y-axis title (once, vertically left)
    fig.add_annotation(
        text=y_label,
        xref="paper",
        yref="paper",
        x=-0.07,
        y=0.5,
        showarrow=False,
        font=dict(size=constants.LABEL_SIZE + font_size_delta, color="black"),
        xanchor="center",
        textangle=-90
    )

    filename = "bw_study.svg" 
    fig.write_image(f"{filename}") 
    return fig

@st.cache_data
def get_total_cycles_plot_3d(
    df,
    title,
    group_by="strategy",
    legend_title='',
    z_label="Total Cycles",
    x_label="Intra Bandwidth (GB/s)",
    y_label="Inter Bandwidth (GB/s)",
):
    all_data = df.copy()

    # Ensure intra_bw and inter_bw are numeric
    all_data["intra_bw"] = pd.to_numeric(all_data["intra_bw"], errors="coerce")
    all_data["inter_bw"] = pd.to_numeric(all_data["inter_bw"], errors="coerce")

    unique_groups = all_data[group_by].unique()
    color_sequence = pc.qualitative.Set2
    color_map = {
        group: color_sequence[i % len(color_sequence)]
        for i, group in enumerate(unique_groups)
    }

    traces = []
    for group in unique_groups:
        group_df = all_data[all_data[group_by] == group]
        traces.append(
            go.Scatter3d(
                x=group_df["intra_bw"],
                y=group_df["inter_bw"],
                z=group_df["total_cycles"],
                mode="markers",
                marker=dict(size=7, color=color_map[group]),
                line=dict(color=color_map[group], width=4),
                name=str(group),
                text=[
                    f"{group_by}: {group}<br>Intra BW: {x}<br>Inter BW: {y}<br>Total Cycles: {z:.0f}"
                    for x, y, z in zip(group_df["intra_bw"], group_df["inter_bw"], group_df["total_cycles"])
                ],
                hoverinfo="text",
            )
        )

    layout = go.Layout(
        scene=dict(
            xaxis=dict(title=x_label, backgroundcolor='rgba(240,240,240,0.8)', gridcolor='gray', showbackground=True),
            yaxis=dict(title=y_label, backgroundcolor='rgba(240,240,240,0.8)', gridcolor='gray', showbackground=True),
            zaxis=dict(title=z_label, backgroundcolor='rgba(240,240,240,0.8)', gridcolor='gray', showbackground=True),
            camera=dict(eye=dict(x=-2.0, y=2.0, z=1.2)),
        ),
        title=title,
        margin=dict(l=10, r=10, b=10, t=50),
        height=700,
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.7)', bordercolor='black', borderwidth=1),
        showlegend=True,
    )

    fig = go.Figure(data=traces, layout=layout)
    fig.update_layout(template='plotly_white')
    return fig


def complete_missing_points(all_data, group_by="strategy"):
    all_prefixes = all_data[group_by].unique()
    all_bw_labels = all_data["bw_label"].unique()
    full_index = pd.MultiIndex.from_product(
        [all_prefixes, all_bw_labels], names=[group_by, "bw_label"]
    )

    all_data = (
        all_data.set_index([group_by, "bw_label"]).reindex(full_index).reset_index()
    )
    all_data["total_cycles"] = all_data["total_cycles"].fillna(np.nan)
    all_data[["intra_bw", "inter_bw"]] = (
        all_data["bw_label"].str.split("_", expand=True).astype(float)
    )

    return all_data


def show_res_across_configs(parallelism_strategy, res_dirs_configs, is_3d=False):
    all_data = combine_data_across_config(parallelism_strategy, res_dirs_configs)
    if all_data:
        combined_df = get_combined_df(all_data, group_by="config")
        dp , tp, sp, pp, fsdp = parallelism_strategy.split('.')[0].split('_')
        npu_count = int(dp) * int(tp) * int(sp) * int(pp)
        if is_3d:
            fig = get_total_cycles_plot_3d(
                combined_df,
                title=f"Total Cycles vs Bandwidth for Different Network Configs - Parallelism Strategy DP:{dp}, TP:{tp}, SP{sp}, PP:{pp}, FSDP:{fsdp} and #NPUs = {npu_count}",
                group_by="config",
                #legend_title='Topology',
            )
        else:
            fig = get_total_cycles_split_by_intra(
                combined_df,
                title=f"Total Cycles vs Bandwidth for Different Network Configs - Parallelism Strategy DP:{dp}, TP:{tp}, SP{sp}, PP:{pp}, FSDP:{fsdp} and #NPUs = {npu_count}",
                group_by="config",
                #legend_title='Topology', 
            )

        if isinstance(fig, list):
            for i, f in enumerate(fig):
                st.plotly_chart(f, use_container_width=True, key=f"plot_{parallelism_strategy}_{i}")
        else:
            st.plotly_chart(fig, use_container_width=True, key=f"plot_{parallelism_strategy}")
    else:
        st.warning(f"No data found for the parallelsim strategy {parallelism_strategy}")

@st.cache_data
def combine_data_across_config(parallelism_strategy, res_dirs_configs):
    all_data = []

    for res_dir, config in res_dirs_configs:
        df_filtered = get_filtered_df(res_dir, parallelism_strategy)
        df_filtered["config"] = config

        if not df_filtered.empty:
            all_data.append(df_filtered)

    return all_data

@st.cache_data
def get_combined_df(data, group_by="strategy"):
    combined_df = pd.concat(data).reset_index(drop=True)
    combined_df_sorted = combined_df.sort_values(["intra_bw", "inter_bw"])
    combined_df_sorted = complete_missing_points(combined_df_sorted, group_by=group_by)

    return combined_df_sorted