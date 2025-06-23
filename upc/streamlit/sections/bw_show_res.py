import glob
import os
from pathlib import Path
import pandas as pd
import regex as re
import matplotlib.pyplot as plt
import plotly.colors as pc
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import numpy as np


def show_inter_intra_sim_res(parallelism_strategies, result_dir, selected_config):
    ###st.title("Execution vs Communication Cycles Analysis")

    if not result_dir.exists() or not any(result_dir.iterdir()):
        st.warning("Results directory is empty. Please run simulations first.")
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

            else:
                st.write(f"No matching files found for strategy {strategy}.")

        if all_data:
            combined_df = get_combined_df(all_data, group_by="strategy")
            ###st.subheader(f"Total Cycles Comparison Across Bandwidth Settings - Network config {selected_config}")
            
            summary_fig = get_total_cycles_plot(
                combined_df,
                title=f"Total Cycles vs Bandwidth for Different Parallelism Strategies - Network config {selected_config}",
                group_by="strategy",
            )
            st.plotly_chart(
                summary_fig, use_container_width=True, key=f"plot_{selected_config}"
            )

            # st.subheader("Raw Combined Data")
            # st.dataframe(combined_df_sorted, key=f"df_{selected_config}")

            # csv = combined_df_sorted.to_csv(index=False).encode("utf-8")
            # st.download_button(
            #    "Download Combined Data as CSV",
            #    data=csv,
            #    file_name="combined_cycles_analysis.csv",
            # )


def plot_overlapped(df, strategy, chunk_size=30):
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
            f"{strategy} - Exec vs Comm Cycles (examples: {start} to {end} from {len(df)})"
        )
        ax.legend()
        plt.tight_layout()
        figures.append(fig)

    return figures


def get_filtered_df(result_path, strategy):
    files = glob.glob(str(result_path / f"{strategy}_*.csv"))
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
    df_filtered = pd.DataFrame(records)
    df_filtered["total_cycles"] = df_filtered["execution_cycles"]
    df_filtered["strategy"] = strategy
    return df_filtered


def get_total_cycles_plot(
    all_data,
    title,
    group_by="strategy",
    y_label="Total Cycles",
    x_label="Intra_Inter Bandwidth (GB/s)",
):
    unique_groups = all_data[group_by].unique()
    color_sequence = pc.qualitative.Set2
    color_map = {
        group: color_sequence[i % len(color_sequence)]
        for i, group in enumerate(unique_groups)
    }

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
            group_by: group_by.replace("_", " ").title(),
        },
        color_discrete_map=color_map,
    )

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
                text=[f"{row['total_cycles']:.0f}"],
                textposition="top center",
                showlegend=False,
            )
        )

    fig.update_layout(xaxis_tickangle=-45, height=600, hovermode="x unified")

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


def show_res_across_configs(parallelism_strategy, res_dirs_configs):
    all_data = combine_data_across_config(parallelism_strategy, res_dirs_configs)

    combined_df = get_combined_df(all_data, group_by="config")

    fig = get_total_cycles_plot(
        combined_df,
        title=f"Total Cycles vs Bandwidth for Different Network configs - Parallelism Strategy {parallelism_strategy}",
        group_by="config",
    )

    st.plotly_chart(fig, use_container_width=True, key=f"plot_{parallelism_strategy}")


def combine_data_across_config(parallelism_strategy, res_dirs_configs):
    all_data = []

    for res_dir, config in res_dirs_configs:
        df_filtered = get_filtered_df(res_dir, parallelism_strategy)
        df_filtered["config"] = config

        if not df_filtered.empty:
            all_data.append(df_filtered)

    return all_data

def get_combined_df(data, group_by="strategy"):
    combined_df = pd.concat(data).reset_index(drop=True)
    combined_df_sorted = combined_df.sort_values(["intra_bw", "inter_bw"])
    combined_df_sorted = complete_missing_points(combined_df_sorted, group_by=group_by)

    return combined_df_sorted