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


def show_inter_intra_sim_res(parallelism_strategies, selected_model):
    result = Path(__file__).parent / "../../results" / selected_model

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
