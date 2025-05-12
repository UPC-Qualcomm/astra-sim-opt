import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_sim_results(csv_file):
    df = pd.read_csv(csv_file)
    df["sys_id"] = df["sys_id"].astype(int)
    df["execution_cycles"] = df["execution_cycles"].astype(float)
    df["exposed_communication_cycles"] = df["exposed_communication_cycles"].astype(
        float
    )

    df = df.sort_values(by="sys_id", ascending=True)
    chunk_size = 100
    num_chunks = (len(df) + chunk_size - 1) // chunk_size

    # Overall averages (absolute cycles)
    avg_exec = df["execution_cycles"].mean()
    avg_exposed = df["exposed_communication_cycles"].mean()

    # Percentages for labeling
    avg_exposed_pct = 100 * avg_exposed / avg_exec
    avg_compute_pct = 100 - avg_exposed_pct

    plots = []
    for i in range(num_chunks):
        chunk = df.iloc[i * chunk_size : (i + 1) * chunk_size]
        fig, ax = plt.subplots(figsize=(20, 6))
        y_pos = np.arange(len(chunk))

        # Base exec cycles
        ax.bar(y_pos, chunk["execution_cycles"], color="royalblue", label="Exec Cycles")

        # Overlay comm cycles
        ax.bar(
            y_pos,
            chunk["exposed_communication_cycles"],
            color="tomato",
            label="Comm Cycles",
        )

        # X-axis labels
        labels = list(chunk["sys_id"])

        if i == num_chunks - 1:
            # Position for average bar (at the end)
            avg_pos = len(chunk)

            # Plot average bar
            ax.bar(avg_pos, avg_exposed, color="lightcoral", label="Avg Comm Cycles")
            ax.bar(
                avg_pos,
                avg_exec - avg_exposed,
                bottom=avg_exposed,
                color="steelblue",
                label="Avg Exec Cycles",
            )

            ax.text(
                avg_pos + 0.1,
                avg_exposed / 2,
                f"{avg_exposed_pct:.1f}%",
                ha="left",
                va="center",
                color="black",
                fontsize=10,
            )

            ax.text(
                avg_pos + 0.1,
                avg_exposed + (avg_exec - avg_exposed) / 2,
                f"{avg_compute_pct:.1f}%",
                ha="left",
                va="center",
                color="black",
                fontsize=10,
            )

            labels.append("Average")
            ax.set_xticks(np.append(y_pos, avg_pos))
        else:
            ax.set_xticks(y_pos)

        ax.set_xticklabels(labels, rotation=45)
        ax.set_ylabel("Cycles")
        ax.set_xlabel("NPU")
        ax.set_title("Execution Cycles with Communication Breakdown")

        # Add legend only to the final plot to avoid duplicate labels
        if i == num_chunks - 1:
            ax.legend()

        plots.append(fig)
        plt.close(fig)

    return plots
