import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import helper.constants as constants

@st.cache_data(show_spinner='Rendering...')
def plot_sim_results(csv_file):
    df = pd.read_csv(csv_file)
    df["sys_id"] = df["sys_id"].astype(int)
    df["exec_cycles"] = df["exec_cycles"].astype(float)
    df["exposed_comm_cycles"] = df["exposed_comm_cycles"].astype(float)
    df["exposed_comp_cycles"] = df["exposed_comp_cycles"].astype(float)
    
    # Calculate overlap cycles
    df["overlap_cycles"] = df["exec_cycles"] - df["exposed_comp_cycles"]

    df = df.sort_values(by="sys_id", ascending=True)
    chunk_size = 32
    num_chunks = (len(df) + chunk_size - 1) // chunk_size

    # Overall averages (absolute cycles)
    avg_exec = df["exec_cycles"].mean()
    avg_exposed_comm = df["exposed_comm_cycles"].mean()
    avg_exposed_comp = df["exposed_comp_cycles"].mean()
    avg_overlap = df["overlap_cycles"].mean()

    # Percentages for labeling
    avg_exposed_comm_pct = 100 * avg_exposed_comm / avg_exec
    avg_exposed_comp_pct = 100 * avg_exposed_comp / avg_exec
    avg_overlap_pct = 100 * avg_overlap / avg_exec

    # Colors for the three divisions
    colors = ['lightblue', 'lightgreen', 'lightcoral']
    # Darker colors for average bars
    avg_colors = ['blue', 'green', 'red']

    plots = []
    for i in range(num_chunks):
        chunk = df.iloc[i * chunk_size : (i + 1) * chunk_size]
        fig, ax = plt.subplots(figsize=(40, 10))
        y_pos = np.arange(len(chunk))

        # Create stacked bars with three divisions
        # Bottom layer: Overlap (blue)
        ax.bar(y_pos, chunk["overlap_cycles"], color=colors[0], label="Overlap (comm, comp)")
        
        # Middle layer: Exposed Comp (lightgreen)
        ax.bar(y_pos, chunk["exposed_comp_cycles"], 
               bottom=chunk["overlap_cycles"], color=colors[1], label="Exposed Comp")
        
        # Top layer: Exposed Comm (lightcoral)
        ax.bar(y_pos, chunk["exposed_comm_cycles"], 
               bottom=chunk["overlap_cycles"] + chunk["exposed_comp_cycles"], 
               color=colors[2], label="Exposed Comm")

        # Add labels for each division of each bar
        for j, pos in enumerate(y_pos):
            overlap_val = chunk["overlap_cycles"].iloc[j]
            exposed_comp_val = chunk["exposed_comp_cycles"].iloc[j]
            exposed_comm_val = chunk["exposed_comm_cycles"].iloc[j]
            total_val = chunk["exec_cycles"].iloc[j]
            
            # Calculate percentages
            overlap_pct = 100 * overlap_val / total_val if total_val > 0 else 0
            exposed_comp_pct = 100 * exposed_comp_val / total_val if total_val > 0 else 0
            exposed_comm_pct = 100 * exposed_comm_val / total_val if total_val > 0 else 0
            
            # Overlap label (black text)
            if overlap_val > 0:
                ax.text(float(pos), overlap_val / 2, f"{overlap_pct:.1f}\n%",
                       ha="center", va="center", color="black", fontsize=constants.IN_PLOT_LABEL_SIZE)
            
            # Exposed Comp label
            if exposed_comp_val > 0:
                ax.text(float(pos), overlap_val + exposed_comp_val / 2, f"{exposed_comp_pct:.1f}\n%",
                       ha="center", va="center", color="black", fontsize=constants.IN_PLOT_LABEL_SIZE)
            
            # Exposed Comm label
            if exposed_comm_val > 0:
                ax.text(float(pos), overlap_val + exposed_comp_val + exposed_comm_val / 2, f"{exposed_comm_pct:.1f}\n%",
                       ha="center", va="center", color="black", fontsize=constants.IN_PLOT_LABEL_SIZE)

        # X-axis labels
        labels = list(chunk["sys_id"])

        if i == num_chunks - 1:
            # Position for average bar (at the end)
            avg_pos = len(chunk)

            # Plot average bar with three divisions (darker colors)
            # Bottom layer: Average Overlap
            ax.bar(avg_pos, avg_overlap, color=avg_colors[0])
            
            # Middle layer: Average Exposed Comp
            ax.bar(avg_pos, avg_exposed_comp, bottom=avg_overlap, 
                   color=avg_colors[1])
            
            # Top layer: Average Exposed Comm
            ax.bar(avg_pos, avg_exposed_comm, 
                   bottom=avg_overlap + avg_exposed_comp,
                   color=avg_colors[2])

            # Add percentage labels
            ax.text(
                avg_pos,
                avg_overlap / 2,
                f"{avg_overlap_pct:.1f}\n%",
                ha="center",
                va="center",
                color="black",
                fontsize=constants.IN_PLOT_LABEL_SIZE,
            )

            ax.text(
                avg_pos,
                avg_overlap + avg_exposed_comp / 2,
                f"{avg_exposed_comp_pct:.1f}\n%",
                ha="center",
                va="center",
                color="black",
                fontsize=constants.IN_PLOT_LABEL_SIZE,
            )
            
            ax.text(
                avg_pos,
                avg_overlap + avg_exposed_comp + avg_exposed_comm / 2,
                f"{avg_exposed_comm_pct:.1f}\n%",
                ha="center",
                va="center",
                color="black",
                fontsize=constants.IN_PLOT_LABEL_SIZE,
            )

            labels.append("Average")
            ax.set_xticks(np.append(y_pos, avg_pos))
        else:
            ax.set_xticks(y_pos)

        ax.set_xticklabels(labels, rotation=45, fontsize=constants.XTICK_SIZE)
        ax.tick_params(axis='y', labelsize=constants.YTICK_SIZE)
        ax.set_ylabel("Time (Cycles)", fontsize=constants.LABEL_SIZE)
        ax.set_xlabel("NPU ID", fontsize=constants.LABEL_SIZE)
        ax.set_title("Execution Cycles Breakdown", fontsize=constants.TITLE_SIZE)

        if i == 0:
            ax.legend(fontsize=constants.LEGEND_SIZE)

        plots.append(fig)
        plt.close(fig)

    return plots
