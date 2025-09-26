import itertools
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from scipy.stats import gmean
from tqdm import tqdm
import matplotlib.pyplot as plt
import streamlit as st

# Custom Modules
import sections.trace_picker as picker
import sections.best_combination as best
import sections.bw_form as bw_form
import sections.bw_show_res as bw_show
import helper.bw_run_sim as bw_run


def render(df, selected_model, selected_config):
    options = list(range(1, len(df) + 1))
    start, end = st.slider(
        "Select Range of Experiment Rankings (by 'total') to Consider:",
        min_value=min(options),
        max_value=max(options),
        value=(2, 10)
    )
    df_sorted = df.sort_values(by="total", ascending=True).reset_index(drop=True)
    df_range = df_sorted.iloc[start-1:end]  
    st.write(f"Showing experiments ranked from {start} to {end}:")
    st.dataframe(df_range)

    figs = picker.plot_experiments_bound_breakdown(df_range, chunk_size=32)
    for fig in figs:
        st.pyplot(fig)

    st.pyplot(get_exposed_comm_fig(df_range))

    st.markdown("---")

    best.best_combinations(df_range, selected_config, selected_model)

    st.title("AstraSim Bandwidth Sweep Runner")
    all_configs = bw_form.get_configuration_options()
    run_button, intra_bw_list, inter_bw_list, selected_config_names_list = (
        bw_form.sweep_bw_form(selected_config, selected_model)
    )

    #if st.button(f"Clean Simulation Directories for {selected_model}"):
    if run_button:
        bw_run.clean_sim_dirs(selected_model, selected_config_names_list)
        st.success(f"Cleaned simulation directories for {selected_model}.")

    st.info("Starting bandwidth sweep simulations...")
    parallelism_strategies = df_range["file_name"].tolist()

    st.subheader("Study the effect of the bandwidth over parallelism strategies")
    all_res_dirs_configs = []
    for config in all_configs:
        if config in selected_config_names_list:
            bw_run.intra_inter_simulations_run(
                parallelism_strategies,
                config,
                selected_model,
                run_button,
                intra_bw_list,
                inter_bw_list,
            )

        #st.markdown("---")
        result_dir = bw_run.get_results_dir(selected_model, config)
        #output_dir = bw_run.get_output_dir(selected_model, selected_config)
        if result_dir.exists() and result_dir.is_dir():
            bw_show.show_inter_intra_sim_res(parallelism_strategies, result_dir, config)
            all_res_dirs_configs.append((result_dir, config))

        st.markdown("---")
    st.subheader("Study the effect of the bandwidth over network configurations")
    for parallelism_strategy in parallelism_strategies:
        bw_show.show_res_across_configs(parallelism_strategy, all_res_dirs_configs)

def get_exposed_comm_fig(df):
    comm_mean, comm_std, comm_gmean = (
        df["comm_percent"].mean(),
        df["comm_percent"].std(),
        gmean(df["comm_percent"][df["comm_percent"] > 0]),
    )
    sorted_comm = np.sort(df["comm_percent"].values)
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(sorted_comm, marker="o", linestyle="-", color="blue")

    ax.set_xticks(range(len(df)))
    ax.set_xlabel("Experiment Number", fontsize=25)
    ax.set_ylabel("Exposed\nComm. Time\n(%)", fontsize=25)
    ax.set_title(
        f"Sorted - Exposed Communication Percentage per Experiment\n(μ={comm_mean:.2f}%, σ={comm_std:.2f}%, gμ={comm_gmean:.2f}%)",
        fontsize=25,
    )
    ax.grid(True)

    ax.tick_params(axis="both", which="major", labelsize=24)
    ax.tick_params(axis="both", which="minor", labelsize=24)

    return fig
