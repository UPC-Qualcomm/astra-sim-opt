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
    num = st.select_slider(
        "Pick The Number of Best Experiments To Consider:", options=options, value=10
    )
    df_top = df.sort_values(by="total", ascending=True).head(num)
    st.subheader(f"Best {num} examples")
    st.dataframe(df_top)

    figs = picker.plot_experiments_bound_breakdown(df_top, chunk_size=32)
    for fig in figs:
        st.pyplot(fig)

    st.pyplot(get_exposed_comm_fig(df_top))

    st.markdown("---")

    best.best_combinations(df_top, selected_config, selected_model)

    st.title("AstraSim Bandwidth Sweep Runner")

    run_button, intra_bw_list, inter_bw_list = bw_form.sweep_bw_form()

    st.info("Starting bandwidth sweep simulations...")
    parallelism_strategies = df_top["file_name"].tolist()
    bw_run.intra_inter_simulations_run(
        parallelism_strategies,
        selected_config,
        selected_model,
        run_button,
        intra_bw_list,
        inter_bw_list,
    )

    st.markdown("---")

    bw_show.show_inter_intra_sim_res(parallelism_strategies, selected_model)


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


@st.cache_data
def run_bandwidth_sweep_parallel(
    base_yml_path,
    output_config_dir,
    parallelism_strategies,
    bw1_values,
    bw2_values,
    app_dir,
    max_workers,
    selected_config,
    selected_model,
):
    combinations = list(
        itertools.product(parallelism_strategies, bw1_values, bw2_values)
    )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                bw_run.run_single_simulation,
                base_yml_path,
                output_config_dir,
                parallelism_strategy,
                bw1,
                bw2,
                app_dir,
                selected_config,
                selected_model,
            )
            for parallelism_strategy, bw1, bw2 in combinations
        ]

        for f in tqdm(
            as_completed(futures), total=len(futures), desc="Running simulations"
        ):
            try:
                f.result()
            except Exception as e:
                st.error(f"Simulation failed: {e}")

    output = app_dir / "../../output" / selected_model
    result = app_dir / "../../results" / selected_model
    subprocess.run(
        f"python ../gather_all_NPUs_results.py --sim_logfile {output}  --output_filename {result}",
        shell=True,
        cwd=None,
    )
