import itertools
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import yaml
from tqdm import tqdm
import streamlit as st
import shutil

# Custom Modules
import sections.trace_picker as picker


parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(parent_dir)


from run_astrasim import run_astrasim

#TODO: Take into consideration the configuration dims

DIR_NAME = "bw_study"

def get_output_dir(selected_model, selected_config):
    return (
            picker._get_output_dir()
            + "/"
            + selected_model
            + "/"
            + selected_config
            + "/"
            + DIR_NAME
        )
@st.cache_data(show_spinner='setting up results directory...')
def get_results_dir(selected_model, selected_config):
    return Path(__file__).parent / "../../results" / selected_model / selected_config / DIR_NAME
@st.cache_data(show_spinner='setting up network directory...')
def get_network_dir(selected_model, selected_config):
    return (
            Path(__file__).parent / "../../network_log" / selected_model / selected_config / DIR_NAME
        )

def intra_inter_simulations_run(
    parallelism_strategies,
    selected_config,
    selected_model,
    run_button,
    intra_bw_list,
    inter_bw_list,
):
    if run_button:
        base_yml_path = picker._get_configs_dir() + "/" + selected_config + ".yml"
        max_workers = os.cpu_count()

        run_bandwidth_sweep_parallel(
            base_yml_path=base_yml_path,
            parallelism_strategies=parallelism_strategies,
            bw1_values=intra_bw_list,
            bw2_values=inter_bw_list,
            max_workers=max_workers,
            selected_config=selected_config,
            selected_model=selected_model,
        )

        # st.success("✅ All simulations completed successfully!")


@st.cache_data(show_spinner='Running Bandwidth Sweep...')
def run_bandwidth_sweep_parallel(
    base_yml_path,
    parallelism_strategies,
    bw1_values,
    bw2_values,
    max_workers,
    selected_config,
    selected_model,
):
    result_dir = get_results_dir(selected_model, selected_config)
    output_dir = get_output_dir(selected_model, selected_config)
    combinations = list(
        itertools.product(parallelism_strategies, bw1_values, bw2_values)
    )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                run_single_simulation,
                base_yml_path,
                parallelism_strategy,
                bw1,
                bw2,
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

    subprocess.run(
        f"python ../gather_all_NPUs_results.py --sim_logfile {output_dir}  --output_filename {result_dir}",
        shell=True,
        cwd=None,
    )


def run_single_simulation(
    base_yml_path,
    parallelism_strategy,
    bw1,
    bw2,
    selected_config,
    selected_model,
):
    if bw1 > bw2:

        result_dir = get_results_dir(selected_model, selected_config)
        output_dir = get_output_dir(selected_model, selected_config)
        network_log = get_network_dir(selected_model, selected_config)
        app_dir = Path(__file__).parent
        workload_configuration = (
            app_dir / "../../workload" / selected_model / parallelism_strategy
        )
        memory_config = app_dir / "../../configuration" / "RemoteMemory.json"
        suffix = f"_bw_{bw1}_{bw2}"
        #TODO: Creat a button to clean the dirs
        #for path in [output_dir, result_dir, network_log]:
        #    os.system(f"rm -rf {path}")

        # delete_files_with_suffix([output_dir, result_dir, network_log], suffix)

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(network_log, exist_ok=True)
        os.makedirs(result_dir, exist_ok=True)

        with open(base_yml_path, "r") as f:
            config = yaml.safe_load(f)

        config["bandwidth"] = [bw1] + [bw2] * (len(config["npus_count"]) - 1)

        output_config_path = (
            output_dir + f"/{parallelism_strategy}_bw_{bw1}_{bw2}_network.yml"
        )

        with open(output_config_path, "w") as f:
            yaml.dump(
                config,
                f,
                default_flow_style=None,
                sort_keys=False,
            )

        failed_cmd = run_astrasim(
            workload_path=str(workload_configuration),
            system=str(app_dir / "../../configuration" / f"{selected_config}_sys.json"),
            network=str(output_config_path),
            memory=str(memory_config),
            output_dir=str(output_dir),
            network_log=str(network_log),
            suffix=suffix,
        )

        if failed_cmd != "":
            raise RuntimeError(f"Simulation failed: {failed_cmd}")


def clean_sim_dirs(selected_model, config_names):
    
    for config in config_names:
        result_dir = Path(get_results_dir(selected_model, config))
        output_dir = Path(get_output_dir(selected_model, config))
        network_log = Path(get_network_dir(selected_model, config))
        
        for dir_path in [result_dir, output_dir, network_log]:
            if dir_path.exists() and dir_path.is_dir():
                shutil.rmtree(dir_path)  