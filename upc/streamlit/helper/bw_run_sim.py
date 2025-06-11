import itertools
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import yaml
from tqdm import tqdm
import streamlit as st

# Custom Modules
import sections.trace_picker as picker


parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(parent_dir)


from run_astrasim import run_astrasim


def intra_inter_simulations_run(
    parallelism_strategies,
    selected_config,
    selected_model,
    run_button,
    intra_bw_list,
    inter_bw_list,
):
    app_dir = Path(__file__).parent
    if run_button:
        base_yml_path = picker._get_configs_dir() + "/" + selected_config + ".yml"
        output_config_dir = picker._get_output_dir() + "/" + selected_config
        max_workers = os.cpu_count()

        run_bandwidth_sweep_parallel(
            base_yml_path=base_yml_path,
            output_config_dir=output_config_dir,
            parallelism_strategies=parallelism_strategies,
            bw1_values=intra_bw_list,
            bw2_values=inter_bw_list,
            app_dir=app_dir,
            max_workers=max_workers,
            selected_config=selected_config,
            selected_model=selected_model,
        )

        # st.success("✅ All simulations completed successfully!")


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
                run_single_simulation,
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


def run_single_simulation(
    base_yml_path,
    output_config_dir,
    model_name,
    bw1,
    bw2,
    app_dir,
    selected_config,
    selected_model,
):
    if bw1 > bw2:
        workload_configuration = (
            app_dir / "../../workload" / selected_model / model_name
        )
        memory_config = app_dir / "../../configuration" / "RemoteMemory.json"
        network_log = app_dir / "../../network_log" / selected_model
        output = app_dir / "../../output" / selected_model
        result = app_dir / "../../results" / selected_model
        suffix = f"_bw_{bw1}_{bw2}"

        for path in [output, result, network_log]:
            new_path = path.with_name(path.name + suffix)
            os.system(f"rm -rf {new_path}")

        os.makedirs(output, exist_ok=True)
        os.makedirs(network_log, exist_ok=True)
        os.makedirs(result, exist_ok=True)

        with open(base_yml_path, "r") as f:
            config = yaml.safe_load(f)

        config["bandwidth"] = [bw1, bw2]

        output_config_path = (
            app_dir
            / "../../output"
            / selected_model
            / f"{model_name}_bw_{bw1}_{bw2}_network.yml"
        )
        output_config_path.parent.mkdir(parents=True, exist_ok=True)

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
            output_dir=str(output),
            network_log=str(network_log),
            suffix=suffix,
        )

        if failed_cmd != "":
            raise RuntimeError(f"Simulation failed: {failed_cmd}")
