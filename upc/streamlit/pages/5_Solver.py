import streamlit as st
import numpy as np
import pandas as pd
import sections.astrasim_solver as astra_solver
import sections.workload_solver as workload_solver
import plotly.express as px
import os
import json
from itertools import product
import yaml
import getpass
import datetime

st.set_page_config(layout="wide")

st.title("Parallelism Strategy Solver")

st.info("""
This page performs a random search to find the best parallelism strategy for a given model and hardware configuration.

**How it works:**
1. You select a network configuration and model parameters.
2. For each parallelism strategy (Data, Pipeline, Tensor, Sequence), you can select multiple possible options.
3. The solver will randomly sample valid combinations (where the product of the selected parallelism factors equals the number of NPUs).
4. For each sampled combination, the tool generates a workload, runs a simulation, and collects the communication and computation cycles.
5. The best strategies are shown in a table and a stacked bar plot.

**Limitations:**
- This is a demo and uses random search, not an exhaustive or optimal search.
- Only a limited number of simulations are run (as set by the slider).
- Some combinations may not be feasible for your hardware or model.
- Only communication and computation cycles are shown; memory cycles and other metrics are not included.
- Results with zero total cycles are ignored in the plots.

In the future, this will be replaced with a more sophisticated search algorithm.
""")

def get_network_configurations():
    network_config_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'configuration')
    try:
        network_files = [f for f in os.listdir(network_config_dir) if f.endswith('.yml')]
        return network_files
    except FileNotFoundError:
        st.error(f"Network configuration directory not found at: {network_config_dir}")
        return []

def load_network_config(network_file):
    network_config_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'configuration')
    try:
        with open(os.path.join(network_config_dir, network_file)) as f:
            raw_content = f.read()
            return raw_content, yaml.safe_load(raw_content)
    except (FileNotFoundError, yaml.YAMLError):
        return None, None

def load_system_config(network_file):
    # Assuming system file has the same base name but with .json extension
    sys_file_name = os.path.splitext(network_file)[0] + "_sys.json"
    sys_config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'configuration', sys_file_name)
    try:
        with open(sys_config_path) as f:
            return f.read()
    except FileNotFoundError:
        st.error(f"System configuration file not found at: {sys_config_path}")
        return None

def get_valid_parallelism_dims(num_npus):
    dims = [1]
    for i in range(2, num_npus + 1):
        if num_npus % i == 0:
            dims.append(i)
    return sorted(list(set(dims)))

st.header("Configuration")
network_files = get_network_configurations()
if network_files:
    selected_network = st.selectbox("Select Network Configuration", network_files, key='network_select')

    if selected_network:
        net_content, net_config_data = load_network_config(selected_network)
        sys_content = load_system_config(selected_network)

        if net_config_data and sys_content:
            num_npu = net_config_data.get("npus_count", 0)
            num_npus = num_npu[0]
            for i in range(1, len(num_npu)):
                num_npus *= num_npu[i]

            st.write(f"Number of NPUs in selected network: **{num_npus}**")

            st.header("Model Parameters")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                batch = st.selectbox("Batch Size", [256, 512, 1024, 2048], index=1, key='batch_select')
            with col2:
                dmodel = st.selectbox("Model Dimension (dmodel)", [512, 1024, 2048, 4096], index=1, key='dmodel_select')
                dff = st.selectbox("Feed-Forward Dimension (dff)", [1024, 2048, 4096, 8192], index=2, key='dff_select')
            with col3:
                head = st.selectbox("Number of Heads", [4, 8, 16, 32, 64], index=1, key='head_select')
                seq = st.selectbox("Sequence Length", [128, 256, 512, 1024, 2048], index=2, key='seq_select')
            with col4:
                num_stacks = st.selectbox("Number of Stacks", [1, 2], index=1, key='num_stacks_select')

            st.header("Parallelism Search Space")
            dims = get_valid_parallelism_dims(num_npus)
            # Allow user to select multiple options for each parallelism strategy
            dp_options = st.multiselect("Data Parallel (DP) options", dims, default=[1])
            pp_options = st.multiselect("Pipeline Parallel (PP) options", dims, default=[1])
            tp_options = st.multiselect("Tensor Parallel (TP) options", dims, default=[1])
            sp_options = st.multiselect("Sequence Parallel (SP) options", dims, default=[1])

            st.header("Search Parameters")
            num_searches = st.slider("Number of Searches", min_value=1, max_value=10, value=5, key='num_searches_slider')

            if st.button("Find Best Parallelism Strategy"):
                st.info(
                    f"Running up to {num_searches} simulations to find the best parallelism strategy for {num_npus} NPUs..."
                )

                results = []
                progress_bar = st.progress(0)

                # Use user-selected options for each parallelism dimension
                all_combinations = list(product(dp_options, pp_options, tp_options, sp_options))

                # Only keep combinations that multiply to num_npus
                valid_combinations = [
                    combo for combo in all_combinations if np.prod(combo) == num_npus
                ]

                if not valid_combinations:
                    st.error("No valid parallelism combinations found for the given number of NPUs and selected options.")
                else:
                    search_indices = np.random.choice(len(valid_combinations), min(num_searches, len(valid_combinations)), replace=False)
                    
                    num_actual_searches = len(search_indices)
    
                    # Create a unique temp directory: temp/{timestamp}
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    TEMP_DIR = os.path.join(os.path.dirname(__file__), "..", "temp", timestamp)
                    os.makedirs(TEMP_DIR, exist_ok=True)

                    st.set_page_config(layout="wide")

                    st.title("Parallelism Strategy Solver")

                    for i, combo_idx in enumerate(search_indices):
                        dp, pp, tp, sp = valid_combinations[combo_idx]

                        # ...existing code...
                        params = {
                            "model_name": "custom",
                            "num_npus": num_npus,
                            "din": dmodel,
                            "dout": dmodel,
                            "dmodel": dmodel,
                            "dff": dff,
                            "batch": batch,
                            "seq": seq,
                            "head": head,
                            "num_stacks": num_stacks,
                            "dp": dp,
                            "pp": pp,
                            "tp": tp,
                            "sp": sp,
                            "sharding": 0,
                            "temp_dir": TEMP_DIR,
                        }

                        workload_solver.generate_workload_for_solver(params)

                        sim_details = astra_solver.run_simulation_for_solver(params, sys_content, net_content)

                        if sim_details:
                            comm_cycles = sim_details.get("comm_cycles")
                            comp_cycles = sim_details.get("comp_cycles")
                            total_cycles = 0
                            for v in [comm_cycles, comp_cycles]:
                                if v is not None:
                                    total_cycles += v
                            results.append({
                                "Data Parallel": dp,
                                "Pipeline Parallel": pp,
                                "Tensor Parallel": tp,
                                "Sequence Parallel": sp,
                                "Comm Cycles": comm_cycles,
                                "Comp Cycles": comp_cycles,
                                "Total Cycles": total_cycles,
                            })
                        
                        progress_bar.progress((i + 1) / num_actual_searches)

                    st.success("Search complete!")

                    if not results:
                        st.error("No successful simulations were run. Please check your parameters.")
                    else:
                        df_results = pd.DataFrame(results)
                        st.dataframe(df_results)

                        # Filter out results with zero or None total cycles
                        df_results = df_results[(df_results["Total Cycles"] > 0) & (df_results["Total Cycles"].notnull())]
                        if df_results.empty:
                            st.warning("No valid results with non-zero cycles to display.")
                        else:
                            df_results = df_results.sort_values(by="Total Cycles").reset_index(drop=True)
                            
                            st.subheader("Best Strategy Found")
                            best_result = df_results.iloc[0]
                            st.json(best_result.to_dict())

                            st.subheader("Top Strategies Comparison")
                            num_to_compare = min(5, len(df_results))
                            top_results = df_results.head(num_to_compare)
                            
                            top_results['Strategy'] = top_results.apply(
                                lambda row: f"DP={row['Data Parallel']}, PP={row['Pipeline Parallel']}, TP={row['Tensor Parallel']}, SP={row['Sequence Parallel']}", axis=1
                            )

                            # Prepare data for stacked bar: one bar per strategy, with comp and comm cycles stacked
                            stacked_df = top_results.melt(
                                id_vars=['Strategy'],
                                value_vars=['Comp Cycles', 'Comm Cycles'],
                                var_name='Cycle Type',
                                value_name='Cycles'
                            )

                            fig = px.bar(
                                stacked_df,
                                x='Strategy',
                                y='Cycles',
                                color='Cycle Type',
                                barmode='stack',
                                title='Top Strategies Cycle Breakdown (Stacked)',
                                labels={'Cycles': 'Cycles', 'Strategy': 'Strategy'},
                            )
                            st.plotly_chart(fig, use_container_width=True)

                            st.balloons()
else:
    st.warning("No network configurations found. Please make sure they are in the correct directory.")