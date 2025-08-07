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
import concurrent.futures
import sections.trace_viewer as tv
import sections.trace_picker as picker

st.set_page_config(layout="wide")
st.markdown("""
    <style>
        .reportview-container {
            margin-top: -2em;
        }
        #MainMenu {visibility: hidden;}
        .stAppDeployButton {display:none;}
    </style>
""", unsafe_allow_html=True)
st.title("Parallelism Strategy Solver")
with st.expander("ℹ️ How does the Parallelism Strategy Solver work?", expanded=False):
    st.markdown("""
    This page performs a random search to find the best parallelism strategy for a given model and hardware configuration.

    **How it works:**
    1. Input the hardware configuration and model parameters.
    2. For each parallelism strategy (Data, Pipeline, Tensor, Sequence), you can select multiple possible options.
    3. The solver will randomly sample valid combinations (where the product of the selected parallelism factors equals the number of NPUs).
    4. For each sampled combination, the tool generates a workload, runs a simulation, and collects the communication and computation cycles.
    5. The best strategies are shown in a table and a stacked bar plot.
    6. Select a strategy to visualize the trace data.

    **Limitations:**
    - This is a demo and uses random search, not an exhaustive or optimal search.
    - Only a limited number of simulations are run (as set by the slider).
    - Some combinations may not be feasible for your hardware or model.
    - Only communication and computation cycles are shown; other metrics are not included.
    - Results with zero total cycles are ignored in the plots.

    In the future, this will be replaced with a more sophisticated search algorithm.
    """)

def initialize_session_state():
    """Initialize all required session state variables"""
    if 'peak_perf' not in st.session_state:
        st.session_state.peak_perf = 989
    if 'peak_bw' not in st.session_state:
        st.session_state.peak_bw = 3350
    if 'show_npu_plots' not in st.session_state:
        st.session_state.show_npu_plots = True

def get_network_configurations():
    """Get list of available network configuration files"""
    network_config_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'configuration')
    try:
        network_files = [f.split('.')[0] for f in os.listdir(network_config_dir) if f.endswith('.yml')]
        return network_files
    except FileNotFoundError:
        st.error(f"Network configuration directory not found at: {network_config_dir}")
        return []

def load_network_config(network_file):
    """Load network configuration from YAML file"""
    network_config_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'configuration')
    try:
        with open(os.path.join(network_config_dir, network_file)) as f:
            raw_content = f.read()
            return raw_content, yaml.safe_load(raw_content)
    except (FileNotFoundError, yaml.YAMLError):
        return None, None

def load_system_config(network_file):
    """Load system configuration from JSON file"""
    sys_file_name = os.path.splitext(network_file)[0] + "_sys.json"
    sys_config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'configuration', sys_file_name)
    try:
        with open(sys_config_path) as f:
            return f.read()
    except FileNotFoundError:
        st.error(f"System configuration file not found at: {sys_config_path}")
        return None

def compute_network_config(total_npus, npus_per_node, is_3d):
    """Validate and compute network configuration dimensions"""
    if npus_per_node > total_npus:
        return None, "NPUs per node cannot exceed total NPU count"
    
    if total_npus % npus_per_node != 0:
        return None, f"Total NPUs ({total_npus}) must be divisible by NPUs per node ({npus_per_node})"
    
    nodes = total_npus // npus_per_node
    
    if is_3d:
        # For 3D: split remaining NPUs by 2, ensuring all dimensions > 1
        remaining_factor = nodes
        if remaining_factor == 1:
            return None, "For 3D configuration, need at least 2 nodes (all dimensions must be > 1)"
        elif remaining_factor == 2:
            return None, "For 3D configuration with 2 nodes, cannot split into 3 dimensions with all > 1"
        elif remaining_factor == 4:
            return [npus_per_node, 2, 2], None
        elif remaining_factor == 8:
            return [npus_per_node, 2, 4], None
        else:
            # Try to split as evenly as possible, ensuring all dims > 1
            import math
            dim2 = int(math.sqrt(remaining_factor))
            while remaining_factor % dim2 != 0 and dim2 > 1:
                dim2 -= 1
            dim3 = remaining_factor // dim2
            
            # Check if all dimensions are > 1
            if dim2 <= 1 or dim3 <= 1:
                return None, f"For 3D configuration, cannot split {remaining_factor} nodes into 3 dimensions with all > 1"
            
            return [npus_per_node, dim2, dim3], None
    else:
        # For 2D
        if nodes < 1:
            return None, "Number of nodes must be at least 1"
        return [npus_per_node, nodes], None

def get_valid_parallelism_dims(num_npus):
    """Get valid parallelism dimensions for given number of NPUs"""
    dims = [1]
    for i in range(2, num_npus + 1):
        if num_npus % i == 0:
            dims.append(i)
    return sorted(list(set(dims)))

def run_simulation_wrapper(args):
    """Wrapper function for running individual simulations"""
    params, sys_content, net_content = args
    dp, pp, tp, sp = params["dp"], params["pp"], params["tp"], params["sp"]
    
    sim_details = astra_solver.run_simulation_for_solver(params, sys_content, net_content)
    if sim_details:
        comm_cycles = sim_details.get("comm_cycles")
        comp_cycles = sim_details.get("comp_cycles")
        total_cycles = 0
        for v in [comm_cycles, comp_cycles]:
            if v is not None:
                total_cycles += v
        return {
            "Data Parallel": dp,
            "Pipeline Parallel": pp,
            "Tensor Parallel": tp,
            "Sequence Parallel": sp,
            "Comm Cycles": comm_cycles,
            "Comp Cycles": comp_cycles,
            "Total Cycles": total_cycles,
        }
    return None

def generate_search_space(valid_combinations, num_searches, search_params):
    """Generate search space parameters for simulations"""
    if not valid_combinations:
        return []
    
    search_indices = np.random.choice(
        len(valid_combinations), 
        min(num_searches, len(valid_combinations)), 
        replace=False
    )
    
    search_space = []
    for combo_idx in search_indices:
        dp, pp, tp, sp = valid_combinations[combo_idx]
        params = {
            "model_name": "custom",
            "num_npus": search_params["num_npus"],
            "din": search_params["dmodel"],
            "dout": search_params["dmodel"],
            "dmodel": search_params["dmodel"],
            "dff": search_params["dff"],
            "batch": search_params["batch"],
            "seq": search_params["seq"],
            "head": search_params["head"],
            "num_stacks": search_params["num_stacks"],
            "dp": dp, "pp": pp, "tp": tp, "sp": sp,
            "sharding": 0,
            "temp_dir": search_params["temp_dir"],
        }
        search_space.append(params)
    return search_space

def execute_parallel_simulations(search_space, sys_content, net_content):
    """Execute simulations in parallel and return results"""
    # Generate workloads first
    with concurrent.futures.ThreadPoolExecutor() as executor:
        list(executor.map(workload_solver.generate_workload_for_solver, search_space))

    # Run simulations
    simulation_inputs = [(params, sys_content, net_content) for params in search_space]
    results = []
    
    progress_bar = st.progress(0)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(run_simulation_wrapper, args) for args in simulation_inputs]
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            result = future.result()
            if result:
                results.append(result)
            progress_bar.progress((i + 1) / len(search_space))
    
    return results

def create_results_dataframe(results):
    """Create and store results dataframe with session caching"""
    if not results:
        return None
    
    df_results = pd.DataFrame(results)
    return df_results

def display_results(df_results, timestamp=None):
    """Display the results in a formatted way"""
    if timestamp:
        st.info(f"📊 Results from: {timestamp}")
    
    st.dataframe(df_results)

    # Filter out results with zero or None total cycles
    df_results_filtered = df_results[(df_results["Total Cycles"] > 0) & (df_results["Total Cycles"].notnull())]
    if df_results_filtered.empty:
        st.warning("No valid results with non-zero cycles to display.")
    else:
        df_results_filtered = df_results_filtered.sort_values(by="Total Cycles").reset_index(drop=True)

        st.subheader("Best Strategy Found", help="The details of the strategy with the lowest total cycles.")
        best_result = df_results_filtered.iloc[0]
        st.json(best_result.to_dict())

        st.subheader("Top Strategies Comparison", help="Showing the top strategies based on total cycles, with a breakdown of computation and communication cycles.")
        num_to_compare = min(5, len(df_results_filtered))
        top_results = df_results_filtered.head(num_to_compare)
        
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

def find_trace_files(df_results, temp_base_dir, seq, batch):
    """Find available trace files for given results"""
    available_temp_dirs = []
    
    for idx, row in df_results.iterrows():
        dp = int(row['Data Parallel'])
        tp = int(row['Tensor Parallel'])
        sp = int(row['Sequence Parallel'])
        pp = int(row['Pipeline Parallel'])
        
        experiment_name = f"{dp}_{tp}_{sp}_{pp}_0.seq_{seq}.batch_{batch}"
        trace_filename = experiment_name + "_trace_matched_timing.csv"
        
        # Look for this trace file in temp directories
        for temp_dir_name in sorted(os.listdir(temp_base_dir), reverse=True):
            temp_dir_path = os.path.join(temp_base_dir, temp_dir_name)
            if os.path.isdir(temp_dir_path):
                trace_file_path = os.path.join(temp_dir_path, trace_filename)
                if os.path.exists(trace_file_path):
                    if temp_dir_path not in available_temp_dirs:
                        available_temp_dirs.append(temp_dir_path)
                    break
    
    return available_temp_dirs

def create_strategy_dropdown(df_results):
    """Create dropdown options for strategy selection"""
    strategy_options = []
    strategy_mapping = {}
    
    for idx, row in df_results.iterrows():
        dp = int(row['Data Parallel'])
        tp = int(row['Tensor Parallel'])
        sp = int(row['Sequence Parallel'])
        pp = int(row['Pipeline Parallel'])
        total_cycles = row['Total Cycles']
        
        strategy_display = f"DP={dp}, TP={tp}, SP={sp}, PP={pp}"
        strategy_options.append(strategy_display)
        strategy_mapping[strategy_display] = {
            'dp': dp, 'tp': tp, 'sp': sp, 'pp': pp,
            'total_cycles': total_cycles
        }
    
    # Sort by total cycles (best first)
    strategy_options.sort(key=lambda x: strategy_mapping[x]['total_cycles'])
    return strategy_options, strategy_mapping

def load_trace_and_render_visualization(df_results, temp_dir, seq, batch, peak_perf, local_mem_bw, selected_strategy=None):
    """Load trace data and render visualization with strategy selection"""
    # Initialize session state variables first
    if 'peak_perf' not in st.session_state:
        st.session_state.peak_perf = peak_perf
    if 'peak_bw' not in st.session_state:
        st.session_state.peak_bw = local_mem_bw
    if 'show_npu_plots' not in st.session_state:
        st.session_state.show_npu_plots = True
    
    # Set session state for peak_perf and peak_bw
    st.session_state.peak_perf = peak_perf
    st.session_state.peak_bw = local_mem_bw
    
    # Create strategy dropdown for user selection
    st.subheader("Select Strategy for Trace Visualization", help="Choose a strategy to visualize its trace data.")
    
    strategy_options, strategy_mapping = create_strategy_dropdown(df_results)
    
    # Use selected strategy or default to best (first in sorted list)
    default_index = 0
    if selected_strategy and selected_strategy in strategy_options:
        default_index = strategy_options.index(selected_strategy)
    
    selected_display = st.selectbox(
        "Choose parallelism strategy to visualize:",
        options=strategy_options,
        index=default_index,
        key=f"strategy_selector_{hash(str(temp_dir))}"
    )
    
    # Get the selected strategy parameters
    selected_params = strategy_mapping[selected_display]
    dp, tp, sp, pp = selected_params['dp'], selected_params['tp'], selected_params['sp'], selected_params['pp']
    
    # Create sim_outputs dictionary with paths to existing simulation data
    experiment_name = f"{dp}_{tp}_{sp}_{pp}_0.seq_{seq}.batch_{batch}"
    trace_file = os.path.join(temp_dir, experiment_name + "_trace_matched_timing.csv")
    
    sim_outputs = {
        "sim_dir": temp_dir,
        "log": os.path.join(temp_dir, experiment_name + ".log"),
        "res_log": os.path.join(temp_dir, experiment_name + "_res.csv"),
    }
    
    # Load trace data into session state if file exists
    if os.path.exists(trace_file):
        try:
            st.session_state.df_matched = pd.read_csv(trace_file)
            st.success(f"✅ Loading trace visualization for: {selected_display}")
            tv.render_sim_ouput_section(sim_outputs)
        except Exception as e:
            st.error(f"Error loading trace data: {str(e)}")
            st.warning("Trace visualization is not available for this result.")
    else:
        st.warning(f"Trace file not found: {trace_file}")
        st.info("Trace visualization is not available for this result.")

def render_hardware_configuration(network_files):
    """Render hardware configuration section"""
    st.header("Hardware Configuration", help="Select the network topology and NPU count for your simulation, then configure system parameters.")
    
    if not network_files:
        st.warning("No network configurations found. Please make sure they are in the correct directory.")
        return None, None, None
    
    # Network topology selection
    col1, col2 = st.columns(2)
    with col1:
        selected_network = st.selectbox("Select Network Topology", network_files, key='network_select')
    with col2:
        total_npu_count = st.selectbox(
            "Total NPU Count", 
            [8, 16, 32], 
            index=0,
            key='total_npu_count'
        )
    
    selected_network = selected_network + '.yml'
    net_content, net_config_data = load_network_config(selected_network)
    sys_content = load_system_config(selected_network)
    
    if not net_config_data or not sys_content:
        return None, None, None
    
    # Parse configurations
    try:
        sys_config_data = json.loads(sys_content)
    except json.JSONDecodeError:
        st.error("Invalid system configuration JSON format")
        return None, None, None
    
    # Calculate NPU count
    num_npu = net_config_data.get("npus_count", 0)
    num_npus = num_npu[0]
    for i in range(1, len(num_npu)):
        num_npus *= num_npu[i]
    
    return selected_network, net_config_data, sys_config_data, num_npus, total_npu_count

def render_system_parameters(net_config_data, sys_config_data, num_npus, total_npu_count):
    """Render system parameter configuration"""
    npus_count_array = net_config_data.get("npus_count", [])
    bandwidth_array = net_config_data.get("bandwidth", [])
    is_3d = len(npus_count_array) == 3
    
    # Calculate NPUs per node options
    npus_per_node_options = [
        i for i in range(2, total_npu_count // (len(npus_count_array) if len(npus_count_array) > 0 else 1) + 1) 
        if total_npu_count % i == 0
    ]
    npus_per_node_default = (npus_count_array[0] if npus_count_array and npus_count_array[0] in npus_per_node_options 
                            else npus_per_node_options[0])
    
    # System parameters in single row
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        peak_perf = st.number_input(
            "Peak Performance (TFLOPS)", 
            min_value=1, max_value=10000, 
            value=int(sys_config_data.get("peak-perf", 989)),
            key='peak_perf_input'
        )
    with col2:
        local_mem_bw = st.number_input(
            "Local Memory BW (GB/s)", 
            min_value=1, max_value=50000, 
            value=int(sys_config_data.get("local-mem-bw", 3350)),
            key='local_mem_bw_input'
        )
    with col3:
        npus_per_node = st.selectbox(
            "NPUs per Node", npus_per_node_options,
            index=npus_per_node_options.index(npus_per_node_default),
            key='npus_per_node'
        )
    with col4:
        intra_node_bw = st.number_input(
            "Intra-Node BW (GB/s)", 
            min_value=1, max_value=10000, 
            value=int(bandwidth_array[0] if bandwidth_array else 900),
            key='intra_node_bw'
        )
    with col5:
        inter_node_bw = st.number_input(
            "Inter-Node BW (GB/s)", 
            min_value=1, max_value=10000, 
            value=int(bandwidth_array[1] if len(bandwidth_array) > 1 else 200),
            key='inter_node_bw'
        )
    
    return {
        'total_npu_count': total_npu_count,
        'peak_perf': peak_perf,
        'local_mem_bw': local_mem_bw,
        'npus_per_node': npus_per_node,
        'intra_node_bw': intra_node_bw,
        'inter_node_bw': inter_node_bw,
        'is_3d': is_3d
    }

def render_model_parameters():
    """Render model parameter configuration"""
    st.header("Model Parameters", help="Configure the model parameters for the solver. These will be used to generate the workloads and run simulations.")
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        batch = st.selectbox("Batch Size", [256, 512, 1024, 2048], index=1, key='batch_select')
    with col2:
        dmodel = st.selectbox("Model Dimension (dmodel)", [512, 1024, 2048, 4096], index=1, key='dmodel_select')
    with col3:
        dff = st.selectbox("Feed-Forward Dimension (dff)", [1024, 2048, 4096, 8192], index=2, key='dff_select')
    with col4:
        head = st.selectbox("Number of Heads", [4, 8, 16, 32], index=1, key='head_select')
    with col5:
        seq = st.selectbox("Sequence Length", [128, 256, 512, 1024, 2048], index=2, key='seq_select')
    with col6:
        num_stacks = st.selectbox("Number of Stacks", [1, 2], index=1, key='num_stacks_select', 
                                 help="The number of layers: limited to 2 for the Demo purposes.", disabled=True)
    
    return {'batch': batch, 'dmodel': dmodel, 'dff': dff, 'head': head, 'seq': seq, 'num_stacks': num_stacks}

def render_parallelism_search_space(num_npus):
    """Render parallelism search space configuration"""
    st.header("Parallelism Search Space", help="Select the possible parallelism strategies degrees. The solver will search for the best combination based on the selected options.")
    dims = get_valid_parallelism_dims(num_npus)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        dp_options = st.multiselect("Data Parallel (DP) options", dims, default=dims)
    with col2:
        sp_options = st.multiselect("Sequence Parallel (SP) options", dims, default=dims)
    with col3:
        tp_options = st.multiselect("Tensor Parallel (TP) options", dims, default=dims)
    with col4:
        pp_options = st.multiselect("Pipeline Parallel (PP) options", [1, 2], default=[1, 2])
    
    return {'dp_options': dp_options, 'sp_options': sp_options, 'tp_options': tp_options, 'pp_options': pp_options}

def render_search_parameters():
    """Render search parameters section"""
    st.header("Search Parameters", help="Set the number of searches to perform. The solver will randomly sample valid combinations of the selected parallelism strategies.")
    num_searches = st.slider("Number of Searches", min_value=1, max_value=10, value=5, key='num_searches_slider')
    return num_searches

def handle_cached_results(results_key, temp_base_dir, model_params, system_params):
    """Handle display and trace visualization for cached results"""
    if results_key not in st.session_state:
        return
    
    st.markdown("---")
    st.header("Previous Results", help="Cached results from the previous search.")
    cached_data = st.session_state[results_key]
    display_results(cached_data['df_results'], cached_data['timestamp'])
    
    # Try to find trace data for cached results
    if os.path.exists(temp_base_dir):
        available_temp_dirs = find_trace_files(
            cached_data['df_results'], temp_base_dir, 
            model_params['seq'], model_params['batch']
        )
        
        if available_temp_dirs:
            temp_dir_to_use = available_temp_dirs[0]  # Use most recent
            load_trace_and_render_visualization(
                cached_data['df_results'], temp_dir_to_use, 
                model_params['seq'], model_params['batch'],
                system_params['peak_perf'], system_params['local_mem_bw']
            )
        else:
            st.info("ℹ️ Trace visualization data is not available for cached results.")
    else:
        st.info("ℹ️ Trace visualization data is not available for cached results.")
    
    # Clear cached results button
    if st.button("🗑️ Clear Cached Results"):
        del st.session_state[results_key]
        st.rerun()

def execute_search(num_searches, num_npus, search_space_params, model_params, system_params, sys_content, net_content, results_key):
    """Execute the parallelism strategy search"""
    st.info(f"Running up to {num_searches} simulations to find the best parallelism strategy for {num_npus} NPUs...")
    
    # Generate valid combinations
    all_combinations = list(product(
        search_space_params['dp_options'], 
        search_space_params['pp_options'], 
        search_space_params['tp_options'], 
        search_space_params['sp_options']
    ))
    
    valid_combinations = [combo for combo in all_combinations if np.prod(combo) == num_npus]
    
    if not valid_combinations:
        st.error("No valid parallelism combinations found for the given number of NPUs and selected options.")
        return
    
    # Create temp directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_dir = os.path.join(os.path.dirname(__file__), "..", "temp", timestamp)
    os.makedirs(temp_dir, exist_ok=True)
    
    # Generate search space
    search_params = {**model_params, **system_params, "num_npus": num_npus, "temp_dir": temp_dir}
    search_space = generate_search_space(valid_combinations, num_searches, search_params)
    
    # Execute simulations
    results = execute_parallel_simulations(search_space, sys_content, net_content)
    
    st.success("Search complete!")
    
    if not results:
        st.error("No successful simulations were run. Please check your parameters.")
        return
    
    # Create and store results
    df_results = create_results_dataframe(results)
    
    # Store in session state
    st.session_state[results_key] = {
        'df_results': df_results,
        'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Set session state for NPU plots
    st.session_state.show_npu_plots = True
    
    # Display results and trace visualization
    display_results(df_results)
    load_trace_and_render_visualization(
        df_results, temp_dir, 
        model_params['seq'], model_params['batch'],
        system_params['peak_perf'], system_params['local_mem_bw']
    )

def main():
    # Initialize session state
    initialize_session_state()

    # Get network configurations
    network_files = get_network_configurations()
    if not network_files:
        st.warning("No network configurations found. Please make sure they are in the correct directory.")
        return

    # Render hardware configuration
    config_result = render_hardware_configuration(network_files)
    if not config_result or len(config_result) != 5:
        return

    selected_network, net_config_data, sys_config_data, num_npus, total_npu_count = config_result

    # Render system parameters
    system_params = render_system_parameters(net_config_data, sys_config_data, num_npus, total_npu_count)

    # Validate network configuration
    new_npus_count, error_msg = compute_network_config(
        system_params['total_npu_count'], 
        system_params['npus_per_node'], 
        system_params['is_3d']
    )

    if error_msg:
        st.error(error_msg)
        return

    # Update configurations
    updated_sys_config = sys_config_data.copy()
    updated_sys_config.update({
        "peak-perf": system_params['peak_perf'],
        "local-mem-bw": system_params['local_mem_bw']
    })

    updated_net_config = net_config_data.copy()
    updated_net_config["npus_count"] = new_npus_count

    if system_params['is_3d']:
        updated_net_config["bandwidth"] = [
            system_params['intra_node_bw'], 
            system_params['inter_node_bw'], 
            system_params['inter_node_bw']
        ]
    else:
        updated_net_config["bandwidth"] = [
            system_params['intra_node_bw'], 
            system_params['inter_node_bw']
        ]

    # Update content strings
    sys_content = json.dumps(updated_sys_config, indent=4)
    net_content = yaml.dump(updated_net_config, default_flow_style=False)

    # Display configuration success
    st.success("Configuration updated successfully!")
    with st.expander("View Updated Configuration", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("System Config")
            st.json(updated_sys_config)
        with col2:
            st.subheader("Network Config")
            st.code(net_content, language='yaml')

    # Update num_npus for the rest of the application
    num_npus = system_params['total_npu_count']

    # Render model and search parameters
    model_params = render_model_parameters()
    search_space_params = render_parallelism_search_space(num_npus)
    num_searches = render_search_parameters()

    # Create session key for caching
    session_key = (f"{selected_network}_{system_params['total_npu_count']}_"
                    f"{system_params['peak_perf']}_{system_params['local_mem_bw']}_"
                    f"{system_params['npus_per_node']}_{system_params['intra_node_bw']}_"
                    f"{system_params['inter_node_bw']}_{model_params['batch']}_"
                    f"{model_params['dmodel']}_{model_params['dff']}_{model_params['head']}_"
                    f"{model_params['seq']}_{model_params['num_stacks']}_"
                    f"{search_space_params['dp_options']}_{search_space_params['sp_options']}_"
                    f"{search_space_params['tp_options']}_{search_space_params['pp_options']}_"
                    f"{num_searches}")

    results_key = f"solver_results_{hash(session_key)}"
    has_cached_results = results_key in st.session_state

    # Show cached results info
    if has_cached_results:
        st.info("Results found for current configuration! Displaying cached results below.")
        st.write("💡 *Tip: Change any parameter above to run a new search.*")

    # Main search execution
    if st.button("Find Best Parallelism Strategy"):
        execute_search(
            num_searches, num_npus, search_space_params, model_params, 
            system_params, sys_content, net_content, results_key
        )

    # Handle cached results display
    if has_cached_results:
        temp_base_dir = os.path.join(os.path.dirname(__file__), "..", "temp")
        handle_cached_results(results_key, temp_base_dir, model_params, system_params)

main()