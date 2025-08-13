import streamlit as st
import subprocess
import time
import scripts.generate_single_workload as gen
import os
import uuid
import json
import yaml
from pathlib import Path

def generate_workload_and_run_simulation():
    #st.header(
    #    "Model Configuration and Simulation",
    #    help=(
    #        "Configure your model parameters, hardware configuration, and parallelism strategy, "
    #        "then generate workload and run simulation in one unified process."
    #    )
    #)

    # Initialize session state for persistence
    if "temp_dir" not in st.session_state:
        # Create a temporary directory for the session
        st.session_state.temp_dir = f"temp/{uuid.uuid4()}/"
        os.makedirs(st.session_state.temp_dir, exist_ok=True)
    
    if "simulation_running" not in st.session_state:
        st.session_state.simulation_running = False
    
    # Initialize session state for all input configurations at the top level
    if 'hw_config' not in st.session_state:
        st.session_state.hw_config = {}
    if 'model_config' not in st.session_state:
        st.session_state.model_config = {}
    if 'parallelism_config' not in st.session_state:
        st.session_state.parallelism_config = {}
    
    temp_dir = st.session_state.temp_dir

    # --- Hardware Configuration Section (Outside form for dynamic updates) ---
    st.subheader("Hardware Configuration", help="Select the network topology and NPU count for your simulation, then configure system parameters.")
    hardware_params = _render_hardware_configuration()
    
    # --- Model Parameters Section (Outside form for dynamic updates) ---
    st.subheader("Model Parameters", help="Configure the model parameters for the solver. These will be used to generate the workloads and run simulations.")
    model_params = _render_model_parameters(hardware_params['total_npu_count'])
    
    # --- Parallelism Strategy Section (Outside form for dynamic updates) ---
    st.subheader("Parallelism Strategy")
    parallelism_params = _render_parallelism_strategy(hardware_params['total_npu_count'])
    
    # --- Configuration Summary ---
    st.markdown("---")
    with st.expander("Configuration Summary", expanded=False):
        col_summary1, col_summary2 = st.columns(2)
        
        with col_summary1:
            st.write("**Hardware Configuration:**")
            st.write(f"• Total NPUs: {hardware_params['total_npu_count']}")
            st.write(f"• NPUs per Node: {hardware_params['npus_per_node']}")
            st.write(f"• Network Topology: {hardware_params['selected_network']}")
            st.write(f"• Peak Performance: {hardware_params['peak_perf']} TFLOPS")
        
        with col_summary2:
            st.write("**Model & Parallelism:**")
            st.write(f"• Model Dimension: {model_params['dmodel']}")
            st.write(f"• Number of Heads: {model_params['head']}")
            st.write(f"• Batch Size: {model_params['batch']}")
            st.write(f"• Parallelism: DP={parallelism_params['dp']}, TP={parallelism_params['tp']}, SP={parallelism_params['sp']}, PP={parallelism_params['pp']}")
    
    # --- Submit Button Section ---
    st.markdown("---")
    if parallelism_params.get('is_valid', False):
        submitted = st.button("Generate Workload and Run Simulation", type="primary")
    else:
        st.warning("Fix the parallelism configuration above before proceeding.")
        submitted = st.button("Generate Workload and Run Simulation", type="primary", disabled=True)

    # --- Submission Section ---
    if submitted and not st.session_state.simulation_running:
        # Check if parallelism configuration is valid before proceeding
        if not parallelism_params.get('is_valid', False):
            st.error("Cannot proceed: Invalid parallelism configuration!")
            st.warning("Please adjust the parallelism factors so their product equals the total NPU count.")
            return _get_default_params(temp_dir)
        
        # Set simulation running flag to prevent double execution
        st.session_state.simulation_running = True
        
        # Clear only temporary files, preserve user input configurations
        _clear_temp_dir(temp_dir)
        
        # Generate network and system configurations
        sys_content, net_content = _create_updated_configurations(hardware_params)
        
        if sys_content is None or net_content is None:
            st.error("Failed to generate configuration files!")
            st.session_state.simulation_running = False
            return _get_default_params(temp_dir)
        
        # Run trace generation
        _run_trace_generation(
            model_params, parallelism_params, temp_dir
        )
        
        # Store configurations for simulation
        simulation_params = {
            "dp": parallelism_params['dp'],
            "tp": parallelism_params['tp'],
            "sp": parallelism_params['sp'],
            "pp": parallelism_params['pp'],
            "sharding": parallelism_params['sharding'],
            "sharding_val": "1" if parallelism_params['sharding'] else "0",
            "temp_dir": temp_dir,
            "selected_model_name": "Custom Model",
            "sys_content": sys_content,
            "net_content": net_content,
        }
        
        # Store configuration hash for comparison
        import hashlib
        import json
        config_str = json.dumps(simulation_params, sort_keys=True, default=str)
        st.session_state['last_config_hash'] = hashlib.md5(config_str.encode()).hexdigest()
        
        st.session_state['sys_content'] = sys_content
        st.session_state['net_content'] = net_content
        st.session_state['hardware_params'] = hardware_params
        st.session_state['model_params'] = model_params
        st.session_state['parallelism_params'] = parallelism_params
        st.session_state['submitted'] = True
        
        # Reset simulation running flag
        st.session_state.simulation_running = False
    
    # --- Return parameters for downstream use ---
    if submitted and parallelism_params.get('is_valid', False):
        return {
            "dp": parallelism_params['dp'],
            "tp": parallelism_params['tp'],
            "sp": parallelism_params['sp'],
            "pp": parallelism_params['pp'],
            "sharding": parallelism_params['sharding'],
            "sharding_val": "1" if parallelism_params['sharding'] else "0",
            "temp_dir": temp_dir,
            "selected_model_name": "Custom Model",
            "sys_content": st.session_state.get('sys_content'),
            "net_content": st.session_state.get('net_content'),
        }
    else:
        return _get_default_params(temp_dir)


def _get_default_params(temp_dir):
    """Return default parameters when no simulation is run"""
    return {
        "dp": 1,
        "tp": 1,
        "sp": 1,
        "pp": 1,
        "sharding": False,
        "sharding_val": "0",
        "temp_dir": temp_dir,
        "selected_model_name": "Custom Model",
    }


def _get_network_configurations():
    """Get list of available network configurations"""
    config_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'configuration')
    try:
        files = os.listdir(config_dir)
        return [f[:-4] for f in files if f.endswith('.yml')]  # Remove .yml extension
    except FileNotFoundError:
        return []

def _load_network_config(network_file):
    """Load network configuration from YAML file"""
    config_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'configuration')
    network_path = os.path.join(config_dir, network_file)
    try:
        with open(network_path, 'r') as f:
            content = f.read()
            config_data = yaml.safe_load(content)
            return content, config_data
    except (FileNotFoundError, yaml.YAMLError) as e:
        print(f"Failed to load network config: {e}")
        return None, None

def _load_system_config(network_file):
    """Load system configuration from JSON file"""
    sys_file_name = os.path.splitext(network_file)[0] + "_sys.json"
    config_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'configuration')
    sys_config_path = os.path.join(config_dir, sys_file_name)
    try:
        with open(sys_config_path) as f:
            return f.read()
    except FileNotFoundError:
        st.error(f"System configuration file not found at: {sys_config_path}")
        return None

def _compute_network_config(total_npus, npus_per_node, is_3d):
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

def _get_valid_parallelism_dims(num_npus):
    """Get valid parallelism dimensions for given number of NPUs"""
    dims = [1]
    for i in range(2, num_npus + 1):
        if num_npus % i == 0:
            dims.append(i)
    return sorted(list(set(dims)))

def _render_hardware_configuration():
    """Render hardware configuration section"""
    network_files = _get_network_configurations()
    if not network_files:
        st.warning("No network configurations found. Please make sure they are in the correct directory.")
        return {'total_npu_count': 8}
    
    # Network topology selection
    col1, col2 = st.columns(2)
    with col1:
        config_display_map = {
            "2D_Torus": "2D Torus",
            "3D_Torus": "3D Torus", 
            "Dragonfly": "Dragonfly",
            "FoldedClos": "Folded-Clos"
        }
        display_options = [config_display_map.get(config, config) for config in network_files]
        if not display_options:
            st.error("No network configurations found")
            return {'total_npu_count': 8}
        
        # Get saved value or use default
        saved_network = st.session_state.hw_config.get('network_topology', display_options[0])
        try:
            default_index = display_options.index(saved_network)
        except ValueError:
            default_index = 0
            
        selected_display_name = st.selectbox("Network Topology", display_options, 
                                           index=default_index, key='network_select')
        
        # Save selection to session state
        st.session_state.hw_config['network_topology'] = selected_display_name
        
        reverse_map = {v: k for k, v in config_display_map.items()}
        selected_network = reverse_map.get(selected_display_name, selected_display_name)
    
    # Load network configuration to get actual dimensions
    if selected_network:
        selected_network_file = selected_network + '.yml'
        net_content, net_config_data = _load_network_config(selected_network_file)
        sys_content = _load_system_config(selected_network_file)
    else:
        st.error("Invalid network selection")
        return {'total_npu_count': 8}
    
    if not net_config_data or not sys_content:
        st.error("Failed to load configuration files")
        return {'total_npu_count': 8}
    
    # Parse system configuration
    try:
        sys_config_data = json.loads(sys_content)
    except json.JSONDecodeError:
        st.error("Invalid system configuration JSON format")
        return {'total_npu_count': 8}
    
    # Get network dimensions from loaded configuration
    npus_count_array = net_config_data.get("npus_count", [])
    bandwidth_array = net_config_data.get("bandwidth", [])
    is_3d = len(npus_count_array) == 3
    
    # Calculate original NPU count from configuration
    original_npu_count = 1
    for dim in npus_count_array:
        original_npu_count *= dim
    
    with col2:
        # Provide NPU count options that are compatible with the topology
        if is_3d:
            # For 3D topologies, ensure we can split into 3 dimensions
            npu_options = [8, 16, 32, 64]  # Common 3D-compatible counts
        else:
            # For 2D topologies
            npu_options = [8, 16, 32, 64]
        
        # Include the original count if not in options
        if original_npu_count not in npu_options:
            npu_options.append(original_npu_count)
            npu_options.sort()
        
        # Get saved value or use default
        saved_npu_count = st.session_state.hw_config.get('total_npu_count', original_npu_count)
        try:
            default_index = npu_options.index(saved_npu_count)
        except ValueError:
            default_index = npu_options.index(original_npu_count) if original_npu_count in npu_options else 0
            
        total_npu_count = st.selectbox(
            "Total NPU Count", 
            npu_options, 
            index=default_index,
            key='total_npu_count'
        )
        
        # Save selection to session state
        st.session_state.hw_config['total_npu_count'] = total_npu_count
    
    # System parameters with values from loaded configuration
    col3, col4, col5, col6, col7 = st.columns(5)
    with col3:
        # Get saved value or use default
        saved_peak_perf = st.session_state.hw_config.get('peak_perf', 989)
        peak_perf_options = [300, 500, 989, 1000]
        try:
            default_index = peak_perf_options.index(saved_peak_perf)
        except ValueError:
            default_index = 2  # Default to 989
            
        peak_perf = st.selectbox(
            "Peak Performance (TFLOPS)", 
            peak_perf_options, 
            index=default_index,
            key='peak_perf_input'
        )
        st.session_state.hw_config['peak_perf'] = peak_perf
        
    with col4:
        # Get saved value or use default
        saved_local_mem_bw = st.session_state.hw_config.get('local_mem_bw', 3350)
        local_mem_bw_options = [1000, 2000, 3350, 5000]
        try:
            default_index = local_mem_bw_options.index(saved_local_mem_bw)
        except ValueError:
            default_index = 2  # Default to 3350
            
        local_mem_bw = st.selectbox(
            "Local Memory BW (GB/s)", 
            local_mem_bw_options, 
            index=default_index,
            key='local_mem_bw_input'
        )
        st.session_state.hw_config['local_mem_bw'] = local_mem_bw
    with col5:
        # Calculate NPUs per node options based on topology
        if is_3d:
            # For 3D, we need at least 2 nodes for proper dimension splitting
            max_npus_per_node = total_npu_count // 2
        else:
            # For 2D, more flexible
            max_npus_per_node = total_npu_count // 2
        
        npus_per_node_options = [i for i in [2, 4, 8, 16] if i <= max_npus_per_node and total_npu_count % i == 0]
        if not npus_per_node_options:
            npus_per_node_options = [1]  # Fallback
        
        # Get saved value or use original config default
        default_npus_per_node = npus_count_array[0] if npus_count_array and npus_count_array[0] in npus_per_node_options else npus_per_node_options[0]
        saved_npus_per_node = st.session_state.hw_config.get('npus_per_node', default_npus_per_node)
        
        try:
            default_index = npus_per_node_options.index(saved_npus_per_node)
        except ValueError:
            default_index = npus_per_node_options.index(default_npus_per_node) if default_npus_per_node in npus_per_node_options else 0
        
        npus_per_node = st.selectbox(
            "NPUs per Node", 
            npus_per_node_options,
            index=default_index,
            key='npus_per_node'
        )
        st.session_state.hw_config['npus_per_node'] = npus_per_node
        
    with col6:
        default_intra_bw = int(bandwidth_array[0]) if bandwidth_array else 900
        saved_intra_bw = st.session_state.hw_config.get('intra_node_bw', default_intra_bw)
        intra_bw_options = [400, 900, 1600]
        try:
            default_index = intra_bw_options.index(saved_intra_bw)
        except ValueError:
            default_index = 1 if default_intra_bw == 900 else 0
            
        intra_node_bw = st.selectbox(
            "Intra-Node BW (GB/s)", 
            intra_bw_options, 
            index=default_index,
            key='intra_node_bw'
        )
        st.session_state.hw_config['intra_node_bw'] = intra_node_bw
        
    with col7:
        default_inter_bw = int(bandwidth_array[1]) if len(bandwidth_array) > 1 else 200
        saved_inter_bw = st.session_state.hw_config.get('inter_node_bw', default_inter_bw)
        inter_bw_options = [100, 200, 400, 800]
        try:
            default_index = inter_bw_options.index(saved_inter_bw)
        except ValueError:
            default_index = 1 if default_inter_bw == 200 else 0
            
        inter_node_bw = st.selectbox(
            "Inter-Node BW (GB/s)", 
            inter_bw_options, 
            index=default_index,
            key='inter_node_bw'
        )
        st.session_state.hw_config['inter_node_bw'] = inter_node_bw
    
    # Display topology information
    st.info(f"**Topology Info**: {selected_display_name} ({'3D' if is_3d else '2D'}) - Original config: {npus_count_array} NPUs, Current: {total_npu_count} NPUs")
    
    return {
        'selected_network': selected_network,
        'total_npu_count': total_npu_count,
        'peak_perf': peak_perf,
        'local_mem_bw': local_mem_bw,
        'npus_per_node': npus_per_node,
        'intra_node_bw': intra_node_bw,
        'inter_node_bw': inter_node_bw,
        'is_3d': is_3d,
        'original_npus_count': npus_count_array,
        'original_bandwidth': bandwidth_array,
        'net_config_data': net_config_data,
        'sys_config_data': sys_config_data
    }

def _render_model_parameters(total_npu_count):
    """Render model parameter configuration without predefined models"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Input Embedding Size
        din_options = [32128, 50257, 51200]
        saved_din = st.session_state.model_config.get('din', 32128)
        try:
            din_index = din_options.index(saved_din)
        except ValueError:
            din_index = 0
            
        din = st.selectbox("Input Embedding Size", din_options, index=din_index, key='din_select')
        st.session_state.model_config['din'] = din
        
        # Model Dimension
        dmodel_options = [512, 768, 1024, 2048, 4096]
        saved_dmodel = st.session_state.model_config.get('dmodel', 1024)
        try:
            dmodel_index = dmodel_options.index(saved_dmodel)
        except ValueError:
            dmodel_index = 2
            
        dmodel = st.selectbox("Model Dimension", dmodel_options, index=dmodel_index, key='dmodel_select')
        st.session_state.model_config['dmodel'] = dmodel
        
    with col2:
        # Feed-Forward Dimension
        dff_options = [2048, 3072, 4096, 8192]
        saved_dff = st.session_state.model_config.get('dff', 4096)
        try:
            dff_index = dff_options.index(saved_dff)
        except ValueError:
            dff_index = 2
            
        dff = st.selectbox("Feed-Forward Dimension", dff_options, index=dff_index, key='dff_select')
        st.session_state.model_config['dff'] = dff
        
        # Batch Size
        batch_options = [128, 512, 1024, 2048]
        saved_batch = st.session_state.model_config.get('batch', 512)
        try:
            batch_index = batch_options.index(saved_batch)
        except ValueError:
            batch_index = 1
            
        batch = st.selectbox("Batch Size", batch_options, index=batch_index, key='batch_select')
        st.session_state.model_config['batch'] = batch
        
    with col3:
        # Sequence Length
        seq_options = [256, 512, 1024, 2048]
        saved_seq = st.session_state.model_config.get('seq', 512)
        try:
            seq_index = seq_options.index(saved_seq)
        except ValueError:
            seq_index = 1
            
        seq = st.selectbox("Sequence Length", seq_options, index=seq_index, key='seq_select')
        st.session_state.model_config['seq'] = seq
        
    with col4:
        # Limit number of heads based on NPU count
        head_options = [h for h in [8, 12, 16, 32, 64] if h <= total_npu_count]
        if not head_options:
            head_options = [8]  # Fallback minimum
            
        saved_head = st.session_state.model_config.get('head', head_options[0])
        try:
            head_index = head_options.index(saved_head)
        except ValueError:
            head_index = 0
            
        head = st.selectbox("Number of Heads", head_options, index=head_index, key='head_select')
        st.session_state.model_config['head'] = head
        
        # Number of Stacks (always 2)
        num_stacks = st.selectbox("Number of Stacks", [2], index=0, key='num_stacks_select', 
                                 disabled=True, help="Limited to 2 for this demo")
    
    # Set output embedding size to be the same as model dimension
    dout = dmodel
    
    return {
        'din': din, 'dout': dout, 'dmodel': dmodel, 'dff': dff,
        'batch': batch, 'seq': seq, 'head': head, 'num_stacks': num_stacks
    }

def _render_parallelism_strategy(total_npu_count):
    """Render parallelism strategy configuration with limited options"""
    # Get valid dimensions limited by NPU count
    dims = _get_valid_parallelism_dims(total_npu_count)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        saved_dp = st.session_state.parallelism_config.get('dp', dims[-1])
        try:
            dp_index = dims.index(saved_dp)
        except ValueError:
            dp_index = -1
        dp = st.selectbox("Data Parallelism (DP)", dims, index=dp_index, key='dp_select')
        st.session_state.parallelism_config['dp'] = dp
        
    with col2:
        saved_tp = st.session_state.parallelism_config.get('tp', dims[0])
        try:
            tp_index = dims.index(saved_tp)
        except ValueError:
            tp_index = 0
        tp = st.selectbox("Tensor Parallelism (TP)", dims, index=tp_index, key='tp_select')
        st.session_state.parallelism_config['tp'] = tp
        
    with col3:
        saved_sp = st.session_state.parallelism_config.get('sp', dims[0])
        try:
            sp_index = dims.index(saved_sp)
        except ValueError:
            sp_index = 0
        sp = st.selectbox("Sequence Parallelism (SP)", dims, index=sp_index, key='sp_select')
        st.session_state.parallelism_config['sp'] = sp
        
    with col4:
        # Pipeline parallelism limited to 1 or 2 only
        pp_options = [p for p in [1, 2] if p <= total_npu_count]
        saved_pp = st.session_state.parallelism_config.get('pp', pp_options[0])
        try:
            pp_index = pp_options.index(saved_pp)
        except ValueError:
            pp_index = 0
        pp = st.selectbox("Pipeline Parallelism (PP)", pp_options, index=pp_index, key='pp_select')
        st.session_state.parallelism_config['pp'] = pp
        
    with col5:
        saved_sharding = st.session_state.parallelism_config.get('sharding', False)
        sharding = st.checkbox("Sharding (FSDP)", value=saved_sharding, key='sharding_checkbox')
        st.session_state.parallelism_config['sharding'] = sharding
    
    # Add spacing
    st.markdown("---")
    
    # Validate that the product equals total NPUs
    product = dp * tp * sp * pp
    is_valid = product == total_npu_count
    
    # Display validation status prominently
    st.subheader("Configuration Validation")
    
    if is_valid:
        st.success(f"**Valid Configuration**: DP({dp}) × TP({tp}) × SP({sp}) × PP({pp}) = **{product} NPUs**")
        st.info("**Status**: Ready to proceed with simulation!")
    else:
        st.error(f"**Invalid Configuration**: DP({dp}) × TP({tp}) × SP({sp}) × PP({pp}) = **{product}** ≠ **{total_npu_count} NPUs**")
        st.warning("**Requirement**: The multiplication of parallelism strategies (DP × TP × SP × PP) must equal the total NPU count!")
        
        # Show valid combinations hint
        st.info("**Suggested Valid Combinations**:")
        example_combinations = []
        for dp_val in dims[:3]:  # Show first 3 DP options
            for tp_val in dims[:3]:
                for sp_val in dims[:2]:  # Show first 2 SP options
                    for pp_val in pp_options:
                        if dp_val * tp_val * sp_val * pp_val == total_npu_count:
                            example_combinations.append(f"DP={dp_val}, TP={tp_val}, SP={sp_val}, PP={pp_val}")
                            if len(example_combinations) >= 3:  # Show max 3 examples
                                break
                    if len(example_combinations) >= 3:
                        break
                if len(example_combinations) >= 3:
                    break
            if len(example_combinations) >= 3:
                break
        
        if example_combinations:
            for combo in example_combinations:
                st.write(f"   • {combo}")
        else:
            st.write("   • Try adjusting DP, TP, SP values from the available options")
        
        st.error("**Status**: Cannot proceed until configuration is valid!")
    
    return {'dp': dp, 'tp': tp, 'sp': sp, 'pp': pp, 'sharding': sharding, 'is_valid': is_valid}

def _create_updated_configurations(hardware_params):
    """Create updated system and network configurations"""
    # Check if we already have loaded configuration data
    if 'net_config_data' in hardware_params and 'sys_config_data' in hardware_params:
        net_config_data = hardware_params['net_config_data']
        sys_config_data = hardware_params['sys_config_data']
    else:
        # Load base configurations
        selected_network = hardware_params['selected_network'] + '.yml'
        net_content, net_config_data = _load_network_config(selected_network)
        sys_content = _load_system_config(selected_network)
        
        if not net_config_data or not sys_content:
            st.error("Failed to load configuration files")
            return None, None
        
        # Parse configurations
        try:
            sys_config_data = json.loads(sys_content)
        except json.JSONDecodeError:
            st.error("Invalid system configuration JSON format")
            return None, None
    
    # Update system configuration
    sys_config_data.update({
        "peak-perf": hardware_params['peak_perf'],
        "local-mem-bw": hardware_params['local_mem_bw']
    })
    
    # Update network configuration
    new_npus_count, error_msg = _compute_network_config(
        hardware_params['total_npu_count'], 
        hardware_params['npus_per_node'], 
        hardware_params['is_3d']
    )
    
    if error_msg:
        st.error(error_msg)
        return None, None
    
    net_config_data["npus_count"] = new_npus_count
    
    if hardware_params['is_3d']:
        net_config_data["bandwidth"] = [
            hardware_params['intra_node_bw'], 
            hardware_params['inter_node_bw'], 
            hardware_params['inter_node_bw']
        ]
    else:
        net_config_data["bandwidth"] = [
            hardware_params['intra_node_bw'], 
            hardware_params['inter_node_bw']
        ]
    
    # Convert back to content strings
    updated_sys_content = json.dumps(sys_config_data, indent=4)
    updated_net_content = yaml.dump(net_config_data, default_flow_style=False)
    
    return updated_sys_content, updated_net_content
    updated_net_content = yaml.dump(net_config_data, default_flow_style=False)
    
    return updated_sys_content, updated_net_content

def _get_models_names():   
    return list(gen.model_display_names.values())

def _workload_form(selected_model_name):
    with st.form("model_config_form"):
        st.subheader("Model parameters")
        display_to_model = {v: k for k, v in gen.model_display_names.items()}
        selected_model = display_to_model[selected_model_name]

        params_list = gen.Model.get_model_params(selected_model)
        param_names = [
            "din",
            "dout",
            "dmodel",
            "dff",
            "batch",
            "seq",
            "head",
            "num_stacks",
        ]   

        params_labels = [                 
            "Input Embedding Size",            
            "Output Embedding Size",            
            "Model Feature Size",
            "FFN Feature Size",
            "Batch Size",
            "Sequance Length",
            "Number of Heads",
            "Number of Layers"
        ]
        cols = st.columns(len(param_names))
        for i, col in enumerate(cols):
            with col:
                params_list[i] = st.number_input(
                    label=params_labels[i], value=params_list[i], key=f"{param_names[i]}_input"
                )

        st.subheader(
            "Parallelism strategy",
            help=(
            "Set the parallelism parameters for the model.\n"
            "- **Data Parallelism Degree (DP)**.\n"
            "- **Tensor Parallelism Degree (TP)**.\n"
            "- **Sequence Parallelism Degree (SP)**.\n"
            "- **Pipeline Parallelism Degree (PP)**.\n"
            "- **FSDP**: Enable Fully Sharded Data Parallelism."
            )
        )
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            dp = st.text_input("Data Parallelism (DP)", 1)
        with col2:
            tp = st.text_input("Tensor Parallelism (TP)", 8)
        with col3:
            sp = st.text_input("Sequence Parallelism (SP)", 2)
        with col4:
            pp = st.text_input("Pipeline Parallelism (PP)", 4)
        with col5:
            sharding = st.checkbox("Sharding (FSDP)", value=False)

        submitted = st.form_submit_button("🚀 Run Model")
        params_list[4] = [params_list[4]]   
    return (params_list, dp, tp, sp, pp, sharding, submitted)


def _run_trace_generation(model_params, parallelism_params, temp_dir):
    """Run trace generation with updated parameters"""
    with st.spinner("Generating trace with custom parameters..."):
        start_time = time.time()
        
        # Convert model parameters to the format expected by generate_trace
        params_list = [
            model_params['din'],
            model_params['dout'], 
            model_params['dmodel'],
            model_params['dff'],
            [model_params['batch']],  # batch needs to be in a list
            model_params['seq'],
            model_params['head'],
            model_params['num_stacks']
        ]
        
        # Convert parallelism parameters to the format expected by generate_trace
        parallelism_strategy = [
            parallelism_params['dp'],
            parallelism_params['tp'],
            parallelism_params['sp'],
            parallelism_params['pp'],
            parallelism_params['sharding']
        ]
        
        gen.generate_trace(parallelism_strategy, params_list, temp_dir)
        elapsed_time = time.time() - start_time
    st.success(f"Trace generation completed in {elapsed_time:.2f} seconds.")


def _clear_session_state():
    # Preserve important session variables that should persist
    preserved_keys = {
        'temp_dir', 'session_id', 'simulation_completed', 
        'last_sim_params', 'last_sim_outputs', 'df_matched',
        'simulation_running', 'sys_content', 'net_content',
        'hardware_params', 'model_params', 'parallelism_params',
        'hw_config', 'model_config', 'parallelism_config'  # Preserve user input configurations
    }
    preserved_values = {}
    
    # Save values we want to keep
    for key in preserved_keys:
        if key in st.session_state:
            preserved_values[key] = st.session_state[key]
    
    # Clear session state keys that might interfere with new runs
    keys_to_clear = [k for k in st.session_state.keys() if k not in preserved_keys]
    for key in keys_to_clear:
        del st.session_state[key]
    
    # Restore preserved values
    for key, value in preserved_values.items():
        st.session_state[key] = value


def _clear_temp_dir(temp_dir):
    subprocess.run(f"rm -rf {temp_dir}*", shell=True, cwd=None)
