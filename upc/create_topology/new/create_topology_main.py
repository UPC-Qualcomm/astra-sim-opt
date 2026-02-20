import random
from create_topology import CustomizedDragonfly, Jellyfish, FoldedClos
from utils import write_ns3_topology_file, write_g2_topology_files

def generate_topology_files(topology, paths_mode, config, output_dir="./", base_filename=None):
    """
    Generates topology and routing files for Astra-Sim.

    Args:
        topology (str): 'FoldedClos', 'Dragonfly', or 'Jellyfish'.
        paths_mode (str): 'ECMP', 'Uniform', or None.
        config (dict): Configuration parameters for the topology.

    Formulas for Node Counts:
    - Dragonfly: 
        Nodes = G * A * concentration
        NPUs = Nodes * npus_per_node
        Switches = G * A
        Config: {'G': int, 'A': int, 'h': int, 'concentration': int,
                 'npus_per_node': int, 'intra_node_topology': str, 'num_nvswitches': int}
    
    - Jellyfish:
        Nodes = num_switches * num_hosts_per_switch
        NPUs = Nodes * npus_per_node
        Switches = num_switches
        Config: {'num_switches': int, 'degree': int, 'num_hosts_per_switch': int,
                 'npus_per_node': int, 'intra_node_topology': str, 'num_nvswitches': int}

    - FoldedClos (Fat-Tree):
        Nodes = K^3 / 4
        NPUs = Nodes * npus_per_node
        Switches = 5/4 * K^2
        Config: {'K': int, 'npus_per_node': int, 'intra_node_topology': str, 'num_nvswitches': int}
    
    Intra-node Topology Options:
    - 'switch': NPUs connect to NVSwitch(es), all NPUs connect to ToR for inter-node
    - 'ring': NPUs (via switches) connected in a ring, all connect to ToR
    - 'fully_connected': NPUs (via switches) fully connected, all connect to ToR
    - None: Direct connection to main switch (no intra-node topology)
    
    Bandwidth Configuration Keys:
    - All topologies: 'intra_node' (bandwidth within node)
    - Dragonfly: 'host_switch' or 'inter_node', 'intra_group', 'inter_group'
    - Jellyfish: 'host_switch' or 'inter_node', 'switch_switch'
    - FoldedClos: 'host_edge' or 'inter_node', 'edge_agg', 'agg_core'
    """
    
    print(f"--- Generating {topology} with {paths_mode} routing ---")
    
    topo_obj = None
    if base_filename is None:
        base_filename = f"{topology}"
    
    # Extract bandwidth config
    bw_config = config.get('bandwidth_config', {})
    bw_unit = config.get('bw_unit', 'GB/s')
    
    # Extract latency config
    lat_config = config.get('latency_config', {})
    lat_unit = config.get('lat_unit', 'ms')
    
    # Extract intra-node configuration
    nodes_per_server = config.get('nodes_per_server', 1)
    npus_per_node = config.get('npus_per_node', 1)
    intra_node_topology = config.get('intra_node_topology', None)
    num_nvswitches = config.get('num_nvswitches', 1)

    # 1. Initialize Topology
    if topology == "Dragonfly":
        G = config.get('G')
        A = config.get('A')
        h = config.get('h', 1)
        conc = config.get('concentration', 1)
        total_servers = G * A * conc
        total_nodes = total_servers * nodes_per_server
        total_npus = total_nodes * npus_per_node
        print(f"Formula: Servers = G({G}) * A({A}) * conc({conc}) = {total_servers}")
        print(f"         Nodes = Servers({total_servers}) * nodes_per_server({nodes_per_server}) = {total_nodes}")
        print(f"         NPUs = Nodes({total_nodes}) * npus_per_node({npus_per_node}) = {total_npus}")
        if intra_node_topology:
            print(f"         Intra-node topology: {intra_node_topology}")
        topo_obj = CustomizedDragonfly(G, A, h, conc, bandwidth_config=bw_config,
                                        latency_config=lat_config,
                                        nodes_per_server=nodes_per_server,
                                        npus_per_node=npus_per_node,
                                        intra_node_topology=intra_node_topology,
                                        num_nvswitches=num_nvswitches)
        
    elif topology == "Jellyfish":
        switches = config.get('num_switches')
        degree = config.get('degree')
        hosts_per_switch = config.get('num_hosts_per_switch', 1)
        total_servers = switches * hosts_per_switch
        total_nodes = total_servers * nodes_per_server
        total_npus = total_nodes * npus_per_node
        print(f"Formula: Servers = Switches({switches}) * Servers/Switch({hosts_per_switch}) = {total_servers}")
        print(f"         Nodes = Servers({total_servers}) * nodes_per_server({nodes_per_server}) = {total_nodes}")
        print(f"         NPUs = Nodes({total_nodes}) * npus_per_node({npus_per_node}) = {total_npus}")
        if intra_node_topology:
            print(f"         Intra-node topology: {intra_node_topology}")
        topo_obj = Jellyfish(switches, degree, num_hosts_per_switch=hosts_per_switch, 
                             bandwidth_config=bw_config,
                             latency_config=lat_config,
                             nodes_per_server=nodes_per_server,
                             npus_per_node=npus_per_node,
                             intra_node_topology=intra_node_topology,
                             num_nvswitches=num_nvswitches)

    elif topology == "FoldedClos":
        K = config.get('K')
        total_servers = int(K**3 / 4)
        total_nodes = total_servers * nodes_per_server
        total_npus = total_nodes * npus_per_node
        print(f"Formula: Servers = K^3 / 4 = {total_servers}")
        print(f"         Nodes = Servers({total_servers}) * nodes_per_server({nodes_per_server}) = {total_nodes}")
        print(f"         NPUs = Nodes({total_nodes}) * npus_per_node({npus_per_node}) = {total_npus}")
        if intra_node_topology:
            print(f"         Intra-node topology: {intra_node_topology}")
        topo_obj = FoldedClos(K, 1, 1, bandwidth_config=bw_config,
                              latency_config=lat_config,
                              nodes_per_server=nodes_per_server,
                              npus_per_node=npus_per_node,
                              intra_node_topology=intra_node_topology,
                              num_nvswitches=num_nvswitches)
        
    else:
        print(f"Error: Unknown topology {topology}")
        return

    # 2. Design Topology & Links
    topo_obj.DesignFullTopology()
    links_output = topo_obj.LinksToG2ConfFile()
    
    # Handle return signature difference (FoldedClos returns tuple)
    if isinstance(links_output, tuple):
        links = links_output[0]
    else:
        links = links_output

    if paths_mode is None:
        print("No paths requested. Finished.")
        return

    # 3. Generate Routing Paths
    final_paths = {}
    
    if paths_mode == "Uniform":
        # Generate standard uniform routing
        # GenerateUniformRouting returns paths[src][dst] = [hop1, hop2, ...]
        # Wrap each path in a list to match the expected format: paths[src][dst] = [[hop1, hop2, ...]]
        raw_uniform = topo_obj.GenerateUniformRouting()
        for src, dests in raw_uniform.items():
            if not src.startswith('h'):
                continue
            final_paths[src] = {}
            for dst, path in dests.items():
                if not dst.startswith('h') or src == dst:
                    continue
                final_paths[src][dst] = [path]
        
    elif paths_mode in ["ECMP", "Random"]:
        topo_obj.GenerateECMPFlowDict(topo_obj.adjacency_matrix)
        raw_paths = topo_obj.paths
        
        all_paths = {}

        if topology in ["Dragonfly", "Jellyfish"]:
            # Post-processing to wrap switch paths with hosts/NPUs
            conc = topo_obj.concentration_factor if topology == "Dragonfly" else topo_obj.num_hosts_per_switch
            total_npus = topo_obj.total_num_hosts
            npus_per_node = topo_obj.npus_per_node
            
            for npu1 in range(total_npus):
                h1 = f'h{npu1+1}'
                all_paths[h1] = {}
                for npu2 in range(total_npus):
                    if npu1 == npu2: 
                        continue
                    
                    # Calculate which node each NPU belongs to
                    node1 = npu1 // npus_per_node
                    node2 = npu2 // npus_per_node
                    
                    # Calculate which switch each node is attached to
                    s1_idx = node1 // conc
                    s2_idx = node2 // conc
                    s1 = f't{s1_idx+1}'
                    s2 = f't{s2_idx+1}'
                    
                    h2 = f'h{npu2+1}'
                    
                    if node1 == node2:
                        # Same node - use intra-node paths (NOT through ToR)
                        all_paths[h1][h2] = topo_obj.generate_intra_node_paths(h1, h2)
                    elif s1_idx == s2_idx:
                        # Same switch, different nodes
                        all_paths[h1][h2] = [[h1, s1, h2]]
                    else:
                        # Inter-switch: raw_paths[s1][s2] is a list of paths
                        switch_paths = raw_paths[s1][s2]
                        all_paths[h1][h2] = [[h1] + p + [h2] for p in switch_paths]
        else:
            # FoldedClos - need to regenerate paths with intra-node handling
            total_npus = int(topo_obj.numHosts)
            npus_per_node = topo_obj.npus_per_node
            
            for npu1 in range(total_npus):
                h1 = f'h{npu1+1}'
                all_paths[h1] = {}
                for npu2 in range(total_npus):
                    if npu1 == npu2:
                        continue
                    
                    h2 = f'h{npu2+1}'
                    
                    # Calculate which node each NPU belongs to
                    node1 = npu1 // npus_per_node
                    node2 = npu2 // npus_per_node
                    
                    if node1 == node2 and intra_node_topology and npus_per_node > 1:
                        # Same node - use intra-node paths (NOT through ToR)
                        all_paths[h1][h2] = topo_obj.generate_intra_node_paths(h1, h2)
                    else:
                        # Inter-node: use standard paths from raw_paths
                        if h1 in raw_paths and h2 in raw_paths.get(h1, {}):
                            path_data = raw_paths[h1][h2]
                            # Ensure it's a list of paths
                            if path_data and isinstance(path_data[0], str):
                                all_paths[h1][h2] = [path_data]
                            else:
                                all_paths[h1][h2] = path_data
                        else:
                            # Fallback - direct path
                            all_paths[h1][h2] = [[h1, h2]]

        # Apply path selection mode
        if paths_mode == "Random":
            for src, dests in all_paths.items():
                final_paths[src] = {}
                for dest, path_list in dests.items():
                    if path_list:
                        final_paths[src][dest] = [random.choice(path_list)]
        else:  # ECMP
            final_paths = all_paths
            
    # 4. Write Output Files
    print(f"Writing files to: {output_dir}")
    
    # Get bandwidths and latencies from topology object
    link_bandwidths = getattr(topo_obj, 'link_bandwidths', None)
    link_latencies = getattr(topo_obj, 'link_latencies', None)
    
    write_g2_topology_files(links, final_paths, base_filename=base_filename, bandwidth=2, link_bandwidths=link_bandwidths, link_latencies=link_latencies, output_dir=output_dir)
    write_ns3_topology_file(links, final_paths, filename=base_filename, bandwidth=2, link_bandwidths=link_bandwidths, link_latencies=link_latencies, bw_unit=bw_unit, lat_unit=lat_unit, output_dir=output_dir)