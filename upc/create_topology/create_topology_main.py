import random
from create_topology import CustomizedDragonfly, Jellyfish, FoldedClos
from utils import write_ns3_topology_file, write_g2_topology_files

def generate_topology_files(topology, paths_mode, config, output_dir="./", base_filename=None):
    """
    Generates topology and routing files for Astra-Sim.

    Args:
        topology (str): 'FoldedClos', 'Dragonfly', or 'Jellyfish'.
        paths_mode (str): 'ECMP', 'Uniform', 'Random', or None.
        config (dict): Configuration parameters for the topology.

    Formulas for Node Counts:
    - Dragonfly: 
        Hosts = G * A * concentration
        Switches = G * A
        Config: {'G': int, 'A': int, 'h': int, 'concentration': int}
    
    - Jellyfish:
        Hosts = num_switches * num_hosts_per_switch
        Switches = num_switches
        Config: {'num_switches': int, 'degree': int, 'num_hosts_per_switch': int}

    - FoldedClos (Fat-Tree):
        Hosts = K^3 / 4
        Switches = 5/4 * K^2
        Config: {'K': int}
    """
    
    print(f"--- Generating {topology} with {paths_mode} routing ---")
    
    topo_obj = None
    if base_filename is None:
        base_filename = f"{topology}"
    
    # Extract bandwidth config
    bw_config = config.get('bandwidth_config', {})
    bw_unit = config.get('bw_unit', 'GB/s')

    # 1. Initialize Topology
    if topology == "Dragonfly":
        G = config.get('G')
        A = config.get('A')
        h = config.get('h', 1)
        conc = config.get('concentration', 1)
        total_hosts = G * A * conc
        print(f"Formula: Hosts = G({G}) * A({A}) * conc({conc}) = {total_hosts}")
        topo_obj = CustomizedDragonfly(G, A, h, conc, bandwidth_config=bw_config)
        #base_filename += f"_{total_hosts}"
        
    elif topology == "Jellyfish":
        switches = config.get('num_switches')
        degree = config.get('degree')
        hosts_per_switch = config.get('num_hosts_per_switch', 1)
        total_hosts = switches * hosts_per_switch
        print(f"Formula: Hosts = Switches({switches}) * Hosts/Switch({hosts_per_switch}) = {total_hosts}")
        topo_obj = Jellyfish(switches, degree, num_hosts_per_switch=hosts_per_switch, bandwidth_config=bw_config)
        #base_filename += f"_{total_hosts}"

    elif topology == "FoldedClos":
        K = config.get('K')
        npus_per_node = config.get('npus_per_node', 1)
        intra_node_topology = config.get('intra_node_topology', 'fully_connected')
        num_intra_node_switches = config.get('num_intra_node_switches', None)
        
        total_nodes = int(K**3 / 4)
        total_hosts = total_nodes * npus_per_node
        print(f"Formula: Nodes = K^3 / 4 = {total_nodes}, Total NPUs = {total_nodes} * {npus_per_node} = {total_hosts}")
        
        topo_obj = FoldedClos(K, 1, 1, bandwidth_config=bw_config, 
                             npus_per_node=npus_per_node, 
                             intra_node_topology=intra_node_topology,
                             num_intra_node_switches=num_intra_node_switches)
        #base_filename += f"_{total_hosts}"
        
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
        raw_paths = topo_obj.GenerateUniformRouting()        
        final_paths = {src: {dst: [path] for dst, path in dests.items() if 'h' in dst} for src, dests in raw_paths.items() if 'h' in src}

        
    elif paths_mode in ["ECMP", "Random"]:
        topo_obj.GenerateECMPFlowDict(topo_obj.adjacency_matrix)
        raw_paths = topo_obj.paths
        
        all_paths = {}

        if topology in ["Dragonfly", "Jellyfish"]:
            # Post-processing to wrap switch paths with hosts for DF and Jellyfish
            conc = topo_obj.concentration_factor if topology == "Dragonfly" else topo_obj.num_hosts_per_switch
            total_hosts = topo_obj.total_num_hosts
            
            for host1 in range(total_hosts):
                h1 = f'h{host1+1}'
                all_paths[h1] = {}
                for host2 in range(total_hosts):
                    if host1 == host2: continue
                    
                    s1_idx = host1 // conc
                    s2_idx = host2 // conc
                    s1 = f's{s1_idx+1}'
                    s2 = f's{s2_idx+1}'
                    
                    h2 = f'h{host2+1}'
                    
                    if s1_idx == s2_idx:
                        # Same switch
                        all_paths[h1][h2] = [[h1, s1, h2]]
                    else:
                        # Inter-switch: raw_paths[s1][s2] is a list of paths
                        switch_paths = raw_paths[s1][s2]
                        all_paths[h1][h2] = [ [h1] + p + [h2] for p in switch_paths ]
        else:
            # FoldedClos handles hosts internally
            all_paths = raw_paths

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
    
    # Get bandwidths from topology object
    link_bandwidths = getattr(topo_obj, 'link_bandwidths', None)
    
    write_g2_topology_files(links, final_paths, base_filename=base_filename, bandwidth=2, link_bandwidths=link_bandwidths, output_dir=output_dir)
    write_ns3_topology_file(links, final_paths, filename=base_filename, bandwidth=2, link_bandwidths=link_bandwidths, bw_unit=bw_unit, output_dir=output_dir)