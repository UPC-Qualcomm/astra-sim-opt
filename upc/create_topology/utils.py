import re
import json

def write_g2_topology_files(links, paths, base_filename="topology", bandwidth=900, link_bandwidths=None, output_dir="./"):
    """
    Writes topology data to two files:
    1. A custom .txt file with a simple, quote-less format.
    2. A standard, machine-readable .json file.
    
    The paths in both files are filtered to only include host-to-host routes.

    Args:
        links (dict): A dictionary of links from the topology generator.
        paths (dict): A dictionary of all possible paths from the generator.
        base_filename (str): The base name for the output files.
        bandwidth (int): The default bandwidth for each link if not specified in link_bandwidths.
        link_bandwidths (dict): Optional dictionary mapping link IDs to bandwidth values.
    """
    # --- 1. Filter paths to keep only host-to-host routes ---
    host_to_host_paths = {}
    for src, dests in paths.items():
        if src.startswith('h'):
            host_dests = {
                dest: path for dest, path in dests.items() 
                if dest.startswith('h') and src != dest
            }
            if host_dests:
                host_to_host_paths[src] = host_dests

    # --- 2. Create and write the custom .txt file ---
    txt_content = []
    
    # Add the edges without quotes
    for link_id, (node1, node2) in links.items():
        bw = bandwidth
        if link_bandwidths and link_id in link_bandwidths:
            bw = link_bandwidths[link_id]
        txt_content.append(f"({node1}, {node2}, {bw})")
    
    # Add a separator for readability
    txt_content.append("\n# Paths\n")

    # Add the paths, handling multiple paths for each src-dest pair
    for src, dests in host_to_host_paths.items():
        for dest, list_of_paths in dests.items():
            for path in list_of_paths:
                # Join the path list into a string like "h1, s1, h2"
                path_str = ", ".join(path)
                txt_content.append(f"{src}: {dest}: [{path_str}]")

    import os
    txt_filename = os.path.join(output_dir, f"{base_filename}.txt")
    with open(txt_filename, "w") as f:
        f.write("\n".join(txt_content))
    print(f"Successfully wrote custom text topology to {txt_filename}")

    # --- 3. Create and write the standard .json file ---
    edges = []
    for link_id, (n1, n2) in links.items():
        bw = bandwidth
        if link_bandwidths and link_id in link_bandwidths:
            bw = link_bandwidths[link_id]
        edges.append([n1, n2, bw])

    json_data = {
        "numEdges": len(links),
        "edges": edges,
        "paths": host_to_host_paths
    }

    json_filename = os.path.join(output_dir, f"{base_filename}.json")
    with open(json_filename, "w") as f:
        json.dump(json_data, f, indent=4)
    print(f"Successfully wrote JSON topology to {json_filename}")


def write_ns3_topology_file(links, paths, filename="ns3_topology.txt", bandwidth="900GiB/s", latency="0.000ms", link_bandwidths=None, bw_unit="GB/s", output_dir="./"):
    """
    Maps node names to sequential IDs and writes a topology file in the NS3 format,
    including pre-computed routes.
    - Hosts are mapped to IDs [0, num_hosts - 1].
    - Switches are mapped to IDs [num_hosts, num_hosts + num_switches - 1].

    Args:
        links (dict): Dictionary of links with string node names (e.g., 'h1', 's1').
        paths (dict): Dictionary of paths with string node names.
        filename (str): The name of the output file.
        bandwidth (str): Default bandwidth for all links.
        latency (str): Link latency.
        link_bandwidths (dict): Optional dictionary mapping link IDs to bandwidth values.
        bw_unit (str): Unit string to append to bandwidth values (e.g., "GB/s").
    """
    def get_num(name):
        """Extracts the integer part of a node name for sorting."""
        match = re.search(r'\d+', name)
        return int(match.group()) if match else -1

    # 1. Discover all unique nodes and categorize them
    all_node_names = set()
    for _, (n1, n2) in links.items():
        all_node_names.add(n1)
        all_node_names.add(n2)

    host_names = sorted([name for name in all_node_names if name.startswith('h')], key=get_num)
    switch_names = sorted([name for name in all_node_names if name.startswith('s')], key=get_num)

    num_hosts = len(host_names)
    num_switches = len(switch_names)
    num_nodes = num_hosts + num_switches
    
    # Calculate unique links and map bandwidths
    unique_links = set()
    pair_to_bw = {}
    
    if link_bandwidths:
        for lid, (n1, n2) in links.items():
            pair = tuple(sorted((n1, n2)))
            pair_to_bw[pair] = link_bandwidths[lid]

    for n1, n2 in links.values():
        # Store links in a canonical order (smaller name first) to handle duplicates
        sorted_pair = tuple(sorted((n1, n2)))
        unique_links.add(sorted_pair)
    num_links = len(unique_links)


    # 2. Create the mapping from name to new sequential ID
    name_to_id_map = {}
    switch_ids = []
    for i, name in enumerate(host_names):
        new_id = i
        name_to_id_map[name] = new_id

    for i, name in enumerate(switch_names):
        new_id = num_hosts + i
        name_to_id_map[name] = new_id
        switch_ids.append(new_id)

    # 3. Process links using the new IDs
    processed_links_with_bw = []
    for n1_str, n2_str in unique_links:
        id1 = name_to_id_map[n1_str]
        id2 = name_to_id_map[n2_str]
        
        pair_key = tuple(sorted((n1_str, n2_str)))
        bw_val = bandwidth
        if pair_key in pair_to_bw:
            val = pair_to_bw[pair_key]
            # Format if int/float to string with unit if needed, or just string
            if isinstance(val, (int, float)):
                bw_val = f"{val}{bw_unit}"
            else:
                bw_val = str(val)
        
        processed_links_with_bw.append((tuple(sorted((id1, id2))), bw_val))

    # 4. Write to file
    import os
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "w") as f:
        # Header
        f.write(f"{num_nodes} {num_switches} {num_links}\n")
        
        # Switch IDs (already sorted)
        for switch_id in switch_ids:
            f.write(f"{switch_id}\n")
            
        # Links
        for (id1, id2), bw in sorted(processed_links_with_bw, key=lambda x: x[0]):
            f.write(f"{id1} {id2} {bw} {latency} 0\n")

        # 5. Write paths
        f.write("\nROUTES\n")
        
        # Filter for host-to-host paths
        host_to_host_paths = {}
        for src, dests in paths.items():
            if src.startswith('h'):
                host_dests = {
                    dest: path_list for dest, path_list in dests.items() 
                    if dest.startswith('h') and src != dest
                }
                if host_dests:
                    host_to_host_paths[src] = host_dests
        
        # Convert names to IDs and write to file
        for src_str, dests in sorted(host_to_host_paths.items(), key=lambda item: get_num(item[0])):
            src_id = name_to_id_map[src_str]
            for dest_str, list_of_paths in sorted(dests.items(), key=lambda item: get_num(item[0])):
                dest_id = name_to_id_map[dest_str]
                for path in list_of_paths:
                    path_ids = [name_to_id_map[hop] for hop in path]
                    path_ids_str = ", ".join(map(str, path_ids))
                    f.write(f"{src_id}:{dest_id}:[{path_ids_str}]\n")


    print(f"Successfully wrote NS3 topology to {filepath}")


def make_node_names_zero_indexed(links, paths):
    """
    Converts 1-indexed node names ('h1', 's1') to 0-indexed names ('h0', 's0').
    This is primarily for G2 configuration files.
    
    Args:
        links (dict): Dictionary of links with 1-indexed string node names.
        paths (dict): Dictionary of paths with 1-indexed string node names.

    Returns:
        tuple: A tuple containing:
            - dict: A new links dictionary with 0-indexed names.
            - dict: A new paths dictionary with 0-indexed names.
    """
    def convert_name(name):
        # Use regex to handle multi-digit numbers correctly
        match = re.match(r"([a-zA-Z]+)([0-9]+)", name)
        if not match:
            return name # Return name unchanged if it doesn't fit the pattern
        
        prefix = match.group(1)
        number = int(match.group(2))
        
        # Subtract 1 only if the number is greater than 0
        if number > 0:
            return f"{prefix}{number - 1}"
        return name

    # Convert links
    new_links = {
        link_id: (convert_name(n1), convert_name(n2)) 
        for link_id, (n1, n2) in links.items()
    }

    # Convert paths
    new_paths = {}
    for src, dests in paths.items():
        new_src = convert_name(src)
        new_paths[new_src] = {}
        for dest, path_list in dests.items():
            new_dest = convert_name(dest)
            new_paths[new_src][new_dest] = [convert_name(hop) for hop in path_list]
            
    return new_links, new_paths