import json
import os
import pandas as pd
import numpy as np

def parse_et_file(file_path):
    """
    Parses a .et.txt file which contains multiple JSON objects.
    """
    nodes = []
    with open(file_path, 'r') as f:
        decoder = json.JSONDecoder()
        content = f.read()
        idx = 0
        while idx < len(content):
            try:
                obj, size = decoder.raw_decode(content[idx:])
                nodes.append(obj)
                idx += size
                while idx < len(content) and content[idx].isspace():
                    idx += 1
            except json.JSONDecodeError:
                break
    
    if not nodes:
        return None, {}

    header = nodes[0]
    node_dict = {node['id']: node for node in nodes[1:] if 'id' in node}
    return header, node_dict

def load_all_graphs(trace_dir):
    """
    Loads all .et.txt trace files from a directory.
    """
    npu_graphs = {}
    for filename in sorted(os.listdir(trace_dir)):
        if filename.endswith(".et.txt"):
            try:
                npu_id = int(filename.split('.')[0])
                file_path = os.path.join(trace_dir, filename)
                _, graph = parse_et_file(file_path)
                if graph:
                    npu_graphs[npu_id] = graph
            except (ValueError, IndexError):
                continue
    return npu_graphs

def count_remaining_comms(graph, start_node_id):
    """
    Counts the number of communication nodes that are descendants of the start_node_id.
    Uses dataDeps and ctrlDeps to build the dependency graph.
    """
    if not graph or start_node_id not in graph:
        return 0

    # Build a reverse dependency map: node_id -> [nodes that depend on it]
    dependents = {}
    for node_id, node in graph.items():
        data_deps = node.get('dataDeps', [])
        ctrl_deps = node.get('ctrlDeps', [])
        all_deps = set(data_deps + ctrl_deps)
        
        for dep_id in all_deps:
            if dep_id not in dependents:
                dependents[dep_id] = []
            dependents[dep_id].append(node_id)
    
    # BFS traversal starting from nodes that depend on start_node_id
    q = []
    for dependent_id in dependents.get(start_node_id, []):
        q.append(dependent_id)
    
    visited = set(q)
    count = 0
    
    while q:
        curr_id = q.pop(0)
        
        node = graph.get(curr_id, {})
        
        # Check if it's a communication node
        node_type = node.get('type', '')
        if 'COMM' in node_type:
            count += 1
        
        # Add nodes that depend on this node to the queue
        for dependent_id in dependents.get(curr_id, []):
            if dependent_id not in visited:
                visited.add(dependent_id)
                q.append(dependent_id)
    
    return count


def get_comm_size(node):
    """
    Extracts comm_size from a node's attributes.
    """
    if 'attr' in node:
        for attr in node['attr']:
            if attr.get('name') == 'comm_size':
                return int(attr.get('int64Val', 0))
    return np.nan

def analyze_flow_rate(trace_file, graphs):
    """
    Analyzes the flow rate and remaining communications for a given trace file.
    """
    try:
        df = pd.read_csv(trace_file)
    except FileNotFoundError:
        print(f"Error: Trace file not found at {trace_file}")
        return pd.DataFrame()

    results = []
    for _, row in df.iterrows():
        sys_id = row['sys_id']
        node_id = str(row['node_id'])  # Ensure node_id is a string for graph lookup
        
        graph = graphs.get(sys_id, {})
        node_data = graph.get(node_id, {})
        
        comm_size = get_comm_size(node_data)
        elapsed_time = row['elapsed_time']

        if pd.isna(comm_size) or elapsed_time == 0:
            flow_rate = np.nan
        else:
            flow_rate = comm_size / elapsed_time

        remaining_comms = count_remaining_comms(graph, node_id)
        
        results.append({
            'sys_id': sys_id,
            'node_id': node_id,
            'node_name': row['node_name'],
            'comm_size': comm_size,
            'elapsed_time': elapsed_time,
            'flow_rate': flow_rate,
            'remaining_comms': remaining_comms
        })
        
    return pd.DataFrame(results)

if __name__ == "__main__":
    # Define paths
    g2_trace_file = '/app/astra-sim/upc/output/comparison_run/Dragonfly/T5_Small_grouped_ecmp/T5_Small_multiple_1_16_1_1_0.seq_2048.batch_1024/run_20260116_165909_386ms/g2/T5_Small_multiple_1_16_1_1_0.seq_2048.batch_1024_trace_matched_timing.csv'
    ns3_trace_file = '/app/astra-sim/upc/output/comparison_run/Dragonfly/T5_Small_grouped_ecmp/T5_Small_multiple_1_16_1_1_0.seq_2048.batch_1024/run_20260116_165911_075ms/ns3/T5_Small_multiple_1_16_1_1_0.seq_2048.batch_1024_trace_matched_timing.csv'
    graph_dir = '/app/astra-sim/upc/zz_analyze_error/temporal_analysis_results/et_txts'

    # Load graphs
    print("Loading graphs...")
    npu_graphs = load_all_graphs(graph_dir)
    if not npu_graphs:
        print("Error: No graphs were loaded. Exiting.")
        exit()
    print(f"Loaded {len(npu_graphs)} graphs.")

    # Analyze g2
    print("\nAnalyzing g2 run...")
    g2_results = analyze_flow_rate(g2_trace_file, npu_graphs)
    if not g2_results.empty:
        print("g2 analysis complete.")
        print(g2_results.head())

    # Analyze ns3
    print("\nAnalyzing ns3 run...")
    ns3_results = analyze_flow_rate(ns3_trace_file, npu_graphs)
    if not ns3_results.empty:
        print("ns3 analysis complete.")
        print(ns3_results.head())

    # Merge results for comparison
    if not g2_results.empty and not ns3_results.empty:
        merged_df = pd.merge(g2_results, ns3_results, on=['sys_id', 'node_name'], suffixes=('_g2', '_ns3'))
        print("\nMerged results:")
        print(merged_df.head())
        
        # Save to CSV
        output_path = '/app/astra-sim/upc/comparing_networks/flow_rate_comparison.csv'
        merged_df.to_csv(output_path, index=False)
        print(f"\nComparison saved to {output_path}")
