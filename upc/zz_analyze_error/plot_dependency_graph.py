import os
import json
import argparse
from graphviz import Digraph

# --- Data Loading (adapted from trace_simulator.py) ---

def parse_et_file(file_path):
    """
    Parses a .et.txt file which contains multiple JSON objects.
    """
    nodes = []
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            # Handle files with multiple JSON objects
            decoder = json.JSONDecoder()
            idx = 0
            while idx < len(content):
                try:
                    obj, consumed = decoder.raw_decode(content[idx:])
                    nodes.append(obj)
                    idx += consumed
                    # Skip whitespace
                    while idx < len(content) and content[idx].isspace():
                        idx += 1
                except json.JSONDecodeError:
                    # Could be trailing characters or malformed json, stop parsing
                    break
    except (IOError, json.JSONDecodeError) as e:
        print(f"Error reading or parsing {file_path}: {e}")
        return None, None

    if not nodes:
        return None, None

    # First element is header, rest are nodes
    header = nodes[0]
    node_dict = {node['id']: node for node in nodes[1:] if 'id' in node}
    return header, node_dict

def load_all_traces(trace_dir):
    """
    Loads all .et.txt trace files from a directory.
    """
    npu_graphs = {}
    for filename in sorted(os.listdir(trace_dir)):
        if filename.endswith(".et.txt"):
            file_path = os.path.join(trace_dir, filename)
            npu_id = int(filename.split('.')[0].split('_')[-1])
            _, graph = parse_et_file(file_path)
            if graph:
                npu_graphs[npu_id] = graph
    return npu_graphs

# --- Graph Drawing Logic ---

def draw_full_npu_graph(npu_graphs, target_npu_id, output_file):
    """
    Creates and renders a dependency graph for all nodes within a single NPU.
    """
    dot = Digraph(comment=f'Full Dependency Graph for NPU {target_npu_id}')
    dot.attr('graph', rankdir='TB', splines='ortho', label=f'NPU {target_npu_id} Graph', labelloc='t', fontsize='16')
    dot.attr('node', fontsize='10', height='0.5')
    dot.attr('edge', fontsize='8')

    npu_graph = npu_graphs.get(target_npu_id)
    if not npu_graph:
        print(f"Error: Graph for NPU {target_npu_id} not found.")
        return

    # 1. Add all nodes from the target NPU to the graph
    for node_id, node in npu_graph.items():
        node_name = node.get('name', f'Node {node_id}')
        node_label = f"{node_name}\nID: {node_id}"
        
        if "comm" in node.get('type', '').lower() or "comm" in node_name.lower():
            dot.node(str(node_id), node_label, shape='box', style='filled', color='salmon')
        else:
            dot.node(str(node_id), node_label, shape='ellipse', style='filled', color='skyblue')

    # 2. Add all intra-NPU dependency edges
    for node_id, node in npu_graph.items():
        dependencies = node.get('ctrlDeps', []) + node.get('dataDeps', [])
        for dep_id in dependencies:
            # Only add an edge if the dependency is also in the same NPU graph
            if dep_id in npu_graph:
                dot.edge(str(dep_id), str(node_id))

    try:
        dot.render(output_file, format='png', cleanup=True)
        print(f"Full NPU dependency graph saved to {output_file}.png")
    except Exception as e:
        print(f"Error rendering graph: {e}")
        print("Please ensure Graphviz is installed and in your system's PATH.")

def main():
    """Main function to parse arguments and generate dependency graphs for all NPUs."""
    parser = argparse.ArgumentParser(description="Generate dependency graphs for all NPUs found in the trace directory.")
    
    base_dir = os.path.dirname(__file__)
    default_jsons_folder = os.path.join(base_dir, 'jsons')
    default_output_dir = os.path.join(base_dir, 'dependency_graphs')

    parser.add_argument("--trace_dir", type=str, default=default_jsons_folder,
                        help=f"Directory containing the .et.txt trace files. Defaults to '{default_jsons_folder}'")
    parser.add_argument("--output_dir", type=str, default=default_output_dir,
                        help=f"Directory to save the generated graph images. Defaults to '{default_output_dir}'")
    
    args = parser.parse_args()

    if not os.path.isdir(args.trace_dir):
        print(f"Error: Trace directory not found at {args.trace_dir}")
        return

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output will be saved to {args.output_dir}")

    npu_graphs = load_all_traces(args.trace_dir)
    if not npu_graphs:
        print("No trace data loaded. Exiting.")
        return

    for npu_id in sorted(npu_graphs.keys()):
        print(f"Generating graph for NPU {npu_id}...")
        output_filename = f"full_graph_npu_{npu_id}"
        output_filepath = os.path.join(args.output_dir, output_filename)
        draw_full_npu_graph(npu_graphs, npu_id, output_filepath)

if __name__ == '__main__':
    main()
