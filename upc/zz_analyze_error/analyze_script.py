import json
import re
import os
from collections import defaultdict

def parse_error_log(log_file):
    """Parses the error log to find the last finished node for each NPU."""
    last_finished_nodes = {}
    sys_id_pattern = re.compile(r"sys\.id=(\d+)")
    node_id_pattern = re.compile(r"GPU comm node id: (\d+)")

    with open(log_file, 'r') as f:
        lines = f.readlines()

    current_sys_id = None
    for line in lines:
        sys_id_match = sys_id_pattern.search(line)
        if sys_id_match:
            current_sys_id = int(sys_id_match.group(1))
        
        node_id_match = node_id_pattern.search(line)
        if node_id_match and current_sys_id is not None:
            node_id = node_id_match.group(1)
            last_finished_nodes[current_sys_id] = node_id
            current_sys_id = None
            
    return last_finished_nodes

def load_npu_graphs(jsons_folder):
    """Loads all node graphs from the jsons directory."""
    npu_graphs = {}
    for filename in os.listdir(jsons_folder):
        if filename.endswith(".et.txt"):
            try:
                npu_id = int(filename.split('.')[0])
                filepath = os.path.join(jsons_folder, filename)
                with open(filepath, 'r') as f:
                    # The file is a stream of JSON objects, not a single array.
                    # We need to decode them one by one.
                    text = f.read()
                    decoder = json.JSONDecoder()
                    nodes = []
                    pos = 0
                    while pos < len(text):
                        # Skip whitespace
                        while pos < len(text) and text[pos].isspace():
                            pos += 1
                        if pos == len(text):
                            break
                        node, size = decoder.raw_decode(text[pos:])
                        nodes.append(node)
                        pos += size

                # Filter out any empty dictionaries that might result from parsing
                nodes = [node for node in nodes if node]
                graph = {node['id']: node for node in nodes if 'id' in node}
                npu_graphs[npu_id] = graph
            except (ValueError, IndexError, json.JSONDecodeError) as e:
                print(f"Warning: Could not parse file {filename}: {e}")
    return npu_graphs

def get_all_dependencies(node, graph):
    """Recursively finds all dependencies for a given node."""
    deps = set()
    
    # Use a stack for iterative traversal to avoid recursion depth issues
    stack = [node['id']]
    visited = set()

    while stack:
        node_id = stack.pop()
        if node_id in visited:
            continue
        visited.add(node_id)
        
        if node_id not in graph:
            continue # Dependency not in the graph, might be an issue or intended.

        current_node = graph[node_id]
        node_deps = current_node.get('dataDeps', []) + current_node.get('ctrlDeps', [])
        
        for dep_id in node_deps:
            if dep_id not in deps:
                deps.add(dep_id)
                stack.append(dep_id)
    return deps

def find_finished_nodes(last_node_id, graph):
    """Finds all finished nodes by traversing dependencies from the last finished node."""
    if last_node_id not in graph:
        return set()
    
    finished = {last_node_id}
    
    # Use a queue for breadth-first traversal of dependencies
    queue = [last_node_id]
    visited = {last_node_id}

    while queue:
        node_id = queue.pop(0)
        node = graph.get(node_id)
        if not node:
            continue

        node_deps = node.get('dataDeps', []) + node.get('ctrlDeps', [])
        for dep_id in node_deps:
            if dep_id not in visited:
                finished.add(dep_id)
                visited.add(dep_id)
                queue.append(dep_id)
                
    return finished

def find_ready_nodes(finished_nodes, graph):
    """Finds all nodes that are ready to run (all dependencies are finished)."""
    ready_nodes = set()
    for node_id, node in graph.items():
        if node_id in finished_nodes:
            continue
        
        deps = node.get('dataDeps', []) + node.get('ctrlDeps', [])
        if not deps or all(dep in finished_nodes for dep in deps):
            ready_nodes.add(node_id)
    return ready_nodes


def main():
    """Main analysis function."""
    log_file = '/app/astra-sim/upc/output/comparison_run/FoldedClos/GPT_3_1300M_grouped/GPT_3_1300M_multiple_4_4_1_1_1.seq_2048.batch_1024/run_20251216_100352_768ms/g2/4_2_1_2_1.seq_2048.batch_1024.log'
    jsons_folder = '/app/astra-sim/upc/zz_analyze_error/jsons'

    # 1. Parse error log
    last_finished_nodes = parse_error_log(log_file)
    if not last_finished_nodes:
        print("Could not find any last finished nodes in the log file.")
        return

    print("Last finished nodes reported:")
    for npu_id in sorted(last_finished_nodes.keys()):
        print(f"  NPU {npu_id}: Node {last_finished_nodes[npu_id]}")
    print("-" * 30)

    # 2. Load NPU graphs
    npu_graphs = load_npu_graphs(jsons_folder)
    if not npu_graphs:
        print(f"Could not load any NPU graphs from '{jsons_folder}'.")
        return

    # 3. & 4. Determine finished and ready nodes for each NPU
    all_npu_ready_nodes = {}
    for npu_id, graph in npu_graphs.items():
        if npu_id in last_finished_nodes:
            last_node_id = last_finished_nodes[npu_id]
            finished_nodes = find_finished_nodes(last_node_id, graph)
            ready_nodes = find_ready_nodes(finished_nodes, graph)
            all_npu_ready_nodes[npu_id] = {graph[nid]['name'] for nid in ready_nodes}

    # 5. Find the first point of divergence
    all_ready_names = set()
    for names in all_npu_ready_nodes.values():
        all_ready_names.update(names)

    divergence_found = False
    for node_name in sorted(list(all_ready_names)):
        ready_on_npus = {npu for npu, names in all_npu_ready_nodes.items() if node_name in names}
        
        # Check if the node is ready on some but not all NPUs
        if ready_on_npus and len(ready_on_npus) < len(all_npu_ready_nodes):
            not_ready_on_npus = set(all_npu_ready_nodes.keys()) - ready_on_npus
            print(f"Potential failure point found for node: '{node_name}'")
            print(f"  - Ready to run on NPUs: {sorted(list(ready_on_npus))}")
            print(f"  - NOT ready on NPUs:   {sorted(list(not_ready_on_npus))}")
            print("-" * 30)
            divergence_found = True
            # We break after the first one, assuming it's the root cause.
            # Remove 'break' to see all divergences.
            break 

    if not divergence_found:
        print("No divergence found. All NPUs appear to be synchronized in their ready nodes.")
        print("This could mean the failure is in a node that is ready on all NPUs, or the log is incomplete.")


if __name__ == '__main__':
    main()