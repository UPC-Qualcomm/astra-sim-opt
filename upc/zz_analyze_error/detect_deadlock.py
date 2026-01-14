import argparse
import sys
import os
import json
import csv
from collections import defaultdict
import re

def parse_comm_groups(json_path):
    """
    Parses the communication group configuration file.
    Expects a JSON dict where keys are group IDs and values are lists of NPU IDs.
    """
    print(f"Loading communication groups from {json_path}")
    with open(json_path, 'r') as f:
        content = f.read()
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            print(f"Error decoding JSON from {json_path}")
            sys.exit(1)
            
    if "comm_groups" in data:
        return data["comm_groups"]
    return data

def parse_et_files(jsons_dir):
    """
    Parses .et.txt files to map node_id to communication group info per NPU.
    Returns: npu_node_map[npu_id][node_id] = { 'pg_name': ..., 'name': ... }
    """
    print(f"Parsing graph files in {jsons_dir}")
    npu_node_map = defaultdict(dict)
    
    et_files = sorted([f for f in os.listdir(jsons_dir) if f.endswith('.et.txt')])
    
    for et_file in et_files:
        try:
            npu_id = int(et_file.split('.')[0])
        except ValueError:
            continue
            
        path = os.path.join(jsons_dir, et_file)
        with open(path, 'r') as f:
            content = f.read()
            decoder = json.JSONDecoder()
            idx = 0
            while idx < len(content):
                # Find start of next JSON object
                start_brace = content.find('{', idx)
                if start_brace == -1: break
                
                try:
                    obj, end_idx = decoder.raw_decode(content, start_brace)
                    idx = end_idx
                    
                    if 'id' in obj:
                        node_id = int(obj['id'])
                        
                        # Extract PG Name (Comm Group ID) if available
                        pg_name = None
                        if 'attr' in obj:
                            for attr in obj['attr']:
                                if attr.get('name') in ['pg_name', 'comm_group', 'communicator_id']:
                                    pg_name = attr.get('stringVal') or attr.get('intVal')
                                    break
                        
                        # Store everything, even if no pg_name, we might need name later
                        npu_node_map[npu_id][node_id] = {
                            'pg_name': str(pg_name) if pg_name is not None else None, 
                            'name': obj.get('name', f"Available_Node_{node_id}"),
                            'type': obj.get('type_id') # If available in your ET format
                        }
                except json.JSONDecodeError:
                    idx = start_brace + 1
                    
    print(f"Loaded graph info for {len(npu_node_map)} NPUs")
    return npu_node_map

def parse_trace(trace_file):
    """
    Parses the trace CSV to find currently active (issued but not finished) nodes.
    Tracks issuance order to identify the 'blocking' node.
    """
    print(f"Parsing trace file {trace_file}")
    active_nodes = defaultdict(dict)
    
    line_count = 0
    issue_count = 0
    callback_count = 0

    with open(trace_file, 'r') as f:
        for line in f:
            line_count += 1
            if '<trace>:' not in line:
                continue
            
            parts = line.split(',')
            
            action = None
            action_idx = -1
            
            for i, p in enumerate(parts):
                p_strip = p.strip()
                if p_strip in ['issue', 'callback']:
                    action = p_strip
                    action_idx = i
                    break
            
            if action_idx != -1 and len(parts) > action_idx + 2:
                try:
                    sys_id = int(parts[action_idx + 1])
                    node_id = int(parts[action_idx + 2])
                    
                    if action == 'issue':
                        node_name = "Unknown"
                        node_type = -1
                        col_type = -1
                        
                        if len(parts) > action_idx + 3:
                            node_name = parts[action_idx + 3].strip()
                        if len(parts) > action_idx + 4:
                            col_type = parts[action_idx + 4].strip()
                        if len(parts) > action_idx + 5:
                            try:
                                node_type = int(parts[action_idx + 5])
                            except ValueError:
                                pass

                        active_nodes[sys_id][node_id] = {
                            'name': node_name,
                            'node_type': node_type,
                            'col_type': col_type,
                            'issue_order': line_count  # Keep track of when it was issued
                        }
                        issue_count += 1
                    elif action == 'callback':
                        if node_id in active_nodes[sys_id]:
                            del active_nodes[sys_id][node_id]
                        else:
                            # Sometimes callbacks happen for nodes we didn't see issue for (if trace is partial)
                            pass
                        callback_count += 1
                except ValueError:
                    continue
                    
    print(f"Processed {line_count} lines. Issues: {issue_count}, Unfinished (Active): {sum(len(x) for x in active_nodes.values())}")
    return active_nodes

def analyze_deadlocks(active_nodes, npu_node_map, comm_groups):
    """
    Detects deadlocks by comparing the 'next' pending collective operation
    for all members of each communication group.
    """
    print("\n--- Analyzing for deadlocks ---")
    deadlocks_found = False
    
    if not active_nodes:
        print("No active nodes found. Trace indicates clean completion or empty log.")
        return

    for group_id, members in comm_groups.items():
        # Map: effective_op_signature -> [list of npus]
        # Signature is (node_id)
        group_status = defaultdict(list)
        idle_members = []
        
        for npu in members:
            # Get all active nodes for this NPU
            npu_active = active_nodes.get(npu, {})
            
            # Find the distinct active node that belongs to THIS group
            # We sort by issue_order to find the one that likely caused the block first
            sorted_active = sorted(npu_active.items(), key=lambda x: x[1]['issue_order'])
            
            matched_node = None
            
            for node_id, trace_info in sorted_active:
                # Check graph info for group ID
                graph_info = npu_node_map.get(npu, {}).get(node_id, {})
                pg_name = graph_info.get('pg_name')
                
                # Check if this node belongs to the current group
                if pg_name and str(pg_name) == str(group_id):
                    matched_node = node_id # Simplification: Signature is just node_id
                    break
            
            if matched_node:
                group_status[matched_node].append(npu)
            else:
                idle_members.append(npu)
        
        # Now analyze the status of this group
        active_ids = list(group_status.keys())
        
        if len(active_ids) > 1:
            deadlocks_found = True
            print(f"\n[DEADLOCK (CROSS-DEPENDENCY)] Group {group_id}")
            print(f"  Group Members: {members}")
            print(f"  Conflicting Collectives:")
            for node_id in active_ids:
                waiting_npus = group_status[node_id]
                print(f"    - Collective Node {node_id} : Waiting NPUs {waiting_npus}")
        
        elif len(active_ids) == 1 and len(idle_members) > 0:
            deadlocks_found = True
            node_id = active_ids[0]
            working_npus = group_status[node_id]
            
            print(f"\n[STALL (WAITING)] Group {group_id}")
            print(f"  Group Members: {members}")
            print(f"  - Collective Node {node_id} : Waiting NPUs {working_npus}")
            print(f"  - Idle/Missing NPUs   : {idle_members}")
    
    if not deadlocks_found:
        print("\nNo group inconsistencies detected.")

    print("\n--- Summary of Stuck Nodes ---")
    # Gather all NPUs we know about from graph files or trace
    all_known_npus = sorted(set(npu_node_map.keys()) | set(active_nodes.keys()))
    
    for npu in all_known_npus:
        if npu not in active_nodes or not active_nodes[npu]:
            print(f"  NPU {npu}: IDLE")
        else:
            # Sort by issue order to find the oldest one (the head of the detailed queue)
            sorted_active = sorted(active_nodes[npu].items(), key=lambda x: x[1]['issue_order'])
            
            first_node_id, _ = sorted_active[0]
            
            # Find involved NPUs based on the collective group of this node
            graph_info = npu_node_map.get(npu, {}).get(first_node_id, {})
            pg_name = graph_info.get('pg_name')
            
            involved_npus_str = "Unknown"
            if pg_name:
                # comm_groups keys are strings
                involved_npus_str = str(comm_groups.get(str(pg_name), "Not Found in comm_groups"))
            
            print(f"  NPU {npu}: STUCK at Node {first_node_id} (Collective Group: {pg_name}, Involved NPUs: {involved_npus_str})")

def main():
    parser = argparse.ArgumentParser(description="Detect deadlock in AstraSim execution traces.")
    parser.add_argument("--trace", required=True, help="Path to the trace CSV file")
    parser.add_argument("--jsons-dir", default="./jsons", help="Directory containing .et.txt files")
    parser.add_argument("--comm-group", default=None, help="Path to JSON file defining comm groups.")
    args = parser.parse_args()
    
    # Auto-detect comm group
    comm_group_file = args.comm_group
    if not comm_group_file:
        potential_files = [f for f in os.listdir(args.jsons_dir) if f.endswith('.json')]
        if potential_files:
            comm_group_file = os.path.join(args.jsons_dir, potential_files[0])
            print(f"Auto-selected comm group file: {comm_group_file}")
    
    if not comm_group_file:
         print("Error: Provide --comm-group or ensure .json file exists in jsons-dir")
         sys.exit(1)

    comm_groups = parse_comm_groups(comm_group_file)
    npu_node_map = parse_et_files(args.jsons_dir)
    active_nodes = parse_trace(args.trace)
    analyze_deadlocks(active_nodes, npu_node_map, comm_groups)

if __name__ == "__main__":
    main()