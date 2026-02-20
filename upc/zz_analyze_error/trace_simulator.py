import json
import os
import random
import heapq
from collections import defaultdict, deque

# --- Data Loading ---

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
                # Skip whitespace
                while idx < len(content) and content[idx].isspace():
                    idx += 1
            except json.JSONDecodeError:
                # This can happen if there's trailing whitespace or other non-json content
                break
    
    if not nodes:
        return None, {}

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
            npu_id = int(filename.split('.')[0])
            file_path = os.path.join(trace_dir, filename)
            _, graph = parse_et_file(file_path)
            if graph:
                npu_graphs[npu_id] = graph
    return npu_graphs

def load_comm_groups(json_path):
    """
    Loads the main JSON file to get communication group information.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data.get('comm_groups', {})

# --- Simulation Logic ---

# Event Types
NODE_START = "NODE_START"
NODE_FINISH = "NODE_FINISH"

class Event:
    """Represents an event in the simulation."""
    def __init__(self, time, event_type, npu_id, node_id, details=None):
        self.time = time
        self.event_type = event_type
        self.npu_id = npu_id
        self.node_id = node_id
        self.details = details or {}

    def __lt__(self, other):
        return self.time < other.time

class TraceSimulator:
    """
    Simulates the execution of distributed traces.
    """
    def __init__(self, npu_graphs, comm_groups, verbose=False):
        self.npu_graphs = npu_graphs
        self.comm_groups = comm_groups
        self.event_queue = []
        self.current_time = 0
        self.finished_nodes = {npu_id: set() for npu_id in npu_graphs}
        self.verbose = verbose
        
        # Track running nodes separately for computation and communication
        self.running_comp_nodes = {npu_id: None for npu_id in npu_graphs}
        self.running_comm_nodes = {npu_id: None for npu_id in npu_graphs}

        self.pending_nodes = {npu_id: deque() for npu_id in npu_graphs}
        self.dependencies = defaultdict(dict)
        self.dependents = defaultdict(lambda: defaultdict(set))
        self.executed_nodes = {npu_id: [] for npu_id in npu_graphs} # To log executed nodes
        self.node_timings = {npu_id: {} for npu_id in npu_graphs} # To store start and stop times
        
        # For collective finish dependency
        self.collective_start_info = defaultdict(dict)
        self.pending_finishes = defaultdict(lambda: defaultdict(dict))

        # Map from any node ID to its host NPU ID, built once.
        self.node_to_npu_map = {
            node_id: npu_id
            for npu_id, graph in self.npu_graphs.items()
            for node_id in graph
        }
        self._build_dependencies()

    def _build_dependencies(self):
        """Pre-calculates dependencies for all nodes, handling cross-NPU dependencies."""
        for npu_id, graph in self.npu_graphs.items():
            for node_id, node in graph.items():
                deps = set(node.get('dataDeps', [])) | set(node.get('ctrlDeps', []))
                self.dependencies[npu_id][node_id] = deps
                if not deps:
                    self.pending_nodes[npu_id].append(node_id)

                # For each dependency, find its NPU and add this node to its dependents list
                for dep_id in deps:
                    if dep_id in self.node_to_npu_map:
                        dep_npu_id = self.node_to_npu_map[dep_id]
                        self.dependents[dep_npu_id][dep_id].add(node_id)

    def get_node_duration(self, node):
        """Calculates a random duration for a node."""
        if node['type'] == 'COMP_NODE':
            tensor_size = next((attr['uint64Val'] for attr in node.get('attr', []) if attr['name'] == 'tensor_size'), 0)
            # Duration proportional to tensor size with some randomness
            return random.uniform(4, 6)
            # return (int(tensor_size) / 1000000) * random.uniform(0.05, 50)
        else: # COMM_NODE, SEND, RECV
            return random.uniform(4, 6) # Random duration for communication

    def try_schedule_node_start(self, npu_id, node_id):
        """
        Check if a node can start and schedule it.
        Returns True if scheduled, False otherwise.
        """
        node = self.npu_graphs[npu_id][node_id]
        is_comm_node = node['type'] != 'COMP_NODE'

        # Check if the required resource (comp or comm) is available
        if is_comm_node:
            if self.running_comm_nodes[npu_id] is not None:
                return False # Comm resource is busy
        else:
            if self.running_comp_nodes[npu_id] is not None:
                return False # Comp resource is busy

        # Schedule the node start event. The start dependency is now purely local.
        heapq.heappush(self.event_queue, Event(self.current_time, NODE_START, npu_id, node_id))
        if is_comm_node:
            self.running_comm_nodes[npu_id] = node_id
        else:
            self.running_comp_nodes[npu_id] = node_id
        return True

    def handle_node_finish(self, event):
        """Handles a NODE_FINISH event."""
        npu_id, node_id = event.npu_id, event.node_id
        node = self.npu_graphs[npu_id][node_id]
        is_comm_node = node['type'] != 'COMP_NODE'

        if self.verbose:
            print(f"Time {event.time:.2f}: NPU {npu_id} finished {'COMM' if is_comm_node else 'COMP'} node {node_id} ({node.get('name', '')})")
        self.finished_nodes[npu_id].add(node_id)
        self.executed_nodes[npu_id].append(node_id)

        # Record finish time
        if node_id not in self.node_timings[npu_id]:
            self.node_timings[npu_id][node_id] = {}
        self.node_timings[npu_id][node_id]['end'] = self.current_time

        if self.verbose:
            print(f"Time {self.current_time}: NPU {npu_id} finishes node {node_id} ({node.get('type')})")

        # Free up the resource
        if is_comm_node:
            self.running_comm_nodes[npu_id] = None
        else:
            self.running_comp_nodes[npu_id] = None

        # Check dependents of the finished node.
        # A finished node can unblock dependents on any NPU.
        for dependent_id in self.dependents[npu_id][node_id]:
            dependent_npu_id = self.node_to_npu_map.get(dependent_id)
            if dependent_npu_id is not None:
                # Remove the finished node from the dependent's dependency list
                self.dependencies[dependent_npu_id][dependent_id].remove(node_id)
                
                # If the dependent has no more dependencies, it's ready to be scheduled
                if not self.dependencies[dependent_npu_id][dependent_id]:
                    self.pending_nodes[dependent_npu_id].append(dependent_id)
                    # Immediately try to schedule nodes on the dependent's NPU
                    self.schedule_pending_nodes_for_npu(dependent_npu_id)
        
        # Also try to schedule new nodes on the current NPU, in case a resource was freed up
        self.schedule_pending_nodes_for_npu(npu_id)

    def write_node_timings(self, output_file):
        """Writes the recorded node timings to a file."""
        with open(output_file, 'w') as f:
            for npu_id, timings in self.node_timings.items():
                f.write(f"NPU {npu_id}:\n")
                # Sort nodes by start time
                sorted_nodes = sorted(timings.items(), key=lambda item: item[1].get('start', float('inf')))
                for node_id, times in sorted_nodes:
                    f.write(f"  Node {node_id}: Start={times.get('start', -1)}, End={times.get('end', -1)}\n")

    def schedule_pending_nodes_for_npu(self, npu_id):
        """
        Tries to schedule all possible nodes from the pending queue for a given NPU.
        This is more robust and checks all pending nodes, not just the first one.
        """
        nodes_to_process = len(self.pending_nodes[npu_id])
        if nodes_to_process == 0:
            return

        still_pending = deque()
        
        # Iterate through all nodes currently in the pending queue
        for _ in range(nodes_to_process):
            node_id = self.pending_nodes[npu_id].popleft()
            
            # Try to schedule the node. If it fails, add it to the still_pending list.
            if not self.try_schedule_node_start(npu_id, node_id):
                still_pending.append(node_id)
        
        # The pending queue for the next round is only the nodes that couldn't be scheduled.
        self.pending_nodes[npu_id] = still_pending

    def schedule_collective_finish(self, npu_id, node_id, local_finish_time):
        """
        Schedules the finish event for a collective node.
        The node can only finish after ALL other nodes in the collective have started.
        """
        node = self.npu_graphs[npu_id][node_id]
        coll_name = node.get('name')
        pg_name = next((attr['stringVal'] for attr in node.get('attr', []) if attr['name'] == 'pg_name'), None)

        if not (coll_name and pg_name and pg_name in self.comm_groups):
            # Not a valid collective for our dependency model, finish locally
            heapq.heappush(self.event_queue, Event(local_finish_time, NODE_FINISH, npu_id, node_id))
            return

        group = self.comm_groups[pg_name]
        if npu_id not in group:
            heapq.heappush(self.event_queue, Event(local_finish_time, NODE_FINISH, npu_id, node_id))
            return

        started_nodes_in_coll = self.collective_start_info.get(coll_name, {})
        
        # Check if all nodes in the group have started this collective
        all_started = all(member_npu_id in started_nodes_in_coll for member_npu_id in group)

        if all_started:
            # Find the latest "arrival time" from any other node in the group
            max_arrival_time = 0
            for member_npu_id in group:
                if member_npu_id != npu_id:
                    start_info = started_nodes_in_coll[member_npu_id]
                    arrival_time = start_info['start_time'] + start_info['duration']
                    if arrival_time > max_arrival_time:
                        max_arrival_time = arrival_time
            
            # Actual finish time is the max of local work completion and the latest data arrival
            finish_time = max(local_finish_time, max_arrival_time)
            heapq.heappush(self.event_queue, Event(finish_time, NODE_FINISH, npu_id, node_id))
        else:
            # Not all nodes have started yet. Put this finish on hold.
            self.pending_finishes[coll_name][npu_id] = {
                'node_id': node_id,
                'local_finish_time': local_finish_time
            }

    def run(self):
        """Runs the simulation."""
        # Initial scheduling: Try to schedule all initially pending nodes across all NPUs.
        # These are the nodes that had no dependencies from the start.
        for npu_id in self.npu_graphs:
            self.schedule_pending_nodes_for_npu(npu_id)

        while self.event_queue:
            event = heapq.heappop(self.event_queue)
            self.current_time = event.time

            if event.event_type == NODE_START:
                npu_id, node_id = event.npu_id, event.node_id
                node = self.npu_graphs[npu_id][node_id]
                
                if self.verbose:
                    is_comm_node = node['type'] != 'COMP_NODE'
                    print(f"Time {self.current_time:.2f}: NPU {npu_id} started {'COMM' if is_comm_node else 'COMP'} node {node_id} ({node.get('name', '')})")
                
                # Record start time
                if node_id not in self.node_timings[npu_id]:
                    self.node_timings[npu_id][node_id] = {}
                self.node_timings[npu_id][node_id]['start'] = self.current_time

                duration = self.get_node_duration(node)
                local_finish_time = self.current_time + duration

                if node['type'] == 'COMM_NODE':
                    # Record start time and duration for dependency tracking
                    coll_name = node.get('name')
                    pg_name = next((attr['stringVal'] for attr in node.get('attr', []) if attr['name'] == 'pg_name'), None)

                    if coll_name and pg_name:
                        self.collective_start_info[coll_name][npu_id] = {
                            'start_time': self.current_time,
                            'duration': duration
                        }
                        
                        # This start might unblock pending finishes from other nodes in the same collective
                        group = self.comm_groups.get(pg_name)
                        if group:
                            started_nodes_in_coll = self.collective_start_info[coll_name]
                            all_started = all(member_npu_id in started_nodes_in_coll for member_npu_id in group)
                            
                            if all_started:
                                # All nodes have now started, check for and schedule any pending finishes
                                pending_for_this_coll = self.pending_finishes.get(coll_name, {})
                                for pending_npu_id, info in list(pending_for_this_coll.items()):
                                    # Re-evaluate and schedule the finish for the pending node
                                    self.schedule_collective_finish(pending_npu_id, info['node_id'], info['local_finish_time'])
                                    # Remove from pending list
                                    del self.pending_finishes[coll_name][pending_npu_id]

                    # Schedule its own finish (which might be immediate or pending)
                    self.schedule_collective_finish(npu_id, node_id, local_finish_time)
                else: # COMP_NODE or other types
                    heapq.heappush(self.event_queue, Event(local_finish_time, NODE_FINISH, npu_id, node_id))

            elif event.event_type == NODE_FINISH:
                self.handle_node_finish(event)

        total_nodes = sum(len(g) for g in self.npu_graphs.values())
        total_executed = sum(len(n) for n in self.executed_nodes.values())
        
        if self.verbose:
            print(f"\nSimulation finished at time {self.current_time:.2f}")
            print(f"Executed {total_executed} out of {total_nodes} total nodes.")

        return total_executed == total_nodes, total_executed, total_nodes

    def write_execution_log(self, output_file):
        """Writes the log of executed nodes to a file."""
        with open(output_file, 'w') as f:
            f.write("Execution Log\n")
            f.write("="*20 + "\n")
            for npu_id in sorted(self.executed_nodes.keys()):
                f.write(f"\n--- NPU {npu_id} ---\n")
                if not self.executed_nodes[npu_id]:
                    f.write("No nodes executed.\n")
                    continue
                
                for node_id in self.executed_nodes[npu_id]:
                    node = self.npu_graphs[npu_id][node_id]
                    f.write(f"  - Node {node_id}: {node.get('name', 'N/A')} (Type: {node.get('type')})\n")

    def write_remaining_nodes_log(self, output_file):
        """Writes a log of nodes that were not executed."""
        with open(output_file, 'w') as f:
            f.write("Remaining (Unexecuted) Nodes Log\n")
            f.write("="*35 + "\n")
            any_remaining = False
            for npu_id in sorted(self.npu_graphs.keys()):
                executed = set(self.executed_nodes[npu_id])
                all_nodes = set(self.npu_graphs[npu_id].keys())
                remaining_nodes = all_nodes - executed
                
                if remaining_nodes:
                    any_remaining = True
                    f.write(f"\n--- NPU {npu_id} ---\n")
                    for node_id in sorted(list(remaining_nodes)):
                        node = self.npu_graphs[npu_id][node_id]
                        deps = self.dependencies[npu_id].get(node_id, set())
                        f.write(f"  - Node {node_id}: {node.get('name', 'N/A')} (Type: {node.get('type')})\n")
                        f.write(f"    - Remaining Dependencies: {deps}\n")
            
            if not any_remaining:
                f.write("All nodes were executed successfully.\n")

def main():
    """Main function to run the simulation."""
    base_dir = os.path.dirname(__file__)
    jsons_folder = os.path.join(base_dir, 'jsons')
    main_json_file = os.path.join(jsons_folder, '4_4_1_1_0.seq_2048.batch_1024.json')
    summary_log_file = os.path.join(base_dir, 'simulation_summary.txt')
    execution_log_file = os.path.join(base_dir, 'execution_log.txt')
    node_timings_file = os.path.join(base_dir, 'node_timings.txt')
    deadlock_log_file = os.path.join(base_dir, 'deadlock_info.txt')

    num_runs = 10000
    success_count = 0
    first_fail_logged = False
    
    with open(summary_log_file, 'w') as f:
        for i in range(num_runs):
            if i % 100 == 0:
                print(f"Run {i}/{num_runs}...")
            
            npu_graphs = load_all_traces(jsons_folder)
            if not npu_graphs:
                print("No trace files found.")
                return

            comm_groups = load_comm_groups(main_json_file)
            
            sim = TraceSimulator(npu_graphs, comm_groups, verbose=(i==0))
            
            try:
                completed, _, _ = sim.run()
                if completed:
                    success_count += 1
                    f.write(f"Run {i}: Success (Completed)\n")
                    if i == 0: # Log first successful run
                        sim.write_execution_log(execution_log_file)
                        sim.write_node_timings(node_timings_file)
                else:
                    f.write(f"Run {i}: Failed (Deadlock)\n")
                    if not first_fail_logged:
                        print(f"\nDeadlock detected on run {i}. Writing logs.")
                        sim.write_execution_log(execution_log_file)
                        sim.write_node_timings(node_timings_file)
                        sim.write_remaining_nodes_log(deadlock_log_file)
                        first_fail_logged = True

            except Exception as e:
                f.write(f"Run {i}: Failed with error: {e}\n")
                # Optionally, print error for the first run to debug
                if i == 0:
                    print(f"Error during first run: {e}")

    print(f"\nSuccesses: {success_count}/{num_runs}")
    print(f"Summary log written to {summary_log_file}")
    if num_runs > 0:
        print(f"Detailed logs for the first run (or first failure) are in:")
        print(f"  - {execution_log_file}")
        print(f"  - {node_timings_file}")
        if success_count < num_runs:
            print(f"  - {deadlock_log_file}")


if __name__ == '__main__':
    main()
