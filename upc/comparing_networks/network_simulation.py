import csv
from collections import defaultdict
import sys
import os

# Add the astra-sim root directory to the Python path
# to allow importing the network module.

print(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from extern.network_backend.g2.network import Network

def parse_log_file(filepath):
    """
    Parses the network log CSV file to extract traffic data,
    grouping messages by their issue tick.
    """
    events = defaultdict(list)
    seen_messages = set()

    with open(filepath, 'r') as f:
        # Skip the initial header line
        next(f, None)
        reader = csv.reader(line for line in f if "update" in line)
        for parts in reader:
            try:
                # Extract relevant fields, converting to int
                src_origin = int(parts[2])
                dst_final = int(parts[3])
                src = int(parts[4])
                dst = int(parts[5])
                tensor_size = int(parts[6])
                tag = int(parts[7])
                workload_node_id = int(parts[8])
                chunk_id = int(parts[9])
                issue_tick = int(parts[10])

                # Unique identifier for a message transfer
                msg_id = (src_origin, dst_final, src, dst, tensor_size, tag, workload_node_id, chunk_id)
                
                # Add the message only on its first appearance (issue tick)
                if msg_id not in seen_messages:
                    seen_messages.add(msg_id)
                    events[issue_tick].append(msg_id)

            except (ValueError, IndexError):
                # Skip malformed lines
                continue
    
    # Return events sorted by time
    return sorted(events.items())


def run_simulation(network, events):
    """
    Runs the event-driven network simulation.
    """
    current_time = 0
    event_idx = 0
    total_completed = 0

    # Main simulation loop
    while event_idx < len(events) or network.len_network():
        
        next_injection_time = float('inf')
        if event_idx < len(events):
            next_injection_time = events[event_idx][0]

        # Determine the time of the next event (either a flow completion or a new message injection)
        next_update_time = network.get_next_messages(current_time) if network.len_network() else -1

        if next_update_time != -1 and next_update_time < next_injection_time:
            # The next event is a flow completion
            current_time = next_update_time
            completed_messages = network.remove_messages(current_time)
            if completed_messages:
                total_completed += len(completed_messages)
                print(f"--- Time: {current_time} ns ---")
                print(f"Completed {len(completed_messages)} messages.")
                for msg in completed_messages:
                    # msg format: (tag, src, dest, size, chunk_id, workload_node_id, times, rates)
                    print(f"  -> Tag: {msg[0]}, Src: {msg[1]}, Dest: {msg[2]}, Size: {msg[3]} B")
        else:
            # The next event is a new message injection
            if event_idx >= len(events):
                # No more injections and no more updates needed
                break
            
            current_time = next_injection_time
            tick, messages_to_add = events[event_idx]
            
            print(f"--- Time: {current_time} ns ---")
            print(f"Injecting {len(messages_to_add)} new messages.")
            for msg_data in messages_to_add:
                (src_origin, dst_final, src, dst, tensor_size, tag, workload_node_id, chunk_id) = msg_data
                network.add_route(tag, src, dst, tensor_size, chunk_id, workload_node_id)
            
            event_idx += 1

    print(f"\n--- Simulation Finished ---")
    print(f"Total messages completed: {total_completed}")


if __name__ == "__main__":
    # Network configuration based on the CSV file name "FullyConnected_G2"
    # and assuming 8 NPUs (0-7) as seen in the logs.
    npus = 32
    net = Network(
        npus_count_per_dim=[npus],
        bandwidth_per_dim=[900],  # Assuming 1000 GB/s bandwidth
        topologies_per_dim=['ring'],
        # latency_per_dim=[0.1]  # Assuming 0.1 ns latency
    )
    
    # csv_filepath = './output/real_all_gather/Ring_G2/astrasim.log_network.csv'
    csv_filepath = './upc/output/Simple/Ring_network.csv'

    # Parse the CSV to get all time-ordered events
    traffic_events = parse_log_file(csv_filepath)
    
    if not traffic_events:
        print("No valid traffic data found in the CSV file.")
    else:
        # Run the full simulation
        run_simulation(net, traffic_events)
        net.print_execution_times()
