import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict
import re
import os

def parse_execution_log(log_file):
    """Parses the execution_log.txt file to extract node timings."""
    node_events = defaultdict(dict)
    
    # Regex to parse each line, accommodating different formats
    log_pattern = re.compile(
        r"Time:\s*(\d+\.\d+|\d+)\s*\|\s*NPU:\s*(\d+)\s*\|\s*"
        r"(NODE_START|NODE_FINISH)\s*\|\s*Node:\s*(\d+)\s*\|\s*Name:\s*(.*)"
    )

    with open(log_file, 'r') as f:
        for line in f:
            match = log_pattern.match(line.strip())
            if match:
                time = float(match.group(1))
                npu_id = int(match.group(2))
                event_type = match.group(3)
                node_id = int(match.group(4))
                node_name = match.group(5).strip()

                key = (npu_id, node_id)
                if event_type == "NODE_START":
                    node_events[key]['start'] = time
                    node_events[key]['name'] = node_name
                    # Infer type from name for coloring
                    if "comm" in node_name.lower():
                        node_events[key]['type'] = 'COMM'
                    else:
                        node_events[key]['type'] = 'COMP'
                elif event_type == "NODE_FINISH":
                    node_events[key]['finish'] = time

    # Filter out incomplete events and format for plotting
    plot_data = []
    for (npu_id, node_id), data in node_events.items():
        if 'start' in data and 'finish' in data:
            plot_data.append({
                'npu': npu_id,
                'start': data['start'],
                'finish': data['finish'],
                'duration': data['finish'] - data['start'],
                'name': data['name'],
                'type': data.get('type', 'UNKNOWN')
            })
    return plot_data

def draw_gantt_chart(plot_data, output_file):
    """Draws a Gantt chart of the NPU execution timeline."""
    if not plot_data:
        print("No data to plot.")
        return

    fig, ax = plt.subplots(figsize=(20, 10))

    # Define colors for different node types
    colors = {'COMP': 'skyblue', 'COMM': 'salmon', 'UNKNOWN': 'lightgrey'}
    
    # Get unique NPUs and sort them
    npus = sorted(list(set(d['npu'] for d in plot_data)))
    npu_map = {npu_id: i for i, npu_id in enumerate(npus)}

    for item in plot_data:
        npu_idx = npu_map[item['npu']]
        ax.barh(npu_idx, item['duration'], left=item['start'], 
                height=0.6, align='center',
                color=colors[item['type']], edgecolor='black')
        
        # Add node name inside the bar if it fits
        if item['duration'] > 0:
             ax.text(item['start'] + item['duration']/2, npu_idx, 
                    item['name'], ha='center', va='center', color='black', fontsize=8)

    # Configure axes and labels
    ax.set_yticks(range(len(npus)))
    ax.set_yticklabels([f'NPU {npu_id}' for npu_id in npus])
    ax.invert_yaxis()  # Puts NPU 0 at the top

    ax.set_xlabel("Time (cycles)")
    ax.set_title("NPU Execution Gantt Chart")
    ax.grid(True, which='major', axis='x', linestyle='--', linewidth=0.5)

    # Create a legend
    patches = [mpatches.Patch(color=color, label=label) for label, color in colors.items()]
    ax.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight')
    print(f"Gantt chart saved to {output_file}")

def main():
    """Main function to generate the Gantt chart."""
    base_dir = os.path.dirname(__file__)
    log_file = os.path.join(base_dir, 'execution_log.txt')
    output_image_file = os.path.join(base_dir, 'gantt_chart.png')

    if not os.path.exists(log_file):
        print(f"Error: Execution log file not found at {log_file}")
        print("Please run trace_simulator.py first to generate the log.")
        return

    plot_data = parse_execution_log(log_file)
    draw_gantt_chart(plot_data, output_image_file)

if __name__ == '__main__':
    main()
