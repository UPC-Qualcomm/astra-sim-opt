import matplotlib.pyplot as plt
import re
import argparse
import os

def extract_comm_sizes_and_npu_counts(file_path):
    """
    Extracts communication sizes and NPU counts from a log file.
    The signature line is expected to contain a tuple where the last element is the communication size,
    and the NPU groups are at the beginning.
    Example: Signature: (((0, 2, 4, 6, 8, 10, 12, 14), (1, 3, 5, 7, 9, 11, 13, 15)), 2, 268435456)
    """
    sizes = []
    npu_counts = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                if 'Signature:' in line:
                    # Extract size
                    size_match = re.search(r'(\d+)\)$', line)
                    if not size_match:
                        continue
                    
                    size = int(size_match.group(1))

                    # Extract NPU groups
                    npu_match = re.search(r'Signature: \((.*)\), \d+, \d+\)$', line)
                    if not npu_match:
                        # Handle cases with no NPU groups but with a size
                        # This can happen for workloads with no collectives with signatures
                        # For now, we can assume 1 NPU if not specified, or skip.
                        # Let's skip to be safe.
                        continue

                    npu_groups_str = npu_match.group(1)
                    
                    # This is a bit tricky, we can use eval but it's unsafe.
                    # A safer way is to parse it manually.
                    # Find all numbers, which represent NPU IDs.
                    npu_ids = set(re.findall(r'\d+', npu_groups_str))
                    num_npus = len(npu_ids)

                    if num_npus > 0:
                        sizes.append(size / num_npus)
                        npu_counts.append(num_npus)

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    return sizes

def plot_comm_sizes(file1, file2, output_file):
    """
    Plots communication sizes from two files on a scatter plot.
    """
    sizes1 = extract_comm_sizes_and_npu_counts(file1)
    sizes2 = extract_comm_sizes_and_npu_counts(file2)

    if not sizes1 and not sizes2:
        print("No communication sizes found in either file. Exiting.")
        return

    plt.figure(figsize=(12, 8))

    # Plot sizes from file1
    if sizes1:
        plt.scatter(range(len(sizes1)), sizes1, color='blue', label=f'{os.path.basename(file1)} ({len(sizes1)} points)')

    # Plot sizes from file2
    if sizes2:
        plt.scatter(range(len(sizes2)), sizes2, color='red', alpha=0.7, label=f'{os.path.basename(file2)} ({len(sizes2)} points)')

    plt.title('Communication Sizes from Workload Traces')
    plt.xlabel('Collective Index')
    plt.ylabel('Communication Size (bytes)')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot communication sizes from two log files.')
    parser.add_argument('file1', type=str, help='Path to the first log file.')
    parser.add_argument('file2', type=str, help='Path to the second log file.')
    parser.add_argument('--output', type=str, default='comm_sizes.png', help='Output file name for the plot.')
    
    args = parser.parse_args()

    plot_comm_sizes(args.file1, args.file2, args.output)
