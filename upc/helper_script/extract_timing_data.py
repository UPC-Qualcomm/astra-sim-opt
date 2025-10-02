#!/usr/bin/env python3

import re
import csv

def extract_timing_data(filename):
    """Extract workload names and execution times from the text file."""
    timing_data = []
    
    # Pattern to match lines like "Total time for 2_1_2_16_0: 2.02 seconds"
    pattern = r'Total time for ([^:]+):\s+([\d.]+)\s+seconds'
    
    with open(filename, 'r') as file:
        for line_num, line in enumerate(file, 1):
            match = re.search(pattern, line.strip())
            if match:
                workload = match.group(1)
                time_seconds = float(match.group(2))
                timing_data.append({
                    'workload': workload,
                    'time_seconds': time_seconds,
                    'line_number': line_num
                })
    
    return timing_data

def create_table(timing_data):
    """Create a formatted table from the timing data."""
    print("Workload Timing Analysis from text_v5.2.txt")
    print("=" * 50)
    print(f"{'Workload Name':<20} {'Time (seconds)':<15} {'Line #':<10}")
    print("-" * 50)
    
    # Sort by time (ascending)
    sorted_data = sorted(timing_data, key=lambda x: x['time_seconds'])
    
    for entry in sorted_data:
        print(f"{entry['workload']:<20} {entry['time_seconds']:<15.2f} {entry['line_number']:<10}")
    
    # Summary statistics
    times = [entry['time_seconds'] for entry in timing_data]
    print("\n" + "=" * 50)
    print("SUMMARY STATISTICS")
    print("=" * 50)
    print(f"Total workloads: {len(timing_data)}")
    print(f"Minimum time: {min(times):.2f} seconds")
    print(f"Maximum time: {max(times):.2f} seconds") 
    print(f"Average time: {sum(times)/len(times):.2f} seconds")
    
    # Find duplicates (if any)
    workloads = [entry['workload'] for entry in timing_data]
    duplicates = []
    seen = set()
    for workload in workloads:
        if workload in seen:
            duplicates.append(workload)
        else:
            seen.add(workload)
    
    if duplicates:
        print(f"\nDuplicate workloads found: {set(duplicates)}")
    else:
        print(f"\nAll workloads are unique: {len(set(workloads))} unique workloads")

def save_to_csv(timing_data, output_filename):
    """Save the timing data to a CSV file."""
    with open(output_filename, 'w', newline='') as csvfile:
        fieldnames = ['workload', 'time_seconds', 'line_number']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for entry in sorted(timing_data, key=lambda x: x['time_seconds']):
            writer.writerow(entry)
    
    print(f"\nData saved to: {output_filename}")

if __name__ == "__main__":
    input_file = "../text_aware.txt"
    output_file = "../results/GPT_3_1300M_v5_unaware/workload_timing_data.csv"
    
    try:
        timing_data = extract_timing_data(input_file)
        
        if timing_data:
            create_table(timing_data)
            save_to_csv(timing_data, output_file)
        else:
            print("No timing data found in the file.")
            
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
    except Exception as e:
        print(f"Error: {e}")