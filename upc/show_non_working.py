#!/usr/bin/python3
import os
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import argparse
import csv

import os
import re
import argparse

def check_log_files(directory, npu):
    """
    Checks all log files in a given directory for occurrences of sys[x] (x = 0 to npu).
    Keeps files where the count of occurrences is exactly npu.
    
    Args:
        directory (str): Path to the directory containing log files.
        npu (int): The expected number of sys[x] occurrences.

    Returns:
        list: Files that do not meet the condition.
    """
    non_matching_files = []
    matching_files = []

    # Regular expression to match "sys[x]" where x is from 0 to npu
    pattern = re.compile(r"sys\[(\d+)\]")

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        
        if not os.path.isfile(file_path) or not filename.endswith(".log"):
            continue  # Skip non-files and non-log files
        
        # Dictionary to track occurrences of sys[x]
        occurrences = set()

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                matches = pattern.findall(line)
                for match in matches:
                    x = int(match)
                    if 0 <= x <= npu:
                        occurrences.add(x)

        # Check if the file meets the condition
        if len(occurrences) != npu and len(occurrences) != 0:
            non_matching_files.append(filename[:-4]+"_" + str(len(occurrences) ))
        elif len(occurrences) != 0:
            matching_files.append(filename[:-4])

    return non_matching_files, matching_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter log files based on sys[x] occurrences.")
    parser.add_argument("--dir", type=str, help="Directory containing log files.")
    parser.add_argument("--npu", type=int, help="Expected number of sys[x] occurrences.")

    args = parser.parse_args()
    
    non_matching, matching = check_log_files(args.dir, args.npu)

    print("Files that do not meet the condition ", len(non_matching), ": ", non_matching)
    print("All files meet the condition ", len(matching), ":", matching)
