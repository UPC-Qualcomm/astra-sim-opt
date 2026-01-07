import os

def count_and_list_files_in_folders(base_path, folder_keywords, file_name_substrings):
    """
    Count folders with specific keywords in their names containing files with specific substrings in their filenames.

    Args:
        base_path (str): The base directory to start the search.
        folder_keywords (list): List of keywords to match folder names.
        file_name_substrings (list): List of substrings to search for in filenames.

    Returns:
        dict: A dictionary with folder keywords as keys and counts and file lists as values.
    """
    results = {keyword: {substring: {"count": 0, "files": []} for substring in file_name_substrings} for keyword in folder_keywords}

    for root, dirs, files in os.walk(base_path):
        for keyword in folder_keywords:
            if keyword in os.path.basename(root):
                for substring in file_name_substrings:
                    matching_files = [file for file in files if substring in file]
                    results[keyword][substring]["count"] += len(matching_files)
                    results[keyword][substring]["files"].extend(os.path.relpath(os.path.join(root, file), base_path) for file in matching_files)

    return results

if __name__ == "__main__":
    base_path = input("Enter the base directory to search: ").strip()

    if not os.path.exists(base_path):
        print(f"The directory '{base_path}' does not exist.")
    else:
        folder_keywords = ["ns3", "g2"]
        file_name_substrings = ["trace.csv", "trace_matched_timing.csv"]

        results = count_and_list_files_in_folders(base_path, folder_keywords, file_name_substrings)

        for keyword, counts in results.items():
            print(f"\nFolders with '{keyword}' in their names:")
            for substring, data in counts.items():
                print(f"  Files containing '{substring}': {data['count']} files")
                for file_path in data['files']:
                    print(f"    - {file_path}")