
def extract_execution_time(log_file):
    """Extract wall time (in cycles) from AstraSim log file. Returns tuple (time_seconds, is_oom)."""
    try:
        with open(log_file, 'r') as f:
            content = f.read()
        
        # Look for wall time in cycles from any system (use sys[0] as reference)
        import re
        match_exec = re.search(r'\[statistics\] \[info\] sys\[0\], Wall time: (\d+)', content)
        match_is_oom = re.search(r'\[workload\] \[info\] sys\[0\] is OOM: (\d+)', content)
        if match_exec and match_is_oom:
            cycles = int(match_exec.group(1))
            # Convert cycles to seconds assuming 1GHz frequency
            # (you can adjust this frequency based on your simulation setup)
            time_seconds = cycles / 1e9
            return time_seconds, int(match_is_oom.group(1))
        
        return None, None
    except Exception as e:
        print(f"Error extracting execution time: {e}")
        return None, None