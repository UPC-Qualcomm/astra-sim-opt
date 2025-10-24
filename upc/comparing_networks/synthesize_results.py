#!/usr/bin/env python3
"""
synthesize_results.py

Scan a run folder with four subfolders (analytical_aware, analytical_unaware, g2, ns3),
parse the relevant files and produce a summarized pandas DataFrame with the computed
time per (src, dst, tag) message for each model.

Usage:
    python synthesize_results.py /path/to/run_folder
"""
import os
import sys
import glob
import re
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

try:
    import pandas as pd
except Exception as e:
    raise SystemExit("pandas is required to run this script. Install with `pip install pandas`") from e


def extract_record_fields_from_log_line(line: str) -> List[str]:
    """
    Extract the comma-separated fields from a log line.
    It handles different prefixes by finding the start of the CSV data,
    which is assumed to start with a comma.
    """
    # Find the start of the CSV data, which is usually after a prefix ending in ': ' or '] '
    match = re.search(r'[:\]]\s*,', line)
    if match:
        # Start from the comma
        data = line[match.start() + 2:]
    else:
        # Fallback for lines that might not have the full prefix
        data = line
    parts = [p.strip() for p in data.split(',')]
    return parts


def parse_csv_like_file(path: str) -> Tuple[List[str], List[Dict[str, str]]]:
    """
    Parse a file that contains a CSV-like header and log lines.
    Returns (header_fields, list_of_records).
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    header_fields: List[str] = []
    records: List[Dict[str, str]] = []

    # Find header line (contains ",action,")
    for ln in lines:
        if ",action," in ln:
            header_fields = extract_record_fields_from_log_line(ln)
            break

    if not header_fields:
        return [], []

    # Normalize header names to lowercase for consistency
    header_fields = [h.strip().lower() for h in header_fields]

    # Parse records
    for ln in lines:
        if not ln.strip() or ",action," in ln:
            continue
        
        parts = extract_record_fields_from_log_line(ln)
        if len(parts) < len(header_fields):
            parts.extend([""] * (len(header_fields) - len(parts)))
        
        rec = {header_fields[i]: parts[i] for i in range(len(header_fields))}
        records.append(rec)
            
    return header_fields, records


def parse_ns3_astrasim_fct(path: str) -> List[Dict[str, any]]:
    """
    Parse ns3/astrasim_fct.txt lines.
    Returns a list of dictionaries, each representing a message.
    """
    results = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            parts = ln.split()
            if len(parts) < 8:
                continue
            
            try:
                # src: 0b000401 -> 4
                src = str(int(parts[0][-4:-2], 16))
                # dst: 0b000101 -> 1
                dst = str(int(parts[1][-4:-2], 16))

                # chunk_id: 10000 -> 0, 10001 -> 1, etc.
                chunk_id = str(int(parts[2]) - 10000)

                issue_tick = float(parts[-3])
                delay = float(parts[-2])

                results.append({
                    "src": src,
                    "dst": dst,
                    "chunk_id": chunk_id,
                    "start_time": issue_tick,
                    "arrival_time": issue_tick + delay,
                    "send_time": delay
                })
            except (ValueError, IndexError):
                pass
    return results


def safe_float(s: str) -> float:
    try:
        return float(s)
    except (ValueError, TypeError):
        return 0.0


def synthesize(run_folder: str) -> pd.DataFrame:
    """
    Main orchestration: parse each subfolder and compute per message times.
    """
    all_messages = []

    # 1) analytical_aware -> based on first and last 'send_mini'
    aware_folder = os.path.join(run_folder, "analytical_aware")
    aware_csv = glob.glob(os.path.join(aware_folder, "*network*.csv"))
    if aware_csv:
        _, records = parse_csv_like_file(aware_csv[0])
        
        # Group records by message
        message_groups = defaultdict(list)
        for r in records:
            if r.get("action", "").lower() == "send_mini":
                key = (
                    r.get("src_origin", ""), r.get("dst_final", ""),
                    r.get("tag", ""), r.get("workload_node_id", ""), r.get("chunk_id", "")
                )
                message_groups[key].append(r)

        for (src, dst, tag, workload, chunk), recs in message_groups.items():
            if not recs:
                continue
            
            first_rec = min(recs, key=lambda x: safe_float(x.get("issue_tick")))
            last_rec = max(recs, key=lambda x: safe_float(x.get("issue_tick")))

            start_time = safe_float(first_rec.get("issue_tick"))
            arrival_time = safe_float(last_rec.get("issue_tick")) + safe_float(last_rec.get("delay"))
            
            all_messages.append({
                "model": "aware", "src": src, "dst": dst, "tag": tag,
                "workload_node_id": workload, "chunk_id": chunk,
                "start_time": start_time,
                "arrival_time": arrival_time,
                "send_time": arrival_time - start_time
            })

    # 2) analytical_unaware -> single 'send' entry per message
    unaware_folder = os.path.join(run_folder, "analytical_unaware")
    unaware_csv = glob.glob(os.path.join(unaware_folder, "*network*.csv"))
    if unaware_csv:
        _, records = parse_csv_like_file(unaware_csv[0])
        for r in records:
            if r.get("action", "").lower() == "send":
                start_time = safe_float(r.get("issue_tick"))
                delay = safe_float(r.get("delay"))
                all_messages.append({
                    "model": "unaware",
                    "src": r.get("src_origin", ""), "dst": r.get("dst_final", ""),
                    "tag": r.get("tag", ""), "workload_node_id": r.get("workload_node_id", ""),
                    "chunk_id": r.get("chunk_id", ""),
                    "start_time": start_time,
                    "arrival_time": start_time + delay,
                    "send_time": delay
                })

    # 3) g2 -> based on first and last 'update'
    g2_folder = os.path.join(run_folder, "g2")
    g2_csv = glob.glob(os.path.join(g2_folder, "*network*.csv"))
    if g2_csv:
        _, records = parse_csv_like_file(g2_csv[0])
        
        message_groups = defaultdict(list)
        for r in records:
            if r.get("action", "").lower() == "update":
                key = (
                    r.get("src_origin", ""), r.get("dst_final", ""),
                    r.get("tag", ""), r.get("workload_node_id", ""), r.get("chunk_id", "")
                )
                message_groups[key].append(r)

        for (src, dst, tag, workload, chunk), recs in message_groups.items():
            if not recs:
                continue

            first_update = min(recs, key=lambda x: safe_float(x.get("issue_tick")))
            last_update = max(recs, key=lambda x: safe_float(x.get("issue_tick")))
            
            start_time = safe_float(first_update.get("issue_tick"))
            arrival_time = safe_float(last_update.get("issue_tick"))

            all_messages.append({
                "model": "g2", "src": src, "dst": dst, "tag": tag,
                "workload_node_id": workload, "chunk_id": chunk,
                "start_time": start_time,
                "arrival_time": arrival_time,
                "send_time": arrival_time - start_time
            })

    # 4) ns3 -> parse astrasim_fct.txt
    ns3_folder = os.path.join(run_folder, "ns3")
    ns3_fct = os.path.join(ns3_folder, "astrasim_fct.txt")
    if os.path.isfile(ns3_fct):
        entries = parse_ns3_astrasim_fct(ns3_fct)
        for entry in entries:
            all_messages.append({
                "model": "ns3",
                "src": entry["src"], "dst": entry["dst"],
                "tag": "",  # Ignored for now
                "workload_node_id": "", # Ignored for now
                "chunk_id": entry["chunk_id"],
                "start_time": entry["start_time"],
                "arrival_time": entry["arrival_time"],
                "send_time": entry["send_time"]
            })

    if not all_messages:
        return pd.DataFrame()

    df = pd.DataFrame(all_messages)
    
    # Define message identifier columns
    id_cols = ['src', 'dst', 'tag', 'workload_node_id', 'chunk_id']
    
    # Pivot for start_time
    df_start = df.pivot_table(index=id_cols, columns='model', values='start_time').reset_index()
    df_start.rename(columns=lambda c: f"start_time_{c}" if c not in id_cols else c, inplace=True)

    # Pivot for send_time
    df_send = df.pivot_table(index=id_cols, columns='model', values='send_time').reset_index()
    df_send.rename(columns=lambda c: f"send_time_{c}" if c not in id_cols else c, inplace=True)

    # Pivot for arrival_time
    df_arrival = df.pivot_table(index=id_cols, columns='model', values='arrival_time').reset_index()
    df_arrival.rename(columns=lambda c: f"arrival_time_{c}" if c not in id_cols else c, inplace=True)

    # Merge the dataframes
    merged_df = pd.merge(df_start, df_send, on=id_cols, how='outer')
    merged_df = pd.merge(merged_df, df_arrival, on=id_cols, how='outer')
    
    # Clean up column names from pivot
    merged_df.columns.name = None
    
    return merged_df.fillna(0)


def main(argv: List[str]) -> None:
    if len(argv) >= 2:
        run_folder = argv[1]
    else:
        # Defaulting to the specified folder for demonstration
        run_folder = "/home/xavid/feina/astra-sim/upc/comparison_run/Ring/all_to_all/run_20251012_234014"

    if not os.path.isdir(run_folder):
        print(f"Run folder not found: {run_folder}", file=sys.stderr)
        sys.exit(1)

    df = synthesize(run_folder)
    
    if df.empty:
        print("No data synthesized.")
        return

    # write to stdout and CSV in run folder
    out_csv = os.path.join(run_folder, "synthesized_results.csv")
    df.to_csv(out_csv, index=False)
    print(df.to_string(index=False))
    print(f"\nSaved summarized results to: {out_csv}")


if __name__ == "__main__":
    main(sys.argv)