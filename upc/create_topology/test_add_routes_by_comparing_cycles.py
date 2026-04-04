#!/usr/bin/env python3
"""
Compare exec_cycles, comm_cycles, exposed_comm_cycles, comp_cycles, exposed_comp_cycles
between two FoldedClos_iter2.csv files, matched by (dp, mp, sp, pp, sharding, sys_id).
"""

import pandas as pd
import sys

FILE_A = (
    "/media/mohammad/extension/experiments/astra-sim/upc/results/"
    "llama_8B_test_g2_my_sync_fix_128_uniform_routes_on_the_fly/FoldedClos_iter2.csv"
)
FILE_B = (
    "/media/mohammad/extension/experiments/astra-sim/upc/results/"
    "llama_8B_test_g2_my_sync_fix_128_uniform_routes_in_file/FoldedClos_iter2.csv"
)

KEY_COLS   = ["dp", "mp", "sp", "pp", "sharding", "sys_id"]
METRIC_COLS = [
    "exec_cycles",
    "comm_cycles",
    "exposed_comm_cycles",
    "comp_cycles",
    "exposed_comp_cycles",
]

LABEL_A = "on_the_fly"
LABEL_B = "in_file"


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def main() -> None:
    df_a = load(FILE_A)
    df_b = load(FILE_B)

    # Keep only what we need
    df_a = df_a[KEY_COLS + METRIC_COLS].copy()
    df_b = df_b[KEY_COLS + METRIC_COLS].copy()

    merged = df_a.merge(df_b, on=KEY_COLS, suffixes=(f"_{LABEL_A}", f"_{LABEL_B}"))

    if merged.empty:
        print("No matching rows found between the two files.")
        sys.exit(1)

    rows = []
    for _, row in merged.iterrows():
        key = {c: row[c] for c in KEY_COLS}
        for metric in METRIC_COLS:
            val_a = row[f"{metric}_{LABEL_A}"]
            val_b = row[f"{metric}_{LABEL_B}"]
            diff  = val_b - val_a
            pct   = (diff / val_a * 100) if val_a != 0 else float("nan")
            rows.append({**key, "metric": metric,
                         LABEL_A: val_a, LABEL_B: val_b,
                         "diff (B-A)": diff, "diff_%": pct})

    result = pd.DataFrame(rows)

    # ── pretty-print ──────────────────────────────────────────────────────────
    key_str = result[KEY_COLS].astype(str).agg("_".join, axis=1)
    result.insert(0, "config", key_str)
    result = result.drop(columns=KEY_COLS)

    pd.set_option("display.max_rows",    None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width",       220)
    pd.set_option("display.float_format", "{:,.2f}".format)

    print(f"\n{'='*80}")
    print(f"  FILE A  ({LABEL_A}):")
    print(f"  {FILE_A}")
    print(f"  FILE B  ({LABEL_B}):")
    print(f"  {FILE_B}")
    print(f"  Matched rows: {len(merged)}")
    print(f"{'='*80}\n")
    print(result.to_string(index=False))

    # ── per-metric summary ────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("  PER-METRIC SUMMARY  (mean absolute & percentage difference)")
    print(f"{'='*80}")
    summary = (
        result.groupby("metric")[["diff (B-A)", "diff_%"]]
        .agg(mean_diff=("diff (B-A)", "mean"),
             mean_abs_diff=("diff (B-A)", lambda x: x.abs().mean()),
             mean_pct=("diff_%", "mean"),
             mean_abs_pct=("diff_%", lambda x: x.abs().mean()))
        .reset_index()
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
