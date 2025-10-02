#!/usr/bin/env python3

import pandas as pd

def merge_csvs():
    """Merge the two CSV files by matching workload identifiers."""
    
    # Read the CSV files
    try:
        # Read the detailed performance data
        base_path = '../results/GPT_3_1300M_v5_unaware/'
        perf_df = pd.read_csv(base_path + '2D_Torus.csv')

        # Read the timing data (excluding line_number column)
        timing_df = pd.read_csv(base_path + 'workload_timing_data.csv')
        timing_df = timing_df.drop('line_number', axis=1)  # Remove line_number column
        
        print(f"Performance data: {len(perf_df)} rows")
        print(f"Timing data: {len(timing_df)} rows")
        
        # Merge the dataframes on dp_mp_sp_pp_sharded = workload
        merged_df = pd.merge(
            perf_df, 
            timing_df, 
            left_on='dp_mp_sp_pp_sharded', 
            right_on='workload', 
            how='inner'
        )
        
        print(f"Merged data: {len(merged_df)} rows")
        
        # Drop the duplicate workload column (keep dp_mp_sp_pp_sharded)
        merged_df = merged_df.drop('workload', axis=1)
        
        # Save the merged data
        output_file = base_path + 'merged_workload_data.csv'
        merged_df.to_csv(output_file, index=False)
        
        print(f"\nMerged data saved to: {output_file}")
        
        # Show some statistics
        print(f"\nColumns in merged data: {list(merged_df.columns)}")
        print("\nSample of merged data:")
        print(merged_df[['dp_mp_sp_pp_sharded', 'time_seconds', 'exec_cycles', 'comm_cycles', 'comp_cycles']].head(10))
        
        # Check for unmatched records
        perf_workloads = set(perf_df['dp_mp_sp_pp_sharded'].unique())
        timing_workloads = set(timing_df['workload'].unique())
        
        unmatched_perf = perf_workloads - timing_workloads
        unmatched_timing = timing_workloads - perf_workloads
        
        if unmatched_perf:
            print(f"\nWorkloads in performance data but not in timing data ({len(unmatched_perf)}):")
            for workload in sorted(unmatched_perf):
                print(f"  {workload}")
        
        if unmatched_timing:
            print(f"\nWorkloads in timing data but not in performance data ({len(unmatched_timing)}):")
            for workload in sorted(unmatched_timing):
                print(f"  {workload}")
        
        if not unmatched_perf and not unmatched_timing:
            print("\n✓ All workloads matched successfully!")
        
        return merged_df
        
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    merged_data = merge_csvs()