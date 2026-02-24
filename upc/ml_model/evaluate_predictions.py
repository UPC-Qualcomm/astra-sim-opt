#!/usr/bin/env python3
"""
Evaluate ML model predictions against actual simulator results.
Parses log file with predictions and matches with ground truth CSV.
"""

import pandas as pd
import numpy as np
import re
import argparse
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import os


def parse_log_file(log_path):
    """Extract predictions from log file."""
    predictions = []
    
    with open(log_path, 'r') as f:
        content = f.read()
    
    # Split by the separator pattern
    blocks = re.split(r'={60,}', content)
    
    for block in blocks:
        if '🤖 ML MODEL PREDICTION:' not in block:
            continue
        
        # Extract parallelism configuration
        dp_match = re.search(r'dp=(\d+)', block)
        mp_match = re.search(r'mp=(\d+)', block)
        sp_match = re.search(r'sp=(\d+)', block)
        pp_match = re.search(r'pp=(\d+)', block)
        sharded_match = re.search(r'weight_sharded=(True|False)', block)
        
        # Extract prediction
        pred_match = re.search(r'🤖 ML MODEL PREDICTION:\s+([\d.]+)\s+GB per NPU', block)
        
        if all([dp_match, mp_match, sp_match, pp_match, sharded_match, pred_match]):
            dp = int(dp_match.group(1))
            mp = int(mp_match.group(1))
            sp = int(sp_match.group(1))
            pp = int(pp_match.group(1))
            sharded = 1 if sharded_match.group(1) == 'True' else 0
            predicted_memory = float(pred_match.group(1))
            
            # Create key matching CSV format
            config_key = f"{dp}_{mp}_{sp}_{pp}_{sharded}"
            
            predictions.append({
                'dp_mp_sp_pp_sharded': config_key,
                'dp': dp,
                'mp': mp,
                'sp': sp,
                'pp': pp,
                'sharded': sharded,
                'predicted_memory_gb': predicted_memory
            })
    
    return pd.DataFrame(predictions)


def load_ground_truth(csv_path):
    """Load ground truth from CSV file."""
    df = pd.read_csv(csv_path)
    
    # Ensure column name matches
    if 'dp_mp_so_pp_sharded' in df.columns:
        df = df.rename(columns={'dp_mp_so_pp_sharded': 'dp_mp_sp_pp_sharded'})
    
    return df


def merge_and_evaluate(predictions_df, ground_truth_df):
    """Merge predictions with ground truth and calculate metrics."""
    
    # Merge on configuration key
    merged = pd.merge(
        predictions_df,
        ground_truth_df,
        on='dp_mp_sp_pp_sharded',
        how='inner',
        suffixes=('_pred', '_gt')
    )
    
    if len(merged) == 0:
        print("Warning: No matching configurations found between predictions and ground truth!")
        return None
    
    # Calculate errors
    # Handle different column names for peak memory
    peak_mem_col = 'peak_memory_gb' if 'peak_memory_gb' in merged.columns else 'peak_memory'
    
    y_true = merged[peak_mem_col].values
    y_pred = merged['predicted_memory_gb'].values
    
    # Calculate metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # Calculate percentage errors
    merged['absolute_error_gb'] = np.abs(y_pred - y_true)
    merged['relative_error_pct'] = (merged['absolute_error_gb'] / y_true) * 100
    
    # Mean Absolute Percentage Error (MAPE)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    metrics = {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'mape': mape,
        'n_samples': len(merged)
    }
    
    return merged, metrics


def plot_results(merged_df, output_dir):
    """Generate visualizations."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Handle different column names
    peak_mem_col = 'peak_memory_gb' if 'peak_memory_gb' in merged_df.columns else 'peak_memory'
    
    y_true = merged_df[peak_mem_col].values
    y_pred = merged_df['predicted_memory_gb'].values
    
    # 1. Predicted vs Actual scatter plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Scatter plot
    axes[0].scatter(y_true, y_pred, alpha=0.6, s=50)
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    axes[0].set_xlabel('Actual Peak Memory (GB)', fontsize=12)
    axes[0].set_ylabel('Predicted Peak Memory (GB)', fontsize=12)
    axes[0].set_title('ML Prediction vs Simulator Ground Truth', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Error distribution
    errors = merged_df['absolute_error_gb'].values
    axes[1].hist(errors, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    axes[1].axvline(errors.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {errors.mean():.2f} GB')
    axes[1].axvline(np.median(errors), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(errors):.2f} GB')
    axes[1].set_xlabel('Absolute Error (GB)', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Error Distribution', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'prediction_evaluation.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to {output_dir}/prediction_evaluation.png")


def main():
    parser = argparse.ArgumentParser(description='Evaluate ML predictions against simulator ground truth')
    parser.add_argument('--log_file', type=str, required=True,
                        help='Log file with ML predictions')
    parser.add_argument('--csv_file', type=str, required=True,
                        help='CSV file with ground truth peak memory')
    parser.add_argument('--output_csv', type=str, default='evaluation_results.csv',
                        help='Output CSV file with combined results')
    parser.add_argument('--output_dir', type=str, default='ml_model/evaluation',
                        help='Output directory for plots and results')
    
    args = parser.parse_args()
    
    # Get base directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_dir, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*60)
    print("ML Model Evaluation: Predictions vs Ground Truth")
    print("="*60)
    
    # Parse log file
    print(f"\nParsing predictions from: {args.log_file}")
    predictions_df = parse_log_file(args.log_file)
    print(f"Found {len(predictions_df)} predictions")
    
    if len(predictions_df) == 0:
        print("Error: No predictions found in log file!")
        return 1
    
    # Load ground truth
    print(f"\nLoading ground truth from: {args.csv_file}")
    ground_truth_df = load_ground_truth(args.csv_file)
    print(f"Found {len(ground_truth_df)} ground truth samples")
    
    # Merge and evaluate
    print("\nMatching predictions with ground truth...")
    result = merge_and_evaluate(predictions_df, ground_truth_df)
    
    if result is None:
        return 1
    
    merged_df, metrics = result
    
    # Print metrics
    print("\n" + "="*60)
    print("EVALUATION METRICS")
    print("="*60)
    print(f"Number of matched samples: {metrics['n_samples']}")
    print(f"Mean Absolute Error (MAE): {metrics['mae']:.3f} GB")
    print(f"Root Mean Squared Error (RMSE): {metrics['rmse']:.3f} GB")
    print(f"R² Score: {metrics['r2']:.4f}")
    print(f"Mean Absolute Percentage Error (MAPE): {metrics['mape']:.2f}%")
    print("="*60)
    
    # Print sample comparisons
    print("\nSample Predictions vs Ground Truth:")
    print("-"*60)
    peak_mem_col = 'peak_memory_gb' if 'peak_memory_gb' in merged_df.columns else 'peak_memory'
    sample_df = merged_df[['dp_mp_sp_pp_sharded', peak_mem_col, 'predicted_memory_gb', 
                            'absolute_error_gb', 'relative_error_pct']].head(10)
    print(sample_df.to_string(index=False))
    
    # Save results - only 3 columns as requested
    output_csv_path = os.path.join(output_dir, args.output_csv)
    peak_mem_col = 'peak_memory_gb' if 'peak_memory_gb' in merged_df.columns else 'peak_memory'
    
    # Create simplified output with only 3 columns
    output_df = merged_df[['dp_mp_sp_pp_sharded', peak_mem_col, 'predicted_memory_gb']].copy()
    output_df.columns = ['dp_mp_sp_pp_sharded', 'peak_memory', 'predicted_memory']
    output_df.to_csv(output_csv_path, index=False)
    print(f"\nDetailed results saved to: {output_csv_path}")
    
    # Also save full detailed results for reference
    full_output_path = os.path.join(output_dir, 'full_' + args.output_csv)
    merged_df.to_csv(full_output_path, index=False)
    print(f"Full detailed results saved to: {full_output_path}")
    
    # Generate plots
    print("\nGenerating visualizations...")
    plot_results(merged_df, output_dir)
    
    # Save metrics to text file
    metrics_path = os.path.join(output_dir, 'metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write("ML Model Evaluation Metrics\n")
        f.write("="*60 + "\n")
        f.write(f"Number of samples: {metrics['n_samples']}\n")
        f.write(f"Mean Absolute Error (MAE): {metrics['mae']:.3f} GB\n")
        f.write(f"Root Mean Squared Error (RMSE): {metrics['rmse']:.3f} GB\n")
        f.write(f"R² Score: {metrics['r2']:.4f}\n")
        f.write(f"Mean Absolute Percentage Error (MAPE): {metrics['mape']:.2f}%\n")
        f.write("="*60 + "\n")
    
    print(f"Metrics saved to: {metrics_path}")
    
    print("\n" + "="*60)
    print("Evaluation completed successfully!")
    print("="*60)
    
    return 0


if __name__ == '__main__':
    exit(main())
