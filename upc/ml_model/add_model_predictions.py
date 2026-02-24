#!/usr/bin/env python3
"""
Add predictions from all three models (Stacked, XGBoost, Gradient Boosting) to evaluation CSV.
"""

import pandas as pd
import re
import os

def parse_log_predictions(log_file):
    """
    Parse log file to extract predictions with their parallelism configurations.
    Returns dict: {config_string: prediction_value}
    """
    predictions = {}
    
    with open(log_file, 'r') as f:
        content = f.read()
    
    # Pattern to match parallelism config and prediction
    # Look for blocks with parallelism info followed by ML prediction
    pattern = r'Parallelism: dp=(\d+), mp=(\d+), sp=(\d+), pp=(\d+), weight_sharded=(True|False).*?🤖 ML MODEL PREDICTION: ([\d.]+) GB per NPU'
    
    matches = re.findall(pattern, content, re.DOTALL)
    
    for match in matches:
        dp, mp, sp, pp, weight_sharded_str, prediction = match
        # Convert weight_sharded from True/False to 1/0
        weight_sharded = '1' if weight_sharded_str == 'True' else '0'
        
        # Create config string matching CSV format: dp_mp_sp_pp_sharded
        config_key = f"{dp}_{mp}_{sp}_{pp}_{weight_sharded}"
        predictions[config_key] = float(prediction)
    
    return predictions

def add_model_predictions(csv_file, log_files, output_csv):
    """
    Add predictions from multiple models to the evaluation CSV.
    
    Args:
        csv_file: Path to existing prediction_evaluation.csv
        log_files: Dict of {model_name: log_file_path}
        output_csv: Path to output CSV with added columns
    """
    # Read existing CSV
    df = pd.read_csv(csv_file)
    
    print(f"Original CSV has {len(df)} rows with columns: {list(df.columns)}")
    
    # Parse each log file and add predictions
    for model_name, log_file in log_files.items():
        if not os.path.exists(log_file):
            print(f"WARNING: Log file not found: {log_file}")
            continue
        
        print(f"\nProcessing {model_name} from {log_file}...")
        predictions = parse_log_predictions(log_file)
        print(f"  Found {len(predictions)} predictions")
        
        # Add column for this model
        column_name = f'predicted_{model_name}'
        df[column_name] = df['dp_mp_sp_pp_sharded'].map(predictions)
        
        # Report matches
        matched = df[column_name].notna().sum()
        print(f"  Matched {matched}/{len(df)} configurations")
        
        # Show any missing configurations
        missing = df[df[column_name].isna()]['dp_mp_sp_pp_sharded'].tolist()
        if missing:
            print(f"  Missing predictions for: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    
    # Save updated CSV
    df.to_csv(output_csv, index=False)
    print(f"\n✓ Updated CSV saved to {output_csv}")
    print(f"  Columns: {list(df.columns)}")
    
    return df

def evaluate_model_performance(df):
    """
    Evaluate which model performed best.
    """
    print("\n" + "="*80)
    print("MODEL PERFORMANCE EVALUATION")
    print("="*80)
    
    # Model columns - include the original predicted_memory as "previous"
    model_columns = []
    
    # Add previous model if it exists
    if 'predicted_memory' in df.columns:
        model_columns.append('predicted_memory')
    
    # Add new models
    for col in df.columns:
        if col.startswith('predicted_') and col != 'predicted_memory':
            model_columns.append(col)
    
    if not model_columns:
        print("No model prediction columns found!")
        return
    
    results = {}
    
    for col in model_columns:
        # Determine model name
        if col == 'predicted_memory':
            model_name = 'previous_model'
        else:
            model_name = col.replace('predicted_', '')
        
        # Calculate metrics (only for rows where prediction exists)
        valid_mask = df[col].notna()
        valid_df = df[valid_mask].copy()
        
        if len(valid_df) == 0:
            print(f"\n{model_name.upper()}: No valid predictions")
            continue
        
        # Calculate errors
        valid_df['error'] = valid_df[col] - valid_df['peak_memory']
        valid_df['abs_error'] = valid_df['error'].abs()
        valid_df['rel_error'] = (valid_df['abs_error'] / valid_df['peak_memory'] * 100)
        valid_df['squared_error'] = valid_df['error'] ** 2
        
        # Metrics
        mae = valid_df['abs_error'].mean()
        rmse = (valid_df['squared_error'].mean()) ** 0.5
        mape = valid_df['rel_error'].mean()
        r2 = 1 - (valid_df['squared_error'].sum() / ((valid_df['peak_memory'] - valid_df['peak_memory'].mean()) ** 2).sum())
        max_error = valid_df['abs_error'].max()
        
        results[model_name] = {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'r2': r2,
            'max_error': max_error,
            'n_samples': len(valid_df)
        }
        
        print(f"\n{model_name.upper()}:")
        print(f"  Samples: {len(valid_df)}")
        print(f"  MAE: {mae:.2f} GB")
        print(f"  RMSE: {rmse:.2f} GB")
        print(f"  MAPE: {mape:.2f}%")
        print(f"  R²: {r2:.4f}")
        print(f"  Max Error: {max_error:.2f} GB")
        
        # Find worst predictions
        worst_5 = valid_df.nlargest(5, 'abs_error')[['dp_mp_sp_pp_sharded', 'peak_memory', col, 'abs_error', 'rel_error']]
        print(f"  Worst 5 predictions:")
        for idx, row in worst_5.iterrows():
            print(f"    {row['dp_mp_sp_pp_sharded']}: actual={row['peak_memory']:.1f}, pred={row[col]:.1f}, error={row['abs_error']:.1f} ({row['rel_error']:.1f}%)")
    
    # Determine best model
    if results:
        print("\n" + "="*80)
        print("OVERALL RANKING (by RMSE):")
        print("="*80)
        
        sorted_models = sorted(results.items(), key=lambda x: x[1]['rmse'])
        
        for rank, (model_name, metrics) in enumerate(sorted_models, 1):
            print(f"\n{rank}. {model_name.upper()}")
            print(f"   RMSE: {metrics['rmse']:.2f} GB")
            print(f"   MAE: {metrics['mae']:.2f} GB")
            print(f"   MAPE: {metrics['mape']:.2f}%")
            print(f"   R²: {metrics['r2']:.4f}")
        
        best_model = sorted_models[0][0]
        print(f"\n🏆 BEST MODEL: {best_model.upper()}")
        print(f"   (Lowest RMSE: {sorted_models[0][1]['rmse']:.2f} GB)")

def main():
    # File paths
    base_dir = '/media/mohammad/extension/experiments/astra-sim/upc'
    csv_file = os.path.join(base_dir, 'ml_model/evaluation/prediction_evaluation_70B.csv')
    
    log_files = {
        #'old_best': os.path.join(base_dir, 'ml_old_best_70B_mod.txt'),
        'stacked': os.path.join(base_dir, 'ml_stacked_70B_mod_log.txt'),
        'xgboost': os.path.join(base_dir, 'ml_xgboost_70B_mod_log.txt'),
        'lightgbm': os.path.join(base_dir, 'ml_lightgbm_70B_mod_log.txt'),
        'mlp': os.path.join(base_dir, 'ml_mlp_70B_mod_log.txt'),
        'random_forest': os.path.join(base_dir, 'ml_random_forest_70B_mod_log.txt'),
        'ridge_regression': os.path.join(base_dir, 'ml_ridge_regression_70B_mod_log.txt'),
        'gradient_boosting': os.path.join(base_dir, 'ml_gradient_boosting_70B_mod_log.txt')
    }
    
    output_csv = os.path.join(base_dir, 'ml_model/evaluation/prediction_evaluation_all_models.csv')
    
    # Add predictions from all models
    df = add_model_predictions(csv_file, log_files, output_csv)
    
    # Evaluate performance
    evaluate_model_performance(df)
    
    print(f"\n{'='*80}")
    print(f"✓ Complete! Results saved to:")
    print(f"  {output_csv}")
    print(f"{'='*80}")

if __name__ == '__main__':
    main()
