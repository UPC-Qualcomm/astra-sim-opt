#!/usr/bin/env python3
"""
Visualize comparison of all three ML models.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def create_comparison_plots(csv_file, output_dir):
    """Create comprehensive comparison plots for all models."""
    
    df = pd.read_csv(csv_file)
    
    # Model columns - include previous model
    model_columns = {}
    
    # Add previous model first
    if 'predicted_memory' in df.columns:
        model_columns['Previous Model'] = 'predicted_memory'
    
    # Add new models
    model_columns.update({
        'Stacked Ensemble': 'predicted_stacked',
        'XGBoost': 'predicted_xgboost',
        'Gradient Boosting': 'predicted_gradient_boosting'
    })
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Scatter plots: Predicted vs Actual for all models
    n_models = len(model_columns)
    fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
    
    for idx, (model_name, col) in enumerate(model_columns.items()):
        ax = axes[idx]
        
        valid_mask = df[col].notna()
        valid_df = df[valid_mask]
        
        # Calculate metrics
        errors = valid_df[col] - valid_df['peak_memory']
        mae = np.abs(errors).mean()
        rmse = np.sqrt((errors ** 2).mean())
        r2 = 1 - ((errors ** 2).sum() / ((valid_df['peak_memory'] - valid_df['peak_memory'].mean()) ** 2).sum())
        
        # Scatter plot
        ax.scatter(valid_df['peak_memory'], valid_df[col], alpha=0.6, s=50)
        
        # Perfect prediction line
        min_val = min(valid_df['peak_memory'].min(), valid_df[col].min())
        max_val = max(valid_df['peak_memory'].max(), valid_df[col].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        ax.set_xlabel('Actual Peak Memory (GB)', fontsize=11)
        ax.set_ylabel('Predicted Peak Memory (GB)', fontsize=11)
        ax.set_title(f'{model_name}\nRMSE={rmse:.2f} GB, R²={r2:.4f}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison_scatter.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: model_comparison_scatter.png")
    
    # 2. Error distribution comparison
    fig, axes = plt.subplots(2, n_models, figsize=(6*n_models, 10))
    
    for idx, (model_name, col) in enumerate(model_columns.items()):
        valid_mask = df[col].notna()
        valid_df = df[valid_mask].copy()
        
        errors = valid_df[col] - valid_df['peak_memory']
        abs_errors = np.abs(errors)
        rel_errors = (abs_errors / valid_df['peak_memory'] * 100)
        
        # Absolute error histogram
        ax1 = axes[0, idx]
        ax1.hist(abs_errors, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
        ax1.axvline(abs_errors.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {abs_errors.mean():.1f} GB')
        ax1.set_xlabel('Absolute Error (GB)', fontsize=10)
        ax1.set_ylabel('Frequency', fontsize=10)
        ax1.set_title(f'{model_name}\nAbsolute Error Distribution', fontsize=11, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Relative error histogram
        ax2 = axes[1, idx]
        ax2.hist(rel_errors, bins=30, alpha=0.7, color='coral', edgecolor='black')
        ax2.axvline(rel_errors.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {rel_errors.mean():.1f}%')
        ax2.set_xlabel('Relative Error (%)', fontsize=10)
        ax2.set_ylabel('Frequency', fontsize=10)
        ax2.set_title(f'{model_name}\nRelative Error Distribution', fontsize=11, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison_errors.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: model_comparison_errors.png")
    
    # 3. Box plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Prepare data for box plots
    abs_error_data = []
    rel_error_data = []
    labels = []
    
    for model_name, col in model_columns.items():
        valid_mask = df[col].notna()
        valid_df = df[valid_mask].copy()
        
        errors = valid_df[col] - valid_df['peak_memory']
        abs_errors = np.abs(errors)
        rel_errors = (abs_errors / valid_df['peak_memory'] * 100)
        
        abs_error_data.append(abs_errors)
        rel_error_data.append(rel_errors)
        labels.append(model_name)
    
    # Absolute error box plot
    bp1 = axes[0].boxplot(abs_error_data, labels=labels, patch_artist=True, showmeans=True)
    for patch in bp1['boxes']:
        patch.set_facecolor('lightblue')
    axes[0].set_ylabel('Absolute Error (GB)', fontsize=11)
    axes[0].set_title('Absolute Error Comparison', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Relative error box plot
    bp2 = axes[1].boxplot(rel_error_data, labels=labels, patch_artist=True, showmeans=True)
    for patch in bp2['boxes']:
        patch.set_facecolor('lightcoral')
    axes[1].set_ylabel('Relative Error (%)', fontsize=11)
    axes[1].set_title('Relative Error Comparison', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison_boxplot.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: model_comparison_boxplot.png")
    
    # 4. Metric comparison bar chart
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    metrics_data = {}
    for model_name, col in model_columns.items():
        valid_mask = df[col].notna()
        valid_df = df[valid_mask].copy()
        
        errors = valid_df[col] - valid_df['peak_memory']
        abs_errors = np.abs(errors)
        rel_errors = (abs_errors / valid_df['peak_memory'] * 100)
        
        mae = abs_errors.mean()
        rmse = np.sqrt((errors ** 2).mean())
        mape = rel_errors.mean()
        r2 = 1 - ((errors ** 2).sum() / ((valid_df['peak_memory'] - valid_df['peak_memory'].mean()) ** 2).sum())
        
        metrics_data[model_name] = {'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'R²': r2}
    
    model_names = list(metrics_data.keys())
    
    # MAE
    maes = [metrics_data[m]['MAE'] for m in model_names]
    axes[0, 0].bar(model_names, maes, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    axes[0, 0].set_ylabel('MAE (GB)', fontsize=11)
    axes[0, 0].set_title('Mean Absolute Error', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(maes):
        axes[0, 0].text(i, v + 0.5, f'{v:.2f}', ha='center', fontsize=10, fontweight='bold')
    
    # RMSE
    rmses = [metrics_data[m]['RMSE'] for m in model_names]
    axes[0, 1].bar(model_names, rmses, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    axes[0, 1].set_ylabel('RMSE (GB)', fontsize=11)
    axes[0, 1].set_title('Root Mean Square Error', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(rmses):
        axes[0, 1].text(i, v + 0.5, f'{v:.2f}', ha='center', fontsize=10, fontweight='bold')
    
    # MAPE
    mapes = [metrics_data[m]['MAPE'] for m in model_names]
    axes[1, 0].bar(model_names, mapes, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    axes[1, 0].set_ylabel('MAPE (%)', fontsize=11)
    axes[1, 0].set_title('Mean Absolute Percentage Error', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(mapes):
        axes[1, 0].text(i, v + 0.1, f'{v:.2f}%', ha='center', fontsize=10, fontweight='bold')
    
    # R²
    r2s = [metrics_data[m]['R²'] for m in model_names]
    axes[1, 1].bar(model_names, r2s, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    axes[1, 1].set_ylabel('R² Score', fontsize=11)
    axes[1, 1].set_title('R² Score', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylim([0.8, 1.0])
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(r2s):
        axes[1, 1].text(i, v + 0.005, f'{v:.4f}', ha='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: model_comparison_metrics.png")

def main():
    base_dir = '/media/mohammad/extension/experiments/astra-sim/upc'
    csv_file = os.path.join(base_dir, 'ml_model/evaluation/prediction_evaluation_all_models.csv')
    output_dir = os.path.join(base_dir, 'ml_model/evaluation/comparison_plots')
    
    print("Creating comparison visualizations...")
    create_comparison_plots(csv_file, output_dir)
    print(f"\n✓ All plots saved to: {output_dir}")

if __name__ == '__main__':
    main()
