#!/usr/bin/env python3
"""
Train ML model to predict AstraSim peak memory usage.
"""

import pandas as pd
import numpy as np
import argparse
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

def load_and_prepare_data(csv_path):
    """Load training data and prepare features."""
    df = pd.read_csv(csv_path)
    
    # Feature columns
    feature_cols = [
        'din', 'dmodel', 'dff', 'batch', 'micro_batch', 'seq', 'head', 'num_stacks',
        'dp', 'mp', 'sp', 'pp', 'fsdp', 'num_npus'
    ]
    
    # Target column
    target_col = 'avg_peak_memory_gb'
    
    X = df[feature_cols].copy()  # Create explicit copy to avoid warnings
    y = df[target_col]
    
    # Create derived features
    X.loc[:, 'total_params_estimate'] = (
        X['dmodel'] * X['din'] + 
        (X['num_stacks'] / X['pp']) * (12 * X['dmodel'] * X['dmodel'] + 13 * X['dmodel']) +
        2 * X['dmodel']
    )
    
    X.loc[:, 'sharding_factor'] = X['mp'] * (1 + X['fsdp'] * X['dp'])
    X.loc[:, 'micro_batch_per_npu'] = X['micro_batch'] / X['dp']
    X.loc[:, 'activation_size_estimate'] = X['micro_batch_per_npu'] * X['seq'] * X['dmodel']
    X.loc[:, 'params_per_npu'] = X['total_params_estimate'] / X['sharding_factor']
    X.loc[:, 'params_size_gb'] = X['params_per_npu'] * 4 / (1024**3)  # FP32
    
    # Log transforms for skewed features
    X.loc[:, 'log_din'] = np.log1p(X['din'])
    X.loc[:, 'log_dmodel'] = np.log1p(X['dmodel'])
    X.loc[:, 'log_dff'] = np.log1p(X['dff'])
    X.loc[:, 'log_batch'] = np.log1p(X['batch'])
    X.loc[:, 'log_seq'] = np.log1p(X['seq'])
    X.loc[:, 'log_total_params'] = np.log1p(X['total_params_estimate'])
    X.loc[:, 'log_params_per_npu'] = np.log1p(X['params_per_npu'])
    
    # Advanced domain-specific features
    # Memory pressure: activation memory per NPU
    X.loc[:, 'memory_pressure_ratio'] = (X['micro_batch_per_npu'] * X['seq'] * X['dmodel']) / X['num_npus']
    X.loc[:, 'log_memory_pressure'] = np.log1p(X['memory_pressure_ratio'])
    
    # Communication overhead (non-linear scaling)
    X.loc[:, 'comm_overhead_sp'] = np.log1p(X['sp']) * X['sp']
    X.loc[:, 'comm_overhead_pp'] = np.log1p(X['pp']) * X['pp']
    X.loc[:, 'total_comm_overhead'] = X['comm_overhead_sp'] + X['comm_overhead_pp']
    
    # Parallelism efficiency factors
    X.loc[:, 'parallelism_product'] = X['dp'] * X['mp'] * X['sp'] * X['pp']
    X.loc[:, 'parallelism_efficiency'] = 1.0 / np.sqrt(X['parallelism_product'])
    
    # Layer distribution per pipeline stage
    X.loc[:, 'layers_per_stage'] = X['num_stacks'] / X['pp']
    X.loc[:, 'memory_per_layer'] = X['params_per_npu'] / X['layers_per_stage']
    
    # Activation recomputation factor (pipeline depth)
    X.loc[:, 'pipeline_depth_factor'] = X['pp'] * X['layers_per_stage']
    
    # Batch efficiency
    X.loc[:, 'batch_efficiency'] = X['batch'] / (X['micro_batch'] * X['dp'])
    X.loc[:, 'tokens_per_npu'] = X['micro_batch_per_npu'] * X['seq']
    
    # Interaction terms for extreme configurations
    X.loc[:, 'sp_pp_interaction'] = X['sp'] * X['pp']
    X.loc[:, 'dp_mp_interaction'] = X['dp'] * X['mp']
    
    # Memory scaling factors
    X.loc[:, 'model_size_ratio'] = X['dmodel'] / X['din']
    X.loc[:, 'ff_expansion_ratio'] = X['dff'] / X['dmodel']
    
    return X, y, df

def train_multiple_models(X_train, X_test, y_train, y_test):
    """Train multiple models and compare performance."""
    
    # Base models with optimized hyperparameters
    rf = RandomForestRegressor(
        n_estimators=300, 
        max_depth=12,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42, 
        n_jobs=-1
    )
    
    gb = GradientBoostingRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        min_samples_split=4,
        min_samples_leaf=2,
        subsample=0.85,
        random_state=42
    )
    
    xgb = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.8,
        reg_alpha=0.1,  # L1 regularization
        reg_lambda=1.0,  # L2 regularization
        random_state=42,
        n_jobs=-1
    )
    
    lgbm = LGBMRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    
    # Stacked ensemble
    stacked = StackingRegressor(
        estimators=[
            ('rf', rf),
            ('gb', gb),
            ('xgb', xgb),
            ('lgbm', lgbm)
        ],
        final_estimator=Ridge(alpha=1.0),
        cv=5,
        n_jobs=-1
    )
    
    models = {
        'Random Forest': rf,
        'Gradient Boosting': gb,
        'XGBoost': xgb,
        'LightGBM': lgbm,
        'Stacked Ensemble': stacked,
        'Ridge Regression': Ridge(alpha=10.0),
        'MLP': MLPRegressor(
            hidden_layer_sizes=(128, 64, 32),
            max_iter=1000,
            alpha=0.01,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42
        )
    }
    
    results = {}
    trained_models = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Cross-validation on training set
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, 
                                     scoring='neg_mean_squared_error', n_jobs=-1)
        cv_rmse = np.sqrt(-cv_scores.mean())
        cv_rmse_std = np.sqrt(cv_scores.std())
        
        model.fit(X_train, y_train)
        
        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Metrics
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        results[name] = {
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_rmse': np.sqrt(train_mse),
            'test_rmse': np.sqrt(test_mse),
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'cv_rmse': cv_rmse,
            'cv_rmse_std': cv_rmse_std,
            'predictions': y_test_pred,
            'overfitting_gap': np.sqrt(test_mse) - np.sqrt(train_mse)  # How much worse on test
        }
        
        trained_models[name] = model
        
        print(f"  CV RMSE: {cv_rmse:.3f} ± {cv_rmse_std:.3f} GB")
        print(f"  Train RMSE: {np.sqrt(train_mse):.3f} GB")
        print(f"  Test RMSE: {np.sqrt(test_mse):.3f} GB")
        print(f"  Overfitting Gap: {results[name]['overfitting_gap']:.3f} GB")
        print(f"  Train MAE: {train_mae:.3f} GB")
        print(f"  Test MAE: {test_mae:.3f} GB")
        print(f"  Train R²: {train_r2:.4f}")
        print(f"  Test R²: {test_r2:.4f}")
    
    return results, trained_models

def plot_results(results, y_test, output_dir):
    """Generate visualizations of model performance."""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Compare model performance
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    models = list(results.keys())
    test_rmse = [results[m]['test_rmse'] for m in models]
    test_mae = [results[m]['test_mae'] for m in models]
    test_r2 = [results[m]['test_r2'] for m in models]
    
    # RMSE comparison
    axes[0, 0].bar(models, test_rmse, color='skyblue')
    axes[0, 0].set_ylabel('RMSE (GB)')
    axes[0, 0].set_title('Test RMSE Comparison')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # MAE comparison
    axes[0, 1].bar(models, test_mae, color='lightcoral')
    axes[0, 1].set_ylabel('MAE (GB)')
    axes[0, 1].set_title('Test MAE Comparison')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # R² comparison
    axes[1, 0].bar(models, test_r2, color='lightgreen')
    axes[1, 0].set_ylabel('R² Score')
    axes[1, 0].set_title('Test R² Comparison')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].set_ylim([0, 1])
    
    # Hide unused subplot
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Prediction vs Actual for each model
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, (name, result) in enumerate(results.items()):
        if idx >= len(axes):
            break
        
        ax = axes[idx]
        y_pred = result['predictions']
        
        ax.scatter(y_test, y_pred, alpha=0.5)
        
        # Perfect prediction line
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        
        ax.set_xlabel('Actual Peak Memory (GB)')
        ax.set_ylabel('Predicted Peak Memory (GB)')
        ax.set_title(f'{name}\nR²={result["test_r2"]:.4f}, RMSE={result["test_rmse"]:.2f}')
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(len(results), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'predictions_vs_actual.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nPlots saved to {output_dir}/")

def analyze_feature_importance(model, feature_names, output_dir):
    """Analyze and plot feature importance for tree-based models."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(12, 8))
        plt.title('Feature Importances')
        plt.bar(range(min(20, len(importances))), importances[indices[:20]])
        plt.xticks(range(min(20, len(importances))), 
                   [feature_names[i] for i in indices[:20]], 
                   rotation=45, ha='right')
        plt.ylabel('Importance')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print("\nTop 10 Most Important Features:")
        for i in range(min(10, len(importances))):
            print(f"  {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")

def main():
    parser = argparse.ArgumentParser(description='Train ML model for AstraSim peak memory prediction')
    parser.add_argument('--input_csv', type=str, default='ml_model/training_data.csv',
                        help='Input CSV file with training data')
    parser.add_argument('--output_dir', type=str, default='ml_model/trained_models',
                        help='Output directory for trained models and plots')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Fraction of data to use for testing')
    
    args = parser.parse_args()
    
    # Get base directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(base_dir, args.input_csv)
    output_dir = os.path.join(base_dir, args.output_dir)
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading and preparing data...")
    X, y, df = load_and_prepare_data(input_path)
    
    print(f"\nDataset shape: {X.shape}")
    print(f"Target range: {y.min():.2f} - {y.max():.2f} GB")
    print(f"Target mean: {y.mean():.2f} GB")
    print(f"Target std: {y.std():.2f} GB")
    
    # Show data distribution by num_npus
    print("\nData distribution by num_npus:")
    npu_counts = X['num_npus'].value_counts().sort_index()
    for npu, count in npu_counts.items():
        print(f"  {npu} NPUs: {count} samples ({count/len(X)*100:.1f}%)")
    
    # Check if dataset is too small
    if len(X) < 50:
        print(f"\n⚠️  WARNING: Dataset is very small ({len(X)} samples)!")
        print("   ML models will likely overfit. Collect more data for reliable predictions.")
        print("   Recommended: at least 100-200 samples for basic models, 1000+ for production.")
    
    # Split data with stratification by num_npus to keep proportional distribution
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=X['num_npus']
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame to preserve column names
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns, index=X_test.index)
    
    print(f"\nTrain set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Train models
    results, trained_models = train_multiple_models(X_train_scaled, X_test_scaled, y_train, y_test)
    
    # Find best model
    best_model_name = min(results.keys(), key=lambda k: results[k]['test_rmse'])
    best_model = trained_models[best_model_name]
    
    print(f"\n{'='*60}")
    print(f"Best Model: {best_model_name}")
    print(f"Test RMSE: {results[best_model_name]['test_rmse']:.3f} GB")
    print(f"Test MAE: {results[best_model_name]['test_mae']:.3f} GB")
    print(f"Test R²: {results[best_model_name]['test_r2']:.4f}")
    print(f"{'='*60}")
    
    # Save ALL trained models
    print(f"\nSaving all trained models to {output_dir}/...")
    for model_name, model in trained_models.items():
        # Create safe filename
        safe_name = model_name.lower().replace(' ', '_').replace('-', '_')
        model_path = os.path.join(output_dir, f'{safe_name}_model.pkl')
        joblib.dump(model, model_path)
        print(f"  ✓ {model_name}: {safe_name}_model.pkl")
    
    # Save best model as 'best_model.pkl' for backward compatibility
    joblib.dump(best_model, os.path.join(output_dir, 'best_model.pkl'))
    print(f"  ✓ Best model (link): best_model.pkl -> {best_model_name}")
    
    # Save scaler (shared by all models)
    joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))
    print(f"  ✓ Feature scaler: scaler.pkl")
    
    # Save feature names
    with open(os.path.join(output_dir, 'feature_names.txt'), 'w') as f:
        for col in X.columns:
            f.write(f"{col}\n")
    print(f"  ✓ Feature names: feature_names.txt")
    
    # Generate plots
    plot_results(results, y_test, output_dir)
    
    # Feature importance for best model
    analyze_feature_importance(best_model, X.columns.tolist(), output_dir)
    
    # Save results summary
    results_df = pd.DataFrame({
        'Model': list(results.keys()),
        'CV_RMSE': [results[m]['cv_rmse'] for m in results.keys()],
        'CV_RMSE_Std': [results[m]['cv_rmse_std'] for m in results.keys()],
        'Train_RMSE': [results[m]['train_rmse'] for m in results.keys()],
        'Test_RMSE': [results[m]['test_rmse'] for m in results.keys()],
        'Overfitting_Gap': [results[m]['overfitting_gap'] for m in results.keys()],
        'Test_MAE': [results[m]['test_mae'] for m in results.keys()],
        'Test_R2': [results[m]['test_r2'] for m in results.keys()],
        'Train_R2': [results[m]['train_r2'] for m in results.keys()]
    })
    results_df.to_csv(os.path.join(output_dir, 'model_results.csv'), index=False)
    print(f"\nResults summary saved to {output_dir}/model_results.csv")

if __name__ == '__main__':
    main()
