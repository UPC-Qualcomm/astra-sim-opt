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
from catboost import CatBoostRegressor
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

BASE_FEATURE_COLS = [
    'din', 'dmodel', 'dff', 'batch', 'micro_batch', 'seq', 'head', 'num_stacks',
    'dp', 'mp', 'sp', 'pp', 'fsdp', 'num_npus'
]
TARGET_COL = 'avg_peak_memory_gb'
VALID_NPUS = {16, 32, 64, 128, 256, 512, 1024, 2048}

def load_and_clean_dataframe(csv_path_or_dir):
    """Load data from file/dir and apply NaN + odd-num_npus cleanup."""

    # Determine if input is a file or directory
    if os.path.isdir(csv_path_or_dir):
        data_dir = csv_path_or_dir
        print(f"\n📁 Loading data from directory: {data_dir}")
        
        # Find all CSV files in the directory
        csv_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.csv')])
        
        if not csv_files:
            raise ValueError(f"No CSV files found in {data_dir}")
        
        print(f"Found {len(csv_files)} data file(s):")
        dfs = []
        
        for csv_file in csv_files:
            csv_path = os.path.join(data_dir, csv_file)
            print(f"  - {csv_file}", end="")
            df_chunk = pd.read_csv(csv_path)
            print(f" ({len(df_chunk)} rows)")
            dfs.append(df_chunk)
        
        # Combine all data files
        df = pd.concat(dfs, ignore_index=True)
        print(f"\n✓ Combined {len(csv_files)} files → {len(df)} total rows")
    else:
        # Single file
        print(f"\n📄 Loading data from file: {csv_path_or_dir}")
        df = pd.read_csv(csv_path_or_dir)
    
    # Report initial state
    initial_rows = len(df)
    print(f"\n📊 Initial dataset: {initial_rows} rows")
    
    # Check for missing values
    missing_counts = df[BASE_FEATURE_COLS + [TARGET_COL]].isnull().sum()
    if missing_counts.sum() > 0:
        print(f"\n⚠️  Missing values found:")
        for col, count in missing_counts[missing_counts > 0].items():
            print(f"   {col}: {count} NaN values")
        
        # Remove rows with NaN values
        df_cleaned = df.dropna(subset=BASE_FEATURE_COLS + [TARGET_COL])
        removed_nan = initial_rows - len(df_cleaned)
        print(f"   → Removed {removed_nan} rows with NaN values")
        df = df_cleaned
    
    # Identify odd num_npus values (not powers of 2)
    odd_npu_mask = ~df['num_npus'].isin(VALID_NPUS)
    
    if odd_npu_mask.sum() > 0:
        print(f"\n⚠️  Found {odd_npu_mask.sum()} rows with odd/non-standard num_npus values:")
        odd_npu_values = df[odd_npu_mask]['num_npus'].unique()
        print(f"   Values: {sorted(odd_npu_values)}")
        
        # Separate odd and even (valid) records
        df_valid = df[~odd_npu_mask].copy()
        df_odd = df[odd_npu_mask].copy()
        
        print(f"   → Keeping {len(df_valid)} rows with valid num_npus")
        print(f"   → Replicating {len(df_odd)} odd-npu rows to nearest valid num_npus")
        
        # Replicate odd-npu records to nearest valid NPU counts
        replicated_dfs = [df_valid]
        for _, row in df_odd.iterrows():
            odd_npu = row['num_npus']
            # Find nearest valid NPU count
            nearest_npu = min(VALID_NPUS, key=lambda x: abs(x - odd_npu))
            row_copy = row.copy()
            row_copy['num_npus'] = nearest_npu
            replicated_dfs.append(pd.DataFrame([row_copy]))
        
        df_replicated = pd.concat(replicated_dfs, ignore_index=True)
        
        for odd_val in sorted(odd_npu_values):
            nearest = min(VALID_NPUS, key=lambda x: abs(x - odd_val))
            count = (df_odd['num_npus'] == odd_val).sum()
            print(f"     {int(odd_val)} → {int(nearest)} ({count} records)")
        
        df = df_replicated
        print(f"   → Total after replication: {len(df)} rows")

    return df

def filter_outliers_by_peak_per_npu(df, threshold):
    """Keep rows where peak_memory/num_npus <= threshold."""
    before = len(df)
    ratio = df[TARGET_COL] / df['num_npus']
    filtered = df[ratio <= threshold].copy()
    removed = before - len(filtered)
    print(f"\n🧹 Outlier filtering enabled")
    print(f"   Condition: {TARGET_COL}/num_npus <= {threshold:.6f}")
    print(f"   Kept: {len(filtered)} / {before} rows ({len(filtered)/before*100:.2f}%)")
    print(f"   Removed outliers: {removed}")
    return filtered

def build_features_from_dataframe(df):
    """Build training features and log-transformed target from cleaned dataframe."""
    X = df[BASE_FEATURE_COLS].copy()  # Create explicit copy to avoid warnings
    y = df[TARGET_COL]
    
    # Apply log1p transform to target (log(1 + x) to handle zeros)
    y = np.log1p(y)
    
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

    # Domain-informed memory/communication proxies
    X.loc[:, 'tokens_global'] = X['batch'] * X['seq']
    X.loc[:, 'tokens_per_dp_rank'] = X['tokens_global'] / X['dp']
    X.loc[:, 'attention_matrix_proxy'] = X['batch'] * X['head'] * (X['seq'] ** 2)

    # Approximate transformer parameter components
    X.loc[:, 'attn_params_proxy'] = 4 * X['dmodel'] * X['dmodel'] * X['num_stacks']
    X.loc[:, 'ffn_params_proxy'] = 8 * X['dmodel'] * X['dmodel'] * X['num_stacks']
    X.loc[:, 'embedding_params_proxy'] = X['din'] * X['dmodel']
    X.loc[:, 'model_params_proxy'] = (
        X['attn_params_proxy'] + X['ffn_params_proxy'] + X['embedding_params_proxy']
    )

    # Parallelism-aware per-rank memory pressure
    X.loc[:, 'per_npu_params_dp_mp_pp'] = X['model_params_proxy'] / (X['dp'] * X['mp'] * X['pp'])
    X.loc[:, 'per_npu_activation_dp'] = X['activation_size_estimate'] / X['dp']
    X.loc[:, 'optimizer_state_proxy'] = 2.0 * X['per_npu_params_dp_mp_pp']

    # Communication/buffering pressure
    X.loc[:, 'allreduce_volume_proxy'] = X['model_params_proxy'] / X['dp']
    X.loc[:, 'pipeline_buffer_proxy'] = (X['micro_batch'] * X['seq'] * X['dmodel']) / X['pp']
    X.loc[:, 'comm_compute_pressure'] = X['allreduce_volume_proxy'] / (X['tokens_global'] + 1.0)

    # Regime and interaction features
    X.loc[:, 'log_num_npus'] = np.log1p(X['num_npus'])
    X.loc[:, 'large_cluster_flag'] = (X['num_npus'] >= 256).astype(int)
    X.loc[:, 'seq_per_pipeline_stage'] = X['seq'] / X['pp']
    X.loc[:, 'batch_seq_per_dp'] = (X['batch'] * X['seq']) / X['dp']
    X.loc[:, 'parallelism_imbalance'] = X[['dp', 'mp', 'sp', 'pp']].max(axis=1) / X[['dp', 'mp', 'sp', 'pp']].min(axis=1)

    # Log transforms for new skewed proxies
    X.loc[:, 'log_tokens_global'] = np.log1p(X['tokens_global'])
    X.loc[:, 'log_attention_matrix_proxy'] = np.log1p(X['attention_matrix_proxy'])
    X.loc[:, 'log_model_params_proxy'] = np.log1p(X['model_params_proxy'])
    X.loc[:, 'log_per_npu_params_dp_mp_pp'] = np.log1p(X['per_npu_params_dp_mp_pp'])
    X.loc[:, 'log_per_npu_activation_dp'] = np.log1p(X['per_npu_activation_dp'])
    X.loc[:, 'log_allreduce_volume_proxy'] = np.log1p(X['allreduce_volume_proxy'])
    X.loc[:, 'log_pipeline_buffer_proxy'] = np.log1p(X['pipeline_buffer_proxy'])
    X.loc[:, 'log_comm_compute_pressure'] = np.log1p(X['comm_compute_pressure'])
    X.loc[:, 'log_batch_seq_per_dp'] = np.log1p(X['batch_seq_per_dp'])

    return X, y

def load_and_prepare_data(csv_path_or_dir, peak_per_npu_threshold=None):
    """Load training data, optionally filter outliers, and prepare features."""
    df = load_and_clean_dataframe(csv_path_or_dir)

    if peak_per_npu_threshold is not None:
        df = filter_outliers_by_peak_per_npu(df, peak_per_npu_threshold)

    X, y = build_features_from_dataframe(df)

    return X, y, df

def tune_peak_per_npu_threshold(csv_path_or_dir, test_size=0.2, min_keep_ratio=0.85):
    """Tune outlier threshold using validation RMSE in log-space on a proxy model."""
    print("\n🔍 Tuning peak_per_npu outlier threshold...")
    df = load_and_clean_dataframe(csv_path_or_dir)

    ratio = df[TARGET_COL] / df['num_npus']
    quantiles = [0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.985, 0.99, 0.995, 0.999]
    candidates = sorted(set([float(np.quantile(ratio, q)) for q in quantiles] + [float(ratio.max())]))

    tuning_rows = []
    for threshold in candidates:
        df_filtered = df[ratio <= threshold].copy()
        keep_ratio = len(df_filtered) / len(df)

        if keep_ratio < min_keep_ratio:
            continue
        if len(df_filtered) < 500:
            continue
        npu_counts = df_filtered['num_npus'].value_counts()
        if (npu_counts < 2).any():
            continue

        X, y = build_features_from_dataframe(df_filtered)

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=X['num_npus']
            )
        except ValueError:
            continue

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns, index=X_test.index)

        proxy_model = GradientBoostingRegressor(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.05,
            random_state=42
        )
        proxy_model.fit(X_train_scaled, y_train)
        y_pred = proxy_model.predict(X_test_scaled)
        rmse_log = np.sqrt(mean_squared_error(y_test, y_pred))

        tuning_rows.append({
            'threshold': threshold,
            'rmse_log': rmse_log,
            'kept_rows': len(df_filtered),
            'keep_ratio': keep_ratio,
        })

    if not tuning_rows:
        fallback = float(np.quantile(ratio, 0.995))
        print(f"   No valid threshold candidate satisfied constraints. Using fallback: {fallback:.6f}")
        return fallback

    tuning_df = pd.DataFrame(tuning_rows).sort_values('rmse_log').reset_index(drop=True)
    best = tuning_df.iloc[0]

    print("\n   Threshold tuning leaderboard (top 5):")
    top_n = min(5, len(tuning_df))
    for i in range(top_n):
        row = tuning_df.iloc[i]
        print(
            f"   {i+1}. thr={row['threshold']:.6f} | rmse_log={row['rmse_log']:.4f} | "
            f"kept={int(row['kept_rows'])} ({row['keep_ratio']*100:.2f}%)"
        )

    print(f"\n   ✅ Selected threshold: {best['threshold']:.6f} (proxy rmse_log={best['rmse_log']:.4f})")
    return float(best['threshold'])

def train_multiple_models(X_train, X_test, y_train, y_test):
    """Train multiple models and compare performance."""
    y_train_actual = np.expm1(y_train)
    y_test_actual = np.expm1(y_test)
    
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
    
    # Hyperparameter optimization for XGBoost
    print("\nOptimizing XGBoost hyperparameters...")
    xgb_base = XGBRegressor(random_state=42, n_jobs=-1)
    xgb_param_dist = {
        'n_estimators': randint(200, 500),
        'max_depth': randint(4, 8),
        'learning_rate': uniform(0.01, 0.1),
        'subsample': uniform(0.7, 0.25),
        'colsample_bytree': uniform(0.7, 0.25),
        'reg_alpha': uniform(0.0, 0.5),
        'reg_lambda': uniform(0.5, 1.5),
        'min_child_weight': randint(1, 5)
    }
    
    xgb_search = RandomizedSearchCV(
        xgb_base,
        xgb_param_dist,
        n_iter=20,
        cv=3,
        scoring='neg_mean_squared_error',
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    xgb_search.fit(X_train, y_train)
    xgb = xgb_search.best_estimator_
    print(f"  Best XGBoost params: {xgb_search.best_params_}")
    print(f"  Best CV score: {np.sqrt(-xgb_search.best_score_):.3f} GB RMSE")
    
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
    
    catboost = CatBoostRegressor(
        iterations=300,
        depth=6,
        learning_rate=0.05,
        l2_leaf_reg=3.0,
        random_state=42,
        verbose=0,
        thread_count=-1
    )
    
    # Stacked ensemble
    stacked = StackingRegressor(
        estimators=[
            ('rf', rf),
            ('gb', gb),
            ('xgb', xgb),
            ('lgbm', lgbm),
            ('catboost', catboost)
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
        'CatBoost': catboost,
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
        cv_rmse_folds = np.sqrt(-cv_scores)
        cv_rmse_log = cv_rmse_folds.mean()
        cv_rmse_std_log = cv_rmse_folds.std()
        
        model.fit(X_train, y_train)
        
        # Predictions (in log scale)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Metrics in log space (comparable to CV scores)
        train_mse_log = mean_squared_error(y_train, y_train_pred)
        test_mse_log = mean_squared_error(y_test, y_test_pred)
        train_mae_log = mean_absolute_error(y_train, y_train_pred)
        test_mae_log = mean_absolute_error(y_test, y_test_pred)
        train_r2_log = r2_score(y_train, y_train_pred)
        test_r2_log = r2_score(y_test, y_test_pred)
        train_rmse_log = np.sqrt(train_mse_log)
        test_rmse_log = np.sqrt(test_mse_log)
        
        # Convert predictions back to original scale for metrics
        y_train_pred_actual = np.expm1(y_train_pred)
        y_test_pred_actual = np.expm1(y_test_pred)
        
        # Metrics in original scale (GB)
        train_mse = mean_squared_error(y_train_actual, y_train_pred_actual)
        test_mse = mean_squared_error(y_test_actual, y_test_pred_actual)
        train_mae = mean_absolute_error(y_train_actual, y_train_pred_actual)
        test_mae = mean_absolute_error(y_test_actual, y_test_pred_actual)
        train_r2 = r2_score(y_train_actual, y_train_pred_actual)
        test_r2 = r2_score(y_test_actual, y_test_pred_actual)
        
        results[name] = {
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_rmse': np.sqrt(train_mse),
            'test_rmse': np.sqrt(test_mse),
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_mse_log': train_mse_log,
            'test_mse_log': test_mse_log,
            'train_rmse_log': train_rmse_log,
            'test_rmse_log': test_rmse_log,
            'train_mae_log': train_mae_log,
            'test_mae_log': test_mae_log,
            'train_r2_log': train_r2_log,
            'test_r2_log': test_r2_log,
            'cv_rmse_log': cv_rmse_log,
            'cv_rmse_std_log': cv_rmse_std_log,
            'predictions_log': y_test_pred,
            'predictions_actual': y_test_pred_actual,
            'overfitting_gap_log': test_rmse_log - train_rmse_log,
            'overfitting_gap': np.sqrt(test_mse) - np.sqrt(train_mse)
        }
        
        trained_models[name] = model
        
        print("  Testing summary (log space; directly comparable to CV):")
        print(f"    CV RMSE (log): {cv_rmse_log:.4f} +- {cv_rmse_std_log:.4f}")
        print(f"    Train RMSE (log): {train_rmse_log:.4f}")
        print(f"    Test RMSE (log): {test_rmse_log:.4f}")
        print(f"    Overfitting Gap (log): {results[name]['overfitting_gap_log']:.4f}")
        print(f"    Train MAE (log): {train_mae_log:.4f}")
        print(f"    Test MAE (log): {test_mae_log:.4f}")
        print(f"    Train R2 (log): {train_r2_log:.4f}")
        print(f"    Test R2 (log): {test_r2_log:.4f}")
        print("  Testing summary (original GB scale):")
        print(f"    Train RMSE: {np.sqrt(train_mse):.3f} GB")
        print(f"    Test RMSE: {np.sqrt(test_mse):.3f} GB")
        print(f"    Overfitting Gap: {results[name]['overfitting_gap']:.3f} GB")
        print(f"    Train MAE: {train_mae:.3f} GB")
        print(f"    Test MAE: {test_mae:.3f} GB")
        print(f"    Train R2: {train_r2:.4f}")
        print(f"    Test R2: {test_r2:.4f}")
    
    return results, trained_models

def plot_results(results, y_test, output_dir):
    """Generate visualizations of model performance."""
    os.makedirs(output_dir, exist_ok=True)
    y_test_actual = np.expm1(y_test)
    
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
        y_pred = result['predictions_actual']
        
        ax.scatter(y_test_actual, y_pred, alpha=0.5)
        
        # Perfect prediction line
        min_val = min(y_test_actual.min(), y_pred.min())
        max_val = max(y_test_actual.max(), y_pred.max())
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

def print_model_rankings(results):
    """Print compact rankings for key test metrics."""
    summary_df = pd.DataFrame([
        {
            'Model': model_name,
            'Test_RMSE_Log': metrics['test_rmse_log'],
            'Test_RMSE_GB': metrics['test_rmse'],
            'Test_MAE_GB': metrics['test_mae']
        }
        for model_name, metrics in results.items()
    ])

    def _print_rank_table(df, metric_col, title):
        ranked = df.sort_values(metric_col, ascending=True).reset_index(drop=True)
        ranked.insert(0, 'Rank', ranked.index + 1)
        print(f"\n{title}")
        print(ranked[['Rank', 'Model', 'Test_RMSE_Log', 'Test_RMSE_GB', 'Test_MAE_GB']].to_string(
            index=False,
            formatters={
                'Test_RMSE_Log': '{:.4f}'.format,
                'Test_RMSE_GB': '{:.3f}'.format,
                'Test_MAE_GB': '{:.3f}'.format,
            }
        ))

    print("\n" + "=" * 60)
    print("Model Rankings")
    print("=" * 60)
    _print_rank_table(summary_df, 'Test_RMSE_Log', 'Ranking by Test RMSE (log)')
    _print_rank_table(summary_df, 'Test_RMSE_GB', 'Ranking by Test RMSE (GB)')
    _print_rank_table(summary_df, 'Test_MAE_GB', 'Ranking by Test MAE (GB)')

def main():
    parser = argparse.ArgumentParser(description='Train ML model for AstraSim peak memory prediction')
    parser.add_argument('--input_csv', type=str, default='ml_model/data',
                        help='Input CSV file or directory with training data files')
    parser.add_argument('--output_dir', type=str, default='ml_model/trained_models',
                        help='Output directory for trained models and plots')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Fraction of data to use for testing')
    parser.add_argument('--peak_per_npu_threshold', type=float, default=None,
                        help='Keep rows with avg_peak_memory_gb/num_npus <= this threshold')
    parser.add_argument('--tune_peak_per_npu_threshold', action='store_true',
                        help='Auto-tune peak-per-NPU threshold using validation RMSE in log space')
    parser.add_argument('--min_keep_ratio', type=float, default=0.85,
                        help='Minimum data keep ratio when tuning threshold (default: 0.85)')
    
    args = parser.parse_args()
    
    # Get base directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(base_dir, args.input_csv)
    output_dir = os.path.join(base_dir, args.output_dir)
    
    os.makedirs(output_dir, exist_ok=True)

    selected_threshold = args.peak_per_npu_threshold
    if selected_threshold is None and args.tune_peak_per_npu_threshold:
        selected_threshold = tune_peak_per_npu_threshold(
            input_path,
            test_size=args.test_size,
            min_keep_ratio=args.min_keep_ratio
        )
    elif selected_threshold is not None:
        print(f"\nUsing user-provided peak_per_npu threshold: {selected_threshold:.6f}")
    else:
        print("\nOutlier filter: disabled")
    
    print("Loading and preparing data...")
    X, y, df = load_and_prepare_data(input_path, peak_per_npu_threshold=selected_threshold)
    
    print(f"\nDataset shape: {X.shape}")
    print(f"Target: log(peak_memory_gb + 1)")
    print(f"Target range: {y.min():.4f} - {y.max():.4f} (log scale)")
    print(f"Target mean: {y.mean():.4f} (log scale)")
    print(f"Target std: {y.std():.4f} (log scale)")
    print(f"Original peak memory range: {np.expm1(y.min()):.2f} - {np.expm1(y.max()):.2f} GB")
    
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

    # Print ranked summaries for quick model comparison
    print_model_rankings(results)
    
    # Find best model
    best_model_name = min(results.keys(), key=lambda k: results[k]['test_rmse_log'])
    best_model = trained_models[best_model_name]
    
    print(f"\n{'='*60}")
    print(f"Best Model: {best_model_name}")
    print(f"Test RMSE (log): {results[best_model_name]['test_rmse_log']:.4f}")
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
        'CV_RMSE_Log': [results[m]['cv_rmse_log'] for m in results.keys()],
        'CV_RMSE_Std_Log': [results[m]['cv_rmse_std_log'] for m in results.keys()],
        'Train_RMSE_Log': [results[m]['train_rmse_log'] for m in results.keys()],
        'Test_RMSE_Log': [results[m]['test_rmse_log'] for m in results.keys()],
        'Overfitting_Gap_Log': [results[m]['overfitting_gap_log'] for m in results.keys()],
        'Train_MAE_Log': [results[m]['train_mae_log'] for m in results.keys()],
        'Test_MAE_Log': [results[m]['test_mae_log'] for m in results.keys()],
        'Train_R2_Log': [results[m]['train_r2_log'] for m in results.keys()],
        'Test_R2_Log': [results[m]['test_r2_log'] for m in results.keys()],
        'Train_RMSE': [results[m]['train_rmse'] for m in results.keys()],
        'Test_RMSE': [results[m]['test_rmse'] for m in results.keys()],
        'Overfitting_Gap': [results[m]['overfitting_gap'] for m in results.keys()],
        'Train_MAE': [results[m]['train_mae'] for m in results.keys()],
        'Test_MAE': [results[m]['test_mae'] for m in results.keys()],
        'Test_R2': [results[m]['test_r2'] for m in results.keys()],
        'Train_R2': [results[m]['train_r2'] for m in results.keys()]
    })
    results_df.to_csv(os.path.join(output_dir, 'model_results.csv'), index=False)
    print(f"\nResults summary saved to {output_dir}/model_results.csv")

if __name__ == '__main__':
    main()
