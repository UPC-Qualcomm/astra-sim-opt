#!/usr/bin/env python3
"""
Use trained model to predict peak memory for new configurations.
"""

import pandas as pd
import numpy as np
import argparse
import os
import joblib

def prepare_features(config):
    """Prepare features from configuration dictionary."""
    features = {
        'din': config['din'],
        'dmodel': config['dmodel'],
        'dff': config['dff'],
        'batch': config['batch'],
        'micro_batch': config['micro_batch'],
        'seq': config['seq'],
        'head': config['head'],
        'num_stacks': config['num_stacks'],
        'dp': config['dp'],
        'mp': config['mp'],
        'sp': config['sp'],
        'pp': config['pp'],
        'fsdp': config['fsdp'],
        'num_npus': config['num_npus']
    }
    
    # Create derived features (same as in training)
    features['total_params_estimate'] = (
        features['dmodel'] * features['din'] + 
        (features['num_stacks'] / features['pp']) * (12 * features['dmodel'] * features['dmodel'] + 13 * features['dmodel']) +
        2 * features['dmodel']
    )
    
    features['sharding_factor'] = features['mp'] * (1 + features['fsdp'] * features['dp'])
    features['micro_batch_per_npu'] = features['micro_batch'] / features['dp']
    features['activation_size_estimate'] = features['micro_batch_per_npu'] * features['seq'] * features['dmodel']
    features['params_per_npu'] = features['total_params_estimate'] / features['sharding_factor']
    features['params_size_gb'] = features['params_per_npu'] * 4 / (1024**3)
    
    # Log transforms
    features['log_din'] = np.log1p(features['din'])
    features['log_dmodel'] = np.log1p(features['dmodel'])
    features['log_dff'] = np.log1p(features['dff'])
    features['log_batch'] = np.log1p(features['batch'])
    features['log_seq'] = np.log1p(features['seq'])
    features['log_total_params'] = np.log1p(features['total_params_estimate'])
    features['log_params_per_npu'] = np.log1p(features['params_per_npu'])
    
    # Advanced domain-specific features
    features['memory_pressure_ratio'] = (features['micro_batch_per_npu'] * features['seq'] * features['dmodel']) / features['num_npus']
    features['log_memory_pressure'] = np.log1p(features['memory_pressure_ratio'])
    
    features['comm_overhead_sp'] = np.log1p(features['sp']) * features['sp']
    features['comm_overhead_pp'] = np.log1p(features['pp']) * features['pp']
    features['total_comm_overhead'] = features['comm_overhead_sp'] + features['comm_overhead_pp']
    
    features['parallelism_product'] = features['dp'] * features['mp'] * features['sp'] * features['pp']
    features['parallelism_efficiency'] = 1.0 / np.sqrt(features['parallelism_product'])
    
    features['layers_per_stage'] = features['num_stacks'] / features['pp']
    features['memory_per_layer'] = features['params_per_npu'] / features['layers_per_stage']
    
    features['pipeline_depth_factor'] = features['pp'] * features['layers_per_stage']
    
    features['batch_efficiency'] = features['batch'] / (features['micro_batch'] * features['dp'])
    features['tokens_per_npu'] = features['micro_batch_per_npu'] * features['seq']
    
    features['sp_pp_interaction'] = features['sp'] * features['pp']
    features['dp_mp_interaction'] = features['dp'] * features['mp']
    
    features['model_size_ratio'] = features['dmodel'] / features['din']
    features['ff_expansion_ratio'] = features['dff'] / features['dmodel']
    
    return features

def predict_peak_memory(config, model_dir, model_name='best'):
    """Predict peak memory for given configuration.
    
    Args:
        config: Configuration dictionary
        model_dir: Directory containing trained models
        model_name: Model to use. Options:
            - 'best' (default): Uses best_model.pkl
            - 'xgboost': Uses xgboost_model.pkl
            - 'gradient_boosting': Uses gradient_boosting_model.pkl
            - 'random_forest': Uses random_forest_model.pkl
            - 'lightgbm': Uses lightgbm_model.pkl
            - 'stacked_ensemble': Uses stacked_ensemble_model.pkl
            - 'ridge_regression': Uses ridge_regression_model.pkl
            - 'mlp': Uses mlp_model.pkl
    """
    # Load model and scaler
    if model_name == 'best':
        model_path = os.path.join(model_dir, 'best_model.pkl')
    else:
        model_path = os.path.join(model_dir, f'{model_name}_model.pkl')
    
    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    feature_names_path = os.path.join(model_dir, 'feature_names.txt')
    
    if not all(os.path.exists(p) for p in [model_path, scaler_path, feature_names_path]):
        raise FileNotFoundError(f"Model files not found. Please train the model first. Missing: {model_path}")
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    with open(feature_names_path, 'r') as f:
        feature_names = [line.strip() for line in f]
    
    # Prepare features
    features = prepare_features(config)
    
    # Create DataFrame with correct column order
    X = pd.DataFrame([features])[feature_names]
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Predict (model predicts log(peak_memory + 1))
    log_prediction = model.predict(X_scaled)[0]
    
    # Convert back from log scale to actual GB
    prediction = np.expm1(log_prediction)
    
    return prediction

def main():
    parser = argparse.ArgumentParser(description='Predict peak memory for AstraSim configuration')
    parser.add_argument('--model_dir', type=str, default='ml_model/trained_models',
                        help='Directory containing trained model')
    parser.add_argument('--model', type=str, default='best',
                        choices=['best', 'xgboost', 'gradient_boosting', 'random_forest', 
                                'lightgbm', 'stacked_ensemble', 'ridge_regression', 'mlp'],
                        help='Which model to use for prediction (default: best)')
    
    # Model parameters
    parser.add_argument('--din', type=int, required=True, help='Input vocabulary size')
    parser.add_argument('--dmodel', type=int, required=True, help='Model dimension')
    parser.add_argument('--dff', type=int, required=True, help='Feed-forward dimension')
    parser.add_argument('--batch', type=int, required=True, help='Batch size')
    parser.add_argument('--micro_batch', type=int, required=True, help='Micro-batch size')
    parser.add_argument('--seq', type=int, required=True, help='Sequence length')
    parser.add_argument('--head', type=int, required=True, help='Number of attention heads')
    parser.add_argument('--num_stacks', type=int, required=True, help='Number of transformer layers')
    
    # Parallelism strategy
    parser.add_argument('--dp', type=int, required=True, help='Data parallelism degree')
    parser.add_argument('--mp', type=int, required=True, help='Model parallelism degree')
    parser.add_argument('--sp', type=int, required=True, help='Spatial parallelism degree')
    parser.add_argument('--pp', type=int, required=True, help='Pipeline parallelism degree')
    parser.add_argument('--fsdp', type=int, required=True, choices=[0, 1], 
                        help='FSDP enabled (1) or not (0)')
    parser.add_argument('--num_npus', type=int, required=True, help='Total number of NPUs')
    
    args = parser.parse_args()
    
    # Validate that dp * mp * sp * pp == num_npus
    total = args.dp * args.mp * args.sp * args.pp
    if total != args.num_npus:
        raise ValueError(f"dp * mp * sp * pp ({total}) must equal num_npus ({args.num_npus})")
    
    # Get base directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(base_dir, args.model_dir)
    
    # Build config
    config = {
        'din': args.din,
        'dmodel': args.dmodel,
        'dff': args.dff,
        'batch': args.batch,
        'micro_batch': args.micro_batch,
        'seq': args.seq,
        'head': args.head,
        'num_stacks': args.num_stacks,
        'dp': args.dp,
        'mp': args.mp,
        'sp': args.sp,
        'pp': args.pp,
        'fsdp': args.fsdp,
        'num_npus': args.num_npus
    }
    
    print("Configuration:")
    print(f"  Model: din={args.din}, dmodel={args.dmodel}, dff={args.dff}")
    print(f"         batch={args.batch}, micro_batch={args.micro_batch}, seq={args.seq}")
    print(f"         head={args.head}, num_stacks={args.num_stacks}")
    print(f"  Parallelism: dp={args.dp}, mp={args.mp}, sp={args.sp}, pp={args.pp}, fsdp={args.fsdp}")
    print(f"  NPUs: {args.num_npus}")
    print(f"  Using model: {args.model}")
    
    # Predict
    try:
        predicted_memory = predict_peak_memory(config, model_dir, args.model)
        print(f"\nPredicted Peak Memory: {predicted_memory:.2f} GB per NPU")
    except Exception as e:
        print(f"\nError: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())
