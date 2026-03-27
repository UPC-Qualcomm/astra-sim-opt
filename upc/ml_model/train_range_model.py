#!/usr/bin/env python3
"""
Train quantile-based range models to predict memory bounds.

This script learns lower/upper quantiles of peak memory so inference returns
[min_memory_gb, max_memory_gb] instead of a single point estimate.
"""

import argparse
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

from train_model import load_and_prepare_data, tune_peak_per_npu_threshold


def evaluate_range(y_true_gb, y_low_gb, y_high_gb):
    """Compute coverage and safety-oriented range metrics."""
    # Enforce valid intervals in case two quantile models cross.
    y_min = np.minimum(y_low_gb, y_high_gb)
    y_max = np.maximum(y_low_gb, y_high_gb)

    in_range = (y_true_gb >= y_min) & (y_true_gb <= y_max)
    below_range = y_true_gb < y_min
    above_range = y_true_gb > y_max

    width = y_max - y_min

    metrics = {
        "coverage": float(in_range.mean()),
        "below_range_rate": float(below_range.mean()),
        "above_range_rate": float(above_range.mean()),
        "avg_interval_width_gb": float(np.mean(width)),
        "median_interval_width_gb": float(np.median(width)),
        "p90_interval_width_gb": float(np.percentile(width, 90)),
    }
    return metrics


def conformal_qhat(y_true_gb, y_low_gb, y_high_gb, target_coverage):
    """Compute conformal calibration term for target marginal coverage.

    This follows split-conformal prediction for regression intervals.
    """
    if not (0.0 < target_coverage < 1.0):
        raise ValueError("target_coverage must be in (0, 1)")

    y_min = np.minimum(y_low_gb, y_high_gb)
    y_max = np.maximum(y_low_gb, y_high_gb)

    # Nonconformity score: how far truth falls outside predicted interval.
    scores = np.maximum(np.maximum(y_min - y_true_gb, y_true_gb - y_max), 0.0)
    n = len(scores)
    if n == 0:
        return 0.0

    # Finite-sample conformal quantile index for coverage >= target_coverage.
    rank = int(np.ceil((n + 1) * target_coverage))
    rank = min(max(rank, 1), n)
    qhat = float(np.partition(scores, rank - 1)[rank - 1])
    return qhat


def train_range_models(X_train, y_train_log, lower_alpha, upper_alpha):
    """Train two quantile regressors in log-space."""
    common_kwargs = {
        "n_estimators": 400,
        "max_depth": 5,
        "learning_rate": 0.05,
        "min_samples_split": 4,
        "min_samples_leaf": 2,
        "subsample": 0.9,
        "random_state": 42,
    }

    low_model = GradientBoostingRegressor(loss="quantile", alpha=lower_alpha, **common_kwargs)
    high_model = GradientBoostingRegressor(loss="quantile", alpha=upper_alpha, **common_kwargs)

    low_model.fit(X_train, y_train_log)
    high_model.fit(X_train, y_train_log)

    return low_model, high_model


def main():
    parser = argparse.ArgumentParser(
        description="Train memory range model (lower/upper quantiles) for AstraSim"
    )
    parser.add_argument("--input_csv", type=str, default="ml_model/data",
                        help="Input CSV file or directory with training data")
    parser.add_argument("--output_dir", type=str, default="ml_model/trained_range_models",
                        help="Output directory for range models")
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="Fraction of data for test split")
    parser.add_argument("--calibration_size", type=float, default=0.2,
                        help="Fraction of train split used for conformal calibration")

    parser.add_argument("--lower_quantile", type=float, default=0.10,
                        help="Lower quantile alpha (default: 0.10)")
    parser.add_argument("--upper_quantile", type=float, default=0.90,
                        help="Upper quantile alpha (default: 0.90)")
    parser.add_argument("--target_coverage", type=float, default=0.90,
                        help="Desired final interval coverage after calibration")

    parser.add_argument("--peak_per_npu_threshold", type=float, default=None,
                        help="Keep rows with avg_peak_memory_gb/num_npus <= this threshold")
    parser.add_argument("--tune_peak_per_npu_threshold", action="store_true",
                        help="Auto-tune peak-per-NPU threshold before training")
    parser.add_argument("--min_keep_ratio", type=float, default=0.85,
                        help="Minimum keep ratio while tuning threshold")

    args = parser.parse_args()

    if not (0.0 < args.lower_quantile < args.upper_quantile < 1.0):
        raise ValueError("Require 0 < lower_quantile < upper_quantile < 1")
    if not (0.0 < args.calibration_size < 0.5):
        raise ValueError("calibration_size must be in (0, 0.5)")
    if not (0.0 < args.target_coverage < 1.0):
        raise ValueError("target_coverage must be in (0, 1)")

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(base_dir, args.input_csv)
    output_dir = os.path.join(base_dir, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    selected_threshold = args.peak_per_npu_threshold
    if selected_threshold is None and args.tune_peak_per_npu_threshold:
        selected_threshold = tune_peak_per_npu_threshold(
            input_path, test_size=args.test_size, min_keep_ratio=args.min_keep_ratio
        )

    print("Loading and preparing data for range training...")
    X, y_log, _ = load_and_prepare_data(
        input_path,
        peak_per_npu_threshold=selected_threshold,
    )

    y_gb = np.expm1(y_log)

    print(f"\nDataset shape: {X.shape}")
    print(f"Quantile targets: [{args.lower_quantile:.2f}, {args.upper_quantile:.2f}]")
    print(f"Target calibrated coverage: {args.target_coverage:.2f}")

    X_train_full, X_test, y_train_full_log, y_test_log = train_test_split(
        X,
        y_log,
        test_size=args.test_size,
        random_state=42,
        stratify=X["num_npus"],
    )

    X_train, X_calib, y_train_log, y_calib_log = train_test_split(
        X_train_full,
        y_train_full_log,
        test_size=args.calibration_size,
        random_state=42,
        stratify=X_train_full["num_npus"],
    )

    y_calib_gb = np.expm1(y_calib_log)
    y_test_gb = np.expm1(y_test_log)

    print(f"Train set size: {len(X_train)}")
    print(f"Calibration set size: {len(X_calib)}")
    print(f"Test set size: {len(X_test)}")

    low_model, high_model = train_range_models(
        X_train,
        y_train_log,
        args.lower_quantile,
        args.upper_quantile,
    )

    # Calibrate interval width with split-conformal method.
    y_low_calib_gb = np.expm1(low_model.predict(X_calib))
    y_high_calib_gb = np.expm1(high_model.predict(X_calib))
    qhat = conformal_qhat(y_calib_gb, y_low_calib_gb, y_high_calib_gb, args.target_coverage)

    y_low_log = low_model.predict(X_test)
    y_high_log = high_model.predict(X_test)

    y_low_gb_raw = np.expm1(y_low_log)
    y_high_gb_raw = np.expm1(y_high_log)

    # Apply symmetric conformal expansion.
    y_low_gb = np.minimum(y_low_gb_raw, y_high_gb_raw) - qhat
    y_high_gb = np.maximum(y_low_gb_raw, y_high_gb_raw) + qhat
    y_low_gb = np.maximum(y_low_gb, 0.0)

    # Optional center estimate for reference.
    y_mid_gb = (np.minimum(y_low_gb, y_high_gb) + np.maximum(y_low_gb, y_high_gb)) / 2.0
    rmse_mid = float(np.sqrt(mean_squared_error(y_test_gb, y_mid_gb)))
    mae_mid = float(mean_absolute_error(y_test_gb, y_mid_gb))

    range_metrics = evaluate_range(y_test_gb, y_low_gb, y_high_gb)

    print("\n" + "=" * 70)
    print("Range Model Evaluation (Test Set)")
    print("=" * 70)
    print(f"Conformal qhat expansion:          {qhat:.3f} GB")
    print(f"Coverage (inside predicted range): {range_metrics['coverage'] * 100:.2f}%")
    print(f"Below-range miss rate:            {range_metrics['below_range_rate'] * 100:.2f}%")
    print(f"Above-range miss rate:            {range_metrics['above_range_rate'] * 100:.2f}%")
    print(f"Avg interval width:               {range_metrics['avg_interval_width_gb']:.3f} GB")
    print(f"Median interval width:            {range_metrics['median_interval_width_gb']:.3f} GB")
    print(f"P90 interval width:               {range_metrics['p90_interval_width_gb']:.3f} GB")
    print(f"Mid-point RMSE (reference only):  {rmse_mid:.3f} GB")
    print(f"Mid-point MAE (reference only):   {mae_mid:.3f} GB")
    print("=" * 70)

    # Save models + metadata.
    joblib.dump(low_model, os.path.join(output_dir, "range_lower_model.pkl"))
    joblib.dump(high_model, os.path.join(output_dir, "range_upper_model.pkl"))

    with open(os.path.join(output_dir, "feature_names.txt"), "w") as f:
        for col in X.columns:
            f.write(f"{col}\n")

    metadata = {
        "lower_quantile": args.lower_quantile,
        "upper_quantile": args.upper_quantile,
        "target_coverage": args.target_coverage,
        "conformal_qhat_gb": qhat,
        "selected_peak_per_npu_threshold": selected_threshold,
        "test_size": args.test_size,
        "calibration_size": args.calibration_size,
        "metrics": range_metrics,
        "midpoint_rmse_gb": rmse_mid,
        "midpoint_mae_gb": mae_mid,
        "n_samples": int(len(X)),
        "n_train": int(len(X_train)),
        "n_calib": int(len(X_calib)),
        "n_test": int(len(X_test)),
    }
    joblib.dump(metadata, os.path.join(output_dir, "range_model_metadata.pkl"))

    summary_df = pd.DataFrame([metadata])
    summary_df.to_csv(os.path.join(output_dir, "range_model_summary.csv"), index=False)

    print(f"\nSaved range models and metadata to: {output_dir}")


if __name__ == "__main__":
    main()
