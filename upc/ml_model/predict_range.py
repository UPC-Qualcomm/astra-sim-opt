#!/usr/bin/env python3
"""
Predict memory range [min_gb, max_gb] for a given configuration.
"""

import argparse
import os

import joblib
import numpy as np
import pandas as pd

from predict import prepare_features


def predict_memory_range(config, model_dir):
    lower_model_path = os.path.join(model_dir, "range_lower_model.pkl")
    upper_model_path = os.path.join(model_dir, "range_upper_model.pkl")
    feature_names_path = os.path.join(model_dir, "feature_names.txt")
    metadata_path = os.path.join(model_dir, "range_model_metadata.pkl")

    required = [lower_model_path, upper_model_path, feature_names_path]
    if not all(os.path.exists(p) for p in required):
        raise FileNotFoundError(
            "Missing range model artifacts. Train first with train_range_model.py"
        )

    lower_model = joblib.load(lower_model_path)
    upper_model = joblib.load(upper_model_path)

    metadata = None
    if os.path.exists(metadata_path):
        metadata = joblib.load(metadata_path)

    with open(feature_names_path, "r") as f:
        feature_names = [line.strip() for line in f]

    features = prepare_features(config)
    X = pd.DataFrame([features])[feature_names]

    lower_log = lower_model.predict(X)[0]
    upper_log = upper_model.predict(X)[0]

    lower_gb = float(np.expm1(lower_log))
    upper_gb = float(np.expm1(upper_log))

    qhat = 0.0
    if metadata is not None:
        qhat = float(metadata.get("conformal_qhat_gb", 0.0))

    raw_min = min(lower_gb, upper_gb)
    raw_max = max(lower_gb, upper_gb)

    min_gb = max(0.0, raw_min - qhat)
    max_gb = raw_max + qhat

    return min_gb, max_gb, metadata


def main():
    parser = argparse.ArgumentParser(description="Predict peak-memory range for AstraSim configuration")
    parser.add_argument("--model_dir", type=str, default="ml_model/trained_range_models",
                        help="Directory containing trained range models")

    parser.add_argument("--din", type=int, required=True)
    parser.add_argument("--dmodel", type=int, required=True)
    parser.add_argument("--dff", type=int, required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--micro_batch", type=int, required=True)
    parser.add_argument("--seq", type=int, required=True)
    parser.add_argument("--head", type=int, required=True)
    parser.add_argument("--num_stacks", type=int, required=True)

    parser.add_argument("--dp", type=int, required=True)
    parser.add_argument("--mp", type=int, required=True)
    parser.add_argument("--sp", type=int, required=True)
    parser.add_argument("--pp", type=int, required=True)
    parser.add_argument("--fsdp", type=int, required=True, choices=[0, 1])
    parser.add_argument("--num_npus", type=int, required=True)

    parser.add_argument("--oom_capacity_gb", type=float, default=None,
                        help="If provided, print definitive OOM/safe/uncertain decision")

    args = parser.parse_args()

    total = args.dp * args.mp * args.sp * args.pp
    if total != args.num_npus:
        raise ValueError(f"dp * mp * sp * pp ({total}) must equal num_npus ({args.num_npus})")

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(base_dir, args.model_dir)

    config = {
        "din": args.din,
        "dmodel": args.dmodel,
        "dff": args.dff,
        "batch": args.batch,
        "micro_batch": args.micro_batch,
        "seq": args.seq,
        "head": args.head,
        "num_stacks": args.num_stacks,
        "dp": args.dp,
        "mp": args.mp,
        "sp": args.sp,
        "pp": args.pp,
        "fsdp": args.fsdp,
        "num_npus": args.num_npus,
    }

    print("Configuration:")
    print(f"  Model: din={args.din}, dmodel={args.dmodel}, dff={args.dff}")
    print(f"         batch={args.batch}, micro_batch={args.micro_batch}, seq={args.seq}")
    print(f"         head={args.head}, num_stacks={args.num_stacks}")
    print(f"  Parallelism: dp={args.dp}, mp={args.mp}, sp={args.sp}, pp={args.pp}, fsdp={args.fsdp}")
    print(f"  NPUs: {args.num_npus}")

    min_gb, max_gb, metadata = predict_memory_range(config, model_dir)

    print(f"\nPredicted Peak Memory Range: [{min_gb:.2f}, {max_gb:.2f}] GB per NPU")
    print(f"Range width: {max_gb - min_gb:.2f} GB")

    if metadata is not None:
        lq = metadata.get("lower_quantile")
        uq = metadata.get("upper_quantile")
        target_cov = metadata.get("target_coverage")
        qhat = metadata.get("conformal_qhat_gb")
        cov = metadata.get("metrics", {}).get("coverage")
        if lq is not None and uq is not None:
            print(f"Model quantiles: [{lq:.2f}, {uq:.2f}]")
        if target_cov is not None:
            print(f"Target coverage: {target_cov * 100:.2f}%")
        if qhat is not None:
            print(f"Conformal qhat: {qhat:.3f} GB")
        if cov is not None:
            print(f"Validation coverage: {cov * 100:.2f}%")

    if args.oom_capacity_gb is not None:
        cap = args.oom_capacity_gb
        print(f"\nCapacity check (cap={cap:.2f} GB):")
        if min_gb > cap:
            print("  Decision: DEFINITELY OOM")
        elif max_gb <= cap:
            print("  Decision: DEFINITELY SAFE")
        else:
            print("  Decision: UNCERTAIN (in risk band)")


if __name__ == "__main__":
    main()
