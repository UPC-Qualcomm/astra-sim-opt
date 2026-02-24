"""
OOM (Out-Of-Memory) Analysis Script
====================================
Compares peak_memory (ground truth) vs predicted_xgboost (predictor)
against a fixed memory capacity to evaluate OOM detection accuracy.

Datasets:
  GPT_1 : evaluation/prediction_evaluation_all_models.csv
  GPT_2 : evaluation_40B/prediction_evaluation_all_models.csv
  GPT_3 : evaluation_no_log/prediction_evaluation_all_models.csv
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ─── Configuration ────────────────────────────────────────────────────────────
MEMORY_CAPACITY = 96  # GB

BASE = os.path.dirname(os.path.abspath(__file__))
FILES = {
    "GPT_1":        os.path.join(BASE, "evaluation",        "prediction_evaluation_all_models.csv"),
    "GPT_2":        os.path.join(BASE, "evaluation_40B",    "prediction_evaluation_all_models.csv"),
    "GPT_3": os.path.join(BASE, "evaluation_no_log", "prediction_evaluation_all_models.csv"),
    "GPT_4": os.path.join(BASE, "evaluation_70B_worse", "prediction_evaluation_all_models.csv"),
}

# Colours for each category in the confusion matrix
CM_COLORS = {
    "TP": "#27ae60",   # green   – correctly flagged OOM
    "FN": "#e67e22",   # orange  – missed OOM (acceptable)
    "FP": "#e74c3c",   # red     – safe flagged as OOM (dangerous!)
    "TN": "#2980b9",   # blue    – correctly flagged safe
}

# Line colours for the three datasets in the threshold plot
DS_COLORS = {
    "GPT_1":          "#3498db",
    "GPT_2":          "#2ecc71",
    "GPT_3":          "#e74c3c",
    "GPT_4": "#9b59b6",
}

DS_MARKERS = {
    "GPT_1":          "o",
    "GPT_2":          "s",
    "GPT_3":          "^",
    "GPT_4": "D",
}

OUTPUT_DIR = os.path.join(BASE, "oom_analysis")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Helper functions ─────────────────────────────────────────────────────────

def compute_stats(peak, pred, capacity, threshold_pct=0.0):
    """Return (TP, TN, FP, FN). Subtracts threshold_pct% of capacity from
    prediction to shift borderline OOM predictions toward Safe (reduces FP)."""
    adj_pred = pred - (capacity * threshold_pct / 100.0)
    actual_oom = peak    > capacity
    pred_oom   = adj_pred > capacity
    TP = int((actual_oom  &  pred_oom).sum())
    TN = int((~actual_oom & ~pred_oom).sum())
    FP = int((~actual_oom &  pred_oom).sum())
    FN = int((actual_oom  & ~pred_oom).sum())
    return TP, TN, FP, FN


def draw_confusion_matrix(ax, TP, TN, FP, FN, title, threshold=None):
    """
    Draw a 2 × 2 coloured confusion-matrix on *ax*.
    Layout (rows=Actual, cols=Predicted):
        [TP  FN]
        [FP  TN]
    """
    total   = TP + TN + FP + FN
    cells   = [("TP", TP, "True Positive\n(OOM Detected ✓)",        (0, 1)),
               ("FN", FN, "False Negative\n(Missed OOM ⚠)",        (1, 1)),
               ("FP", FP, "False Positive\n(Safe → OOM ✗ DANGER)", (0, 0)),
               ("TN", TN, "True Negative\n(Safe Detected ✓)",       (1, 0))]

    ax.set_xlim(-0.45, 2.0)
    ax.set_ylim(-0.55,  2.4)
    ax.set_aspect("equal")
    ax.axis("off")

    for key, count, label, (col, row) in cells:
        pct   = 100.0 * count / total if total else 0
        color = CM_COLORS[key]
        rect  = mpatches.FancyBboxPatch(
            (col, row), 0.92, 0.92,
            boxstyle="round,pad=0.04",
            facecolor=color, edgecolor="white", linewidth=2, alpha=0.90
        )
        ax.add_patch(rect)
        # count (large)
        ax.text(col + 0.46, row + 0.58, str(count),
                ha="center", va="center", fontsize=18,
                fontweight="bold", color="white")
        # percentage
        ax.text(col + 0.46, row + 0.30, f"{pct:.1f}%",
                ha="center", va="center", fontsize=10, color="white")
        # small label
        ax.text(col + 0.46, row + 0.83, label,
                ha="center", va="top", fontsize=7.5,
                color="white", style="italic",
                multialignment="center")

    # Column headers
    ax.text(0.46, 2.15, "Predicted\nOOM",  ha="center", va="center",
            fontsize=9, fontweight="bold", color="#333333")
    ax.text(1.46, 2.15, "Predicted\nSafe", ha="center", va="center",
            fontsize=9, fontweight="bold", color="#333333")

    # Row headers (rotated)
    ax.text(-0.35, 1.46, "Actual\nOOM",  ha="center", va="center",
            fontsize=9, fontweight="bold", color="#333333", rotation=90)
    ax.text(-0.35, 0.46, "Actual\nSafe", ha="center", va="center",
            fontsize=9, fontweight="bold", color="#333333", rotation=90)

    # Dividers
    ax.plot([0, 1.92], [1.0, 1.0], color="#aaaaaa", linewidth=1.0)
    ax.plot([1.0, 1.0], [0, 2.0],  color="#aaaaaa", linewidth=1.0)

    prec = 100.0 * TP / (TP + FP) if (TP + FP) else 100.0
    fpr  = 100.0 * FP / (FP + TN) if (FP + TN) else 0.0
    thresh_str = f"  |  Safety threshold: −{threshold:.0f}%" if threshold is not None else ""
    ax.set_title(f"{title}{thresh_str}\nPrecision: {prec:.1f}%   FPR: {fpr:.1f}%   Total: {total}",
                 fontsize=11, fontweight="bold", pad=6)


# ─── Load data ────────────────────────────────────────────────────────────────
datasets = {}
for name, path in FILES.items():
    df = pd.read_csv(path)
    datasets[name] = df
    actual_oom_n = (df["peak_memory"] > MEMORY_CAPACITY).sum()
    print(f"Loaded {name}: {len(df)} records,  actual OOM (>{MEMORY_CAPACITY} GB): {actual_oom_n}")


# ─── Figure 1 : Confusion matrix – NO threshold ───────────────────────────────
print("\n" + "=" * 60)
print(f"OOM Analysis  (capacity = {MEMORY_CAPACITY} GB, threshold = 0%)")
print("=" * 60)

fig1, axes1 = plt.subplots(1, len(FILES), figsize=(20, 5.5))
fig1.suptitle(
    f"XGBoost OOM Prediction  –  Capacity = {MEMORY_CAPACITY} GB  |  No Safety Threshold",
    fontsize=13, fontweight="bold"
)

for ax, (name, df) in zip(axes1, datasets.items()):
    TP, TN, FP, FN = compute_stats(df["peak_memory"], df["predicted_xgboost"], MEMORY_CAPACITY)
    total = TP + TN + FP + FN
    print(f"\n{name}:")
    print(f"  Total={total}  |  Actual OOM={(TP+FN)}  |  Actual Safe={(TN+FP)}")
    print(f"  TP={TP}  TN={TN}  FP={FP} (DANGEROUS)  FN={FN} (acceptable)")
    prec = 100.0 * TP / (TP + FP) if (TP + FP) else 100.0
    print(f"  Precision: {prec:.1f}%  |  False OOM alarms (FP): {FP}  |  Missed OOM (FN): {FN}")
    draw_confusion_matrix(ax, TP, TN, FP, FN, name)
    # MAPE + FPR annotation at the bottom
    mape = float(np.mean(np.abs((df["peak_memory"].values - df["predicted_xgboost"].values)
                                / df["peak_memory"].values)) * 100)
    ax.text(0.775, -0.50, f"MAPE = {mape:.1f}%",
            ha="center", va="top", fontsize=9,
            color="#333333", style="italic",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f0f0",
                      edgecolor="#aaaaaa", linewidth=0.8))

plt.tight_layout(rect=[0, 0, 1, 0.95])
out1 = os.path.join(OUTPUT_DIR, "fig1_oom_no_threshold.png")
plt.savefig(out1, dpi=150, bbox_inches="tight")
print(f"\nSaved → {out1}")


# ─── Figure 2 : Scatter plot – peak_memory vs predicted_xgboost ───────────────
print("\n" + "=" * 60)
print("Scatter plot: ground truth vs XGBoost prediction")
print("=" * 60)

fig2, axes2 = plt.subplots(1, len(FILES), figsize=(15, 5.5))
fig2.suptitle(
    f"Peak Memory vs XGBoost Prediction  –  Capacity threshold = {MEMORY_CAPACITY} GB",
    fontsize=13, fontweight="bold"
)

_QUAD_LABEL = {
    "TP": "TP – Actual OOM, Predicted OOM",
    "FN": "FN – Actual OOM, Missed by predictor",
    "FP": "FP – Actual Safe, Over-predicted OOM",
    "TN": "TN – Actual Safe, Predicted Safe",
}

for ax, (name, df) in zip(axes2, datasets.items()):
    peak = df["peak_memory"].values
    pred = df["predicted_xgboost"].values

    actual_oom = peak > MEMORY_CAPACITY
    pred_oom   = pred > MEMORY_CAPACITY

    masks = {
        "TP": actual_oom  &  pred_oom,
        "FN": actual_oom  & ~pred_oom,
        "FP": ~actual_oom &  pred_oom,
        "TN": ~actual_oom & ~pred_oom,
    }

    for key, mask in masks.items():
        if mask.any():
            ax.scatter(peak[mask], pred[mask],
                       color=CM_COLORS[key], label=_QUAD_LABEL[key],
                       s=40, edgecolors="white", linewidths=0.4, alpha=0.85, zorder=3)

    # Axis range with a bit of padding
    all_vals = np.concatenate([peak, pred])
    lo = max(0, all_vals.min() * 0.9)
    hi = all_vals.max() * 1.08

    # Perfect-prediction diagonal
    ax.plot([lo, hi], [lo, hi], color="#555555", linewidth=1.0,
            linestyle="--", alpha=0.5, label="Perfect prediction (y = x)")

    # Capacity lines
    ax.axvline(MEMORY_CAPACITY, color="#c0392b", linewidth=1.4,
               linestyle=":", alpha=0.8, label=f"Capacity = {MEMORY_CAPACITY} GB")
    ax.axhline(MEMORY_CAPACITY, color="#c0392b", linewidth=1.4,
               linestyle=":", alpha=0.8)

    # Shade the four quadrants very lightly
    ax.axvspan(MEMORY_CAPACITY, hi, ymin=0, ymax=1, color="#e74c3c", alpha=0.04)
    ax.axhspan(MEMORY_CAPACITY, hi, xmin=0, xmax=1, color="#e74c3c", alpha=0.04)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Ground Truth  peak_memory (GB)", fontsize=9)
    ax.set_ylabel("Predicted  predicted_xgboost (GB)", fontsize=9)
    ax.set_title(name, fontsize=11, fontweight="bold")
    ax.legend(fontsize=6.5, loc="upper left")
    ax.grid(True, alpha=0.25)
    # MAPE annotation at the bottom
    mape = float(np.mean(np.abs((df["peak_memory"].values - df["predicted_xgboost"].values)
                                / df["peak_memory"].values)) * 100)
    ax.text(0.5, -0.2, f"MAPE = {mape:.1f}%",
            ha="center", va="bottom", fontsize=9,
            transform=ax.transAxes,
            color="#333333", style="italic",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f0f0",
                      edgecolor="#aaaaaa", linewidth=0.8, alpha=0.85))
    
plt.tight_layout(rect=[0, 0, 1, 0.95])
out2s = os.path.join(OUTPUT_DIR, "fig2_scatter.png")
plt.savefig(out2s, dpi=150, bbox_inches="tight")
print(f"Saved → {out2s}")


# ─── Threshold sweep ──────────────────────────────────────────────────────────
# Find the minimum threshold per dataset to make FN = 0,
# then set a common upper bound slightly above the worst case.

max_needed = 0
for name, df in datasets.items():
    peak  = df["peak_memory"].values
    pred  = df["predicted_xgboost"].values
    fp_mask = (peak <= MEMORY_CAPACITY) & (pred > MEMORY_CAPACITY)
    if fp_mask.any():
        # need  pred - capacity*t/100 <= capacity  →  t >= (pred/capacity - 1)*100
        needed = (pred[fp_mask] / MEMORY_CAPACITY - 1) * 100.0
        max_needed = max(max_needed, needed.max())

THRESH_MAX  = max(int(np.ceil(max_needed)) + 20, 50)   # at least 50 for a nice plot
thresholds  = np.arange(0, THRESH_MAX + 1, 1, dtype=float)

sweep = {}   # name → array of FN counts
opt_thresh = {}  # name → first threshold where FN == 0

for name, df in datasets.items():
    fp_arr = []
    for t in thresholds:
        _, _, FP, _ = compute_stats(df["peak_memory"], df["predicted_xgboost"], MEMORY_CAPACITY, t)
        fp_arr.append(FP)
    fp_arr = np.array(fp_arr)
    sweep[name] = fp_arr
    zero_idx = np.where(fp_arr == 0)[0]
    opt_thresh[name] = float(thresholds[zero_idx[0]]) if len(zero_idx) else float(thresholds[-1])


# ─── Figure 3 : Threshold study ───────────────────────────────────────────────
print("\n" + "=" * 60)
print("Threshold study (x = safety %, y = false OOM alarms / FP)")
print("=" * 60)

fig3, ax3 = plt.subplots(figsize=(11, 5.5))

for name, fp_arr in sweep.items():
    t_opt = opt_thresh[name]
    color  = DS_COLORS[name]
    marker = DS_MARKERS[name]
    ax3.plot(thresholds, fp_arr, color=color, linewidth=2.5,
             label=f"{name}  (FP→0 at {t_opt:.0f}% of max capacity)",
             marker=marker, markevery=max(1, THRESH_MAX//20), markersize=5)
    # vertical dashed line at the optimal threshold
    ax3.axvline(t_opt, color=color, linestyle="--", linewidth=1.2, alpha=0.6)
    # annotation
    ax3.annotate(
        f"{t_opt:.0f}%",
        xy=(t_opt, 0),
        xytext=(t_opt + 1, fp_arr[0] * 0.07 + 1.5),
        fontsize=9, color=color,
        arrowprops=dict(arrowstyle="->", color=color, lw=1.2)
    )
    print(f"  {name}: FP=0 first reached at threshold = −{t_opt:.0f}%")

ax3.set_xlabel("Safety Threshold Subtracted from Predicted Memory (%)", fontsize=12)
ax3.set_ylabel("False OOM Alarms  (False Positives)", fontsize=12)
ax3.set_title(
    f"Safety Threshold Study  –  Capacity = {MEMORY_CAPACITY} GB\n"
    "Subtracting X% of capacity from XGBoost prediction to eliminate false OOM alarms (FP)",
    fontsize=12, fontweight="bold"
)
ax3.legend(fontsize=11)
ax3.grid(True, alpha=0.3)
ax3.set_xlim(0, THRESH_MAX)
ax3.set_ylim(bottom=-0.5)

plt.tight_layout()
out3 = os.path.join(OUTPUT_DIR, "fig3_threshold_study.png")
plt.savefig(out3, dpi=150, bbox_inches="tight")
print(f"Saved → {out3}")


# ─── Figure 4 : Confusion matrix – WITH optimal threshold ────────────────────
print("\n" + "=" * 60)
print("OOM Analysis with optimal (FP→0) safety threshold")
print("=" * 60)

fig4, axes4 = plt.subplots(1, len(FILES), figsize=(20, 5.5))
fig4.suptitle(
    f"XGBoost OOM Prediction  –  Capacity = {MEMORY_CAPACITY} GB  |  With Optimal Threshold (FP → 0)",
    fontsize=13, fontweight="bold"
)

for ax, (name, df) in zip(axes4, datasets.items()):
    t = opt_thresh[name]
    TP, TN, FP, FN = compute_stats(df["peak_memory"], df["predicted_xgboost"], MEMORY_CAPACITY, t)
    total = TP + TN + FP + FN
    print(f"\n{name}  (threshold = −{t:.0f}%):")
    print(f"  TP={TP}  TN={TN}  FP={FP} (DANGEROUS)  FN={FN} (acceptable)")
    prec = 100.0 * TP / (TP + FP) if (TP + FP) else 100.0
    print(f"  Precision: {prec:.1f}%  |  False OOM alarms (FP): {FP}  |  Missed OOM (FN): {FN}")
    draw_confusion_matrix(ax, TP, TN, FP, FN, name, threshold=t)

plt.tight_layout(rect=[0, 0, 1, 0.95])
out4 = os.path.join(OUTPUT_DIR, "fig4_oom_with_threshold.png")
plt.savefig(out4, dpi=150, bbox_inches="tight")
print(f"Saved → {out4}")

plt.show()
print("\nDone.")
