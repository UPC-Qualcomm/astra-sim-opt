#!/usr/bin/env python3
"""
Standalone script to plot Pareto front from DeepHyper results CSV file.

This script allows you to visualize multi-objective optimization results
without re-running the entire optimization. Supports both interactive (Plotly)
and static (Matplotlib) plots with automatic outlier removal.

Usage:
    python plot_pareto_front.py <results_file.csv>
    python plot_pareto_front.py results.csv --format interactive
    python plot_pareto_front.py results.csv --format static
    python plot_pareto_front.py results.csv --format both
    
Optional arguments:
    --format: Plot format: 'interactive' (HTML), 'static' (PNG), 'both' (default: both)
    --obj0-name: Name for objective 0 (default: "Objective 0")
    --obj1-name: Name for objective 1 (default: "Objective 1")
    --output: Output filename (default: auto-generated)
    --no-labels: Don't show configuration details on hover (interactive only)
    --no-outlier-removal: Disable automatic outlier removal
    --iqr-multiplier: IQR multiplier for outlier detection (default: 1.5)
"""

import argparse
import pandas as pd
from pathlib import Path
import numpy as np

# Try importing plotly for interactive plots
try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Import matplotlib for static plots
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def remove_outliers_iqr(df: pd.DataFrame, columns: list, iqr_multiplier: float = 1.5) -> pd.DataFrame:
    """Remove outliers using Interquartile Range (IQR) method.
    
    Args:
        df: DataFrame to filter
        columns: List of column names to check for outliers
        iqr_multiplier: IQR multiplier for outlier detection (default: 1.5)
        
    Returns:
        DataFrame with outliers removed
    """
    # First, remove failure values (like -10000000000.0 = -1e10)
    mask = pd.Series([True] * len(df), index=df.index)
    
    for col in columns:
        if col not in df.columns:
            continue
        
        # Remove failure markers (-1e10) and NaN values
        # Using abs() to catch both positive and negative failure markers
        valid_mask = (df[col].notna()) & (df[col].abs() < 9e9)
        mask = mask & valid_mask
    
    # Now apply IQR filtering on the valid values
    df_valid = df[mask].copy()
    
    if len(df_valid) == 0:
        return df_valid
    
    for col in columns:
        if col not in df_valid.columns:
            continue
        
        values = df_valid[col]
        
        if len(values) < 4:  # Need at least 4 points for IQR
            continue
        
        # Calculate IQR
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        
        if iqr == 0:  # All values are the same
            continue
        
        # Define outlier bounds
        lower_bound = q1 - iqr_multiplier * iqr
        upper_bound = q3 + iqr_multiplier * iqr
        
        # Update mask to keep only points within bounds
        df_valid = df_valid[(df_valid[col] >= lower_bound) & (df_valid[col] <= upper_bound)]
    
    return df_valid


def compute_pareto_front(df, obj0_col="objective_0", obj1_col="objective_1"):
    """Compute Pareto-efficient points (assuming minimization for both objectives).
    
    Args:
        df: DataFrame with objectives
        obj0_col: Column name for objective 0
        obj1_col: Column name for objective 1
    
    Returns:
        Boolean array indicating which points are Pareto-efficient
    """
    # Filter out failed evaluations
    valid_df = df[[obj0_col, obj1_col]].dropna()
    
    if len(valid_df) == 0:
        return np.array([False] * len(df))
    
    # Extract objectives
    objectives = valid_df[[obj0_col, obj1_col]].values
    
    # Compute Pareto front
    is_efficient = np.ones(len(objectives), dtype=bool)
    
    for i, point in enumerate(objectives):
        if is_efficient[i]:
            # Check if any other point dominates this point
            # For minimization: point is dominated if another point is better in all objectives
            is_efficient[is_efficient] = np.any(objectives[is_efficient] >= point, axis=1)
    
    # Map back to original dataframe indices
    pareto_mask = np.array([False] * len(df))
    pareto_mask[valid_df.index] = is_efficient
    
    return pareto_mask


def create_hover_text(df, param_cols):
    """Create detailed hover text for each point showing ALL available data.
    
    Args:
        df: DataFrame with results
        param_cols: List of parameter column names
    
    Returns:
        List of hover text strings
    """
    hover_texts = []
    
    # Columns to exclude from "other fields" (we'll show them explicitly)
    exclude_cols = {'objective_0', 'objective_1', 'job_id', 'job_status', 'exec_time', 
                    'pareto_efficient', 'system_config', 'network_config', 'memory_config'}
    exclude_cols.update(set(param_cols))
    exclude_cols.update({c for c in df.columns if c.startswith('m:')})
    
    for idx, row in df.iterrows():
        text_parts = [f"<b>Evaluation #{row.get('job_id', idx)}</b>"]
        
        # Add job status
        if "job_status" in row:
            text_parts.append(f"Status: {row['job_status']}")
        
        # Add objectives
        text_parts.append("<br><b>Objectives:</b>")
        if "objective_0" in row:
            text_parts.append(f"  Obj 0: {row['objective_0']:.6e}")
        if "objective_1" in row:
            text_parts.append(f"  Obj 1: {row['objective_1']:.6e}")
        
        # Add execution time if available
        if "exec_time" in row and pd.notna(row["exec_time"]):
            text_parts.append(f"  Exec Time: {row['exec_time']/1e9:,.5f} s")
        
        # Add Pareto status
        if "pareto_efficient" in row:
            status = "✓ Pareto-Efficient" if row["pareto_efficient"] else "Non-Pareto"
            text_parts.append(f"  {status}")
        
        # Add ALL configuration parameters (p: columns)
        text_parts.append("<br><b>Configuration Parameters:</b>")
        for col in param_cols:
            if col in row and pd.notna(row[col]):
                value = row[col]
                param_name = col.replace('p:', '').replace('-', ' ').title()
                # Format value
                if isinstance(value, float):
                    text_parts.append(f"  {param_name}: {value:.2f}")
                elif isinstance(value, bool):
                    text_parts.append(f"  {param_name}: {value}")
                else:
                    text_parts.append(f"  {param_name}: {value}")
        
        # Add system config (parsed if it's JSON)
        if "system_config" in row and pd.notna(row["system_config"]):
            text_parts.append("<br><b>System Config:</b>")
            try:
                import json
                config = json.loads(row["system_config"])
                for key, val in list(config.items())[:15]:  # Show first 15 items
                    text_parts.append(f"  {key}: {val}")
                if len(config) > 15:
                    text_parts.append(f"  ... and {len(config) - 15} more settings")
            except:
                text_parts.append(f"  {str(row['system_config'])[:100]}...")
        
        # Add network config (parsed if it's JSON)
        if "network_config" in row and pd.notna(row["network_config"]):
            text_parts.append("<br><b>Network Config:</b>")
            try:
                import json
                config = json.loads(row["network_config"])
                for key, val in config.items():
                    text_parts.append(f"  {key}: {val}")
            except:
                text_parts.append(f"  {str(row['network_config'])[:100]}...")
        
        # Add any other fields not covered above
        other_cols = [c for c in row.index if c not in exclude_cols and not c.startswith('m:') and not c.startswith('p:')]
        if other_cols:
            text_parts.append("<br><b>Other Fields:</b>")
            for col in other_cols:
                if pd.notna(row[col]):
                    text_parts.append(f"  {col}: {row[col]}")
        
        hover_texts.append("<br>".join(text_parts))
    
    return hover_texts


def plot_pareto_front_static(
    df,
    obj0_name="Objective 0",
    obj1_name="Objective 1",
    output_file="pareto_front.png"
):
    """Create static Pareto front plot using Matplotlib.
    
    Args:
        df: DataFrame with results (already filtered)
        obj0_name: Name for objective 0 axis
        obj1_name: Name for objective 1 axis
        output_file: Output PNG file path
    
    Returns:
        Path to saved plot
    """
    if not MATPLOTLIB_AVAILABLE:
        print("❌ Matplotlib not available. Install with: pip install matplotlib")
        return None
    
    # Create plot
    fig, ax = plt.subplots(figsize=(9, 7), tight_layout=True)
    
    # Check if pareto_efficient column exists
    if "pareto_efficient" in df.columns:
        # Plot non-Pareto efficient points
        non_pareto = df[~df["pareto_efficient"]]
        if len(non_pareto) > 0:
            ax.plot(
                -non_pareto["objective_0"],
                -non_pareto["objective_1"],
                "o",
                color="blue",
                alpha=0.7,
                label="Non Pareto-Efficient",
                markersize=6
            )
        
        # Plot Pareto efficient points
        pareto = df[df["pareto_efficient"]]
        if len(pareto) > 0:
            ax.plot(
                -pareto["objective_0"],
                -pareto["objective_1"],
                "o",
                color="red",
                alpha=0.9,
                label="Pareto-Efficient",
                markersize=8
            )
    else:
        # Plot all points if pareto_efficient column doesn't exist
        ax.plot(
            -df["objective_0"],
            -df["objective_1"],
            "o",
            color="blue",
            alpha=0.7,
            label="All Evaluations",
            markersize=6
        )
    
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    ax.set_xlabel(f"{obj0_name}", fontsize=12)
    ax.set_ylabel(f"{obj1_name}", fontsize=12)
    ax.set_title("Pareto Front: Multi-Objective Optimization", fontsize=14, fontweight="bold")
    
    # Save figure
    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"✓ Static plot saved to: {output_file}")
    
    plt.close(fig)
    return output_file


def plot_pareto_front_interactive(
    df,
    obj0_name="Objective 0",
    obj1_name="Objective 1",
    output_file="pareto_front_interactive.html",
    show_labels=True
):
    """Create interactive Pareto front plot using Plotly.
    
    Args:
        df: DataFrame with results (already filtered)
        obj0_name: Name for objective 0 axis
        obj1_name: Name for objective 1 axis
        output_file: Output HTML file path
        show_labels: Whether to show configuration details on hover
    
    Returns:
        Path to saved plot
    """
    if not PLOTLY_AVAILABLE:
        print("❌ Plotly not available. Install with: pip install plotly")
        return None
    
    # Filter valid evaluations
    valid_mask = df["objective_0"].notna() & df["objective_1"].notna()
    df_valid = df[valid_mask].copy()
    
    if len(df_valid) == 0:
        print("❌ No valid evaluations found")
        return None
    
    # Ensure pareto_efficient column exists
    if "pareto_efficient" not in df.columns:
        df["pareto_efficient"] = compute_pareto_front(df)
    
    # Separate Pareto and non-Pareto points
    pareto_df = df[df["pareto_efficient"] & valid_mask]
    non_pareto_df = df[~df["pareto_efficient"] & valid_mask]
    
    # Get parameter columns for hover text
    param_cols = [col for col in df.columns if col.startswith("p:")]
    
    # Create hover text
    if show_labels and param_cols:
        pareto_hover = create_hover_text(pareto_df, param_cols)
        non_pareto_hover = create_hover_text(non_pareto_df, param_cols)
    else:
        pareto_hover = None
        non_pareto_hover = None
    
    # Create figure
    fig = go.Figure()
    
    # Plot non-Pareto points
    if len(non_pareto_df) > 0:
        fig.add_trace(go.Scatter(
            x=-non_pareto_df["objective_0"],  # Negate for maximization display
            y=-non_pareto_df["objective_1"],
            mode="markers",
            name="Non Pareto-Efficient",
            marker=dict(
                size=8,
                color="rgba(100, 149, 237, 0.6)",  # Cornflower blue
                line=dict(width=1, color="white")
            ),
            text=non_pareto_hover,
            hovertemplate="%{text}<extra></extra>" if non_pareto_hover else None
        ))
    
    # Plot Pareto-efficient points
    if len(pareto_df) > 0:
        fig.add_trace(go.Scatter(
            x=-pareto_df["objective_0"],
            y=-pareto_df["objective_1"],
            mode="markers",
            name="Pareto-Efficient",
            marker=dict(
                size=12,
                color="rgba(220, 20, 60, 0.8)",  # Crimson
                symbol="star",
                line=dict(width=1, color="white")
            ),
            text=pareto_hover,
            hovertemplate="%{text}<extra></extra>" if pareto_hover else None
        ))
    
    # Update layout
    fig.update_layout(
        title=dict(
            text="<b>Pareto Front: Multi-Objective Optimization</b>",
            x=0.5,
            xanchor="center"
        ),
        xaxis=dict(
            title=f"<b>{obj0_name}</b>",
            gridcolor="rgba(200, 200, 200, 0.3)",
            showgrid=True
        ),
        yaxis=dict(
            title=f"<b>{obj1_name}</b>",
            gridcolor="rgba(200, 200, 200, 0.3)",
            showgrid=True
        ),
        plot_bgcolor="white",
        hovermode="closest",
        width=1800,
        height=900,
        legend=dict(
            x=1.02,
            y=1,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="rgba(0, 0, 0, 0.2)",
            borderwidth=1
        ),
        font=dict(size=12),
        hoverlabel=dict(
            font_size=11,
            font_family="monospace",
            align="left",
            namelength=-1
        )
    )
    
    # Add statistics annotation
    df_valid = df[df["objective_0"].notna() & df["objective_1"].notna()]
    stats_text = (
        f"<b>Statistics:</b><br>"
        f"Total evaluations: {len(df_valid)}<br>"
        f"Pareto-efficient: {len(pareto_df)} ({len(pareto_df)/len(df_valid)*100:.1f}%)<br>"
        f"{obj0_name} range: [{-df_valid['objective_0'].max():.2e}, {-df_valid['objective_0'].min():.2e}]<br>"
        f"{obj1_name} range: [{-df_valid['objective_1'].max():.2e}, {-df_valid['objective_1'].min():.2e}]"
    )
    
    fig.add_annotation(
        text=stats_text,
        xref="paper", yref="paper",
        x=0.98, y=0.98,
        xanchor="right", yanchor="top",
        showarrow=False,
        bgcolor="rgba(255, 255, 255, 0.9)",
        bordercolor="rgba(0, 0, 0, 0.2)",
        borderwidth=1,
        borderpad=10,
        font=dict(size=10)
    )
    
    # Save to HTML
    fig.write_html(output_file, config={
        'displayModeBar': True,
        'displaylogo': False,
        'modeBarButtonsToRemove': ['select2d', 'lasso2d']
    })
    
    print(f"✓ Interactive plot saved to: {output_file}")
    
    # Also try to save as static PNG
    try:
        static_file = output_file.replace(".html", "_plotly.png")
        fig.write_image(static_file, width=900, height=700, scale=2)
        print(f"✓ Static PNG (from Plotly) saved to: {static_file}")
    except Exception:
        pass  # Silently fail if kaleido not installed
    
    return output_file


def plot_pareto_front(
    results_file,
    obj0_name="Objective 0",
    obj1_name="Objective 1",
    output_file=None,
    plot_format="both",
    show_labels=True,
    remove_outliers=False,
    iqr_multiplier=1.5,
    obj0_min=None,
    obj0_max=None,
    obj1_min=None,
    obj1_max=None
):
    """Main function to plot Pareto front in specified format(s).
    
    Args:
        results_file: Path to DeepHyper results CSV file
        obj0_name: Name for objective 0 axis
        obj1_name: Name for objective 1 axis
        output_file: Output file path (auto-generated if None)
        plot_format: 'interactive', 'static', or 'both' (default: both)
        show_labels: Whether to show configuration details on hover (interactive only)
        remove_outliers: Whether to remove outliers using IQR method (default: False)
        iqr_multiplier: IQR multiplier for outlier detection (default: 1.5)
        obj0_min: Manual minimum value for objective 0 (optional)
        obj0_max: Manual maximum value for objective 0 (optional)
        obj1_min: Manual minimum value for objective 1 (optional)
        obj1_max: Manual maximum value for objective 1 (optional)
    
    Returns:
        List of paths to saved plots
    """
    # Load results
    print(f"📂 Loading results from: {results_file}")
    df = pd.read_csv(results_file)
    print(f"   Loaded {len(df)} evaluations")
    
    # Check if multi-objective
    if "objective_0" not in df.columns or "objective_1" not in df.columns:
        # Try to parse from exec_time_seconds (random optimizer format)
        if "exec_time_seconds" in df.columns:
            print("   Parsing objectives from exec_time_seconds column...")
            try:
                # Parse tuple strings like "(10000000000, 10000000000)"
                import ast
                def parse_tuple(s):
                    try:
                        return ast.literal_eval(s)
                    except:
                        return (None, None)
                
                df[["objective_0", "objective_1"]] = df["exec_time_seconds"].apply(
                    lambda x: pd.Series(parse_tuple(x))
                )
                print(f"   ✓ Parsed objectives from exec_time_seconds")
            except Exception as e:
                print(f"   ❌ Failed to parse exec_time_seconds: {e}")
                return []
        else:
            print("❌ Not a multi-objective optimization results file")
            print("   Expected columns: objective_0, objective_1 or exec_time_seconds")
            return []
    
    # Remove failure markers first (always filter out -1e10 values)
    df_before = len(df)
    df = df[(df["objective_0"].notna()) & (df["objective_1"].notna())]
    n_failures = df_before - len(df)
    if n_failures > 0:
        print(f"   Removed {n_failures} failed evaluations ({n_failures/df_before*100:.1f}%)")
    
    # Apply manual range filtering if specified
    if obj0_min is not None or obj0_max is not None or obj1_min is not None or obj1_max is not None:
        df_before_manual = len(df)
        if obj0_min is not None:
            df = df[df["objective_0"] >= obj0_min]
        if obj0_max is not None:
            df = df[df["objective_0"] <= obj0_max]
        if obj1_min is not None:
            df = df[df["objective_1"] >= obj1_min]
        if obj1_max is not None:
            df = df[df["objective_1"] <= obj1_max]
        n_manual = df_before_manual - len(df)
        if n_manual > 0:
            print(f"   Manual filter removed {n_manual} points ({n_manual/df_before_manual*100:.1f}%)")
    
    # Apply IQR outlier removal if requested
    if remove_outliers:
        df_before_iqr = len(df)
        df = remove_outliers_iqr(df, ["objective_0", "objective_1"], iqr_multiplier)
        n_iqr = df_before_iqr - len(df)
        if n_iqr > 0:
            print(f"   IQR filter removed {n_iqr} outliers ({n_iqr/df_before_iqr*100:.1f}%)")
    
    if len(df) == 0:
        print("❌ No valid data points after filtering")
        return []
    
    # Compute or use existing Pareto front
    if "pareto_efficient" in df.columns:
        print("   Using existing pareto_efficient column")
    else:
        print("   Computing Pareto front...")
        df["pareto_efficient"] = compute_pareto_front(df)
    
    valid_df = df[df["objective_0"].notna() & df["objective_1"].notna()]
    pareto_count = df["pareto_efficient"].sum()
    print(f"   Valid evaluations: {len(valid_df)}")
    print(f"   Pareto-efficient points: {pareto_count} ({pareto_count/len(valid_df)*100:.1f}%)")
    
    # Generate base filename if not provided
    if output_file is None:
        csv_name = Path(results_file).stem
        if csv_name.startswith("deephyper_results_"):
            model_name = csv_name.replace("deephyper_results_", "")
        else:
            model_name = csv_name
        output_base = Path(results_file).parent / f"pareto_front_{model_name}"
    else:
        output_base = Path(output_file).parent / Path(output_file).stem
    
    saved_files = []
    
    # Generate plots based on format
    if plot_format in ["interactive", "both"]:
        if PLOTLY_AVAILABLE:
            interactive_file = str(output_base) + ".html"
            result = plot_pareto_front_interactive(
                df=df,
                obj0_name=obj0_name,
                obj1_name=obj1_name,
                output_file=interactive_file,
                show_labels=show_labels
            )
            if result:
                saved_files.append(result)
        else:
            print("⚠️  Plotly not available for interactive plots")
            print("   Install with: pip install plotly")
    
    if plot_format in ["static", "both"]:
        if MATPLOTLIB_AVAILABLE:
            static_file = str(output_base) + ".png"
            result = plot_pareto_front_static(
                df=df,
                obj0_name=obj0_name,
                obj1_name=obj1_name,
                output_file=static_file
            )
            if result:
                saved_files.append(result)
        else:
            print("⚠️  Matplotlib not available for static plots")
            print("   Install with: pip install matplotlib")
    
    return saved_files


def main():
    parser = argparse.ArgumentParser(
        description="Plot Pareto front from DeepHyper results (interactive and/or static)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate both interactive and static plots (default - no filtering)
  python plot_pareto_front.py results.csv
  
  # With custom axis labels
  python plot_pareto_front.py results.csv --obj0-name "Time (s)" --obj1-name "Memory (GB)"
  
  # Apply IQR statistical outlier removal
  python plot_pareto_front.py results.csv --use-iqr
  
  # Manual filtering by value range (note: use = sign for negative values)
  python plot_pareto_front.py results.csv --obj0-min=-1e10 --obj0-max=-1e8
  
  # Combine manual and IQR filtering
  python plot_pareto_front.py results.csv --obj0-min=-5e9 --use-iqr
        """
    )
    
    parser.add_argument(
        "results_file",
        help="Path to DeepHyper results CSV file"
    )
    
    parser.add_argument(
        "--format",
        choices=["interactive", "static", "both"],
        default="both",
        help="Plot format: interactive (HTML), static (PNG), or both (default: both)"
    )
    
    parser.add_argument(
        "--obj0-name",
        default="Objective 0",
        help="Name for objective 0 axis (default: Objective 0)"
    )
    
    parser.add_argument(
        "--obj1-name",
        default="Objective 1",
        help="Name for objective 1 axis (default: Objective 1)"
    )
    
    parser.add_argument(
        "--output",
        help="Output file path (without extension, default: auto-generated)"
    )
    
    parser.add_argument(
        "--no-labels",
        action="store_true",
        help="Don't show configuration details on hover (interactive only)"
    )
    
    parser.add_argument(
        "--use-iqr",
        action="store_true",
        help="Apply IQR statistical outlier removal (default: disabled)"
    )
    
    parser.add_argument(
        "--iqr-multiplier",
        type=float,
        default=1.5,
        help="IQR multiplier for outlier detection (default: 1.5, only used with --use-iqr)"
    )
    
    parser.add_argument(
        "--obj0-min",
        type=float,
        help="Manual minimum value for objective 0 (use = for negative values: --obj0-min=-1e9)"
    )
    
    parser.add_argument(
        "--obj0-max",
        type=float,
        help="Manual maximum value for objective 0 (use = for negative values: --obj0-max=-1e6)"
    )
    
    parser.add_argument(
        "--obj1-min",
        type=float,
        help="Manual minimum value for objective 1 (use = for negative values: --obj1-min=-1e9)"
    )
    
    parser.add_argument(
        "--obj1-max",
        type=float,
        help="Manual maximum value for objective 1 (use = for negative values: --obj1-max=-1e6)"
    )
    
    args = parser.parse_args()
    
    # Check if file exists
    if not Path(args.results_file).exists():
        print(f"❌ File not found: {args.results_file}")
        return 1
    
    # Check if at least one backend is available
    if not PLOTLY_AVAILABLE and not MATPLOTLIB_AVAILABLE:
        print("❌ Neither Plotly nor Matplotlib is available")
        print("   Install at least one: pip install plotly  OR  pip install matplotlib")
        return 1
    
    # Create plot(s)
    saved_files = plot_pareto_front(
        results_file=args.results_file,
        obj0_name=args.obj0_name,
        obj1_name=args.obj1_name,
        output_file=args.output,
        plot_format=args.format,
        show_labels=not args.no_labels,
        remove_outliers=args.use_iqr,
        iqr_multiplier=args.iqr_multiplier,
        obj0_min=args.obj0_min,
        obj0_max=args.obj0_max,
        obj1_min=args.obj1_min,
        obj1_max=args.obj1_max
    )
    
    if saved_files:
        print(f"\n✅ Generated {len(saved_files)} plot(s) successfully")
        return 0
    else:
        print("\n❌ Failed to generate plots")
        return 1


if __name__ == "__main__":
    exit(main())
