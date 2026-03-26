#!/usr/bin/env bash
set -u

# ============================================================================
# COMPREHENSIVE OPTIMIZATION SWEEP ACROSS MODELS AND SEARCH SPACES
# ============================================================================
# This script runs DeepHyper Bayesian Optimization for multiple models,
# objectives, and search space configurations.
#
# Models:
#   - GPT_1300M (num 5)
#   - GPT_40B (num 19)
#   - GPT_175B (num 10)
#   - LLama_8B (num 17)
#   - llama_3_70B (num 14)
#
# Configurations per model:
#   1. Parallelism strategy only
#   2. Parallelism strategy + network (intra and inter-node bandwidth)
#
# Objectives (all available):
#   Power and energy objectives, EDP variants, custom multi-objective options
#
# Usage:
#   ./run_all_models_objectives.sh [--skip-failed] [--max-runs N]
#
# ============================================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
ASTRA_SIM_ROOT="${ASTRA_SIM_ROOT:-$(cd -- "${SCRIPT_DIR}/../../.." && pwd)}"
export ASTRA_SIM_ROOT
export PYTHONPATH="${ASTRA_SIM_ROOT}/upc${PYTHONPATH:+:${PYTHONPATH}}"

# Parse command-line arguments
SKIP_FAILED=false
MAX_RUNS=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-failed) SKIP_FAILED=true; shift ;;
        --max-runs) MAX_RUNS="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Define models: NAME NUM
declare -A MODELS=(
    #["GPT_1300M"]=5
    ["llama_8B"]=17
    ["GPT_40B"]=19
    #["GPT_175B"]=10
    ["llama_3_70B"]=14
)

# Define search space configurations
declare -A CONFIGS=(
    ["parallelism_only"]="parallelism_strategy_params.json"
    ["parallelism_network"]="parallelism_strategy_params_g2_intra.json"
)

# Load objectives dynamically from Python
load_objectives() {
    "$PYTHON_BIN" -c "
import os
import sys
sys.path.insert(0, os.path.join(os.environ.get('ASTRA_SIM_ROOT', '/app/astra-sim'), 'upc'))
from Optimization import get_available_objective_types
for obj in get_available_objective_types():
    print(obj)
" 2>/dev/null || echo "Failed to load objectives"
}

mapfile -t OBJECTIVES < <(load_objectives)

if [[ ${#OBJECTIVES[@]} -eq 0 ]] || [[ "${OBJECTIVES[0]}" == "Failed to load objectives" ]]; then
    echo "ERROR: Could not load objectives dynamically. Falling back to hardcoded list." >&2
    OBJECTIVES=(
        "time"
        "time_and_network_bw"
        "power"
        "energy"
        "power_and_time"
        "energy_and_time"
        "edp"
        "ed2p"
        "e2d"
        "edp_and_network_bw"
        "ed2p_and_network_bw"
        "e2d_and_network_bw"
        "energy_cycles_and_network_bw"
        "power_cycles_network_bw"
        "latency_network"
        "latency_network_raw"
        "latency_network_minmax"
        "latency_network_sqrt"
        "latency_network_power"
        "latency_memory"
        "network_memory"
        "latency_network_memory"
        "latency_total_network"
    )
fi

# Statistics
TOTAL_RUNS=0
COMPLETED_RUNS=0
FAILED_RUNS=0
SKIPPED_RUNS=0
declare -a FAILED_RUNS_LIST=()

# Helper function to count total runs
count_total_runs() {
    local model_count=${#MODELS[@]}
    local config_count=${#CONFIGS[@]}
    local objective_count=${#OBJECTIVES[@]}
    echo $((model_count * config_count * objective_count))
}

TOTAL_RUNS=$(count_total_runs)

echo "════════════════════════════════════════════════════════════════════════════════"
echo "COMPREHENSIVE DEEPHYPER MULTI-OBJECTIVE OPTIMIZATION SWEEP"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "Models to optimize (${#MODELS[@]}):"
for model in "${!MODELS[@]}"; do
    echo "  - $model (Model #${MODELS[$model]})"
done
echo ""
echo "Search space configurations (${#CONFIGS[@]}):"
for config_name in "${!CONFIGS[@]}"; do
    echo "  - $config_name (${CONFIGS[$config_name]})"
done
echo ""
echo "Objective functions (${#OBJECTIVES[@]}):"
for obj in "${OBJECTIVES[@]}"; do
    echo "  - $obj"
done
echo ""
echo "Total planned optimization runs: $TOTAL_RUNS"
echo "Estimated runtime: ~$((TOTAL_RUNS * 30)) minutes (30 min per run with 4 workers)"
echo ""
if [ -n "$MAX_RUNS" ]; then
    echo "⚠️  Running only first $MAX_RUNS runs (--max-runs $MAX_RUNS)"
    echo ""
fi
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

cd "$SCRIPT_DIR" || exit 1

check_python_runtime() {
    if ! "$PYTHON_BIN" - <<'PYCHECK' >/dev/null 2>&1
import importlib
for name in ["numpy", "pandas", "yaml", "matplotlib", "deephyper"]:
    importlib.import_module(name)
print("ok")
PYCHECK
    then
        echo "ERROR: Python dependencies are missing for ${PYTHON_BIN}."
        echo "Install them first, for example:"
        echo "  ${PYTHON_BIN} -m pip install -r ${ASTRA_SIM_ROOT}/requirements.txt"
        return 1
    fi
    return 0
}

if ! check_python_runtime; then
    exit 2
fi

RUN_NUMBER=0

# Create temporary Python script generator
generate_optimization_script() {
    local model_num=$1
    local model_name=$2
    local config_path=$3
    local config_name=$4
    local objective=$5
    local script_file=$6
    
    cat > "$script_file" << 'PYTHON_CODE_BLOCK'
import os
import sys

astra_sim_root = os.environ.get("ASTRA_SIM_ROOT", "/app/astra-sim")
sys.path.insert(0, os.path.join(astra_sim_root, "upc"))

from Optimization import (
    create_search_space,
    RandomSampler,
    SimulationRunner,
    DeepHyperOptimizer,
    create_objective,
)
from Optimization.core.base_optimizer import format_score

try:
    from plot_pareto_front import plot_pareto_front
    HAS_PLOT_PARETO = True
except ImportError:
    HAS_PLOT_PARETO = False

# Get environment parameters
model_num = int(os.environ['MODEL_NUM'])
model_name_base = os.environ['MODEL_NAME']
config_path = os.environ['CONFIG_PATH']
objective_key = os.environ['OBJECTIVE']
config_name = os.environ['CONFIG_NAME']

# Build descriptive name
MODEL_NAME = f"{model_name_base}_{config_name}_{objective_key}"
NETWORK_NAME = "FoldedClos"
BUDGET = 150
INIT_SAMPLES = 30
N_WORKERS = 4
TOP_K = 5

print(f"\n{'='*75}")
print(f"Optimization Configuration")
print(f"{'='*75}")
print(f"  Model: {model_name_base} (num {model_num})")
print(f"  Config: {config_name}")
print(f"  Search space: {config_path}")
print(f"  Objective: {objective_key}")
print(f"  Budget: {BUDGET} evaluations")
print(f"  Workers: {N_WORKERS}")
print(f"{'='*75}\n")

# Determine search space categories
include_categories = []
if 'intra' in config_path or 'network' in config_path.lower():
    include_categories = ["parallelism_strategy", "network"]
else:
    include_categories = ["parallelism_strategy"]

print(f"1. Creating search space...")
print(f"   Categories: {include_categories}")

try:
    search_space = create_search_space(config_path, include_categories=include_categories)
    print(f"   ✓ Search space created ({search_space.num_npus} NPUs)")
except Exception as e:
    print(f"   ✗ Failed to create search space: {e}")
    sys.exit(1)

print(f"\n2. Creating sampler...")
sampler = RandomSampler(seed=42)
print(f"   ✓ RandomSampler created")

print(f"\n3. Creating simulation runner...")
net_sim_config = {
    "sim_type": "g2",
    "topology": NETWORK_NAME,
    "paths_mode": "None",
    "routing_mode": "foldedclos_uniform",
    "estimate_power": 1,
    "power_config_path": os.path.join(
        os.environ.get("ASTRA_SIM_ROOT", "/app/astra-sim"), 
        "upc", "power_model", "a100_config.json"
    ),
    "topology_config": {
        "num_npus": search_space.num_npus,
        "npus_per_node": 8,
        "intra_node_topology": "switch",
        "num_nvswitches": 4,
        "bandwidth_config": {
            "host_edge": 100,
            "edge_agg": 100,
            "agg_core": 100,
            "intra_node": 450,
        },
        "bw_unit": "GB/s",
    },
}

try:
    sim_runner = SimulationRunner(
        model_num=model_num,
        model_name=MODEL_NAME,
        network_name=NETWORK_NAME,
        folder_prefix="SWEEP_DEEPHYPER",
        verbose=False,
        net_sim_config=net_sim_config,
    )
    print(f"   ✓ SimulationRunner created")
except Exception as e:
    print(f"   ✗ Failed to create simulation runner: {e}")
    sys.exit(1)

print(f"\n4. Creating objective function: {objective_key}...")
try:
    objective = create_objective(objective_type=objective_key)
    print(f"   ✓ Objective: {objective.name}")
except Exception as e:
    print(f"   ✗ Failed to create objective: {e}")
    sys.exit(1)

# Infer number of objectives
n_obj = 1
if hasattr(objective, 'is_multi_objective') and objective.is_multi_objective:
    obj_dirs = getattr(objective, 'objective_directions', None)
    if isinstance(obj_dirs, (list, tuple)) and len(obj_dirs) > 0:
        n_obj = len(obj_dirs)
    elif objective_key in ('energy_cycles_and_network_bw', 'power_cycles_network_bw', 'latency_network_memory'):
        n_obj = 3
    else:
        n_obj = 2

moo_weight = [1.0 / n_obj] * n_obj if n_obj > 1 else None

print(f"\n5. Creating DeepHyper optimizer...")
print(f"   Objectives: {n_obj} | Surr. model: ExtraTrees | Acq: UCBd")

try:
    optimizer = DeepHyperOptimizer(
        search_space=search_space,
        sampler=sampler,
        simulation_runner=sim_runner,
        budget=BUDGET,
        objective=objective,
        init_samples=INIT_SAMPLES,
        n_workers=N_WORKERS,
        acq_func="UCBd",
        acq_func_kwargs={"kappa": 10.0, "scheduler": {"type": "periodic-exp-decay", "period": 15, "kappa_final": 0.01}},
        surrogate_model="ET",
        surrogate_model_kwargs={"max_features": "sqrt"},
        acq_optimizer="mixedga",
        random_state=42,
        verbose=False,
        keep_top_k=TOP_K,
        profile_time=True,
        evaluator_method="process",
        acq_optimizer_kwargs={"max_total_failures": -1, "acq_optimizer_freq": 2},
        moo_scalarization_strategy="AugChebyshev",
        moo_scalarization_weight=moo_weight,
        enable_tracker=True,
        tracker_kill_multiplier=1.5,
        tracker_initial_threshold=1e15,
        cleanup_batch_size=10,
        compress_and_clean_is_enabled=False
    )
    print(f"   ✓ Optimizer configured")
except Exception as e:
    print(f"   ✗ Failed to create optimizer: {e}")
    sys.exit(1)

print(f"\n{'='*75}")
print(f"STARTING OPTIMIZATION")
print(f"{'='*75}\n")

try:
    best_config, history = optimizer.run()
    
    if best_config is not None:
        print(f"\n{'='*75}")
        print(f"OPTIMIZATION COMPLETE - BEST CONFIGURATION FOUND")
        print(f"{'='*75}\n")
        
        print(f"✓ Best score: {format_score(optimizer.best_score)}")
        print(f"  Evaluations: {len(history)}/{BUDGET}")
        print(f"  Results: {optimizer.save_dir}\n")
        
        # Generate plots for 2-objective optimizations
        try:
            obj_dir = getattr(objective, 'objective_directions', None)
            n_obj = 1
            if hasattr(objective, 'is_multi_objective') and objective.is_multi_objective:
                if isinstance(obj_dir, (list, tuple)):
                    n_obj = len(obj_dir)
                elif objective_key in ('energy_cycles_and_network_bw', 'power_cycles_network_bw', 'latency_network_memory'):
                    n_obj = 3
                else:
                    n_obj = 2
            
            if n_obj == 2 and HAS_PLOT_PARETO:
                print("\nGenerating Pareto front plots...")
                try:
                    csv_path = os.path.join(optimizer.save_dir, optimizer.results_filename)
                    output_base = os.path.join(optimizer.save_dir, f"pareto_front_{MODEL_NAME}")
                    
                    # Map objective keys to plot labels
                    plot_labels_map = {
                        'latency_network': ['log10(Exec Time)', 'log10(Network BW)'],
                        'latency_total_network': ['Exec Time', 'Network BW'],
                        'latency_network_raw': ['Exec Time', 'Network BW'],
                        'latency_network_minmax': ['Normalized Time', 'Normalized BW'],
                        'latency_network_sqrt': ['sqrt(Time)', 'sqrt(Network BW)'],
                        'latency_memory': ['log10(Exec Time)', 'log10(Memory)'],
                        'network_memory': ['log10(Network BW)', 'log10(Memory)'],
                        'power_and_time': ['Power (W)', 'Exec Time'],
                        'energy_and_time': ['Energy (J)', 'Exec Time'],
                        'edp_and_network_bw': ['EDP', 'Network BW'],
                        'ed2p_and_network_bw': ['ED²P', 'Network BW'],
                        'e2d_and_network_bw': ['E²D', 'Network BW'],
                    }
                    plot_labels = plot_labels_map.get(objective_key, ['Obj 0', 'Obj 1'])
                    
                    pareto_plots = plot_pareto_front(
                        results_file=csv_path,
                        obj0_name=plot_labels[0],
                        obj1_name=plot_labels[1],
                        output_file=output_base,
                        plot_format="both",
                        show_labels=True,
                        remove_outliers=True,
                        iqr_multiplier=1.5,
                    )
                    
                    print(f"✓ Pareto plots saved:")
                    for plot_path in pareto_plots:
                        if plot_path.endswith('.html'):
                            print(f"    - Interactive: {plot_path}")
                        elif plot_path.endswith('.png'):
                            print(f"    - Static: {plot_path}")
                except Exception as e:
                    print(f"⚠️  Could not generate Pareto plots: {e}")
            elif n_obj > 2 and HAS_PLOT_PARETO:
                print(f"ℹ️  Skipping Pareto plots for {n_obj}-objective optimization")
        except Exception as e:
            print(f"⚠️  Error during visualization: {e}")
        
        sys.exit(0)
    else:
        print(f"\n❌ No best configuration found")
        sys.exit(1)
        
except KeyboardInterrupt:
    print(f"\n⚠️  Interrupted by user")
    sys.exit(1)
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
PYTHON_CODE_BLOCK
}

# Main optimization loop
for model_name in "${!MODELS[@]}"; do
    model_num=${MODELS[$model_name]}
    
    for config_name in "${!CONFIGS[@]}"; do
        config_file="${CONFIGS[$config_name]}"
        config_path="../search_space/${config_file}"
        
        # Validate search space file exists
        if [ ! -f "$config_path" ]; then
            echo "⚠️  WARNING: Search space not found: $config_path"
            echo "    Skipping $config_name for $model_name"
            continue
        fi
        
        for objective in "${OBJECTIVES[@]}"; do
            ((RUN_NUMBER++))
            
            # Honor max-runs limit
            if [ -n "$MAX_RUNS" ] && [ "$RUN_NUMBER" -gt "$MAX_RUNS" ]; then
                ((SKIPPED_RUNS++))
                continue
            fi
            
            MODEL_DISPLAY="${model_name}_${config_name}_${objective}"
            SCRIPT_FILE="/tmp/astra_opt_${RUN_NUMBER}.py"
            
            echo "[${RUN_NUMBER}/${TOTAL_RUNS}] $MODEL_DISPLAY"
            
            # Generate and run the optimization script
            generate_optimization_script "$model_num" "$model_name" "$config_path" "$config_name" "$objective" "$SCRIPT_FILE"
            
            export MODEL_NUM="$model_num"
            export MODEL_NAME="$model_name"
            export CONFIG_PATH="$config_path"
            export CONFIG_NAME="$config_name"
            export OBJECTIVE="$objective"
            
            if PYTHONUNBUFFERED=1 "$PYTHON_BIN" "$SCRIPT_FILE" 2>&1; then
                ((COMPLETED_RUNS++))
                echo "✓ Completed"
                echo ""
            else
                ((FAILED_RUNS++))
                FAILED_RUNS_LIST+=("$MODEL_DISPLAY")
                if [ "$SKIP_FAILED" = true ]; then
                    echo "✗ Failed (skipping remaining variants of this model)"
                    break 2
                fi
                echo "✗ Failed"
                echo ""
            fi
            
            rm -f "$SCRIPT_FILE"
        done
    done
done

# Summary
echo "════════════════════════════════════════════════════════════════════════════════"
echo "OPTIMIZATION SWEEP SUMMARY"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "  Total runs:       $TOTAL_RUNS"
echo "  Completed: ✓      $COMPLETED_RUNS"
echo "  Failed: ✗         $FAILED_RUNS"
if [ -n "$MAX_RUNS" ]; then
    echo "  Skipped:          $SKIPPED_RUNS (--max-runs limit)"
fi
SUCCESS_RATE=$((COMPLETED_RUNS * 100 / (TOTAL_RUNS - SKIPPED_RUNS) ))
echo "  Success rate:     $SUCCESS_RATE%"
echo ""
if [ $FAILED_RUNS -gt 0 ]; then
    echo "Failed optimization runs:"
    for failed in "${FAILED_RUNS_LIST[@]}"; do
        echo "  - $failed"
    done
    echo ""
fi
echo "Results saved in: SWEEP_DEEPHYPER_* directories"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

[ $FAILED_RUNS -eq 0 ]
