#!/usr/bin/env bash
#SBATCH -q large
set -euo pipefail

EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$EXP_DIR/config.env"

mkdir -p "$EXP_DIR/logs" "$EXP_DIR/outputs"

# Resolve core paths robustly so the suite remains valid after moving directories.
DEFAULT_SWEEP_SCRIPT="$(cd "$EXP_DIR/../../../Optimization/examples" && pwd)/example_deephyper_opt_sweep.py"
DEFAULT_START_ENV_SCRIPT="$(cd "$EXP_DIR/../../../../../" && pwd)/start_py_311.sh"
DEFAULT_SEARCH_SPACE_PATH="$EXP_DIR/inputs/search_space.json"
DEFAULT_RESULT_FOLDER_PREFIX="$EXP_DIR/outputs"

if [[ -z "${SWEEP_SCRIPT:-}" || ! -f "$SWEEP_SCRIPT" ]]; then
  SWEEP_SCRIPT="$DEFAULT_SWEEP_SCRIPT"
fi
if [[ -z "${START_ENV_SCRIPT:-}" || ! -f "$START_ENV_SCRIPT" ]]; then
  START_ENV_SCRIPT="$DEFAULT_START_ENV_SCRIPT"
fi
if [[ -z "${SEARCH_SPACE_PATH:-}" || ! -f "$SEARCH_SPACE_PATH" ]]; then
  SEARCH_SPACE_PATH="$DEFAULT_SEARCH_SPACE_PATH"
fi
if [[ -z "${RESULT_FOLDER_PREFIX:-}" ]]; then
  RESULT_FOLDER_PREFIX="$DEFAULT_RESULT_FOLDER_PREFIX"
fi

# Expand relative paths from config.env against EXP_DIR.
if [[ "$SWEEP_SCRIPT" != /* ]]; then
  SWEEP_SCRIPT="$EXP_DIR/$SWEEP_SCRIPT"
fi
if [[ "$START_ENV_SCRIPT" != /* ]]; then
  START_ENV_SCRIPT="$EXP_DIR/$START_ENV_SCRIPT"
fi
if [[ "$SEARCH_SPACE_PATH" != /* ]]; then
  SEARCH_SPACE_PATH="$EXP_DIR/$SEARCH_SPACE_PATH"
fi
if [[ "$RESULT_FOLDER_PREFIX" != /* ]]; then
  RESULT_FOLDER_PREFIX="$EXP_DIR/$RESULT_FOLDER_PREFIX"
fi

if [[ ! -f "$START_ENV_SCRIPT" ]]; then
  echo "start_py_311.sh not found: $START_ENV_SCRIPT" >&2
  exit 1
fi
if [[ ! -f "$SWEEP_SCRIPT" ]]; then
  echo "Sweep script not found: $SWEEP_SCRIPT" >&2
  exit 1
fi
if [[ ! -f "$SEARCH_SPACE_PATH" ]]; then
  echo "Search space file not found: $SEARCH_SPACE_PATH" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "$START_ENV_SCRIPT"

if [[ -n "${SLURM_CPUS_PER_TASK:-}" ]]; then
  export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
  export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
  export NUMEXPR_NUM_THREADS="$SLURM_CPUS_PER_TASK"
fi

N_WORKERS_EFFECTIVE="${N_WORKERS_OVERRIDE:-${SLURM_CPUS_PER_TASK:-$N_WORKERS}}"

TS="$(date +%Y%m%d_%H%M%S)"
RUN_PREFIX="$RESULT_FOLDER_PREFIX/run_${TS}"
mkdir -p "$RUN_PREFIX"

LOG_FILE="$EXP_DIR/logs/optimization_${TS}.log"

echo "Starting experiment: $EXP_NAME" | tee -a "$LOG_FILE"
echo "SLURM job: ${SLURM_JOB_ID:-N/A}" | tee -a "$LOG_FILE"
echo "Node: ${SLURMD_NODENAME:-$(hostname)}" | tee -a "$LOG_FILE"
echo "CPUs: ${SLURM_CPUS_PER_TASK:-N/A}" | tee -a "$LOG_FILE"

time python "$SWEEP_SCRIPT" \
  --objective "$OBJECTIVE_KEY" \
  --model-num "$MODEL_NUM" \
  --model-name "$MODEL_NAME" \
  --num-npus "$NUM_NPUS" \
  --network-name "FoldedClos" \
  --budget "$BUDGET" \
  --init-samples "$INIT_SAMPLES" \
  --n-workers "$N_WORKERS_EFFECTIVE" \
  --top-k "$TOP_K" \
  --cleanup-batch-size "$CLEANUP_BATCH_SIZE" \
  --folder-prefix "$RUN_PREFIX" \
  --search-space-path "$SEARCH_SPACE_PATH" \
  --sim-type "$SIM_TYPE" \
  --topology "$TOPOLOGY" \
  --routing-mode "$ROUTING_MODE" \
  --compress-and-clean \
  2>&1 | tee -a "$LOG_FILE"

echo "Completed experiment: $EXP_NAME" | tee -a "$LOG_FILE"
