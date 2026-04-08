#!/usr/bin/env bash
ulimit -c 0
#SBATCH -q large
set -euo pipefail

###############################################################################
# run_experiment_campaign.sh
#
# Campaign-aware variant of run_experiment.sh.
# Reads the original experiment config.env from ORIG_EXP_DIR, then applies
# campaign overrides via CAMPAIGN_* environment variables set by
# launch_campaigns.sh before sbatch submission.
#
# Required env vars (set by launch_campaigns.sh):
#   ORIG_EXP_DIR               Path to the original experiment directory
#   CAMPAIGN_NAME              Name of this campaign run
#   CAMPAIGN_ENABLE_TRACKER    1 = enable tracker flag, 0 = omit it
#   CAMPAIGN_EARLY_STOPPING    Patience value for --early-stopping-patience
#   CAMPAIGN_N_WORKERS         Number of workers
#   CAMPAIGN_SEARCH_TYPE       DeepHyper search type: cbo (default) or random
#   CAMPAIGN_OUTPUT_DIR        Where to write outputs and logs for this campaign
###############################################################################

# ── Resolve experiment directory ──────────────────────────────────────────────
if [[ -z "${ORIG_EXP_DIR:-}" ]]; then
  echo "Error: ORIG_EXP_DIR not set. This script must be launched via launch_campaigns.sh" >&2
  exit 1
fi
if [[ ! -f "$ORIG_EXP_DIR/config.env" ]]; then
  echo "Error: config.env not found in $ORIG_EXP_DIR" >&2
  exit 1
fi

EXP_DIR="$ORIG_EXP_DIR"
source "$EXP_DIR/config.env"

# ── Apply campaign overrides ──────────────────────────────────────────────────
CAMPAIGN_NAME="${CAMPAIGN_NAME:?CAMPAIGN_NAME not set}"
CAMPAIGN_OUTPUT_DIR="${CAMPAIGN_OUTPUT_DIR:?CAMPAIGN_OUTPUT_DIR not set}"
CAMPAIGN_ENABLE_TRACKER="${CAMPAIGN_ENABLE_TRACKER:-0}"
CAMPAIGN_EARLY_STOPPING="${CAMPAIGN_EARLY_STOPPING:-1}"
CAMPAIGN_N_WORKERS="${CAMPAIGN_N_WORKERS:-1}"
CAMPAIGN_SEARCH_TYPE="${CAMPAIGN_SEARCH_TYPE:-cbo}"

mkdir -p "$CAMPAIGN_OUTPUT_DIR/logs" "$CAMPAIGN_OUTPUT_DIR/outputs"

# ── Resolve core paths ───────────────────────────────────────────────────────
DEFAULT_SWEEP_SCRIPT="$(cd "$EXP_DIR/../../../Optimization/examples" && pwd)/example_deephyper_opt_sweep.py"
DEFAULT_START_ENV_SCRIPT="$(cd "$EXP_DIR/../../../../../" && pwd)/start_py_311.sh"
DEFAULT_SEARCH_SPACE_PATH="$EXP_DIR/inputs/search_space.json"

if [[ -z "${SWEEP_SCRIPT:-}" || ! -f "$SWEEP_SCRIPT" ]]; then
  SWEEP_SCRIPT="$DEFAULT_SWEEP_SCRIPT"
fi
if [[ -z "${START_ENV_SCRIPT:-}" || ! -f "$START_ENV_SCRIPT" ]]; then
  START_ENV_SCRIPT="$DEFAULT_START_ENV_SCRIPT"
fi
if [[ -z "${SEARCH_SPACE_PATH:-}" || ! -f "$SEARCH_SPACE_PATH" ]]; then
  SEARCH_SPACE_PATH="$DEFAULT_SEARCH_SPACE_PATH"
fi

# Expand relative paths against EXP_DIR.
if [[ "$SWEEP_SCRIPT" != /* ]]; then
  SWEEP_SCRIPT="$EXP_DIR/$SWEEP_SCRIPT"
fi
if [[ "$START_ENV_SCRIPT" != /* ]]; then
  START_ENV_SCRIPT="$EXP_DIR/$START_ENV_SCRIPT"
fi
if [[ "$SEARCH_SPACE_PATH" != /* ]]; then
  SEARCH_SPACE_PATH="$EXP_DIR/$SEARCH_SPACE_PATH"
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

# ── Workers: campaign override takes precedence ──────────────────────────────
N_WORKERS_EFFECTIVE="$CAMPAIGN_N_WORKERS"

# ── Early stopping: use campaign setting directly ────────────────────────────
# The sweep script multiplies this value by 30 internally.
# Pass 1 to effectively disable, or a desired base patience value.
EARLY_STOPPING_PATIENCE_ARG="$CAMPAIGN_EARLY_STOPPING"
EARLY_STOPPING_MIN_EVALUATIONS="${EARLY_STOPPING_MIN_EVALUATIONS:-${INIT_SAMPLES}}"

# ── Output paths ─────────────────────────────────────────────────────────────
TS="$(date +%Y%m%d_%H%M%S)"
RESULT_FOLDER_PREFIX="$CAMPAIGN_OUTPUT_DIR/outputs/run_${TS}"
mkdir -p "$RESULT_FOLDER_PREFIX"

UPC_ROOT="${ASTRA_SIM_ROOT}/upc"
if [[ "$CAMPAIGN_OUTPUT_DIR" == "$UPC_ROOT"/* ]]; then
  SIM_FOLDER_PREFIX="${CAMPAIGN_OUTPUT_DIR#"$UPC_ROOT"/}/run_${TS}"
else
  SIM_FOLDER_PREFIX="campaigns/${CAMPAIGN_NAME}/${EXP_NAME}/run_${TS}"
fi

LOG_FILE="$CAMPAIGN_OUTPUT_DIR/logs/optimization_${TS}.log"

echo "═══════════════════════════════════════════════════" | tee -a "$LOG_FILE"
echo "Campaign experiment: $EXP_NAME" | tee -a "$LOG_FILE"
echo "Campaign:      $CAMPAIGN_NAME" | tee -a "$LOG_FILE"
echo "SLURM job:     ${SLURM_JOB_ID:-N/A}" | tee -a "$LOG_FILE"
echo "Node:          ${SLURMD_NODENAME:-$(hostname)}" | tee -a "$LOG_FILE"
echo "CPUs:          ${SLURM_CPUS_PER_TASK:-N/A}" | tee -a "$LOG_FILE"
echo "Workers:       $N_WORKERS_EFFECTIVE" | tee -a "$LOG_FILE"
echo "Tracker:       $CAMPAIGN_ENABLE_TRACKER" | tee -a "$LOG_FILE"
echo "Early stop:    $EARLY_STOPPING_PATIENCE_ARG (×30 multiplier in sweep)" | tee -a "$LOG_FILE"
echo "Search type:   $CAMPAIGN_SEARCH_TYPE" | tee -a "$LOG_FILE"
echo "Output dir:    $CAMPAIGN_OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "═══════════════════════════════════════════════════" | tee -a "$LOG_FILE"

# ── Tracker flag: only pass --enable-tracker when actually enabled ────────────
TRACKER_FLAG=()
if [[ "$CAMPAIGN_ENABLE_TRACKER" == "1" ]]; then
  TRACKER_FLAG=("--enable-tracker")
fi

# Use node-local temp storage for Python multiprocessing artifacts.
TMP_BASE="${SLURM_TMPDIR:-/tmp}"
JOB_TMP_DIR="${TMP_BASE%/}/astra_tmp_${SLURM_JOB_ID:-$$}"
mkdir -p "$JOB_TMP_DIR"
export TMPDIR="$JOB_TMP_DIR"
export TMP="$JOB_TMP_DIR"
export TEMP="$JOB_TMP_DIR"
export STG_TMP_DIR="/scratch/nas/4/nasser/tmp/stg_${SLURM_JOB_ID:-$$}"
mkdir -p "$STG_TMP_DIR"

_EXP_START=$(date +%s)
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
  --folder-prefix "$SIM_FOLDER_PREFIX" \
  --search-space-path "$SEARCH_SPACE_PATH" \
  --sim-type "$SIM_TYPE" \
  --topology "$TOPOLOGY" \
  --routing-mode "$ROUTING_MODE" \
  --compress-and-clean \
  --include-categories "$INCLUDE_CATEGORIES" \
  "${TRACKER_FLAG[@]}" \
  --early-stopping-patience "$EARLY_STOPPING_PATIENCE_ARG" \
  --early-stopping-min-evaluations "$EARLY_STOPPING_MIN_EVALUATIONS" \
  --search-type "$CAMPAIGN_SEARCH_TYPE" \
  2>&1 | tee -a "$LOG_FILE"

_EXP_END=$(date +%s)
_ELAPSED=$(( _EXP_END - _EXP_START ))
printf "Experiment wall time: %02dh %02dm %02ds (%ds total)\n" \
  $(( _ELAPSED/3600 )) $(( (_ELAPSED%3600)/60 )) $(( _ELAPSED%60 )) "$_ELAPSED" \
  | tee -a "$LOG_FILE"

echo "Completed campaign experiment: $CAMPAIGN_NAME / $EXP_NAME" | tee -a "$LOG_FILE"
