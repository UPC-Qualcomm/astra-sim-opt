#!/usr/bin/env bash
# launch_continuous.sh
#
# Continuous scheduler: keeps looping over all nodes until every experiment in
# the manifest is submitted to SLURM on a node that has sufficient resources.
#
# Strategy per pass:
#   For each still-pending experiment, scan all nodes in order.  The first node
#   that has enough free CPUs *and* memory gets the job.  After the whole
#   pending list has been scanned once, if no progress was made we sleep
#   RETRY_SLEEP_SECONDS and try again (jobs may finish and free resources).
#
# CPU policy : 32% of the node's total CPUs, capped at MAX_CPUS_PER_EXPERIMENT.
# Memory policy: DEFAULT_MEM_PER_CPU_GB (2 GB/core) for all experiments,
#                overridden to 10 GB/core for any experiment whose config.env
#                sets MEM_PER_CPU_GB_OVERRIDE=10 (i.e. the gpt175b jobs).
#
# The same node may host multiple jobs as long as resources allow.
# Local "committed" counters per node are updated on every submission so that
# back-to-back submissions within a single pass don't double-book the same CPUs.
# These committed values are reset at the start of each new pass when real-time
# data from scontrol is refreshed.

set -euo pipefail

ROOT_DIR="/scratch/nas/4/nasser/astra-sim/upc/slurm_optimization_suite_500"
EXPERIMENTS_DIR="$ROOT_DIR/experiments"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LAUNCH_LOG_DIR="$ROOT_DIR/launch_logs/$TIMESTAMP"
mkdir -p "$LAUNCH_LOG_DIR"

if [[ ! -d "$EXPERIMENTS_DIR" ]]; then
  echo "Error: Experiments directory not found at: $EXPERIMENTS_DIR" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

NODE_STATES="idle,mix"

SKIP_NODES=("sert-2201" "sert-1430" "sert-1419" "sert-1434" "sert-1433" "sert-1431" "sert-1425" "sert-1424" "sert-1906")

DEFAULT_CORES_PERCENT=32
DEFAULT_MEM_PER_CPU_GB=2
MAX_CPUS_PER_EXPERIMENT=8
MIN_CPUS_PER_EXPERIMENT=4   # Never schedule fewer than this many CPUs; skip node if memory can't fit even this many0

# How long to sleep (seconds) between passes when no experiment could be placed.
RETRY_SLEEP_SECONDS=60

# ---------------------------------------------------------------------------
# Experiment manifest — comment out experiments you don't want to run
# ---------------------------------------------------------------------------
ACTIVE_EXPERIMENTS=(
  "llama70b_128npus_edp"
  "llama70b_128npus_edp_and_bw"
  "llama70b_128npus_energy_and_time"
  "llama70b_128npus_memory_and_time"
  "llama70b_128npus_time"
  "llama70b_128npus_time_and_bw"
  "llama70b_128npus_time_and_throughput_per_energy"
  "gpt60b_128npus_edp"
  "gpt60b_128npus_edp_and_bw"
  "gpt60b_128npus_energy_and_time"
  "gpt60b_128npus_memory_and_time"
  "gpt60b_128npus_time"
  "gpt60b_128npus_time_and_bw"
  "gpt60b_128npus_time_and_throughput_per_energy"
  "gpt175b_1024npus_edp"
  "gpt175b_1024npus_edp_and_bw"
  "gpt175b_1024npus_energy_and_time"
  "gpt175b_1024npus_memory_and_time"
  "gpt175b_1024npus_time"
  "gpt175b_1024npus_time_and_bw"
  "gpt175b_1024npus_time_and_throughput_per_energy"
  "llama8b_32npus_edp"
  "llama8b_32npus_edp_and_bw"
  "llama8b_32npus_energy_and_time"
  "llama8b_32npus_memory_and_time"
  "llama8b_32npus_time"
  "llama8b_32npus_time_and_bw"
  "llama8b_32npus_time_and_throughput_per_energy"
)

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
usage() {
  cat <<USAGE
Usage: bash launch_continuous.sh [--dry-run] [--partition PARTITION] [--max-jobs N] [--sleep SECONDS]

Options:
  --dry-run            Print sbatch commands without submitting.
  --partition PART     Force SLURM partition for all jobs.
  --max-jobs N         Submit at most N experiments in total.
  --sleep SECONDS      Seconds to wait between passes when no job fits (default: $RETRY_SLEEP_SECONDS).
  -h|--help            Show this message.
USAGE
}

DRY_RUN=0
FORCED_PARTITION=""
MAX_JOBS=0
SLURM_QOS="large"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)    DRY_RUN=1; shift ;;
    --partition)  FORCED_PARTITION="${2:-}"; shift 2 ;;
    --max-jobs)   MAX_JOBS="${2:-0}"; shift 2 ;;
    --sleep)      RETRY_SLEEP_SECONDS="${2:-60}"; shift 2 ;;
    -h|--help)    usage; exit 0 ;;
    *)            echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

if ! command -v sinfo >/dev/null 2>&1 || ! command -v sbatch >/dev/null 2>&1; then
  echo "Error: sinfo/sbatch not available in PATH. Run on a SLURM login node." >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Build experiment list
# ---------------------------------------------------------------------------
ALL_EXP_DIRS=()
for exp_name in "${ACTIVE_EXPERIMENTS[@]}"; do
  exp_path="$EXPERIMENTS_DIR/$exp_name"
  if [[ -d "$exp_path" ]]; then
    ALL_EXP_DIRS+=("$exp_path")
  else
    echo "Warning: experiment directory not found: $exp_path" >&2
  fi
done

if [[ ${#ALL_EXP_DIRS[@]} -eq 0 ]]; then
  echo "No active experiments found in $EXPERIMENTS_DIR" >&2
  exit 1
fi

if [[ "$MAX_JOBS" -gt 0 ]] && [[ "$MAX_JOBS" -lt "${#ALL_EXP_DIRS[@]}" ]]; then
  ALL_EXP_DIRS=("${ALL_EXP_DIRS[@]:0:$MAX_JOBS}")
fi

# ---------------------------------------------------------------------------
# CSV log setup
# ---------------------------------------------------------------------------
ASSIGNMENT_CSV="$LAUNCH_LOG_DIR/assignments.csv"
cat > "$ASSIGNMENT_CSV" <<CSV
experiment,node,partition,node_cores_total,node_mem_total_mb,cpu_free_at_submit,mem_free_mb_at_submit,cpus_per_task,mem_per_cpu_gb,mem_per_cpu_slurm,job_name,submit_status,job_id
CSV

# ---------------------------------------------------------------------------
# Helper: get real-time free resources for a node.
# Echoes: cpu_tot|mem_tot_mb|cpu_free|mem_free_mb
# ---------------------------------------------------------------------------
get_node_free_resources() {
  local node_name="$1"
  local node_info
  node_info="$(scontrol show node "$node_name" 2>/dev/null || true)"

  local cpu_tot cpu_alloc mem_tot alloc_mem
  cpu_tot="$(grep -oE 'CPUTot=[0-9]+' <<< "$node_info" | head -n1 | cut -d= -f2)"
  cpu_alloc="$(grep -oE 'CPUAlloc=[0-9]+' <<< "$node_info" | head -n1 | cut -d= -f2)"
  mem_tot="$(grep -oE 'RealMemory=[0-9]+' <<< "$node_info" | head -n1 | cut -d= -f2)"
  alloc_mem="$(grep -oE 'AllocMem=[0-9]+' <<< "$node_info" | head -n1 | cut -d= -f2)"

  cpu_tot="${cpu_tot:-0}"; cpu_alloc="${cpu_alloc:-0}"
  mem_tot="${mem_tot:-0}"; alloc_mem="${alloc_mem:-0}"

  local cpu_free=$(( cpu_tot - cpu_alloc ))
  local mem_free=$(( mem_tot - alloc_mem ))
  (( cpu_free < 0 )) && cpu_free=0
  (( mem_free < 0 )) && mem_free=0

  echo "${cpu_tot}|${mem_tot}|${cpu_free}|${mem_free}"
}

# ---------------------------------------------------------------------------
# Helper: check whether a given node name should be globally excluded.
# Returns 0 if excluded, 1 if usable.
# ---------------------------------------------------------------------------
is_skipped_node() {
  local node="$1"
  for excluded in "${SKIP_NODES[@]}"; do
    [[ "$node" == "$excluded" ]] && return 0
  done
  return 1
}

# ---------------------------------------------------------------------------
# Build node list from sinfo (refreshed at the start of every pass)
# ---------------------------------------------------------------------------
# Arrays populated by refresh_nodes:
NODES=()
PARTITIONS=()
NODE_CORES_TOTAL=()
NODE_MEM_TOTAL_MB=()
# Per-node CPUs/mem committed by US this pass (reset on every refresh)
NODE_COMMITTED_CPUS=()
NODE_COMMITTED_MEM_MB=()

refresh_nodes() {
  local sinfo_out
  sinfo_out="$(sinfo -N -h -t "$NODE_STATES" -o "%N|%P|%t|%c|%m" | sort -rV)"

  NODES=()
  PARTITIONS=()
  NODE_CORES_TOTAL=()
  NODE_MEM_TOTAL_MB=()
  NODE_COMMITTED_CPUS=()
  NODE_COMMITTED_MEM_MB=()

  while IFS='|' read -r node partition _state cores mem_mb; do
    [[ -z "$node" ]] && continue
    is_skipped_node "$node" && continue
    partition="${partition%%\**}"
    NODES+=("$node")
    PARTITIONS+=("$partition")
    NODE_CORES_TOTAL+=("$cores")
    NODE_MEM_TOTAL_MB+=("$mem_mb")
    NODE_COMMITTED_CPUS+=("0")
    NODE_COMMITTED_MEM_MB+=("0")
  done <<< "$sinfo_out"
}

# ---------------------------------------------------------------------------
# Find a suitable node for an experiment.
# Sets globals: CHOSEN_NODE_IDX, CHOSEN_CPUS, CHOSEN_MEM_MB
# Returns 0 on success, 1 if no node is suitable right now.
# ---------------------------------------------------------------------------
CHOSEN_NODE_IDX=-1
CHOSEN_CPUS=0
CHOSEN_MEM_MB=0

find_node_for_experiment() {
  local mem_per_cpu_gb="$1"     # required GB per CPU
  local cpus_override="$2"      # "" or hard cpu count

  CHOSEN_NODE_IDX=-1
  CHOSEN_CPUS=0
  CHOSEN_MEM_MB=0

  for ((ni=0; ni<${#NODES[@]}; ni++)); do
    local node="${NODES[$ni]}"

    # Real-time resource snapshot from SLURM
    local res_line
    res_line="$(get_node_free_resources "$node")"
    IFS='|' read -r cpu_tot mem_tot_mb cpu_free mem_free_mb <<< "$res_line"

    # Subtract CPUs/mem we already committed to this node in this pass
    local effective_cpu_free=$(( cpu_free - NODE_COMMITTED_CPUS[$ni] ))
    local effective_mem_free=$(( mem_free_mb - NODE_COMMITTED_MEM_MB[$ni] ))
    (( effective_cpu_free < 0 )) && effective_cpu_free=0
    (( effective_mem_free < 0 )) && effective_mem_free=0

    (( effective_cpu_free < 1 )) && continue

    # CPU budget: 32% of total, capped at MAX_CPUS_PER_EXPERIMENT
    local policy_max_cpus=$(( cpu_tot * DEFAULT_CORES_PERCENT / 100 ))
    (( policy_max_cpus < 1 )) && policy_max_cpus=1

    local candidate_cpus
    if [[ -n "$cpus_override" ]]; then
      candidate_cpus="$cpus_override"
    else
      candidate_cpus="$policy_max_cpus"
      (( candidate_cpus > MAX_CPUS_PER_EXPERIMENT )) && candidate_cpus="$MAX_CPUS_PER_EXPERIMENT"
    fi

    # Can't use more than what is currently free
    (( candidate_cpus > effective_cpu_free )) && candidate_cpus="$effective_cpu_free"
    (( candidate_cpus < MIN_CPUS_PER_EXPERIMENT )) && continue

    # Check memory: required = candidate_cpus * mem_per_cpu_gb * 1024 MB
    local required_mem_mb=$(( candidate_cpus * mem_per_cpu_gb * 1024 ))

    if (( required_mem_mb > effective_mem_free )); then
      # Try fewer CPUs to fit within memory budget
      local max_by_mem=$(( effective_mem_free / (mem_per_cpu_gb * 1024) ))
      (( max_by_mem < MIN_CPUS_PER_EXPERIMENT )) && continue
      (( max_by_mem < candidate_cpus )) && candidate_cpus="$max_by_mem"
      required_mem_mb=$(( candidate_cpus * mem_per_cpu_gb * 1024 ))
    fi

    # Final check: enforce minimum after all adjustments
    (( candidate_cpus < MIN_CPUS_PER_EXPERIMENT )) && continue

    # This node fits — select it
    CHOSEN_NODE_IDX="$ni"
    CHOSEN_CPUS="$candidate_cpus"
    CHOSEN_MEM_MB="$required_mem_mb"
    return 0
  done

  return 1
}

# ---------------------------------------------------------------------------
# Submit a single experiment to the already-selected node.
# Returns 0 on success, 1 on sbatch error.
# ---------------------------------------------------------------------------
submit_experiment() {
  local exp_dir="$1"
  local node_idx="$2"
  local cpus_per_task="$3"
  local requested_mem_mb="$4"
  local mem_per_cpu_gb="$5"

  local node="${NODES[$node_idx]}"
  local partition="${PARTITIONS[$node_idx]}"

  local submit_partition="$partition"
  [[ -n "$FORCED_PARTITION" ]] && submit_partition="$FORCED_PARTITION"
  [[ -n "${PARTITION_OVERRIDE:-}" ]] && submit_partition="${PARTITION_OVERRIDE}"

  local mem_per_cpu_slurm="${mem_per_cpu_gb}G"
  local job_name="${JOB_NAME:-$(basename "$exp_dir")}"
  local logs_dir="$exp_dir/logs"
  mkdir -p "$logs_dir"

  local run_script="$exp_dir/run_experiment.sh"
  local wrap_cmd="bash \"$run_script\""

  local sbatch_cmd=(
    sbatch
    --job-name "$job_name"
    --chdir "$exp_dir"
    #--nodelist "$node"
    --partition "$submit_partition"
    --cpus-per-task "$cpus_per_task"
    --mem-per-cpu "$mem_per_cpu_slurm"
    --export "ALL,N_WORKERS_OVERRIDE=$cpus_per_task"
    --output "$logs_dir/slurm-%j.out"
    --error  "$logs_dir/slurm-%j.err"
    --wrap "$wrap_cmd"
  )

  [[ -n "${SLURM_ACCOUNT:-}" ]] && sbatch_cmd+=(--account "$SLURM_ACCOUNT")
  [[ -n "${SLURM_QOS:-}"     ]] && sbatch_cmd+=(--qos    "$SLURM_QOS")
  [[ -n "${SLURM_EXTRA_ARGS:-}" ]] && sbatch_cmd+=($SLURM_EXTRA_ARGS)

  # Gather current node totals for the log
  local res_line
  res_line="$(get_node_free_resources "$node")"
  IFS='|' read -r cpu_tot_log mem_tot_log cpu_free_log mem_free_log <<< "$res_line"

  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[DRY-RUN] ${sbatch_cmd[*]}"
    echo "$(basename "$exp_dir"),$node,$submit_partition,$cpu_tot_log,$mem_tot_log,$cpu_free_log,$mem_free_log,$cpus_per_task,$mem_per_cpu_gb,$mem_per_cpu_slurm,$job_name,DRY_RUN," >> "$ASSIGNMENT_CSV"
    return 0
  fi

  set +e
  local out rc
  out="$("${sbatch_cmd[@]}" 2>&1)"
  rc=$?
  set -e

  if [[ $rc -eq 0 ]]; then
    local job_id
    job_id="$(awk '{print $NF}' <<< "$out")"
    echo "  Submitted $(basename "$exp_dir") -> $node  (job $job_id, cpus=$cpus_per_task, mem/cpu=${mem_per_cpu_slurm})"
    echo "$(basename "$exp_dir"),$node,$submit_partition,$cpu_tot_log,$mem_tot_log,$cpu_free_log,$mem_free_log,$cpus_per_task,$mem_per_cpu_gb,$mem_per_cpu_slurm,$job_name,SUBMITTED,$job_id" >> "$ASSIGNMENT_CSV"
    return 0
  else
    echo "  sbatch failed for $(basename "$exp_dir") on $node: $out" >&2
    echo "$(basename "$exp_dir"),$node,$submit_partition,$cpu_tot_log,$mem_tot_log,$cpu_free_log,$mem_free_log,$cpus_per_task,$mem_per_cpu_gb,$mem_per_cpu_slurm,$job_name,SBATCH_ERROR," >> "$ASSIGNMENT_CSV"
    return 1
  fi
}

# ---------------------------------------------------------------------------
# Main scheduling loop
# ---------------------------------------------------------------------------

# Pending list: indices into ALL_EXP_DIRS
PENDING_INDICES=()
for ((i=0; i<${#ALL_EXP_DIRS[@]}; i++)); do
  PENDING_INDICES+=("$i")
done

total_submitted=0
pass=0

echo "============================================================"
echo "launch_continuous.sh  started at $(date)"
echo "Experiments to submit : ${#ALL_EXP_DIRS[@]}"
echo "Logs                  : $LAUNCH_LOG_DIR"
echo "============================================================"
echo

while [[ ${#PENDING_INDICES[@]} -gt 0 ]]; do
  pass=$(( pass + 1 ))
  echo "--- Pass $pass  (pending: ${#PENDING_INDICES[@]})  $(date '+%H:%M:%S') ---"

  # Refresh node list + reset committed counters at the top of every pass
  refresh_nodes

  if [[ ${#NODES[@]} -eq 0 ]]; then
    echo "  No nodes in states '$NODE_STATES' — sleeping ${RETRY_SLEEP_SECONDS}s ..."
    sleep "$RETRY_SLEEP_SECONDS"
    continue
  fi

  progress_this_pass=0
  STILL_PENDING=()

  for idx in "${PENDING_INDICES[@]}"; do
    exp_dir="${ALL_EXP_DIRS[$idx]}"
    exp_name="$(basename "$exp_dir")"
    cfg="$exp_dir/config.env"
    run_script="$exp_dir/run_experiment.sh"

    if [[ ! -f "$cfg" || ! -f "$run_script" ]]; then
      echo "  SKIP $exp_name (missing config.env or run_experiment.sh)"
      continue   # drop from pending permanently
    fi

    # Source config to pick up MEM_PER_CPU_GB_OVERRIDE / CPUS_PER_TASK_OVERRIDE
    # shellcheck disable=SC1090
    (
      # Run in a subshell so variables don't leak between experiments
      source "$cfg"
      mem_per_cpu_gb="${MEM_PER_CPU_GB_OVERRIDE:-$DEFAULT_MEM_PER_CPU_GB}"
      cpus_override="${CPUS_PER_TASK_OVERRIDE:-}"
      echo "${mem_per_cpu_gb}|${cpus_override}|${JOB_NAME:-$exp_name}|${PARTITION_OVERRIDE:-}"
    ) > /tmp/_launch_cfg_$$
    IFS='|' read -r _mem_per_cpu _cpus_override _job_name _part_override < /tmp/_launch_cfg_$$
    rm -f /tmp/_launch_cfg_$$

    mem_per_cpu_gb="$_mem_per_cpu"
    cpus_override="$_cpus_override"
    JOB_NAME="$_job_name"
    PARTITION_OVERRIDE="$_part_override"

    if find_node_for_experiment "$mem_per_cpu_gb" "$cpus_override"; then
      ni="$CHOSEN_NODE_IDX"
      cpus="$CHOSEN_CPUS"
      req_mem_mb="$CHOSEN_MEM_MB"

      # Reserve resources in our in-pass committed tracking so the next
      # experiment in this same pass doesn't double-book the same node.
      NODE_COMMITTED_CPUS[$ni]=$(( NODE_COMMITTED_CPUS[$ni] + cpus ))
      NODE_COMMITTED_MEM_MB[$ni]=$(( NODE_COMMITTED_MEM_MB[$ni] + req_mem_mb ))

      if submit_experiment "$exp_dir" "$ni" "$cpus" "$req_mem_mb" "$mem_per_cpu_gb"; then
        (( total_submitted++ ))
        (( progress_this_pass++ ))
        # Experiment is no longer pending — do NOT add to STILL_PENDING
      else
        # sbatch error — put back on pending list so we retry next pass
        STILL_PENDING+=("$idx")
        # Undo the committed reservation since the job wasn't actually placed
        NODE_COMMITTED_CPUS[$ni]=$(( NODE_COMMITTED_CPUS[$ni] - cpus ))
        NODE_COMMITTED_MEM_MB[$ni]=$(( NODE_COMMITTED_MEM_MB[$ni] - req_mem_mb ))
        (( NODE_COMMITTED_CPUS[$ni] < 0 )) && NODE_COMMITTED_CPUS[$ni]=0
        (( NODE_COMMITTED_MEM_MB[$ni] < 0 )) && NODE_COMMITTED_MEM_MB[$ni]=0
      fi
    else
      echo "  WAIT  $exp_name — no node has enough resources right now"
      STILL_PENDING+=("$idx")
    fi
  done

  PENDING_INDICES=("${STILL_PENDING[@]+"${STILL_PENDING[@]}"}")

  echo "  Progress this pass: $progress_this_pass submitted, ${#PENDING_INDICES[@]} still pending"

  if [[ ${#PENDING_INDICES[@]} -gt 0 ]] && [[ $progress_this_pass -eq 0 ]]; then
    echo "  No progress — sleeping ${RETRY_SLEEP_SECONDS}s before next pass ..."
    sleep "$RETRY_SLEEP_SECONDS"
  fi
done

echo
echo "============================================================"
echo "All experiments submitted."
echo "  Total submitted : $total_submitted"
echo "  Passes taken    : $pass"
echo "  Assignment log  : $ASSIGNMENT_CSV"
echo "  Finished at     : $(date)"
echo "============================================================"
