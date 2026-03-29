#!/usr/bin/env bash
set -euo pipefail

# Use /scratch for SLURM accessibility on compute nodes
ROOT_DIR="/scratch/nas/4/nasser/astra-sim/upc/slurm_optimization_suite"
EXPERIMENTS_DIR="$ROOT_DIR/experiments"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LAUNCH_LOG_DIR="$ROOT_DIR/launch_logs/$TIMESTAMP"
mkdir -p "$LAUNCH_LOG_DIR"

# Verify that the scratch directory is accessible
if [[ ! -d "$EXPERIMENTS_DIR" ]]; then
  echo "Error: Experiments directory not found at: $EXPERIMENTS_DIR" >&2
  echo "Please ensure experiments are synced to /scratch/nas/4/nasser/astra-sim/upc/slurm_optimization_suite" >&2
  exit 1
fi

# Node states considered usable for scheduling.
NODE_STATES="idle,mix"

# Optional SLURM parameters (uncomment and set if needed)
# SLURM_ACCOUNT=""      # e.g., --account myaccount
# SLURM_QOS=""          # e.g., --qos large (if valid for your account/partition)
# SLURM_EXTRA_ARGS=""   # Additional sbatch arguments

# Retry configuration for failed sbatch submissions
MAX_RETRIES_PER_EXPERIMENT=3

# Scheduling guards to avoid pending jobs when a node cannot run immediately.
ENFORCE_REALTIME_NODE_CHECK=1
SBATCH_IMMEDIATE_SECONDS=1

# Global defaults (can be overridden per experiment in config.env)
DEFAULT_CORES_PERCENT=32
DEFAULT_MEM_PER_CORE_GB=2
MAX_CPUS_PER_EXPERIMENT=8

#################################################################################
# EXPERIMENT MANIFEST - Comment out experiments you DON'T want to run
# (Uncommented experiments will be submitted to SLURM)
#################################################################################
ACTIVE_EXPERIMENTS=(
  "gpt175b_1024npus_edp"
  "gpt175b_1024npus_edp_and_bw"
  "gpt175b_1024npus_energy_and_time"
  "gpt175b_1024npus_memory_and_time"
  "gpt175b_1024npus_time"
  "gpt175b_1024npus_time_and_bw"
  "gpt175b_1024npus_time_and_throughput_per_energy"
  "gpt60b_128npus_edp"
  "gpt60b_128npus_edp_and_bw"
  "gpt60b_128npus_energy_and_time"
  "gpt60b_128npus_memory_and_time"
  "gpt60b_128npus_time"
  "gpt60b_128npus_time_and_bw"
  "gpt60b_128npus_time_and_throughput_per_energy"
  "llama70b_128npus_edp"
  "llama70b_128npus_edp_and_bw"
  "llama70b_128npus_energy_and_time"
  "llama70b_128npus_memory_and_time"
  "llama70b_128npus_time"
  "llama70b_128npus_time_and_bw"
  "llama70b_128npus_time_and_throughput_per_energy"
  "llama8b_32npus_edp"
  "llama8b_32npus_edp_and_bw"
  "llama8b_32npus_energy_and_time"
  "llama8b_32npus_memory_and_time"
  "llama8b_32npus_time"
  "llama8b_32npus_time_and_bw"
  "llama8b_32npus_time_and_throughput_per_energy"
  # "llama8b_32npus_time"          # Example: uncomment to RUN this experiment
)

usage() {
  cat <<USAGE
Usage: bash launch_all.sh [--dry-run] [--partition PARTITION] [--max-jobs N] [--allow-node-reuse]

Options:
  --dry-run            Print assignments and sbatch commands without submitting.
  --partition PART     Force SLURM partition for all jobs.
  --max-jobs N         Submit at most N experiments (in manifest order).
  --allow-node-reuse   Compatibility flag (scheduler already packs multiple jobs per node).

Experiment Selection:
  Edit the ACTIVE_EXPERIMENTS array above to choose which experiments to run.
  Comment out (#) experiments you don't want to run.
USAGE
}

DRY_RUN=0
FORCED_PARTITION=""
MAX_JOBS=0
ALLOW_NODE_REUSE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --partition)
      FORCED_PARTITION="${2:-}"
      shift 2
      ;;
    --max-jobs)
      MAX_JOBS="${2:-0}"
      shift 2
      ;;
    --allow-node-reuse)
      ALLOW_NODE_REUSE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if ! command -v sinfo >/dev/null 2>&1 || ! command -v sbatch >/dev/null 2>&1; then
  echo "Error: sinfo/sbatch not available in PATH. Run on SLURM login node." >&2
  exit 1
fi

# Filter experiments by ACTIVE_EXPERIMENTS list
EXPERIMENT_PATHS=()
for exp_name in "${ACTIVE_EXPERIMENTS[@]}"; do
  exp_path="$EXPERIMENTS_DIR/$exp_name"
  if [[ -d "$exp_path" ]]; then
    EXPERIMENT_PATHS+=("$exp_path")
  else
    echo "Warning: experiment directory not found: $exp_path" >&2
  fi
done

if [[ ${#EXPERIMENT_PATHS[@]} -eq 0 ]]; then
  echo "No active experiments found in $EXPERIMENTS_DIR" >&2
  exit 1
fi

if [[ "$MAX_JOBS" -gt 0 ]] && [[ "$MAX_JOBS" -lt "${#EXPERIMENT_PATHS[@]}" ]]; then
  EXPERIMENT_PATHS=("${EXPERIMENT_PATHS[@]:0:$MAX_JOBS}")
fi

SINFO_OUT="$LAUNCH_LOG_DIR/sinfo_nodes.txt"

# Sort nodes in descending order (higher IDs first = more resources)
sinfo -N -h -t "$NODE_STATES" -o "%N|%P|%t|%c|%m" | sort -rV > "$SINFO_OUT"

if [[ ! -s "$SINFO_OUT" ]]; then
  echo "No available nodes in states: $NODE_STATES" >&2
  exit 1
fi

# Parse node inventory.
NODES=()
PARTITIONS=()
NODE_CORES=()
NODE_MEM_MB=()
while IFS='|' read -r node partition state cores mem_mb; do
  [[ -z "$node" ]] && continue
  partition="${partition%%\**}"
  NODES+=("$node")
  PARTITIONS+=("$partition")
  NODE_CORES+=("$cores")
  NODE_MEM_MB+=("$mem_mb")
done < "$SINFO_OUT"

if [[ ${#NODES[@]} -eq 0 ]]; then
  echo "No parseable nodes found from sinfo output." >&2
  exit 1
fi

ASSIGNMENT_CSV="$LAUNCH_LOG_DIR/assignments.csv"
cat > "$ASSIGNMENT_CSV" <<CSV
experiment,node,partition,node_cores_total,node_mem_total_mb,node_cores_free,node_mem_free_mb,cpus_per_task,mem_per_cpu_gb,mem_per_cpu_slurm,job_name,submit_status,job_id
CSV

get_node_free_resources() {
  local node_name="$1"
  local node_info
  node_info="$(scontrol show node "$node_name" 2>/dev/null || true)"

  local cpu_tot cpu_alloc mem_tot alloc_mem
  cpu_tot="$(grep -oE 'CPUTot=[0-9]+' <<< "$node_info" | head -n1 | cut -d= -f2)"
  cpu_alloc="$(grep -oE 'CPUAlloc=[0-9]+' <<< "$node_info" | head -n1 | cut -d= -f2)"
  mem_tot="$(grep -oE 'RealMemory=[0-9]+' <<< "$node_info" | head -n1 | cut -d= -f2)"
  alloc_mem="$(grep -oE 'AllocMem=[0-9]+' <<< "$node_info" | head -n1 | cut -d= -f2)"

  [[ -z "$cpu_tot" ]] && cpu_tot=0
  [[ -z "$cpu_alloc" ]] && cpu_alloc=0
  [[ -z "$mem_tot" ]] && mem_tot=0
  [[ -z "$alloc_mem" ]] && alloc_mem=0

  local cpu_free mem_free
  cpu_free=$(( cpu_tot - cpu_alloc ))
  mem_free=$(( mem_tot - alloc_mem ))
  (( cpu_free < 1 )) && cpu_free=1
  (( mem_free < 1 )) && mem_free=1

  echo "$cpu_tot|$mem_tot|$cpu_free|$mem_free"
}

node_has_realtime_capacity() {
  local node_name="$1"
  local req_cpus="$2"
  local req_mem_mb="$3"

  local resource_line_now
  resource_line_now="$(get_node_free_resources "$node_name")"

  local cpu_tot_now mem_tot_now cpu_free_now mem_free_now
  IFS='|' read -r cpu_tot_now mem_tot_now cpu_free_now mem_free_now <<< "$resource_line_now"

  (( cpu_free_now >= req_cpus )) || return 1
  (( mem_free_now >= req_mem_mb )) || return 1

  local pool_cpus_now
  pool_cpus_now=$(( cpu_free_now * DEFAULT_CORES_PERCENT / 100 ))
  (( pool_cpus_now < 1 )) && pool_cpus_now=1
  (( pool_cpus_now >= req_cpus )) || return 1

  return 0
}

SYNCED_NODES=()

sync_astraenv_to_node() {
  local node_name="$1"
  local source_env="/scratch/nas/4/nasser/astra-sim/astraenv"
  local target_env="/scratch/nas/4/nasser/astra-sim/astraenv"

  # Check if already synced this node
  for synced in "${SYNCED_NODES[@]}"; do
    if [[ "$synced" == "$node_name" ]]; then
      return 0
    fi
  done

  # Check if venv already exists on target node
  if ssh "$node_name" "[[ -d '$target_env/bin' && -f '$target_env/bin/python' ]]" 2>/dev/null; then
    echo "  ✓ astraenv already exists on $node_name"
    SYNCED_NODES+=("$node_name")
    return 0
  fi

  # Sync environment to node
  echo "  ⟳ Syncing astraenv to $node_name..."
  if ssh "$node_name" "mkdir -p $(dirname "$target_env")" && \
     rsync -az --delete "$source_env/" "$node_name:$target_env/" 2>/dev/null; then
    echo "  ✓ astraenv synced to $node_name"
    SYNCED_NODES+=("$node_name")
    return 0
  else
    echo "  ✗ Failed to sync astraenv to $node_name" >&2
    return 1
  fi
}

declare -i node_idx=0
submitted=0
failed=0

NODE_CORES_TOTAL_RUNTIME=()
NODE_MEM_TOTAL_RUNTIME=()
NODE_FREE_CORES_INITIAL=()
NODE_FREE_MEM_INITIAL_MB=()
NODE_POOL_CPUS=()
NODE_REMAINING_CPUS=()
NODE_REMAINING_MEM_MB=()
NODE_ASSIGNED_JOBS=()
NODE_ASSIGNED_CPUS=()

# Build per-node scheduling pool based on 32% of currently free CPUs.
for i in "${!NODES[@]}"; do
  resource_line="$(get_node_free_resources "${NODES[$i]}")"
  IFS='|' read -r node_cores_total node_mem_total_mb node_cores_free node_mem_free_mb <<< "$resource_line"

  if [[ "$node_cores_total" -le 0 ]]; then
    node_cores_total="${NODE_CORES[$i]}"
    node_cores_free="${NODE_CORES[$i]}"
  fi
  if [[ "$node_mem_total_mb" -le 0 ]]; then
    node_mem_total_mb="${NODE_MEM_MB[$i]}"
    node_mem_free_mb="${NODE_MEM_MB[$i]}"
  fi

  pool_cpus=$(( node_cores_free * DEFAULT_CORES_PERCENT / 100 ))
  (( pool_cpus < 1 )) && pool_cpus=1

  NODE_CORES_TOTAL_RUNTIME+=("$node_cores_total")
  NODE_MEM_TOTAL_RUNTIME+=("$node_mem_total_mb")
  NODE_FREE_CORES_INITIAL+=("$node_cores_free")
  NODE_FREE_MEM_INITIAL_MB+=("$node_mem_free_mb")
  NODE_POOL_CPUS+=("$pool_cpus")
  NODE_REMAINING_CPUS+=("$pool_cpus")
  NODE_REMAINING_MEM_MB+=("$node_mem_free_mb")
  NODE_ASSIGNED_JOBS+=("0")
  NODE_ASSIGNED_CPUS+=("0")
done

for exp_dir in "${EXPERIMENT_PATHS[@]}"; do
  cfg="$exp_dir/config.env"
  run_script="$exp_dir/run_experiment.sh"

  if [[ ! -f "$cfg" || ! -x "$run_script" ]]; then
    echo "Skipping $exp_dir (missing config.env or executable run_experiment.sh)" >&2
    ((failed+=1))
    continue
  fi

  # shellcheck disable=SC1090
  source "$cfg"

  cpus_override="${CPUS_PER_TASK_OVERRIDE:-}"
  mem_override="${MEM_PER_CPU_GB_OVERRIDE:-}"

  if [[ -n "$mem_override" ]]; then
    mem_per_cpu_gb="$mem_override"
  else
    mem_per_cpu_gb="$DEFAULT_MEM_PER_CORE_GB"
  fi

  # Retry loop for sbatch submission
  submission_successful=0
  retry_count=0
  TRIED_NODES=()

  while [[ $submission_successful -eq 0 ]] && [[ $retry_count -lt $MAX_RETRIES_PER_EXPERIMENT ]]; do

    # Choose a node slot: each node can host multiple experiments,
    # skipping nodes that already failed for this experiment
    selected_node_pos=-1
    selected_cpus=0
    selected_req_mem_mb=0

    for ((try_i=0; try_i<${#NODES[@]}; try_i++)); do
      node_pos=$(( (node_idx + try_i) % ${#NODES[@]} ))
      node="${NODES[$node_pos]}"

      # Skip nodes already tried for this experiment
      skip_node=0
      for tried_node in "${TRIED_NODES[@]}"; do
        if [[ "$tried_node" == "$node" ]]; then
          skip_node=1
          break
        fi
      done
      [[ $skip_node -eq 1 ]] && continue

      remaining_cpus="${NODE_REMAINING_CPUS[$node_pos]}"
      remaining_mem_mb="${NODE_REMAINING_MEM_MB[$node_pos]}"
      (( remaining_cpus < 1 )) && continue

      if [[ -n "$cpus_override" ]]; then
        candidate_cpus="$cpus_override"
        (( candidate_cpus > remaining_cpus )) && candidate_cpus="$remaining_cpus"
      else
        candidate_cpus="$remaining_cpus"
        (( candidate_cpus > MAX_CPUS_PER_EXPERIMENT )) && candidate_cpus="$MAX_CPUS_PER_EXPERIMENT"
      fi

      (( candidate_cpus < 1 )) && continue
      candidate_req_mem_mb=$(( candidate_cpus * mem_per_cpu_gb * 1024 ))

      if (( candidate_req_mem_mb > remaining_mem_mb )); then
        max_cpus_by_mem=$(( remaining_mem_mb / (mem_per_cpu_gb * 1024) ))
        (( max_cpus_by_mem < 1 )) && continue
        (( max_cpus_by_mem < candidate_cpus )) && candidate_cpus="$max_cpus_by_mem"
        candidate_req_mem_mb=$(( candidate_cpus * mem_per_cpu_gb * 1024 ))
      fi

      if [[ "$ENFORCE_REALTIME_NODE_CHECK" -eq 1 ]]; then
        node_has_realtime_capacity "$node" "$candidate_cpus" "$candidate_req_mem_mb" || continue
      fi

      selected_node_pos="$node_pos"
      selected_cpus="$candidate_cpus"
      selected_req_mem_mb="$candidate_req_mem_mb"
      break
    done

    if (( selected_node_pos < 0 )); then
      echo "FAILED scheduling $(basename "$exp_dir"): no remaining capacity on available/untried nodes" >&2
      echo "$(basename "$exp_dir"),N/A,N/A,0,0,0,0,0,$mem_per_cpu_gb,${mem_per_cpu_gb}G,${JOB_NAME:-$(basename "$exp_dir")},FAILED_NO_CAPACITY," >> "$ASSIGNMENT_CSV"
      ((failed+=1))
      break
    fi

    node_pos="$selected_node_pos"
    node="${NODES[$node_pos]}"
    partition="${PARTITIONS[$node_pos]}"
    node_cores_total="${NODE_CORES_TOTAL_RUNTIME[$node_pos]}"
    node_mem_total_mb="${NODE_MEM_TOTAL_RUNTIME[$node_pos]}"
    node_cores_free="${NODE_FREE_CORES_INITIAL[$node_pos]}"
    node_mem_free_mb="${NODE_FREE_MEM_INITIAL_MB[$node_pos]}"
    cpus_per_task="$selected_cpus"
    requested_mem_mb="$selected_req_mem_mb"

    NODE_REMAINING_CPUS[$node_pos]=$(( ${NODE_REMAINING_CPUS[$node_pos]} - cpus_per_task ))
    NODE_REMAINING_MEM_MB[$node_pos]=$(( ${NODE_REMAINING_MEM_MB[$node_pos]} - requested_mem_mb ))
    NODE_ASSIGNED_JOBS[$node_pos]=$(( ${NODE_ASSIGNED_JOBS[$node_pos]} + 1 ))
    NODE_ASSIGNED_CPUS[$node_pos]=$(( ${NODE_ASSIGNED_CPUS[$node_pos]} + cpus_per_task ))
    (( NODE_REMAINING_CPUS[$node_pos] < 0 )) && NODE_REMAINING_CPUS[$node_pos]=0
    (( NODE_REMAINING_MEM_MB[$node_pos] < 0 )) && NODE_REMAINING_MEM_MB[$node_pos]=0
    node_idx=$(( node_pos + 1 ))

    mem_per_cpu_slurm="${mem_per_cpu_gb}G"
    job_name="${JOB_NAME:-$(basename "$exp_dir")}"
    logs_dir="$exp_dir/logs"
    mkdir -p "$logs_dir"

    submit_partition="$partition"
    if [[ -n "$FORCED_PARTITION" ]]; then
      submit_partition="$FORCED_PARTITION"
    elif [[ -n "${PARTITION_OVERRIDE:-}" ]]; then
      submit_partition="$PARTITION_OVERRIDE"
    fi

    # Sync environment to compute node before submission
    echo "Preparing node $node for $(basename "$exp_dir")..."
    if ! sync_astraenv_to_node "$node"; then
      echo "Warning: Failed to sync astraenv to $node, job may fail" >&2
    fi

    wrap_cmd="bash \"$run_script\""

    sbatch_cmd=(
      sbatch
      --job-name "$job_name"
      --chdir "$exp_dir"
      --nodelist "$node"
      --partition "$submit_partition"
      --cpus-per-task "$cpus_per_task"
      --mem-per-cpu "$mem_per_cpu_slurm"
      --export "ALL,N_WORKERS_OVERRIDE=$cpus_per_task"
      --output "$logs_dir/slurm-%j.out"
      --error "$logs_dir/slurm-%j.err"
      --wrap "$wrap_cmd"
    )

    if [[ "$SBATCH_IMMEDIATE_SECONDS" -ge 0 ]]; then
      sbatch_cmd+=(--immediate="$SBATCH_IMMEDIATE_SECONDS")
    fi

    # Add optional SLURM parameters if configured
    [[ -n "${SLURM_ACCOUNT:-}" ]] && sbatch_cmd+=(--account "$SLURM_ACCOUNT")
    [[ -n "${SLURM_QOS:-}" ]] && sbatch_cmd+=(--qos "$SLURM_QOS")
    [[ -n "${SLURM_EXTRA_ARGS:-}" ]] && sbatch_cmd+=($SLURM_EXTRA_ARGS)

    if [[ "$DRY_RUN" -eq 1 ]]; then
      echo "[DRY-RUN] ${sbatch_cmd[*]}"
      echo "$(basename "$exp_dir"),$node,$submit_partition,$node_cores_total,$node_mem_total_mb,$node_cores_free,$node_mem_free_mb,$cpus_per_task,$mem_per_cpu_gb,$mem_per_cpu_slurm,$job_name,DRY_RUN," >> "$ASSIGNMENT_CSV"
      ((submitted+=1))
      submission_successful=1
    else
      set +e
      out="$("${sbatch_cmd[@]}" 2>&1)"
      rc=$?
      set -e

      if [[ $rc -eq 0 ]]; then
        job_id="$(awk '{print $NF}' <<< "$out")"
        echo "Submitted $(basename "$exp_dir") to $node (job $job_id, cpus=$cpus_per_task, mem/cpu=$mem_per_cpu_slurm)"
        echo "$(basename "$exp_dir"),$node,$submit_partition,$node_cores_total,$node_mem_total_mb,$node_cores_free,$node_mem_free_mb,$cpus_per_task,$mem_per_cpu_gb,$mem_per_cpu_slurm,$job_name,SUBMITTED,$job_id" >> "$ASSIGNMENT_CSV"
        ((submitted+=1))
        submission_successful=1
      else
        echo "⚠️  Failed on $node (retry $(($retry_count+1))/$MAX_RETRIES_PER_EXPERIMENT): $out" >&2
        TRIED_NODES+=("$node")
        ((retry_count+=1))
        
        # Restore resources since this node failed
        NODE_REMAINING_CPUS[$node_pos]=$(( ${NODE_REMAINING_CPUS[$node_pos]} + cpus_per_task ))
        NODE_REMAINING_MEM_MB[$node_pos]=$(( ${NODE_REMAINING_MEM_MB[$node_pos]} + requested_mem_mb ))
        NODE_ASSIGNED_JOBS[$node_pos]=$(( ${NODE_ASSIGNED_JOBS[$node_pos]} - 1 ))
        NODE_ASSIGNED_CPUS[$node_pos]=$(( ${NODE_ASSIGNED_CPUS[$node_pos]} - cpus_per_task ))
        (( NODE_ASSIGNED_JOBS[$node_pos] < 0 )) && NODE_ASSIGNED_JOBS[$node_pos]=0
        (( NODE_ASSIGNED_CPUS[$node_pos] < 0 )) && NODE_ASSIGNED_CPUS[$node_pos]=0
      fi
    fi
  done

  if [[ $submission_successful -eq 0 ]]; then
    echo "FAILED submitting $(basename "$exp_dir") after $MAX_RETRIES_PER_EXPERIMENT retries" >&2
    echo "$(basename "$exp_dir"),RETRY_EXHAUSTED,N/A,0,0,0,0,0,$mem_per_cpu_gb,${mem_per_cpu_gb}G,${JOB_NAME:-$(basename "$exp_dir")},FAILED_EXHAUSTED," >> "$ASSIGNMENT_CSV"
    ((failed+=1))
  fi
done

echo
echo "Per-node scheduling summary"
echo "node,partition,jobs_assigned,pool_cpus,cpus_assigned,cpus_remaining,mem_free_initial_mb,mem_used_mb,mem_remaining_mb"
for i in "${!NODES[@]}"; do
  mem_free_initial_mb="${NODE_FREE_MEM_INITIAL_MB[$i]}"
  mem_remaining_mb="${NODE_REMAINING_MEM_MB[$i]}"
  mem_used_mb=$(( mem_free_initial_mb - mem_remaining_mb ))
  (( mem_used_mb < 0 )) && mem_used_mb=0
  echo "${NODES[$i]},${PARTITIONS[$i]},${NODE_ASSIGNED_JOBS[$i]},${NODE_POOL_CPUS[$i]},${NODE_ASSIGNED_CPUS[$i]},${NODE_REMAINING_CPUS[$i]},${mem_free_initial_mb},${mem_used_mb},${mem_remaining_mb}"
done

echo
echo "Launch summary"
echo "- Total considered: ${#EXPERIMENT_PATHS[@]}"
echo "- Submitted: $submitted"
echo "- Failed/skipped: $failed"
echo "- Node snapshot: $SINFO_OUT"
echo "- Assignment file: $ASSIGNMENT_CSV"
