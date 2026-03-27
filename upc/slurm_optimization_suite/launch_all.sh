#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENTS_DIR="$ROOT_DIR/experiments"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LAUNCH_LOG_DIR="$ROOT_DIR/launch_logs/$TIMESTAMP"
mkdir -p "$LAUNCH_LOG_DIR"

# Node states considered usable for scheduling.
NODE_STATES="idle,mix"

# Global defaults (can be overridden per experiment in config.env)
DEFAULT_CORES_PERCENT=32
DEFAULT_MEM_PER_CORE_GB=2
MAX_CPUS_PER_EXPERIMENT=8

usage() {
  cat <<USAGE
Usage: bash launch_all.sh [--dry-run] [--partition PARTITION] [--max-jobs N] [--allow-node-reuse]

Options:
  --dry-run            Print assignments and sbatch commands without submitting.
  --partition PART     Force SLURM partition for all jobs.
  --max-jobs N         Submit at most N experiments (in manifest order).
  --allow-node-reuse   Compatibility flag (scheduler already packs multiple jobs per node).
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

mapfile -t EXPERIMENT_PATHS < <(find "$EXPERIMENTS_DIR" -mindepth 1 -maxdepth 1 -type d | sort)
if [[ ${#EXPERIMENT_PATHS[@]} -eq 0 ]]; then
  echo "No experiment folders found in $EXPERIMENTS_DIR" >&2
  exit 1
fi

if [[ "$MAX_JOBS" -gt 0 ]] && [[ "$MAX_JOBS" -lt "${#EXPERIMENT_PATHS[@]}" ]]; then
  EXPERIMENT_PATHS=("${EXPERIMENT_PATHS[@]:0:$MAX_JOBS}")
fi

SINFO_OUT="$LAUNCH_LOG_DIR/sinfo_nodes.txt"
sinfo -N -h -t "$NODE_STATES" -o "%N|%P|%t|%c|%m" > "$SINFO_OUT"

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

  # Choose a node slot: each node can host multiple experiments,
  # consuming only 32% of its free CPUs in chunks of up to 8 CPUs/job.
  selected_node_pos=-1
  selected_cpus=0
  selected_req_mem_mb=0

  for ((try_i=0; try_i<${#NODES[@]}; try_i++)); do
    node_pos=$(( (node_idx + try_i) % ${#NODES[@]} ))
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

    selected_node_pos="$node_pos"
    selected_cpus="$candidate_cpus"
    selected_req_mem_mb="$candidate_req_mem_mb"
    break
  done

  if (( selected_node_pos < 0 )); then
    echo "FAILED scheduling $(basename "$exp_dir"): no remaining 32%-pool capacity on available nodes" >&2
    echo "$(basename "$exp_dir"),N/A,N/A,0,0,0,0,0,$mem_per_cpu_gb,${mem_per_cpu_gb}G,${JOB_NAME:-$(basename "$exp_dir")},FAILED," >> "$ASSIGNMENT_CSV"
    ((failed+=1))
    continue
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

  sbatch_cmd=(
    sbatch
    --job-name "$job_name"
    --chdir "$exp_dir"
    --nodelist "$node"
    --partition "$submit_partition"
    --qos "large"
    --cpus-per-task "$cpus_per_task"
    --mem-per-cpu "$mem_per_cpu_slurm"
    --export "ALL,N_WORKERS_OVERRIDE=$cpus_per_task"
    --output "$logs_dir/slurm-%j.out"
    --error "$logs_dir/slurm-%j.err"
    "$run_script"
  )

  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[DRY-RUN] ${sbatch_cmd[*]}"
    echo "$(basename "$exp_dir"),$node,$submit_partition,$node_cores_total,$node_mem_total_mb,$node_cores_free,$node_mem_free_mb,$cpus_per_task,$mem_per_cpu_gb,$mem_per_cpu_slurm,$job_name,DRY_RUN," >> "$ASSIGNMENT_CSV"
    ((submitted+=1))
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
    else
      echo "FAILED submitting $(basename "$exp_dir"): $out" >&2
      echo "$(basename "$exp_dir"),$node,$submit_partition,$node_cores_total,$node_mem_total_mb,$node_cores_free,$node_mem_free_mb,$cpus_per_task,$mem_per_cpu_gb,$mem_per_cpu_slurm,$job_name,FAILED," >> "$ASSIGNMENT_CSV"
      ((failed+=1))
    fi
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
