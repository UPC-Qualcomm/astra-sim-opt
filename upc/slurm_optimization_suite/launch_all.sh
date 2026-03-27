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
DEFAULT_MEM_PER_CORE_GB_SMALL=4
DEFAULT_MEM_PER_CORE_GB_LARGE=10

usage() {
  cat <<USAGE
Usage: bash launch_all.sh [--dry-run] [--partition PARTITION] [--max-jobs N] [--allow-node-reuse]

Options:
  --dry-run            Print assignments and sbatch commands without submitting.
  --partition PART     Force SLURM partition for all jobs.
  --max-jobs N         Submit at most N experiments (in manifest order).
  --allow-node-reuse   Reuse nodes round-robin if experiments > available nodes.
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

if [[ "$ALLOW_NODE_REUSE" -ne 1 ]] && [[ ${#EXPERIMENT_PATHS[@]} -gt ${#NODES[@]} ]]; then
  echo "Not enough available nodes for one-to-one assignment." >&2
  echo "Experiments: ${#EXPERIMENT_PATHS[@]}, nodes: ${#NODES[@]}" >&2
  echo "Either reduce jobs with --max-jobs or enable --allow-node-reuse." >&2
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

  if [[ "$ALLOW_NODE_REUSE" -eq 1 ]]; then
    node_pos=$((node_idx % ${#NODES[@]}))
  else
    node_pos=$node_idx
  fi

  node="${NODES[$node_pos]}"
  partition="${PARTITIONS[$node_pos]}"
  state_cores="${NODE_CORES[$node_pos]}"
  state_mem_mb="${NODE_MEM_MB[$node_pos]}"
  ((node_idx+=1))

  resource_line="$(get_node_free_resources "$node")"
  IFS='|' read -r node_cores_total node_mem_total_mb node_cores_free node_mem_free_mb <<< "$resource_line"
  if [[ "$node_cores_total" -le 0 ]]; then
    node_cores_total="$state_cores"
    node_cores_free="$state_cores"
  fi
  if [[ "$node_mem_total_mb" -le 0 ]]; then
    node_mem_total_mb="$state_mem_mb"
    node_mem_free_mb="$state_mem_mb"
  fi

  cores_percent="${CORES_PERCENT:-$DEFAULT_CORES_PERCENT}"
  cpus_override="${CPUS_PER_TASK_OVERRIDE:-}"
  mem_override="${MEM_PER_CPU_GB_OVERRIDE:-}"

  if [[ -n "$cpus_override" ]]; then
    cpus_per_task="$cpus_override"
  else
    cpus_per_task=$(( node_cores_free * cores_percent / 100 ))
    (( cpus_per_task < 1 )) && cpus_per_task=1
  fi

  if [[ -n "$mem_override" ]]; then
    mem_per_cpu_gb="$mem_override"
  else
    if [[ "${NUM_NPUS:-0}" -ge 1024 ]]; then
      mem_per_cpu_gb="$DEFAULT_MEM_PER_CORE_GB_LARGE"
    else
      mem_per_cpu_gb="$DEFAULT_MEM_PER_CORE_GB_SMALL"
    fi
  fi

  # Ensure request fits node memory. If not, reduce cpus_per_task.
  requested_mem_mb=$(( cpus_per_task * mem_per_cpu_gb * 1024 ))
  if (( requested_mem_mb > node_mem_free_mb )); then
    max_cpus_by_mem=$(( node_mem_free_mb / (mem_per_cpu_gb * 1024) ))
    (( max_cpus_by_mem < 1 )) && max_cpus_by_mem=1
    cpus_per_task="$max_cpus_by_mem"
  fi

  if (( cpus_per_task > node_cores_free )); then
    cpus_per_task="$node_cores_free"
    (( cpus_per_task < 1 )) && cpus_per_task=1
  fi

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
echo "Launch summary"
echo "- Total considered: ${#EXPERIMENT_PATHS[@]}"
echo "- Submitted: $submitted"
echo "- Failed/skipped: $failed"
echo "- Node snapshot: $SINFO_OUT"
echo "- Assignment file: $ASSIGNMENT_CSV"
