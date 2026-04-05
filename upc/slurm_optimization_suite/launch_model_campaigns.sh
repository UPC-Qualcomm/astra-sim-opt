#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# launch_model_campaigns.sh
#
# Submits all 7 optimization campaigns for llama8b, llama70b, and gpt60b.
# At launch time, idle SLURM nodes are discovered via sinfo and each job is
# dispatched to a randomly selected one, spreading load across the cluster.
# Override with --nodelist <node> to pin all jobs to a specific node instead.
#
# For each model:   7 campaigns × 2 objectives = 14 jobs per model.
#
# Objectives run for every model (tracker-compatible subset):
#   time  |  latency_network
#
# Outputs land in the same campaigns/ tree as launch_campaigns.sh so results
# from both launchers are directly comparable.
#
# Usage:
#   bash launch_model_campaigns.sh [options]
#
# Options:
#   --model <name|all>     Model to launch: gpt60b, llama70b, llama8b, or all.
#                          Default: all
#   --campaign <name|all>  Campaign name (see --list), or all. Default: all
#   --nodelist NODE         Pin ALL jobs to this specific node (skips auto-discovery).
#   --partition PART        Override SLURM partition for all jobs.
#   --dry-run              Print sbatch commands without submitting.
#   --list                 Show configuration and exit.
#   -h|--help              Show this help.
###############################################################################

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT_DIR="/scratch/nas/4/nasser/astra-sim/upc/slurm_optimization_suite"
EXPERIMENTS_DIR="$ROOT_DIR/experiments"
CAMPAIGNS_DIR="$ROOT_DIR/campaigns"
CAMPAIGN_SCRIPT="$ROOT_DIR/run_experiment_campaign.sh"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LAUNCH_LOG_DIR="$ROOT_DIR/launch_logs/model_campaigns_${TIMESTAMP}"
mkdir -p "$LAUNCH_LOG_DIR"

# ─── Node selection ─────────────────────────────────────────────────────────────
# Leave NODE_OVERRIDE empty (default): idle nodes are discovered automatically
# via sinfo at runtime and each job is sent to a randomly chosen one.
# Set via --nodelist to pin every job to one specific node instead.
NODE_OVERRIDE=""  # set by --nodelist
NODE_PARTITION="production"  # default partition; overridden by --partition
FORCED_PARTITION=""

# Suffix used in experiment directory names, e.g. llama8b_32npus_time
declare -A MODEL_NPUS=(
  [gpt60b]=128
  [llama70b]=128
  [llama8b]=32
)

# Display order (heaviest first)
ALL_MODELS=("gpt60b" "llama70b" "llama8b")

# ─── Objectives ───────────────────────────────────────────────────────────────
# These three are tracker-compatible and common to all 6 campaigns.
OBJECTIVES=("time" "latency_network")

# ─── Campaign definitions ─────────────────────────────────────────────────────
# Format: NAME|ENABLE_TRACKER|EARLY_STOPPING|N_WORKERS|SEARCH_TYPE
#   ENABLE_TRACKER  1 = pass --enable-tracker, 0 = omit it
#   EARLY_STOPPING  -1 = disabled; positive = patience (×30 applied in sweep)
#   N_WORKERS       parallel DeepHyper workers (= CPUs requested per job)
#   SEARCH_TYPE     cbo (Bayesian, default) or random
CAMPAIGN_DEFS=(
  "1_baseline_1w|0|-1|1|cbo"
  "2_earlystop30_1w|0|30|1|cbo"
  "3_tracker_1w|1|-1|1|cbo"
  "4_tracker_es30_1w|1|30|1|cbo"
  "5_tracker_es15_4w|1|15|4|cbo"
  "6_notracker_4w|0|-1|4|cbo"
  "7_random_tracker_es15_4w|1|15|4|random"
)

# ─── SLURM config ─────────────────────────────────────────────────────────────
SLURM_QOS="large"
DEFAULT_MEM_PER_CPU_GB=10
MAX_RETRIES=3

# ─── CLI parsing ──────────────────────────────────────────────────────────────
usage() {
  sed -n '3,/^###/p' "$0" | head -n -1
  echo
  echo "Models:"
  for m in "${ALL_MODELS[@]}"; do
    printf "  %-12s → %s NPUs\n" "$m" "${MODEL_NPUS[$m]}"
  done
  echo
  if [[ -n "$NODE_OVERRIDE" ]]; then
    echo "Node pinning: $NODE_OVERRIDE (${FORCED_PARTITION:-$NODE_PARTITION})"
  else
    echo "Node selection: random idle node via sinfo"
    echo "Partition:      ${FORCED_PARTITION:-$NODE_PARTITION}"
  fi
  echo
  echo "Campaigns:"
  for cdef in "${CAMPAIGN_DEFS[@]}"; do
    IFS='|' read -r name tracker es workers search_type <<< "$cdef"
    tracker_label="tracker"   ; [[ "$tracker" == "0" ]] && tracker_label="no-tracker"
    es_label="es=$es"         ; [[ "$es" == "-1" ]]     && es_label="no-early-stop"
    printf "  %-28s %s | %s | %d worker(s) | search=%s\n" "$name" "$tracker_label" "$es_label" "$workers" "${search_type:-cbo}"
  done
}

SELECTED_MODEL="all"
SELECTED_CAMPAIGN="all"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)     SELECTED_MODEL="$2";    shift 2 ;;
    --campaign)  SELECTED_CAMPAIGN="$2"; shift 2 ;;
    --nodelist)  NODE_OVERRIDE="$2";     shift 2 ;;
    --partition) FORCED_PARTITION="$2";   shift 2 ;;
    --dry-run)   DRY_RUN=1;              shift   ;;
    --list)      usage; exit 0 ;;
    -h|--help)   usage; exit 0 ;;
    *)           echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
  esac
done

# ─── Build model list ─────────────────────────────────────────────────────────
MODELS_TO_RUN=()
if [[ "$SELECTED_MODEL" == "all" ]]; then
  MODELS_TO_RUN=("${ALL_MODELS[@]}")
else
  if [[ -z "${MODEL_NPUS[$SELECTED_MODEL]+_}" ]]; then
    echo "Error: unknown model '$SELECTED_MODEL'. Valid: ${ALL_MODELS[*]}" >&2
    exit 1
  fi
  MODELS_TO_RUN=("$SELECTED_MODEL")
fi

# ─── Build campaign list ──────────────────────────────────────────────────────
CAMPAIGNS_TO_RUN=()
if [[ "$SELECTED_CAMPAIGN" == "all" ]]; then
  CAMPAIGNS_TO_RUN=("${CAMPAIGN_DEFS[@]}")
else
  found=0
  for cdef in "${CAMPAIGN_DEFS[@]}"; do
    IFS='|' read -r name _ _ _ <<< "$cdef"
    if [[ "$name" == "$SELECTED_CAMPAIGN" ]]; then
      CAMPAIGNS_TO_RUN=("$cdef")
      found=1
      break
    fi
  done
  if [[ $found -eq 0 ]]; then
    echo "Error: unknown campaign '$SELECTED_CAMPAIGN'. Use --list." >&2
    exit 1
  fi
fi

# ─── Validate prerequisites ───────────────────────────────────────────────────
if ! command -v sbatch >/dev/null 2>&1; then
  echo "Error: sbatch not available. Run on a SLURM login node." >&2
  exit 1
fi
if [[ ! -d "$EXPERIMENTS_DIR" ]]; then
  echo "Error: experiments directory not found: $EXPERIMENTS_DIR" >&2
  exit 1
fi
if [[ ! -f "$CAMPAIGN_SCRIPT" ]]; then
  echo "Error: run_experiment_campaign.sh not found: $CAMPAIGN_SCRIPT" >&2
  exit 1
fi

# ─── Seed randomness ────────────────────────────────────────────────────────────
# Read 2 bytes from /dev/urandom so $RANDOM is truly different on every run.
RANDOM=$(od -An -N2 -tu2 < /dev/urandom | tr -d ' ')

# ─── Discover idle SLURM nodes ──────────────────────────────────────────────
IDLE_NODES=()
if [[ -z "$NODE_OVERRIDE" ]]; then
  if ! command -v sinfo >/dev/null 2>&1; then
    echo "Warning: sinfo not available; jobs will submit without --nodelist." >&2
  else
    while IFS=' ' read -r n state; do
      [[ "$state" == "idle" || "$state" == "idle~" ]] || continue
      IDLE_NODES+=("$n")
    done < <(sinfo -N -h -o "%N %T" 2>/dev/null)

    if [[ ${#IDLE_NODES[@]} -eq 0 ]]; then
      echo "Warning: no idle nodes found; jobs will submit without --nodelist." >&2
    else
      echo "Discovered ${#IDLE_NODES[@]} idle node(s)."
    fi
  fi
fi

# Returns a random idle node, or the override node, or empty string (no pinning).
pick_random_node() {
  if [[ -n "$NODE_OVERRIDE" ]]; then
    echo "$NODE_OVERRIDE"
  elif [[ ${#IDLE_NODES[@]} -gt 0 ]]; then
    echo "${IDLE_NODES[$(( RANDOM % ${#IDLE_NODES[@]} ))]}"
  else
    echo ""
  fi
}

# ─── Launch ───────────────────────────────────────────────────────────────────
ASSIGNMENT_CSV="$LAUNCH_LOG_DIR/assignments.csv"
echo "model,campaign,experiment,node,partition,cpus,mem_per_cpu,job_name,status,job_id" \
  > "$ASSIGNMENT_CSV"

total_submitted=0
total_failed=0

total_jobs=$(( ${#MODELS_TO_RUN[@]} * ${#CAMPAIGNS_TO_RUN[@]} * ${#OBJECTIVES[@]} ))
echo "Models: ${#MODELS_TO_RUN[@]}  Campaigns: ${#CAMPAIGNS_TO_RUN[@]}  Objectives: ${#OBJECTIVES[@]}"
echo "Total jobs to submit: $total_jobs"
echo

for model in "${MODELS_TO_RUN[@]}"; do
  partition="${FORCED_PARTITION:-$NODE_PARTITION}"
  npus="${MODEL_NPUS[$model]}"

  echo "════════════════════════════════════════════════════════════════"
  echo "Model: $model  (${#CAMPAIGNS_TO_RUN[@]} campaigns × ${#OBJECTIVES[@]} objectives, partition=$partition)"
  echo "  Each job dispatched to a randomly selected idle node."
  echo "════════════════════════════════════════════════════════════════"

  for cdef in "${CAMPAIGNS_TO_RUN[@]}"; do
    IFS='|' read -r camp_name camp_tracker camp_es camp_workers camp_search_type <<< "$cdef"
    camp_search_type="${camp_search_type:-cbo}"
    tracker_label="tracker"   ; [[ "$camp_tracker" == "0" ]] && tracker_label="no-tracker"
    es_label="es=$camp_es"    ; [[ "$camp_es" == "-1" ]]     && es_label="no-early-stop"
    echo "  Campaign: $camp_name  ($tracker_label | $es_label | ${camp_workers}w | search=$camp_search_type)"

    for objective in "${OBJECTIVES[@]}"; do
      exp_name="${model}_${npus}npus_${objective}"
      exp_dir="$EXPERIMENTS_DIR/$exp_name"

      if [[ ! -d "$exp_dir" || ! -f "$exp_dir/config.env" ]]; then
        echo "    ⚠  Skipping $exp_name (missing directory or config.env)" >&2
        echo "$model,$camp_name,$exp_name,$node,$partition,,,${camp_name}__${exp_name},MISSING," \
          >> "$ASSIGNMENT_CSV"
        total_failed=$(( total_failed + 1 ))
        continue
      fi

      # Read per-experiment resource overrides from config.env
      MEM_PER_CPU_GB_OVERRIDE=""
      # shellcheck disable=SC1090
      source "$exp_dir/config.env"
      mem_per_cpu="${MEM_PER_CPU_GB_OVERRIDE:-$DEFAULT_MEM_PER_CPU_GB}"

      # Pick a fresh random idle node for each job.
      node="$(pick_random_node)"

      camp_output_dir="$CAMPAIGNS_DIR/${camp_name}/${exp_name}"
      mkdir -p "$camp_output_dir/logs" "$camp_output_dir/outputs"

      job_name="${camp_name}__${exp_name}"
      cpus_per_task="$camp_workers"
      wrap_cmd="ulimit -c 0; bash \"$CAMPAIGN_SCRIPT\""

      sbatch_cmd=(
        sbatch
        --job-name "$job_name"
        --chdir "$exp_dir"
        --partition "$partition"
        --cpus-per-task "$cpus_per_task"
        --mem-per-cpu "${mem_per_cpu}G"
        --export "ALL,ORIG_EXP_DIR=$exp_dir,CAMPAIGN_NAME=$camp_name,CAMPAIGN_ENABLE_TRACKER=$camp_tracker,CAMPAIGN_EARLY_STOPPING=$camp_es,CAMPAIGN_N_WORKERS=$camp_workers,CAMPAIGN_SEARCH_TYPE=$camp_search_type,CAMPAIGN_OUTPUT_DIR=$camp_output_dir"
        --output "$camp_output_dir/logs/slurm-%j.out"
        --error "$camp_output_dir/logs/slurm-%j.err"
        --wrap "$wrap_cmd"
      )

      [[ -n "${SLURM_QOS:-}" ]] && sbatch_cmd+=(--qos "$SLURM_QOS")
      [[ -n "$node" ]]           && sbatch_cmd+=(--nodelist "$node")

      if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "    [DRY-RUN] $exp_name → $node (cpus=$cpus_per_task, mem=${mem_per_cpu}G, search=$camp_search_type)"
        echo "$model,$camp_name,$exp_name,$node,$partition,$cpus_per_task,${mem_per_cpu}G,$job_name,DRY_RUN," \
          >> "$ASSIGNMENT_CSV"
        total_submitted=$(( total_submitted + 1 ))
      else
        retry=0
        submitted=0
        while [[ $retry -lt $MAX_RETRIES ]] && [[ $submitted -eq 0 ]]; do
          set +e
          out="$("${sbatch_cmd[@]}" 2>&1)"
          rc=$?
          set -e

          if [[ $rc -eq 0 ]]; then
            job_id="$(awk '{print $NF}' <<< "$out")"
            echo "    ✓ $exp_name → $node  (job $job_id, cpus=$cpus_per_task, search=$camp_search_type)"
            echo "$model,$camp_name,$exp_name,$node,$partition,$cpus_per_task,${mem_per_cpu}G,$job_name,SUBMITTED,$job_id" \
              >> "$ASSIGNMENT_CSV"
            total_submitted=$(( total_submitted + 1 ))
            submitted=1
          else
            retry=$(( retry + 1 ))
            echo "    ⚠  $exp_name retry $retry/$MAX_RETRIES: $out" >&2
          fi
        done

        if [[ $submitted -eq 0 ]]; then
          echo "    ✗ FAILED $exp_name after $MAX_RETRIES retries" >&2
          echo "$model,$camp_name,$exp_name,$node,$partition,$cpus_per_task,${mem_per_cpu}G,$job_name,FAILED," \
            >> "$ASSIGNMENT_CSV"
          total_failed=$(( total_failed + 1 ))
        fi
      fi
    done
    echo
  done
done

# ─── Summary ──────────────────────────────────────────────────────────────────
echo "════════════════════════════════════════════════════════════════"
  if [[ -n "$NODE_OVERRIDE" ]]; then
    echo "Node pinning:  $NODE_OVERRIDE (${FORCED_PARTITION:-$NODE_PARTITION})"
  elif [[ ${#IDLE_NODES[@]} -gt 0 ]]; then
    echo "Idle nodes:    ${#IDLE_NODES[@]} available (randomly selected per job)"
    echo "Partition:     ${FORCED_PARTITION:-$NODE_PARTITION}"
  else
    echo "Partition:     ${FORCED_PARTITION:-$NODE_PARTITION} (no node pinning)"
  fi
  echo
  echo "Launch summary"
echo "  Models launched:  ${#MODELS_TO_RUN[@]}"
echo "  Campaigns:        ${#CAMPAIGNS_TO_RUN[@]}"
echo "  Objectives:       ${#OBJECTIVES[@]}"
echo "  Total submitted:  $total_submitted"
echo "  Total failed:     $total_failed"
echo "  Assignments CSV:  $ASSIGNMENT_CSV"
echo "  Campaign outputs: $CAMPAIGNS_DIR/"
