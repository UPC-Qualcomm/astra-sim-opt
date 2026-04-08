#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# launch_campaigns.sh
#
# Launches 6 pre-defined experiment campaigns with different tracker,
# early-stopping, and worker configurations.  Each campaign saves results
# under  campaigns/<campaign_name>/<experiment>/  so multiple runs are
# cleanly separated.
#
# Node pinning: every experiment (model+objective) is deterministically
# mapped to the same SLURM node across all campaigns, so the only variable
# between campaigns is the optimizer configuration.
#
# Usage:
#   bash launch_campaigns.sh --campaign <name|all> [options]
#
# Options:
#   --campaign <name|all>   Campaign to launch (see list below), or "all".
#   --dry-run               Print sbatch commands without submitting.
#   --partition PART        Force SLURM partition for all jobs.
#   --list                  List available campaigns and exit.
#
# Campaigns:
#   1_baseline_1w           No tracker, no early stop, 1 worker  (all experiments)
#   2_earlystop30_1w        No tracker, 30 early stop, 1 worker  (all experiments)
#   3_tracker_1w            Tracker, no early stop, 1 worker     (subset: time objectives)
#   4_tracker_es30_1w       Tracker, 30 early stop, 1 worker     (subset: time objectives)
#   5_tracker_es15_4w       Tracker, 15 early stop/w, 4 workers  (all experiments)
#   6_notracker_4w          No tracker, no early stop, 4 workers (subset: time objectives)
#
# "subset" = only objectives: time, time_and_bw, latency_network
#
# Early stopping note:
#   The sweep script (example_deephyper_opt_sweep.py) applies a 30× multiplier
#   to the --early-stopping-patience value.  Effective patience = 30 × value.
#   A value of -1 disables early stopping.
###############################################################################

# ─── Paths ────────────────────────────────────────────────────────────────────
# Use /scratch for SLURM accessibility on compute nodes
ROOT_DIR="/scratch/nas/4/nasser/astra-sim/upc/slurm_optimization_suite_last_hope"
EXPERIMENTS_DIR="$ROOT_DIR/experiments"
CAMPAIGNS_DIR="$ROOT_DIR/campaigns"
CAMPAIGN_SCRIPT="$ROOT_DIR/run_experiment_campaign.sh"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LAUNCH_LOG_DIR="$ROOT_DIR/launch_logs/campaigns_${TIMESTAMP}"
mkdir -p "$LAUNCH_LOG_DIR"

if [[ ! -d "$EXPERIMENTS_DIR" ]]; then
  echo "Error: Experiments directory not found at: $EXPERIMENTS_DIR" >&2
  echo "Please ensure experiments are synced to $ROOT_DIR" >&2
  exit 1
fi
if [[ ! -f "$CAMPAIGN_SCRIPT" ]]; then
  echo "Error: run_experiment_campaign.sh not found at: $CAMPAIGN_SCRIPT" >&2
  exit 1
fi

# ─── SLURM config ────────────────────────────────────────────────────────────
SLURM_QOS="large"
DEFAULT_MEM_PER_CPU_GB=10
MAX_RETRIES=3

# ─── Campaign definitions ────────────────────────────────────────────────────
# Format: NAME|ENABLE_TRACKER|EARLY_STOPPING|N_WORKERS|FILTER|SEARCH_TYPE
#   ENABLE_TRACKER:  1 = yes, 0 = no
#   EARLY_STOPPING:  patience value passed to sweep (-1 = disabled)
#   N_WORKERS:       parallel workers
#   FILTER:          "all" or "subset"
#   SEARCH_TYPE:     cbo (Bayesian, default) or random
CAMPAIGN_DEFS=(
  "1_baseline_1w|0|-1|1|all|cbo"
  "2_earlystop30_1w|0|30|1|all|cbo"
  "3_tracker_1w|1|-1|1|subset|cbo"
  "4_tracker_es30_1w|1|30|1|subset|cbo"
  "5_tracker_es15_4w|1|15|4|all|cbo"
  "6_notracker_4w|0|-1|4|subset|cbo"
  "7_random_tracker_es15_4w|1|15|4|subset|random"
)

# Objectives included in "subset" campaigns (tracker-compatible objectives)
SUBSET_OBJECTIVES=("time" "time_and_bw" "latency_network")

# ─── All experiments — used for subset filtering only ────────────────────────
# Add or remove lines when experiments are created or deleted.
ALL_EXPERIMENTS=(
  # ── gpt175b / 1024 NPUs  (heaviest) ─────────────────────────────────────
  gpt175b_1024npus_time
  gpt175b_1024npus_time_and_bw
  gpt175b_1024npus_latency_network
  gpt175b_1024npus_memory_and_time
  gpt175b_1024npus_energy_and_time
  gpt175b_1024npus_time_and_throughput_per_energy
  gpt175b_1024npus_edp
  gpt175b_1024npus_edp_and_bw
  # ── llama70b / 128 NPUs ──────────────────────────────────────────────────
  llama70b_128npus_time
  llama70b_128npus_time_and_bw
  llama70b_128npus_latency_network
  llama70b_128npus_memory_and_time
  llama70b_128npus_energy_and_time
  llama70b_128npus_time_and_throughput_per_energy
  llama70b_128npus_edp
  llama70b_128npus_edp_and_bw
  # ── gpt60b / 128 NPUs ────────────────────────────────────────────────────
  gpt60b_128npus_time
  gpt60b_128npus_time_and_bw
  gpt60b_128npus_latency_network
  gpt60b_128npus_memory_and_time
  gpt60b_128npus_energy_and_time
  gpt60b_128npus_time_and_throughput_per_energy
  gpt60b_128npus_edp
  gpt60b_128npus_edp_and_bw
  # ── llama8b / 32 NPUs  (lightest) ────────────────────────────────────────
  llama8b_32npus_time
  llama8b_32npus_time_and_bw
  llama8b_32npus_latency_network
  llama8b_32npus_memory_and_time
  llama8b_32npus_energy_and_time
  llama8b_32npus_time_and_throughput_per_energy
  llama8b_32npus_edp
  llama8b_32npus_edp_and_bw
)

# ─── CLI parsing ─────────────────────────────────────────────────────────────
usage() {
  sed -n '3,/^###/p' "$0" | head -n -1
  echo
  echo "Available campaigns:"
  for cdef in "${CAMPAIGN_DEFS[@]}"; do
    IFS='|' read -r name tracker es workers filter search_type <<< "$cdef"
    tracker_label="tracker"   ; [[ "$tracker" == "0" ]] && tracker_label="no-tracker"
    es_label="es=$es"         ; [[ "$es" == "-1" ]]     && es_label="no-early-stop"
    printf "  %-28s %s, %s, %d worker(s), %s experiments, search=%s\n" \
      "$name" "$tracker_label" "$es_label" "$workers" "$filter" "${search_type:-cbo}"
  done
}

SELECTED_CAMPAIGN=""
DRY_RUN=0
FORCED_PARTITION=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --campaign)   SELECTED_CAMPAIGN="$2"; shift 2 ;;
    --dry-run)    DRY_RUN=1; shift ;;
    --partition)  FORCED_PARTITION="$2"; shift 2 ;;
    --list)       usage; exit 0 ;;
    -h|--help)    usage; exit 0 ;;
    *)            echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "$SELECTED_CAMPAIGN" ]]; then
  echo "Error: --campaign is required (use --list to see options)" >&2
  exit 1
fi

# ─── Build list of campaigns to run ──────────────────────────────────────────
CAMPAIGNS_TO_RUN=()
if [[ "$SELECTED_CAMPAIGN" == "all" ]]; then
  CAMPAIGNS_TO_RUN=("${CAMPAIGN_DEFS[@]}")
else
  found=0
  for cdef in "${CAMPAIGN_DEFS[@]}"; do
    IFS='|' read -r name _ _ _ _ <<< "$cdef"
    if [[ "$name" == "$SELECTED_CAMPAIGN" ]]; then
      CAMPAIGNS_TO_RUN=("$cdef")
      found=1
      break
    fi
  done
  if [[ $found -eq 0 ]]; then
    echo "Error: Unknown campaign '$SELECTED_CAMPAIGN'. Use --list." >&2
    exit 1
  fi
fi

# ─── Validate SLURM tools ────────────────────────────────────────────────────
if ! command -v sbatch >/dev/null 2>&1; then
  echo "Error: sbatch not available. Run on a SLURM login node." >&2
  exit 1
fi

echo "Campaigns to launch: ${#CAMPAIGNS_TO_RUN[@]}"
echo

# ─── Filter experiments for a campaign ────────────────────────────────────────
get_campaign_experiments() {
  local filter="$1"
  if [[ "$filter" == "all" ]]; then
    echo "${ALL_EXPERIMENTS[@]}"
    return
  fi
  # "subset": only experiments whose objective matches SUBSET_OBJECTIVES
  local result=()
  for exp in "${ALL_EXPERIMENTS[@]}"; do
    # Extract objective: everything after the last _<npus>_ pattern
    # e.g. llama8b_32npus_time_and_bw → time_and_bw
    local obj
    obj=$(echo "$exp" | sed 's/^[^_]*_[0-9]*npus_//')
    for allowed in "${SUBSET_OBJECTIVES[@]}"; do
      if [[ "$obj" == "$allowed" ]]; then
        result+=("$exp")
        break
      fi
    done
  done
  echo "${result[@]}"
}

# ─── Launch ───────────────────────────────────────────────────────────────────
ASSIGNMENT_CSV="$LAUNCH_LOG_DIR/assignments.csv"
echo "campaign,experiment,node,partition,cpus,mem_per_cpu,search_type,job_name,status,job_id" > "$ASSIGNMENT_CSV"

total_submitted=0
total_failed=0

for cdef in "${CAMPAIGNS_TO_RUN[@]}"; do
  IFS='|' read -r camp_name camp_tracker camp_es camp_workers camp_filter camp_search_type <<< "$cdef"
  camp_search_type="${camp_search_type:-cbo}"

  tracker_label="tracker"   ; [[ "$camp_tracker" == "0" ]] && tracker_label="no-tracker"
  es_label="es=$camp_es"    ; [[ "$camp_es" == "-1" ]]     && es_label="no-early-stop"

  echo "════════════════════════════════════════════════════════════════"
  echo "Campaign: $camp_name"
  echo "  $tracker_label | $es_label | ${camp_workers} worker(s) | $camp_filter | search=$camp_search_type"
  echo "════════════════════════════════════════════════════════════════"

  # Get experiments for this campaign
  read -ra exps <<< "$(get_campaign_experiments "$camp_filter")"

  if [[ ${#exps[@]} -eq 0 ]]; then
    echo "  No experiments matched filter '$camp_filter'. Skipping."
    continue
  fi
  echo "  Experiments: ${#exps[@]}"

  for exp_name in "${exps[@]}"; do
    exp_dir="$EXPERIMENTS_DIR/$exp_name"
    if [[ ! -d "$exp_dir" || ! -f "$exp_dir/config.env" ]]; then
      echo "  ⚠  Skipping $exp_name (missing directory or config.env)" >&2
      ((total_failed++))
      continue
    fi

    # Read per-experiment resource overrides from config.env
    MEM_PER_CPU_GB_OVERRIDE=""
    PARTITION_OVERRIDE=""
    # shellcheck disable=SC1090
    source "$exp_dir/config.env"

    mem_per_cpu="${MEM_PER_CPU_GB_OVERRIDE:-$DEFAULT_MEM_PER_CPU_GB}"
    partition="${FORCED_PARTITION:-${PARTITION_OVERRIDE:-production}}"

    # Campaign output directory
    camp_output_dir="$CAMPAIGNS_DIR/${camp_name}/${exp_name}"
    mkdir -p "$camp_output_dir/logs" "$camp_output_dir/outputs"

    job_name="${camp_name}__${exp_name}"
    logs_dir="$camp_output_dir/logs"

    # CPUs = workers (each worker needs a core)
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
      --output "$logs_dir/slurm-%j.out"
      --error "$logs_dir/slurm-%j.err"
      --wrap "$wrap_cmd"
    )

    [[ -n "${SLURM_QOS:-}" ]] && sbatch_cmd+=(--qos "$SLURM_QOS")

    if [[ "$DRY_RUN" -eq 1 ]]; then
      echo "  [DRY-RUN] $exp_name ($partition, cpus=$cpus_per_task, search=$camp_search_type)"
      echo "$camp_name,$exp_name,any,$partition,$cpus_per_task,${mem_per_cpu}G,$job_name,DRY_RUN," >> "$ASSIGNMENT_CSV"
      ((total_submitted++))
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
          echo "  ✓ $exp_name (job $job_id, cpus=$cpus_per_task, search=$camp_search_type)"
          echo "$camp_name,$exp_name,any,$partition,$cpus_per_task,${mem_per_cpu}G,$camp_search_type,$job_name,SUBMITTED,$job_id" >> "$ASSIGNMENT_CSV"
          ((total_submitted++))
          submitted=1
        else
          ((retry++))
          echo "  ⚠  $exp_name retry $retry/$MAX_RETRIES: $out" >&2
        fi
      done

      if [[ $submitted -eq 0 ]]; then
        echo "  ✗ FAILED $exp_name after $MAX_RETRIES retries" >&2
        echo "$camp_name,$exp_name,any,$partition,$cpus_per_task,${mem_per_cpu}G,$camp_search_type,$job_name,FAILED," >> "$ASSIGNMENT_CSV"
        ((total_failed++))
      fi
    fi
  done
  echo
done

# ─── Summary ─────────────────────────────────────────────────────────────────
echo "════════════════════════════════════════════════════════════════"
echo "Launch summary"
echo "  Partition:          ${FORCED_PARTITION:-production (default)}"
echo "  Campaigns launched: ${#CAMPAIGNS_TO_RUN[@]}"
echo "  Total submitted:    $total_submitted"
echo "  Total failed:       $total_failed"
echo "  Assignments CSV:    $ASSIGNMENT_CSV"
echo "  Campaign outputs:   $CAMPAIGNS_DIR/"
