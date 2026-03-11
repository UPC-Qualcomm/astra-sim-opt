#!/bin/bash
# =================================================================================
# Experiment 8 - Multi-scale simulator comparison
# Varies NPU count (2-512), model shapes, and parallelization strategies
# Uses FoldedClos1024 topology for all scales
#
# Jobs are collected per NPU count and executed group-by-group, each group
# using the parallelism and VMEM budget configured in get_npu_parallel/get_npu_vmem.
# =================================================================================

export EXPERIMENT_NUM=8
export EXPERIMENT_NAME="experiment8"
export MODE="ecmp"
export TIMEOUT="1000m"
export NUM_RUNS=1
export BASE_ECMP_SEED=25
export TOPOLOGY_NAMES="FoldedClosECMP1024"
export NS3_CONFIG_INDICES="3"
export SIM_TYPES="g2"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source the template for function definitions
source "$SCRIPT_DIR/compare_networks_template.sh"

# =================================================================================
# --- PER-NPU CONFIGURATION TABLE ---
# Adjust these to tune resource usage per scale.
# get_npu_parallel <npus>  → max parallel jobs while running that NPU count
# get_npu_vmem     <npus>  → total virtual memory budget (KB) for that NPU group
# Per-job limit = get_npu_vmem / get_npu_parallel
# =================================================================================
get_npu_parallel() {
    case "$1" in
        2|4|8)    echo 20 ;;
        16|32)    echo 15 ;;
        64|128)   echo 10 ;;
        256|512)  echo 8 ;;
        1024)     echo 5 ;;
        *)        echo 4 ;;
    esac
}
get_npu_vmem() {
    case "$1" in
        2|4|8)    echo 45000000 ;;
        16|32)    echo 45000000 ;;
        64|128)   echo 45000000 ;;
        256|512)  echo 45000000 ;;
        1024)     echo 45000000 ;;
        *)        echo 45000000 ;;
    esac
}
export -f get_npu_parallel get_npu_vmem

# Set paths that are constant across all NPU counts
export BASE_OUTPUT_DIR="$ASTRA_SIM_ROOT/upc/output/comparison_run/experiment8"
export NS3_TOPOLOGY_BASE="$ASTRA_SIM_ROOT/upc/experiments_files/experiment8/configuration/ns3/topologies/"
export G2_TOPOLOGY_BASE="$ASTRA_SIM_ROOT/upc/experiments_files/experiment8/configuration/g2/topologies/"

# Override execute_job: extract NPUS_COUNT from the 7th field of the job string
execute_job_exp8() {
    local job_string="$1"
    IFS='|' read -r sim_type workload_dir sys_name topo_name run_num ns3_conf_idx npus_count <<< "$job_string"

    # Apply per-NPU virtual memory limit (per-job = total / parallel)
    local npu_vmem npu_parallel
    npu_vmem=$(get_npu_vmem "$npus_count")
    npu_parallel=$(get_npu_parallel "$npus_count")
    if [ "$npu_vmem" -gt 0 ] 2>/dev/null; then
        ulimit -v $(( npu_vmem / npu_parallel )) 2>/dev/null
    fi

    # Skip if output already exists for this sim/workload/run
    local workload_name topo_short run_padded output_pattern
    workload_name=$(basename "$workload_dir")
    topo_short="${topo_name%%_*}"
    run_padded=$(printf '%02d' "${run_num}")
    output_pattern="${BASE_OUTPUT_DIR}/${topo_short}/npu_${npus_count}/${workload_name}/run_${sim_type}_${run_padded}_*"
    if compgen -G "$output_pattern" > /dev/null 2>&1; then
        echo ">>> SKIP (exists): [$sim_type] npu=${npus_count} | ${workload_name} | run=${run_num}"
        return 0
    fi

    export NPUS_COUNT="$npus_count"
    run_single_simulation "$sim_type" "$workload_dir" "$sys_name" "$topo_name" "$run_num" "$ns3_conf_idx"
}
export -f execute_job_exp8

# --- Collect jobs into per-NPU files, then execute each group ---
JOBS_TMPDIR=$(mktemp -d)
trap "rm -rf $JOBS_TMPDIR" EXIT

for NPUS in 2 4 8 16 32 64 128 256 512 1024; do
    export NPUS_COUNT=$NPUS
    export SYSTEM_CONFIG_NAMES="FoldedClos${NPUS}"
    BASE_WORKLOAD_DIR="$ASTRA_SIM_ROOT/upc/experiments_files/experiment8/workload/npu_${NPUS}"

    # Re-build arrays for generate_job_list
    IFS=' ' read -r -a SYSTEM_CONFIG_NAMES_ARRAY <<< "$SYSTEM_CONFIG_NAMES"
    IFS=' ' read -r -a TOPOLOGY_NAMES_ARRAY <<< "$TOPOLOGY_NAMES"
    IFS=' ' read -r -a NS3_CONFIG_INDICES_ARRAY <<< "$NS3_CONFIG_INDICES"
    SIM_TYPES_RESOLVED="$SIM_TYPES"
    if [ "$SIM_TYPES_RESOLVED" == "all" ]; then
        SIM_TYPES_RESOLVED="analytical g2 ns3"
    fi
    IFS=' ' read -r -a SIM_TYPES_ARRAY <<< "$SIM_TYPES_RESOLVED"

    # Skip if no workloads for this NPU count
    if [ ! -d "$BASE_WORKLOAD_DIR" ]; then
        echo ">>> No workloads for NPU count $NPUS, skipping."
        continue
    fi

    workload_count=$(find "$BASE_WORKLOAD_DIR" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
    if [ "$workload_count" -eq 0 ]; then
        echo ">>> No workload directories under npu_${NPUS}, skipping."
        continue
    fi

    echo "--- Collecting jobs for NPU count: $NPUS (Workloads: $workload_count) ---"

    # Generate jobs and save to per-NPU file
    TEMP_JOBS=$(mktemp)
    generate_job_list "$TEMP_JOBS"
    sed "s/$/|${NPUS}/" "$TEMP_JOBS" > "$JOBS_TMPDIR/npu_${NPUS}.jobs"
    rm -f "$TEMP_JOBS"
done

total_jobs=$(cat "$JOBS_TMPDIR"/*.jobs 2>/dev/null | wc -l)

# Merge per-NPU job files in ascending NPU order into one ordered list
ALL_ORDERED="$JOBS_TMPDIR/all_jobs.ordered"
for NPUS in 2 4 8 16 32 64 128 256 512 1024; do
    [ -f "$JOBS_TMPDIR/npu_${NPUS}.jobs" ] && cat "$JOBS_TMPDIR/npu_${NPUS}.jobs" >> "$ALL_ORDERED"
done

echo ""
echo "========================================================================="
echo "--- EXPERIMENT 8 | Total jobs: $total_jobs ---"
echo "--- Topology: $TOPOLOGY_NAMES | Simulators: $SIM_TYPES ---"
echo "--- Runs per config: $NUM_RUNS | ECMP seed base: $BASE_ECMP_SEED ---"
echo "--- Dispatcher: dynamic slot limit per NPU group (see get_npu_parallel) ---"
echo "========================================================================="
echo ""
echo "Slot limits by NPU: 2-8→$(get_npu_parallel 8) | 16-32→$(get_npu_parallel 32) | 64-128→$(get_npu_parallel 128) | 256-512→$(get_npu_parallel 512) | 1024→$(get_npu_parallel 1024)"
echo ""

# =================================================================================
# Dynamic dispatcher — reads jobs in ascending NPU order.
# Before launching each job, waits until running_count < get_npu_parallel(job_npus).
# This means the next NPU group starts as soon as its slot opens, without waiting
# for all jobs of the previous group to finish.
#   e.g. NPU=128 limit=4, NPU=256 limit=2:
#        NPU=256 jobs start once only 1 NPU=128 job remains (running < 2).
# =================================================================================
RUNNING_PIDS=()

_reap_finished() {
    local alive=()
    for pid in "${RUNNING_PIDS[@]}"; do
        kill -0 "$pid" 2>/dev/null && alive+=("$pid")
    done
    RUNNING_PIDS=("${alive[@]}")
}

while IFS= read -r job_string; do
    npus_count="${job_string##*|}"
    max_slots=$(get_npu_parallel "$npus_count")

    # Wait until a slot is free for this NPU group's parallelism limit
    while true; do
        _reap_finished
        [ "${#RUNNING_PIDS[@]}" -lt "$max_slots" ] && break
        sleep 0.5
    done

    echo ">>> LAUNCH npu=${npus_count} | max_slots=${max_slots} | running=${#RUNNING_PIDS[@]}"
    execute_job_exp8 "$job_string" &
    RUNNING_PIDS+=($!)
done < "$ALL_ORDERED"

# Drain remaining jobs
wait

echo ""
echo "#################################################################"
echo "--- ALL NPU COUNTS FINISHED: EXPERIMENT 8 ---"
echo "#################################################################"
