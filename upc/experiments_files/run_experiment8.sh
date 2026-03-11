#!/bin/bash
# =================================================================================
# Experiment 8 - Multi-scale simulator comparison
# Varies NPU count (2-512), model shapes, and parallelization strategies
# Uses FoldedClos1024 topology for all scales
#
# Jobs from ALL NPU counts are collected first, then executed together
# so that MAX_PARALLEL_JOBS is utilized across the full set of workloads.
# =================================================================================

export EXPERIMENT_NUM=8
export EXPERIMENT_NAME="experiment8"
export MODE="ecmp"
export TIMEOUT="1000m"
export MAX_PARALLEL_JOBS=8
export TOTAL_VMEM_LIMIT=45000000
export NUM_RUNS=1
export BASE_ECMP_SEED=25
export TOPOLOGY_NAMES="FoldedClosECMP1024"
export NS3_CONFIG_INDICES="3"
export SIM_TYPES="g2 analytical"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source the template for function definitions
source "$SCRIPT_DIR/compare_networks_template.sh"

# Set paths that are constant across all NPU counts
export BASE_OUTPUT_DIR="$ASTRA_SIM_ROOT/upc/output/comparison_run/experiment8"
export NS3_TOPOLOGY_BASE="$ASTRA_SIM_ROOT/upc/experiments_files/experiment8/configuration/ns3/topologies/"
export G2_TOPOLOGY_BASE="$ASTRA_SIM_ROOT/upc/experiments_files/experiment8/configuration/g2/topologies/"

# Override execute_job: extract NPUS_COUNT from the 7th field of the job string
execute_job_exp8() {
    local job_string="$1"
    # Apply per-job virtual memory limit
    if [ -n "$TOTAL_VMEM_LIMIT" ] && [ "$TOTAL_VMEM_LIMIT" -gt 0 ] 2>/dev/null; then
        local per_job_limit=$((TOTAL_VMEM_LIMIT / MAX_PARALLEL_JOBS))
        ulimit -v "$per_job_limit" 2>/dev/null
    fi
    IFS='|' read -r sim_type workload_dir sys_name topo_name run_num ns3_conf_idx npus_count <<< "$job_string"

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

# --- Collect all jobs across all NPU counts ---
ALL_JOBS_FILE=$(mktemp)
trap "rm -f $ALL_JOBS_FILE" EXIT

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

    # Generate jobs into a temp file, then append |NPUS_COUNT to each line
    TEMP_JOBS=$(mktemp)
    generate_job_list "$TEMP_JOBS"
    sed "s/$/|${NPUS}/" "$TEMP_JOBS" >> "$ALL_JOBS_FILE"
    rm -f "$TEMP_JOBS"
done

total_jobs=$(wc -l < "$ALL_JOBS_FILE")

echo ""
echo "========================================================================="
echo "--- EXPERIMENT 8 | Total jobs across all NPU counts: $total_jobs ---"
echo "--- Max parallel jobs: $MAX_PARALLEL_JOBS ---"
echo "--- Topology: $TOPOLOGY_NAMES ---"
echo "--- Simulators: $SIM_TYPES ---"
echo "--- Runs per config: $NUM_RUNS | ECMP seed base: $BASE_ECMP_SEED ---"
echo "========================================================================="

# Execute ALL jobs in a single parallel batch
cat "$ALL_JOBS_FILE" | xargs -P "$MAX_PARALLEL_JOBS" -I {} bash -c 'execute_job_exp8 "{}"'

echo ""
echo "#################################################################"
echo "--- ALL NPU COUNTS FINISHED: EXPERIMENT 8 ---"
echo "#################################################################"
