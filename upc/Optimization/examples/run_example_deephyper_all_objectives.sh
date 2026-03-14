#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
EXAMPLE_SCRIPT="${SCRIPT_DIR}/example_deephyper_opt_sweep.py"

OBJECTIVES=(
  "power_cycles_network_bw"
  #"power"
  #"energy"
  #"power_and_time"
  #"energy_and_time"
  #"edp"
  #"edp_and_network_bw"
  #"ed2p_and_network_bw"
  #"e2d_and_network_bw"
  "energy_cycles_and_network_bw"
  #"weighted_edp"
  #"ed2p"
  #"e2d"
  #"energy_and_cycles"
)

cd "$SCRIPT_DIR" || exit 1

failed=0
for objective in "${OBJECTIVES[@]}"; do
  log_file="${objective}.txt"
  echo "======================================================================"
  echo "Running objective: ${objective}"
  echo "Log file: ${log_file}"
  echo "======================================================================"

  if "$PYTHON_BIN" "$EXAMPLE_SCRIPT" "$objective" > "$log_file" 2>&1; then
    echo "✓ Completed: ${objective}"
  else
    echo "✗ Failed: ${objective} (see ${log_file})"
    failed=1
  fi
  echo
 done

exit "$failed"
