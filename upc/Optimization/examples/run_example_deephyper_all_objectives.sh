#!/usr/bin/env bash
set -u

# ============================================================================
# DeepHyper Sweep: All Available Objectives
# ============================================================================
# Script to sweep through all available objectives in DeepHyper optimization.
#
# Usage:
#   ./run_example_deephyper_all_objectives.sh [objectives...]
#   
#   If no objectives are specified, all available objectives are swept.
#   Results are organized by objective with plots saved for 2-objective runs.
#
# ============================================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
EXAMPLE_SCRIPT="${SCRIPT_DIR}/example_deephyper_opt_sweep.py"
RESULTS_DIR="${SCRIPT_DIR}/SWEEP_RESULTS_$(date +%Y%m%d_%H%M%S)"

cd "$SCRIPT_DIR" || exit 1

# Load objectives: either from args or dynamically discover
if [[ "$#" -gt 0 ]]; then
  OBJECTIVES=("$@")
else
  mapfile -t OBJECTIVES < <("$PYTHON_BIN" "$EXAMPLE_SCRIPT" --list-objectives)
fi

if [[ "${#OBJECTIVES[@]}" -eq 0 ]]; then
  echo "❌ No objectives found to sweep."
  exit 1
fi

# Create results directory
mkdir -p "$RESULTS_DIR"

echo "════════════════════════════════════════════════════════════════════════"
echo "DeepHyper Sweep: All Objectives"
echo "════════════════════════════════════════════════════════════════════════"
echo ""
echo "Results directory: $RESULTS_DIR"
echo "Objectives to sweep (${#OBJECTIVES[@]}):"
for obj in "${OBJECTIVES[@]}"; do
  echo "  - $obj"
done
echo ""
echo "════════════════════════════════════════════════════════════════════════"
echo ""

failed=0
completed=0
for objective in "${OBJECTIVES[@]}"; do
  log_file="${RESULTS_DIR}/${objective}.log"
  summary_file="${RESULTS_DIR}/${objective}_summary.txt"
  
  echo "[$(( ++completed ))/${#OBJECTIVES[@]}] Running objective: ${objective}"
  
  if "$PYTHON_BIN" "$EXAMPLE_SCRIPT" "$objective" > "$log_file" 2>&1; then
    echo "  ✓ Completed (log: ${log_file})"
    
    # Extract key info and save summary
    if grep -q "BEST CONFIGURATION" "$log_file"; then
      {
        echo "Objective: $objective"
        echo "Status: ✓ Completed"
        echo ""
        grep -A 5 "BEST CONFIGURATION" "$log_file" | head -10
      } > "$summary_file"
      
      # Check for pareto plots
      if grep -q "Pareto.*saved" "$log_file"; then
        echo "  📊 Pareto plots generated"
      fi
    fi
  else
    echo "  ✗ Failed (log: ${log_file})"
    failed=1
    {
      echo "Objective: $objective"
      echo "Status: ✗ Failed"
      echo ""
      tail -20 "$log_file"
    } > "$summary_file"
  fi
  echo ""
done

echo "════════════════════════════════════════════════════════════════════════"
echo "Sweep Summary"
echo "════════════════════════════════════════════════════════════════════════"
echo ""
echo "Results saved in: $RESULTS_DIR"
echo "Total objectives: ${#OBJECTIVES[@]}"
echo ""
echo "Log files:"
ls -1 "$RESULTS_DIR"/*.log 2>/dev/null | head -5
if [[ $(ls -1 "$RESULTS_DIR"/*.log 2>/dev/null | wc -l) -gt 5 ]]; then
  echo "  ... and $(( $(ls -1 "$RESULTS_DIR"/*.log 2>/dev/null | wc -l) - 5 )) more"
fi
echo ""
echo "Summary files:"
ls -1 "$RESULTS_DIR"/*_summary.txt 2>/dev/null | head -5
echo ""
echo "Pareto plots are saved in individual optimization result directories"
echo "════════════════════════════════════════════════════════════════════════"
echo ""

exit "$failed"