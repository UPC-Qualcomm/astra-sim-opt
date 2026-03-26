#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
EXAMPLE_SCRIPT="${SCRIPT_DIR}/example_deephyper_opt_sweep.py"

cd "$SCRIPT_DIR" || exit 1

if [[ "$#" -gt 0 ]]; then
  OBJECTIVES=("$@")
else
  mapfile -t OBJECTIVES < <("$PYTHON_BIN" "$EXAMPLE_SCRIPT" --list-objectives)
fi

if [[ "${#OBJECTIVES[@]}" -eq 0 ]]; then
  echo "No objectives found to sweep."
  exit 1
fi

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

echo "All objectives completed. Check log files for details."
  exit "$failed"