#!/bin/bash

# Check if a time is provided as an argument
if [ -z "$1" ]; then
  echo "Usage: $0 HHMM"
  echo "Example: $0 1146 to delete runs at 11:46"
  exit 1
fi

TIME_TO_DELETE=$1
SEARCH_DIR="upc/output"

echo "Searching for directories named 'run_*_${TIME_TO_DELETE}*' in ${SEARCH_DIR}."

# Find the directories
DIRS_TO_DELETE=$(find "${SEARCH_DIR}" -type d -name "run_*_${TIME_TO_DELETE}*")

if [ -z "$DIRS_TO_DELETE" ]; then
  echo "No directories found matching the pattern."
  exit 0
fi

echo "The following directories will be deleted:"
echo "$DIRS_TO_DELETE"
echo ""

read -p "Are you sure you want to delete these directories? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]
then
    # Delete the directories
    find "${SEARCH_DIR}" -type d -name "run_*_${TIME_TO_DELETE}*" -exec rm -rf {} +
    echo "Deletion complete."
else
    echo "Deletion cancelled."
fi
