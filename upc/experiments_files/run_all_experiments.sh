#!/bin/bash
# Run all experiments sequentially

# ./run_experiment1.sh
# ./run_experiment2_deterministic.sh
# ./run_experiment2_ecmp.sh
./run_experiment3_deterministic.sh
./run_experiment4_deterministic.sh
./run_experiment3_ecmp.sh
./run_experiment4_ecmp.sh