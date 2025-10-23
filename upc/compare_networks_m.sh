#!/bin/bash

# Common settings
NPUS_COUNT=16
NET_CONFIG="/home/xavid/feina/astra-sim/upc/configuration/g2/FoldedClos_16_config.yml"
NS3_CONFIG="/home/xavid/feina/astra-sim/upc/configuration/ns3/FoldedClos_16_config.txt"
LOGICAL_CONFIG="/home/xavid/feina/astra-sim/upc/configuration/ns3/16_nodes_logical.json"
COMM_SIZE=150000
SEED=1
# GROUPS variable is removed to avoid quoting issues.

# List of collectives to test
COLLECTIVES=(
    # "reduce"
    # "broadcast"
    # "gather"
    "all_gather"
    "all_reduce"
    "all_to_all"
    "reduce_scatter"
)

# List of system configurations to test
SYSTEMS=(
    "FullyConnected"
    "Ring"
    "Switch"
)

echo "--- Starting All Test Scenarios ---"

# Loop through each collective
for collective in "${COLLECTIVES[@]}"; do
    # Loop through each system configuration
    for system in "${SYSTEMS[@]}"; do
        
        SYS_CONFIG_FILE="/home/xavid/feina/astra-sim/upc/configuration/${system}_sys.json"
        
        # --- Running Test ---
        echo "--- SCENARIO: Collective: $collective, System: $system ---"
        
        # The --groups argument is now hardcoded with single quotes for safety.
        python3 compare_networks.py \
              --npus-count $NPUS_COUNT \
              --collectives "$collective" \
              --groups '{"1": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]}' \
              --system-config "$SYS_CONFIG_FILE" \
              --network-config $NET_CONFIG \
              --ns3-config $NS3_CONFIG \
              --logical-topology-config $LOGICAL_CONFIG \
              --comm-size $COMM_SIZE \
              --seed $SEED
              
        echo "" # Add a newline for better readability
    done
done

echo "--- All scenarios completed. ---"