#!/bin/bash

# export ASTRA_SIM_ROOT=$(git rev-parse --show-toplevel)

# Common settings
NPUS_COUNT=16
LOGICAL_CONFIG="$ASTRA_SIM_ROOT/upc/configuration/ns3/16_nodes_logical.json"
COMM_SIZE=150000

# System configs
G2_SYS_CONFIG="$ASTRA_SIM_ROOT/upc/configuration/g2/Switch_sys.json"
NS3_SYS_CONFIG="$ASTRA_SIM_ROOT/upc/configuration/g2/Switch_sys.json"
ANALYTICAL_SYS_CONFIG="$ASTRA_SIM_ROOT/upc/configuration/Switch_sys.json"

# Network configs
G2_NET_CONFIG="$ASTRA_SIM_ROOT/upc/configuration/g2/FoldedClos_16_config.yml"
ANALYTICAL_NET_CONFIG="$ASTRA_SIM_ROOT/upc/configuration/Switch.yml" 
# NS3_CONFIG is now set inside the loop


# This mode skips workload generation and uses the directory provided in --workload-dir.
# The script will infer the collective name from the directory path.
echo "--- SCENARIO: Running simulations on a pre-existing workload ---"

# for i in {1..39}
# do
#   echo "--- RUNNING NS3 CONFIGURATION $i ---"
#   NS3_CONFIG="$ASTRA_SIM_ROOT/upc/configuration/ns3/FoldedClos_16_config${i}.txt"

#   timeout 2m python3 compare_networks.py \
#         --workload-dir "$ASTRA_SIM_ROOT/upc/comparing_networks/workload/sends_recv_easy" \
#         --npus-count $NPUS_COUNT \
#         --comm-size $COMM_SIZE \
#         --logical-topology-config $LOGICAL_CONFIG \
#         --g2-system-config $G2_SYS_CONFIG \
#         --g2-network-config $G2_NET_CONFIG \
#         --ns3-system-config $NS3_SYS_CONFIG \
#         --ns3-network-config "$NS3_CONFIG" \
#         --analytical-system-config $ANALYTICAL_SYS_CONFIG \
#         --analytical-network-config $ANALYTICAL_NET_CONFIG
# done


NS3_SYS_CONFIG="$ASTRA_SIM_ROOT/upc/configuration/g2/FullyConnected_sys.json"

for i in {2..2}
do
  echo "--- RUNNING NS3 CONFIGURATION $i ---"
  NS3_CONFIG="$ASTRA_SIM_ROOT/upc/configuration/ns3/FoldedClos_16_config${i}.txt"

  timeout 50m python3 compare_networks.py \
        --workload-dir "$ASTRA_SIM_ROOT/upc/comparing_networks/workload/toy_all_to_all_one_collective" \
        --npus-count $NPUS_COUNT \
        --comm-size $COMM_SIZE \
        --logical-topology-config $LOGICAL_CONFIG \
        --g2-system-config $G2_SYS_CONFIG \
        --g2-network-config $G2_NET_CONFIG \
        --ns3-system-config $NS3_SYS_CONFIG \
        --ns3-network-config "$NS3_CONFIG" \
        --analytical-system-config $ANALYTICAL_SYS_CONFIG \
        --analytical-network-config $ANALYTICAL_NET_CONFIG
done


# for i in {5..5}
# do
#   echo "--- RUNNING NS3 CONFIGURATION $i ---"
#   NS3_CONFIG="$ASTRA_SIM_ROOT/upc/configuration/ns3/FoldedClos_16_config${i}.txt"

#   timeout 5m python3 compare_networks.py \
#         --workload-dir "$ASTRA_SIM_ROOT/upc/comparing_networks/workload/toy_all_reduce_one_collective" \
#         --npus-count $NPUS_COUNT \
#         --comm-size $COMM_SIZE \
#         --logical-topology-config $LOGICAL_CONFIG \
#         --g2-system-config $G2_SYS_CONFIG \
#         --g2-network-config $G2_NET_CONFIG \
#         --ns3-system-config $NS3_SYS_CONFIG \
#         --ns3-network-config "$NS3_CONFIG" \
#         --analytical-system-config $ANALYTICAL_SYS_CONFIG \
#         --analytical-network-config $ANALYTICAL_NET_CONFIG
# done
