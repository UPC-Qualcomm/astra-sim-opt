#!/bin/bash

# Common settings
NPUS_COUNT=16
LOGICAL_CONFIG="/home/xavid/feina/astra-sim/upc/configuration/ns3/16_nodes_logical.json"
COMM_SIZE=150000
SEED=1

# System configs
G2_SYS_CONFIG="/home/xavid/feina/astra-sim/upc/configuration/g2/Switch_sys.json"
NS3_SYS_CONFIG="/home/xavid/feina/astra-sim/upc/configuration/g2/Switch_sys.json"
ANALYTICAL_SYS_CONFIG="/home/xavid/feina/astra-sim/upc/configuration/Switch_sys.json"

# Network configs
G2_NET_CONFIG="/home/xavid/feina/astra-sim/upc/configuration/g2/FoldedClos_16_config.yml"
ANALYTICAL_NET_CONFIG="/home/xavid/feina/astra-sim/upc/configuration/Switch.yml" 
NS3_CONFIG="/home/xavid/feina/astra-sim/upc/configuration/ns3/FoldedClos_16_config.txt"


# This mode skips workload generation and uses the directory provided in --workload-dir.
# The script will infer the collective name from the directory path.
echo "--- SCENARIO: Running simulations on a pre-existing workload ---"
python3 compare_networks.py \
      --workload-dir "/home/xavid/feina/astra-sim/upc/comparing_networks/workload/sends_recv" \
      --npus-count $NPUS_COUNT \
      --comm-size $COMM_SIZE \
      --seed $SEED \
      --logical-topology-config $LOGICAL_CONFIG \
      --g2-system-config $G2_SYS_CONFIG \
      --g2-network-config $G2_NET_CONFIG \
      --ns3-system-config $NS3_SYS_CONFIG \
      --ns3-network-config $NS3_CONFIG \
      --analytical-system-config $ANALYTICAL_SYS_CONFIG \
      --analytical-network-config $ANALYTICAL_NET_CONFIG \


echo "--- All scenarios completed. ---"