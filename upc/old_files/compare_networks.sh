export ASTRA_SIM_ROOT=$(git rev-parse --show-toplevel)

for i in $(seq 1 1)
do
  SEED=$i
  echo "--- Starting Run $i with SEED=$SEED ---"
  python compare_networks.py \
        --npus-count 16 \
        --collectives all_gather \
        --groups '{"1": [0,2,4,6], "2": [1,3,5,7], "3":[8,10,12,14], "4":[9,11,13,15]}' \
        --g2-system-config $ASTRA_SIM_ROOT/upc/configuration/Ring_sys.json \
        --analytical-system-config $ASTRA_SIM_ROOT/upc/configuration/Ring_sys.json \
        --ns3-system-config $ASTRA_SIM_ROOT/upc/configuration/Ring_sys.json \
        --g2-network-config $ASTRA_SIM_ROOT/upc/configuration/g2/FoldedClos_16_config.yml \
        --analytical-network-config $ASTRA_SIM_ROOT/upc/configuration/g2/FoldedClos_16_config.yml \
        --ns3-network-config $ASTRA_SIM_ROOT/upc/configuration/ns3/FoldedClos_16_config.txt \
        --logical-topology-config $ASTRA_SIM_ROOT/upc/configuration/ns3/16_nodes_logical.json \
        --comm-size 150000 \
        --seed $SEED
        #--groups '{"1": [0,1,2,3,4,5,6,7], "2": [8,9,10,11,12,13,14,15]}' \
done

echo "--- All 5 runs completed. ---"