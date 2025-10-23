for i in $(seq 1 1)
do
  SEED=$i
  echo "--- Starting Run $i with SEED=$SEED ---"
  python compare_networks.py \
        --npus-count 16 \
        --collectives all_gather \
        --groups '{"1": [0,2,4,6], "2": [1,3,5,7], "3":[8,10,12,14], "4":[9,11,13,15]}' \
        --system-config /home/xavid/feina/astra-sim/upc/configuration/Ring_sys.json \
        --network-config /home/xavid/feina/astra-sim/upc/configuration/g2/FoldedClos_16_config.yml \
        --ns3-config /home/xavid/feina/astra-sim/upc/configuration/ns3/FoldedClos_16_config.txt \
        --logical-topology-config /home/xavid/feina/astra-sim/upc/configuration/ns3/16_nodes_logical.json \
        --comm-size 150000 \
        --seed $SEED
        #--groups '{"1": [0,1,2,3,4,5,6,7], "2": [8,9,10,11,12,13,14,15]}' \
done

echo "--- All 5 runs completed. ---"