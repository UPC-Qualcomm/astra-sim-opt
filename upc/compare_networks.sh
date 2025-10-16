python compare_networks.py \
      --npus-count 16 \
      --collectives all_gather \
      --groups '{"1": [0,2,4,6], "2": [1,3,5,7], "3": [8,10,12,14], "4": [9,11,13,15]}' \
      --system-config /home/xavid/feina/astra-sim/upc/configuration/Ring_sys.json \
      --network-config /home/xavid/feina/astra-sim/upc/configuration/g2/config.yml \
      --ns3-config /home/xavid/feina/astra-sim/upc/configuration/ns3/config_16_folded_clos.txt \
      --logical-topology-config /home/xavid/feina/astra-sim/upc/configuration/ns3/16_nodes_logical.json \
      --comm-size 1310720