#!/bin/bash

# Asegúrate de que ASTRA_SIM_ROOT está configurado. Si no, intenta obtenerlo con git.
if [ -z "$ASTRA_SIM_ROOT" ]; then
    export ASTRA_SIM_ROOT=$(git rev-parse --show-toplevel)
    echo "ASTRA_SIM_ROOT no estaba configurado. Se ha establecido en: $ASTRA_SIM_ROOT"
fi

# --- CONFIGURACIÓN GENERAL ---
NPUS_COUNT=16
LOGICAL_CONFIG="$ASTRA_SIM_ROOT/upc/configuration/ns3/16_nodes_logical.json"
COMM_SIZE=150000
PYTHON_EXEC="../../../opt/venv/astra-sim/bin/python"

# --- CONFIGURACIÓN DE SISTEMAS ---
G2_SYS_CONFIG="$ASTRA_SIM_ROOT/upc/configuration/g2/FullyConnected_sys.json"
NS3_SYS_CONFIG="$ASTRA_SIM_ROOT/upc/configuration/g2/FullyConnected_sys.json"
ANALYTICAL_SYS_CONFIG="$ASTRA_SIM_ROOT/upc/configuration/FullyConnected_sys.json"

# --- CONFIGURACIÓN DE REDES (Plantillas) ---
G2_NET_CONFIG="$ASTRA_SIM_ROOT/upc/configuration/g2/FoldedClos_16_config.yml"
ANALYTICAL_NET_CONFIG="$ASTRA_SIM_ROOT/upc/configuration/FullyConnected.yml"

# --- LISTAS PARA LOS BUCLES ---
WORKLOAD_DIRS=(
    # "$ASTRA_SIM_ROOT/upc/comparing_networks/workload/toy_all_to_all_one_collective"
    # "$ASTRA_SIM_ROOT/upc/comparing_networks/workload/toy_all_reduce_one_collective"
    # "$ASTRA_SIM_ROOT/upc/comparing_networks/workload/model"
    "$ASTRA_SIM_ROOT/upc/comparing_networks/workload/modelGPT_3_1300M"
    # "$ASTRA_SIM_ROOT/upc/comparing_networks/workload/modelGPT_40B"
    # "$ASTRA_SIM_ROOT/upc/comparing_networks/workload/modelT5_Small"
)
NS3_CONFIG_INDICES=(1 2 3 4 5 6 7 8 9 10 11 12 13 14) # Índices de los ficheros de config de NS3 a usar
TOPOLOGY_INDICES=(0) # Índices de las topologías a usar

# =================================================================================
# --- BUCLE PRINCIPAL ---
# =================================================================================

for workload_dir in "${WORKLOAD_DIRS[@]}"; do
    echo "#################################################################"
    echo "--- PROCESANDO WORKLOAD: $(basename $workload_dir) ---"
    echo "#################################################################"

    # --- 1. EJECUCIÓN DE G2 CON DISTINTAS TOPOLOGÍAS (Bucle original) ---
    echo "--- Iniciando ejecuciones de G2 con distintas topologías ---"
    for topo_idx in "${TOPOLOGY_INDICES[@]}"; do
        echo "  --- Usando Topología G2 N°$topo_idx ---"
        
        TOPOLOGY_FILE="$ASTRA_SIM_ROOT/upc/configuration/g2/topologies/G2_FoldedClos_16.0_topology${topo_idx}.json"

        timeout 5m "$PYTHON_EXEC" compare_networks.py \
            --workload-dir "$workload_dir" \
            --npus-count $NPUS_COUNT \
            --comm-size $COMM_SIZE \
            --logical-topology-config "$LOGICAL_CONFIG" \
            --g2-system-config "$G2_SYS_CONFIG" \
            --g2-network-config "$G2_NET_CONFIG" \
            --g2-topology-file "$TOPOLOGY_FILE" \
            --analytical-system-config "$ANALYTICAL_SYS_CONFIG" \
            --analytical-network-config "$ANALYTICAL_NET_CONFIG" \
            --python-exec "$PYTHON_EXEC"
    done
    echo "--- Finalizadas ejecuciones de G2 con distintas topologías ---"


    # --- 2. EJECUCIÓN DE NS3 CON DISTINTAS TOPOLOGÍAS (Bucle original) ---
    # echo "--- Iniciando ejecuciones de NS3 con distintas topologías ---"
    # for ns3_conf_idx in "${NS3_CONFIG_INDICES[@]}"; do
    #     NS3_CONFIG_FILE="$ASTRA_SIM_ROOT/upc/configuration/ns3/configs/FoldedClos_16_config${ns3_conf_idx}.txt"
        
    #     if [ ! -f "$NS3_CONFIG_FILE" ]; then
    #         echo "  --- AVISO: No se encontró el fichero de configuración NS3 N°$ns3_conf_idx. Saltando... ---"
    #         continue
    #     fi

    #     echo "  --- Usando Configuración NS3 N°$ns3_conf_idx ---"

    #     for topo_idx in "${TOPOLOGY_INDICES[@]}"; do
    #         echo "    --- Usando Topología NS3 N°$topo_idx ---"

    #         TOPOLOGY_FILE="$ASTRA_SIM_ROOT/upc/configuration/ns3/topologies/ns3_FoldedClos_16.0_topology${topo_idx}"

    #         timeout 5m "$PYTHON_EXEC" compare_networks.py \
    #             --workload-dir "$workload_dir" \
    #             --npus-count $NPUS_COUNT \
    #             --comm-size $COMM_SIZE \
    #             --logical-topology-config "$LOGICAL_CONFIG" \
    #             --ns3-system-config "$NS3_SYS_CONFIG" \
    #             --ns3-network-config "$NS3_CONFIG_FILE" \
    #             --ns3-topology-file "$TOPOLOGY_FILE" \
    #             --ns3-precomputed-paths 1 \
    #             --python-exec "$PYTHON_EXEC"
    #     done
    # done
    # echo "--- Finalizadas ejecuciones de NS3 con distintas topologías ---"

    # --- 3. EJECUCIÓN DE G2 (Topología Fija) 5 VECES ---
    # echo "--- Iniciando 5 ejecuciones de G2 con topología fija ---"
    # G2_FIXED_TOPOLOGY="$ASTRA_SIM_ROOT/upc/configuration/g2/topologies/G2_FoldedClos_16.0_topology_all_paths.json"
    # for run in {1..5}; do
    #     echo "  --- Ejecución G2 N°$run/5 ---"
    #     timeout 5m "$PYTHON_EXEC" compare_networks.py \
    #         --workload-dir "$workload_dir" \
    #         --npus-count $NPUS_COUNT \
    #         --comm-size $COMM_SIZE \
    #         --logical-topology-config "$LOGICAL_CONFIG" \
    #         --g2-system-config "$G2_SYS_CONFIG" \
    #         --g2-network-config "$G2_NET_CONFIG" \
    #         --g2-topology-file "$G2_FIXED_TOPOLOGY" \
    #         --python-exec "$PYTHON_EXEC"
    # done
    # echo "--- Finalizadas 5 ejecuciones de G2 con topología fija ---"


    # --- 4. EJECUCIÓN DE NS3 (Topología Fija, sin override de precomputed-paths) 5 VECES ---
    # echo "--- Iniciando 5 ejecuciones de NS3 con topología fija ---"
    # NS3_FIXED_TOPOLOGY="$ASTRA_SIM_ROOT/upc/configuration/ns3/topologies/ns3_FoldedClos_16.0_topology_all_paths"
    # for run in {1..5}; do
    #     echo "  --- Ronda de ejecución NS3 N°$run/5 ---"
    #     for ns3_conf_idx in "${NS3_CONFIG_INDICES[@]}"; do
    #         NS3_CONFIG_FILE="$ASTRA_SIM_ROOT/upc/configuration/ns3/configs/FoldedClos_16_config${ns3_conf_idx}.txt"
            
    #         if [ ! -f "$NS3_CONFIG_FILE" ]; then
    #             echo "    --- AVISO: No se encontró el fichero de configuración NS3 N°$ns3_conf_idx. Saltando... ---"
    #             continue
    #         fi

    #         echo "    --- Usando Configuración NS3 N°$ns3_conf_idx ---"

    #         timeout 5m "$PYTHON_EXEC" compare_networks.py \
    #             --workload-dir "$workload_dir" \
    #             --npus-count $NPUS_COUNT \
    #             --comm-size $COMM_SIZE \
    #             --logical-topology-config "$LOGICAL_CONFIG" \
    #             --ns3-system-config "$NS3_SYS_CONFIG" \
    #             --ns3-network-config "$NS3_CONFIG_FILE" \
    #             --ns3-topology-file "$NS3_FIXED_TOPOLOGY" \
    #             --python-exec "$PYTHON_EXEC"
    #     done
    # done
    # echo "--- Finalizadas 5 ejecuciones de NS3 con topología fija ---"

done

echo "#################################################################"
echo "--- TODAS LAS SIMULACIONES HAN FINALIZADO ---"
echo "#################################################################"