import json
import sys
import os

# --- Lógica para la generación de Workloads (integrada desde generate_real_congestion.py) ---
# Asegúrate de que chakra-tools está instalado y accesible en tu PYTHONPATH
# pip install chakra-tools
try:
    from chakra.src.third_party.utils.protolib import encodeMessage as encode_message
    from chakra.schema.protobuf.et_def_pb2 import (
        Node as ChakraNode,
        GlobalMetadata,
        AttributeProto as ChakraAttr,
        COMM_COLL_NODE,
        COMP_NODE,
        GATHER,
        REDUCE,
        BROADCAST,
        ALL_REDUCE,
        ALL_GATHER,
        ALL_TO_ALL,
        REDUCE_SCATTER,
        BoolList,
    )
except ImportError:
    print("Error: No se pudo importar Chakra. Asegúrate de que 'chakra-tools' está instalado (`pip install chakra-tools`)")
    print("y de que el PYTHONPATH está configurado correctamente si es necesario.")
    sys.exit(1)

# Mapeo de nombres de colectivos a tipos de Chakra
COLLECTIVE_MAP = {
    "gather": GATHER,
    "reduce": REDUCE,
    "broadcast": BROADCAST,
    "all_gather": ALL_GATHER,
    "all_reduce": ALL_REDUCE,
    "all_to_all": ALL_TO_ALL,
    "reduce_scatter": REDUCE_SCATTER,
}

def generate_comm_group_json(json_path: str, groups: dict):
    """Genera el fichero JSON de grupos de comunicadores."""
    with open(json_path, "w") as f:
        json.dump(groups, f, indent=4)

def generate_congested_et_files(npus_count: int, comm_size: int, groups: dict, output_dir: str, coll_type: int, coll_name: str):
    """Genera los ficheros de traza (Execution Trace) para cada NPU."""
    involved_npus = set()
    for group_id, members in groups.items():
        involved_npus.update(members)
        for npu_id in members:
            output_filename = os.path.join(output_dir, f"{coll_name}.{npu_id}.et")
            with open(output_filename, "wb") as et:
                encode_message(et, GlobalMetadata(version="0.0.4"))
                node = ChakraNode(
                    id=1,
                    name=f"{coll_name}_Collective_in_pg_{group_id}",
                    type=COMM_COLL_NODE
                )
                node.attr.extend([
                    ChakraAttr(name="comm_type", int64_val=coll_type),
                    ChakraAttr(name="comm_size", uint64_val=comm_size),
                    ChakraAttr(name="pg_name", string_val=str(group_id)),
                    ChakraAttr(name="involved_dim", bool_list=BoolList(values=[i in members for i in range(npus_count)]))
                ])
                encode_message(et, node)
    
    # Generar ET para nodos no involucrados en la comunicación
    uninvolved_npus = set(range(npus_count)) - involved_npus
    for npu_id in uninvolved_npus:
        output_filename = os.path.join(output_dir, f"{coll_name}.{npu_id}.et")
        with open(output_filename, "wb") as et:
            encode_message(et, GlobalMetadata(version="0.0.4"))
            node = ChakraNode(
                id=1,
                name=f"comp_node_on_{npu_id}",
                type=COMP_NODE,
                duration_micros=1000  # Duración de ejemplo en microsegundos
            )
            encode_message(et, node)

    print(f"Generados ETs para el colectivo '{coll_name}' en: {output_dir}")

def generate_workloads(npus_count: int, comm_size: int, collectives: list, groups: dict, base_workload_dir: str):
    """Función principal para generar todas las cargas de trabajo necesarias."""
    generated_paths = {}
    for coll_name in collectives:
        if coll_name not in COLLECTIVE_MAP:
            print(f"Aviso: El colectivo '{coll_name}' no es válido. Omitiendo.")
            continue
        
        coll_type = COLLECTIVE_MAP[coll_name]
        output_dir = os.path.join(base_workload_dir, f"toy_{coll_name}_real_congestion")
        os.makedirs(output_dir, exist_ok=True)
        
        json_path = os.path.join(output_dir, f"{coll_name}.json")
        generate_comm_group_json(json_path, groups)
        generate_congested_et_files(npus_count, comm_size, groups, output_dir, coll_type, coll_name)
        
        generated_paths[coll_name] = output_dir
    return generated_paths