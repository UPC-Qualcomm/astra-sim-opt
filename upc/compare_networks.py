import os
import sys
import json
import yaml
import subprocess
import shutil
import argparse
from datetime import datetime

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
    for group_id, members in groups.items():
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

# --- Lógica para modificar configuraciones y ejecutar simulaciones ---

def modify_config_file(src_path, dest_path, overrides):
    """
    Carga un fichero de configuración (JSON, YML, o texto plano),
    aplica cambios y lo guarda en una nueva ubicación.
    """
    shutil.copy(src_path, dest_path)
    if not overrides:
        return

    if src_path.endswith(".json"):
        with open(dest_path, 'r') as f:
            config = json.load(f)
        for key, value in overrides.items():
            # Soporte para claves anidadas como "general.num_npus"
            keys = key.split('.')
            d = config
            for k in keys[:-1]:
                d = d.setdefault(k, {})
            d[keys[-1]] = value
        with open(dest_path, 'w') as f:
            json.dump(config, f, indent=4)

    elif src_path.endswith((".yml", ".yaml")):
        with open(dest_path, 'r') as f:
            config = yaml.safe_load(f)
        for key, value in overrides.items():
            keys = key.split('.')
            d = config
            for k in keys[:-1]:
                d = d.setdefault(k, {})
            d[keys[-1]] = value
        with open(dest_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

    elif src_path.endswith(".txt"):
        with open(dest_path, 'r') as f:
            lines = f.readlines()
        
        new_lines = []
        keys_to_update = list(overrides.keys())
        
        for line in lines:
            found = False
            for key in keys_to_update:
                if line.strip().startswith(key):
                    new_lines.append(f"{key} {overrides[key]}\n")
                    keys_to_update.remove(key)
                    found = True
                    break
            if not found:
                new_lines.append(line)

        with open(dest_path, 'w') as f:
            f.writelines(new_lines)

def run_simulation(sim_type, python_exec, workload_dir, system_config, network_config, ns3_config, memory_config, logical_topology_config, output_dir):
    """Ejecuta una única simulación de Astra-Sim."""
    print(f"\n--- Ejecutando simulación para: {sim_type} ---")
    os.makedirs(output_dir, exist_ok=True)
    network_log = os.path.join(output_dir, "network.log")

    cmd = [
        python_exec,
        "--workload_dir", workload_dir,
        "--memory", memory_config,
        "--output_dir", output_dir,
        "--network_log", network_log,
    ]

    if sim_type == "ns3":
        cmd.insert(1, "run_astrasim_ns3.py")
        cmd.extend([
            "--system", system_config,
            "--network_config", ns3_config,
            "--logical_topology", logical_topology_config
        ])
    else:
        cmd.insert(1, "run_astrasim.py")
        cmd.extend([
            "--system", system_config,
            "--network", network_config,
            "--sim_type", sim_type
        ])

    print(f"Comando: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
        print(f"--- Simulación para {sim_type} finalizada con éxito ---")
    except subprocess.CalledProcessError as e:
        print(f"### ERROR al ejecutar la simulación para {sim_type}: {e} ###")
    except FileNotFoundError:
        print(f"### ERROR: No se encuentra el ejecutable '{cmd[0]}' o '{cmd[1]}'. Revisa la ruta. ###")


def main(args):
    """Función principal que orquesta la generación y ejecución."""
    
    # --- 1. Generar Workloads ---
    print("=== Fase 1: Generando Workloads ===")
    base_workload_dir = "./comparing_networks/workload"
    workload_paths = generate_workloads(
        args.npus_count, args.comm_size, args.collectives, args.groups, base_workload_dir
    )

    # --- 2. Preparar y Ejecutar Simulaciones para cada colectivo ---
    for coll_name, workload_dir in workload_paths.items():
        print(f"\n=== Fase 2: Ejecutando simulaciones para el colectivo '{coll_name}' ===")
        
        # Crear directorio de salida único para esta ejecución
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        network_name = os.path.splitext(os.path.basename(args.network_config))[0]
        run_folder_name = f"run_{timestamp}"
        base_run_dir = os.path.join("comparison_run", network_name, coll_name, run_folder_name)
        
        configs_dir = os.path.join(base_run_dir, "configs")
        os.makedirs(configs_dir, exist_ok=True)
        print(f"Directorio de salida para esta ejecución: {base_run_dir}")

        # --- Guardar configuración de la ejecución ---
        config_summary_path = os.path.join(base_run_dir, "run_summary.txt")
        with open(config_summary_path, "w") as f:
            f.write("### AstraSim Experiment Configuration ###\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Run Directory: {base_run_dir}\n")
            f.write("\n")
            f.write("### Workload Characteristics ###\n")
            f.write(f"Collective: {coll_name}\n")
            f.write(f"Workload Directory: {workload_dir}\n")
            f.write(f"NPUs Count: {args.npus_count}\n")
            f.write(f"Communication Size (Bytes): {args.comm_size}\n")
            f.write(f"Communication Groups: {json.dumps(args.groups)}\n")
            f.write("\n")
            f.write("### Simulation Configuration ###\n")
            f.write(f"System Config Template: {args.system_config}\n")
            f.write(f"Network Config Template: {args.network_config}\n")
            f.write(f"NS3 Config Template: {args.ns3_config}\n")
            f.write(f"NS3 Logical Topology Template: {args.logical_topology_config}\n")
            f.write(f"Python Executable: {args.python_exec}\n")
        print(f"Guardada la configuración de la ejecución en: {config_summary_path}")


        # --- 3. Modificar y guardar configuraciones para esta ejecución ---
        # Esto asegura la reproducibilidad
        
        # System Config
        sys_conf_src = args.system_config
        sys_conf_dest = os.path.join(configs_dir, os.path.basename(args.system_config))
        sys_overrides = {"general.num_npus": args.npus_count} if args.npus_count else {}
        modify_config_file(sys_conf_src, sys_conf_dest, sys_overrides)

        # Network Config (para g2/analytical)
        net_conf_src = args.network_config
        net_conf_dest = os.path.join(configs_dir, os.path.basename(args.network_config))
        # Aquí puedes añadir overrides para el YML si es necesario, ej: {"topology.0.npus_count.0": 16}
        modify_config_file(net_conf_src, net_conf_dest, {}) 

        # NS3 Config
        sim_types = ["analytical_unaware", "analytical_aware", "g2", "ns3"]
        sim_types = ["g2", "ns3"]
        ns3_conf_src = args.ns3_config
        ns3_conf_dest = os.path.join(configs_dir, os.path.basename(args.ns3_config))
        
        # Para NS3, preparamos los overrides de las rutas de salida
        ns3_overrides = {}
        if "ns3" in sim_types:
            ns3_output_dir = os.path.join(base_run_dir, "ns3")
            os.makedirs(ns3_output_dir, exist_ok=True)
            ns3_overrides = {
                "TRACE_OUTPUT_FILE": os.path.join("/home/xavid/feina/astra-sim/upc", ns3_output_dir, "astrasim_trace.tr"),
                "FCT_OUTPUT_FILE": os.path.join("/home/xavid/feina/astra-sim/upc", ns3_output_dir, "astrasim_fct.txt"),
                "PFC_OUTPUT_FILE": os.path.join("/home/xavid/feina/astra-sim/upc", ns3_output_dir, "astrasim_pfc.txt"),
                "QLEN_MON_FILE": os.path.join("/home/xavid/feina/astra-sim/upc", ns3_output_dir, "astrasim_qlen.txt"),
            }
        modify_config_file(ns3_conf_src, ns3_conf_dest, ns3_overrides)

        # Memory Config (solo copiar)
        mem_conf_src = "./configuration/RemoteMemory.json"
        mem_conf_dest = os.path.join(configs_dir, "RemoteMemory.json")
        shutil.copy(mem_conf_src, mem_conf_dest)

        # NS3 Logical Topology (solo copiar)
        lt_conf_src = args.logical_topology_config
        lt_conf_dest = os.path.join(configs_dir, os.path.basename(args.logical_topology_config))
        shutil.copy(lt_conf_src, lt_conf_dest)

        # --- 4. Ejecutar las simulaciones ---
        
        # Configurar PYTHONPATH para G2
        g2_path = os.path.abspath("../extern/network_backend/g2")
        original_pythonpath = os.environ.get('PYTHONPATH', '')
        os.environ['PYTHONPATH'] = f"{g2_path}:{original_pythonpath}"

        for sim_type in sim_types:
            output_dir = os.path.join(base_run_dir, sim_type)
            run_simulation(
                sim_type=sim_type,
                python_exec=args.python_exec,
                workload_dir=workload_dir,
                system_config=sys_conf_dest,
                network_config=net_conf_dest,
                ns3_config=ns3_conf_dest,
                memory_config=mem_conf_dest,
                logical_topology_config=lt_conf_dest,
                output_dir=output_dir
            )
        
        # Restaurar PYTHONPATH
        os.environ['PYTHONPATH'] = original_pythonpath

    print("\n=== Todas las simulaciones han finalizado. ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script de comparación de redes para Astra-Sim.")

    # Argumentos para la generación de workloads
    parser.add_argument("--npus-count", type=int, default=8, help="Número total de NPUs en el sistema.")
    parser.add_argument("--comm-size", type=int, default=128*1024*1024, help="Tamaño en bytes de la comunicación colectiva.")
    parser.add_argument("--collectives", nargs='+', default=["all_gather", "all_reduce"], help="Lista de colectivos a generar y probar.")
    parser.add_argument("--groups", type=json.loads, default='{"1": [0, 1, 4, 5], "2": [2, 3, 6, 7]}', help='Grupos de NPUs para los colectivos en formato JSON string.')

    # Argumentos para la configuración de la simulación
    parser.add_argument("--system-config", type=str, default="./configuration/Ring_sys.json", help="Ruta al fichero de configuración del sistema.")
    parser.add_argument("--network-config", type=str, default="./configuration/Ring.yml", help="Ruta al fichero de configuración de red (YML).")
    parser.add_argument("--ns3-config", type=str, default="./configuration/ns3/config_8_ring.txt", help="Ruta al fichero de configuración de NS3.")
    parser.add_argument("--logical-topology-config", type=str, default="./configuration/ns3/8_nodes_logical.json", help="Ruta al fichero de topología lógica para NS3.")
    
    # Otros
    parser.add_argument("--python-exec", type=str, default="../../astraenv39/bin/python3.9", help="Ruta al ejecutable de Python.")

    parsed_args = parser.parse_args()
    main(parsed_args)