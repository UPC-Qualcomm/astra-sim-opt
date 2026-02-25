import os
import sys
import json
import yaml
import subprocess
import shutil
import argparse
from datetime import datetime


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

def run_simulation(sim_type, python_exec, workload_dir, system_config, network_config, ns3_config, memory_config, logical_topology_config, output_dir, project_root):
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
        run_script = os.path.join(project_root, "upc/run_astrasim_ns3.py")
        cmd.insert(1, run_script)
        cmd.extend([
            "--system", system_config,
            "--network_config", ns3_config,
            "--logical_topology", logical_topology_config
        ])
    else:
        run_script = os.path.join(project_root, "upc/run_astrasim.py")
        cmd.insert(1, run_script)
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
    
    # --- Determine Project Root from Environment Variable ---
    try:
        project_root = os.environ["ASTRA_SIM_ROOT"]
        if not os.path.isdir(project_root):
            print(f"Error: ASTRA_SIM_ROOT environment variable '{project_root}' is not a valid directory.")
            sys.exit(1)
    except KeyError:
        print("Error: Please set the 'ASTRA_SIM_ROOT' environment variable to the astra-sim project root directory.")
        sys.exit(1)

    workload_paths = {}
    if args.workload_dir:
        # --- Fase 1 (Opción A): Usar workload existente ---
        print(f"=== Fase 1: Usando workload pre-existente de '{args.workload_dir}' ===")
        if not os.path.isdir(args.workload_dir):
            print(f"Error: El directorio de workload '{args.workload_dir}' no existe.")
            sys.exit(1)
        
        base_name = args.workload_dir.split('workload/')[1] #os.path.basename(args.workload_dir)
        print(base_name)
        workload_paths[base_name] = args.workload_dir

    # --- 2. Preparar y Ejecutar Simulaciones para cada colectivo ---
    for coll_name, workload_dir in workload_paths.items():
        print(f"\n=== Fase 2: Ejecutando para el Colectivo='{coll_name}' ===")
        
        # Crear directorio de salida único para esta ejecución
        run_start_time = datetime.now()
        timestamp = run_start_time.strftime("%Y%m%d_%H%M%S_%f")[:-3] + "ms"
        # if args.g2_network_config:
        #     network_name = os.path.splitext(os.path.basename(args.g2_network_config))[0].split('_')[0]
        # elif args.ns3_network_config:
        #     network_name = os.path.splitext(os.path.basename(args.ns3_network_config))[0].split('_')[0]
        network_name = args.topology_name
        # Include run_number in the folder name if provided
        if args.run_number:
            run_folder_name = f"run_{args.run_number:02d}_{timestamp}"
        else:
            run_folder_name = f"run_{timestamp}"
        base_run_dir = os.path.join(args.base_output_dir, network_name, coll_name, run_folder_name)
        print(base_run_dir)
        
        configs_dir = os.path.join(base_run_dir, "configs")
        os.makedirs(configs_dir, exist_ok=True)
        print(f"Directorio de salida para esta ejecución: {base_run_dir}")

        # --- Guardar configuración de la ejecución ---
        config_summary_path = os.path.join(base_run_dir, "run_summary.txt")
        with open(config_summary_path, "w") as f:
            f.write("### AstraSim Experiment Configuration ###\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Run Directory: {base_run_dir}\n")
            if args.run_number:
                f.write(f"Run Number: {args.run_number}\n")
            f.write("\n")
            f.write("### Workload Characteristics ###\n")
            f.write(f"Collective: {coll_name}\n")
            f.write(f"Workload Directory: {workload_dir}\n")
            f.write(f"NPUs Count: {args.npus_count}\n")
            f.write(f"Communication Size (Bytes): {args.comm_size}\n")
            f.write(f"Communication Groups: {json.dumps(args.groups)}\n")
            f.write("\n")
            f.write("### Simulation Configuration ###\n")
            f.write(f"G2 System Config: {args.g2_system_config}\n")
            f.write(f"Analytical System Config: {args.analytical_system_config}\n")
            f.write(f"NS3 System Config: {args.ns3_system_config}\n")
            f.write(f"G2 Network Config: {args.g2_network_config}\n")
            f.write(f"Analytical Network Config: {args.analytical_network_config}\n")
            f.write(f"NS3 Network Config: {args.ns3_network_config}\n")
            f.write(f"NS3 Logical Topology: {args.logical_topology_config}\n")
            f.write("\n### Overrides ###\n")
            f.write(f"G2 Topology File Override: {args.g2_topology_file}\n")
            f.write(f"NS3 Topology File Override: {args.ns3_topology_file}\n")
            f.write(f"NS3 Precomputed Paths Override: {args.ns3_precomputed_paths}\n")
            if args.ns3_ecmp_seed is not None:
                f.write(f"NS3 ECMP Seed Override: {args.ns3_ecmp_seed}\n")
            f.write("\n")
            f.write(f"Python Executable: {args.python_exec}\n")
        print(f"Guardada la configuración de la ejecución en: {config_summary_path}")


        # --- 3. Copiar y preparar configuraciones para esta ejecución ---
        sim_types = []
        
        # Preparar configs de G2 si se han proporcionado
        g2_sys_conf_dest, g2_net_conf_dest = None, None
        if args.g2_system_config and args.g2_network_config:
            sim_types.append("g2")
            g2_sys_conf_dest = os.path.join(configs_dir, f"g2_{os.path.basename(args.g2_system_config)}")
            shutil.copy(args.g2_system_config, g2_sys_conf_dest)
            
            g2_net_conf_dest = os.path.join(configs_dir, f"g2_{os.path.basename(args.g2_network_config)}")
            g2_overrides = {}
            if args.g2_topology_file:
                g2_overrides["topology_file"] = os.path.abspath(args.g2_topology_file)
            modify_config_file(args.g2_network_config, g2_net_conf_dest, g2_overrides)

            # Leer el fichero de config de red de G2 para encontrar el fichero de topología
            try:
                with open(g2_net_conf_dest, 'r') as f: # Leer desde el fichero modificado
                    g2_net_data = yaml.safe_load(f)
                
                g2_topology_file_src = g2_net_data.get("topology_file")
                if g2_topology_file_src and os.path.exists(g2_topology_file_src):
                    g2_topology_dest = os.path.join(configs_dir, f"g2_{os.path.basename(g2_topology_file_src)}")
                    shutil.copy(g2_topology_file_src, g2_topology_dest)
                    print(f"Copiado fichero de topología G2: {g2_topology_file_src}")
                elif g2_topology_file_src:
                    print(f"Aviso: El 'topology_file' '{g2_topology_file_src}' especificado en la config de red G2 no existe.")

            except (yaml.YAMLError, FileNotFoundError) as e:
                print(f"Aviso: No se pudo leer el fichero de configuración de red G2 para buscar la topología: {e}")


        # Preparar configs analíticas si se han proporcionado
        analytical_sys_conf_dest, analytical_net_conf_dest = None, None
        if args.analytical_system_config and args.analytical_network_config:
            sim_types.extend(["analytical_unaware"])
            analytical_sys_conf_dest = os.path.join(configs_dir, f"analytical_{os.path.basename(args.analytical_system_config)}")
            shutil.copy(args.analytical_system_config, analytical_sys_conf_dest)
            analytical_net_conf_dest = os.path.join(configs_dir, f"analytical_{os.path.basename(args.analytical_network_config)}")
            shutil.copy(args.analytical_network_config, analytical_net_conf_dest)

        # Preparar configs de NS3 si se han proporcionado
        ns3_sys_conf_dest, ns3_conf_dest = None, None
        if args.ns3_system_config and args.ns3_network_config:
            sim_types.append("ns3")
            ns3_sys_conf_dest = os.path.join(configs_dir, f"ns3_{os.path.basename(args.ns3_system_config)}")
            shutil.copy(args.ns3_system_config, ns3_sys_conf_dest)
            
            ns3_conf_dest = os.path.join(configs_dir, os.path.basename(args.ns3_network_config))
            ns3_output_dir = os.path.join(base_run_dir, "ns3")
            os.makedirs(ns3_output_dir, exist_ok=True)
            ns3_overrides = {
                "TRACE_OUTPUT_FILE": os.path.join(project_root, "upc", ns3_output_dir, "astrasim_trace.tr"),
                "FCT_OUTPUT_FILE": os.path.join(project_root, "upc", ns3_output_dir, "astrasim_fct.txt"),
                "QLEN_MON_FILE": os.path.join(project_root, "upc", ns3_output_dir, "astrasim_qlen.txt"),
                "PFC_OUTPUT_FILE": os.path.join(project_root, "upc", ns3_output_dir, "astrasim_pfc.txt"),
            }
            if args.ns3_topology_file:
                ns3_overrides["TOPOLOGY_FILE"] = os.path.abspath(args.ns3_topology_file)
            if args.ns3_precomputed_paths is not None:
                ns3_overrides["USE_PRECOMPUTED_ROUTES"] = args.ns3_precomputed_paths
            if args.ns3_ecmp_seed is not None:
                ns3_overrides["ECMP_SEED"] = args.ns3_ecmp_seed

            modify_config_file(args.ns3_network_config, ns3_conf_dest, ns3_overrides)

        # Memory Config (solo copiar)
        mem_conf_src = os.path.join(project_root, "upc/configuration/RemoteMemory.json")
        mem_conf_dest = os.path.join(configs_dir, "RemoteMemory.json")
        shutil.copy(mem_conf_src, mem_conf_dest)

        # NS3 Logical Topology (solo copiar)
        lt_conf_dest = None
        if args.logical_topology_config:
            lt_conf_dest = os.path.join(configs_dir, os.path.basename(args.logical_topology_config))
            shutil.copy(args.logical_topology_config, lt_conf_dest)

        # --- 4. Ejecutar las simulaciones ---
        
        # Configurar PYTHONPATH para G2
        g2_path = os.path.join(project_root, "extern/network_backend/g2")
        original_pythonpath = os.environ.get('PYTHONPATH', '')
        os.environ['PYTHONPATH'] = f"{g2_path}:{original_pythonpath}"

        for sim_type in sim_types:
            output_dir = os.path.join(base_run_dir, sim_type)
            
            current_sys_config, current_net_config = None, None
            if sim_type == "g2":
                current_sys_config = g2_sys_conf_dest
                current_net_config = g2_net_conf_dest
            elif "analytical" in sim_type:
                current_sys_config = analytical_sys_conf_dest
                current_net_config = analytical_net_conf_dest
            elif sim_type == "ns3":
                current_sys_config = ns3_sys_conf_dest

            if not current_sys_config:
                print(f"Aviso: Omitiendo {sim_type} por falta de configuración de sistema.")
                continue

            run_simulation(
                sim_type=sim_type,
                python_exec=args.python_exec,
                workload_dir=workload_dir,
                system_config=current_sys_config,
                network_config=current_net_config,
                ns3_config=ns3_conf_dest,
                memory_config=mem_conf_dest,
                logical_topology_config=lt_conf_dest,
                output_dir=output_dir,
                project_root=project_root
            )
        
        # Restaurar PYTHONPATH
        os.environ['PYTHONPATH'] = original_pythonpath

        # --- 5. Guardar el tiempo de ejecución ---
        run_end_time = datetime.now()
        runtime = run_end_time - run_start_time
        with open(config_summary_path, "a") as f:
            f.write("\n### Execution Summary ###\n")
            f.write(f"Start Time: {run_start_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}\n")
            f.write(f"End Time: {run_end_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}\n")
            f.write(f"Total Runtime: {str(runtime)}\n")
        print(f"Tiempo de ejecución total para '{coll_name}': {runtime}")

    print("\n=== Todas las simulaciones han finalizado. ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script de comparación de redes para Astra-Sim.")

    # Argumentos para la generación de workloads
    parser.add_argument("--workload-dir", type=str, default=None, help="Ruta a un directorio de workload pre-generado. Si se especifica, se omite la generación.")
    parser.add_argument("--npus-count", type=int, default=8, help="Número total de NPUs en el sistema.")
    parser.add_argument("--comm-size", type=int, default=128*1024*1024, help="Tamaño en bytes de la comunicación colectiva.")
    parser.add_argument("--collectives", nargs='+', default=["all_gather"], help="Lista de colectivos a generar y probar.")
    parser.add_argument("--groups", type=json.loads, default='{"1": [0, 1, 4, 5], "2": [2, 3, 6, 7]}', help='Grupos de NPUs para los colectivos en formato JSON string.')

    # Argumentos para la configuración de la simulación
    parser.add_argument("--g2-system-config", type=str, default=None, help="Ruta al fichero de sistema para G2.")
    parser.add_argument("--analytical-system-config", type=str, default=None, help="Ruta al fichero de sistema para modelos analíticos.")
    parser.add_argument("--ns3-system-config", type=str, default=None, help="Ruta al fichero de sistema para NS3.")
    
    parser.add_argument("--g2-network-config", type=str, default=None, help="Ruta al fichero de configuración de red (YML) para G2.")
    parser.add_argument("--analytical-network-config", type=str, default=None, help="Ruta al fichero de configuración de red (YML) para modelos analíticos.")
    parser.add_argument("--ns3-network-config", type=str, default=None, help="Ruta al fichero de configuración de red para NS3.")
    
    parser.add_argument("--logical-topology-config", type=str, default=None, help="Ruta al fichero de topología lógica para NS3.")
    
    # Argumentos para sobreescribir configuraciones
    parser.add_argument("--g2-topology-file", type=str, default=None, help="Sobrescribe la ruta del fichero de topología en la configuración de red de G2.")
    parser.add_argument("--ns3-topology-file", type=str, default=None, help="Sobrescribe la ruta del fichero de topología en la configuración de red de NS3.")
    parser.add_argument("--ns3-precomputed-paths", type=int, default=None, help="Sobrescribe el valor de PRECOMPUTED_PATHS en la configuración de red de NS3.")

    # Otros
    parser.add_argument("--python-exec", type=str, default="../../../opt/venv/astra-sim/bin/python", help="Ruta al ejecutable de Python.")

    # New: topology name to drive output folder naming (overrides network-config derived name)
    parser.add_argument("--topology-name", type=str, default=None, help="Nombre de la topología (usado para nombrar la carpeta de salida).")
    
    # New: run number for multiple runs
    parser.add_argument("--run-number", type=int, default=None, help="Número de ejecución (para múltiples ejecuciones).")
    
    # New: NS3 ECMP seed override
    parser.add_argument("--ns3-ecmp-seed", type=int, default=None, help="Sobrescribe el valor de ECMP_SEED en la configuración de NS3.")
    
    # New: Base output directory
    parser.add_argument("--base-output-dir", type=str, default="output/comparison_run", help="Directorio base para las salidas (por defecto: output/comparison_run).")

    parsed_args = parser.parse_args()
    main(parsed_args)