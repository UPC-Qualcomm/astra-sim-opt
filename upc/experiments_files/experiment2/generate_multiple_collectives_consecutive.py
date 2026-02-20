from typing import List, Dict
import os
import json

# --- Chakra Imports ---
try:
    from chakra.src.third_party.utils.protolib import encodeMessage as encode_message
    from chakra.schema.protobuf.et_def_pb2 import (
        Node as ChakraNode,
        GlobalMetadata,
        AttributeProto as ChakraAttr,
        COMM_COLL_NODE,
        COMP_NODE,
        ALL_GATHER,
        REDUCE_SCATTER,
        BoolList,
    )
except ImportError:
    print("Error: Could not import Chakra. Please ensure 'chakra-tools' is installed.")
    import sys
    sys.exit(1)

# --- Configuration ---

# Total number of NPUs in the system
NPUS_COUNT = 16

# Base directory to store all generated workloads
BASE_OUTPUT_DIR = "/app/astra-sim/upc/experiments_files/experiment2/workload/multiple_collectives_ecmp"

# Communication size in bytes
COMM_SIZE = 128*1024*1024

# Workload configurations: list of (workload_name, list of (collective_type, size_multiplier, name) tuples)
WORKLOAD_CONFIGS = [
    ("single_allgather", [(ALL_GATHER, 1, "all_gather")]),  # 1 all_gather
    ("single_reducescatter", [(REDUCE_SCATTER, 16, "reduce_scatter")]),  # 1 reduce_scatter
    # ("double_allgather", [(ALL_GATHER, 1, "all_gather"), (ALL_GATHER, 1, "all_gather")]),  # 2 all_gather
    # ("mixed_ag_rs", [(ALL_GATHER, 1, "all_gather"), (ALL_GATHER, 1, "all_gather"), (REDUCE_SCATTER, 16, "reduce_scatter")]),  # 2 all_gather + 1 reduce_scatter
    ("six_allgather", [(ALL_GATHER, 1, "all_gather"), (ALL_GATHER, 1, "all_gather"), (ALL_GATHER, 1, "all_gather"), 
     (ALL_GATHER, 1, "all_gather"), (ALL_GATHER, 1, "all_gather"), (ALL_GATHER, 1, "all_gather")])  # 6 all gather
] 

# --- Trace Generation Logic ---

def generate_comm_group_json(json_path: str, groups: dict):
    """Generates the communicator groups JSON file."""
    with open(json_path, "w") as f:
        json.dump(groups, f, indent=4)

def generate_et_files(npus_count: int, base_comm_size: int, groups: dict, output_dir: str, collectives_list: list, workload_name: str):
    """Generates the Execution Trace (.et) files for each NPU.
    
    Args:
        collectives_list: List of (coll_type, size_multiplier, coll_name) tuples
    """
    # Map each NPU to the list of groups it belongs to.
    npu_to_groups = {i: [] for i in range(npus_count)}
    for group_id, members in groups.items():
        for npu_id in members:
            if npu_id in npu_to_groups:
                npu_to_groups[npu_id].append(group_id)

    # Iterate through all NPUs in the system to create a trace file for each.
    for npu_id in range(npus_count):
        output_filename = os.path.join(output_dir, f"{workload_name}.{npu_id}.et")
        with open(output_filename, "wb") as et:
            encode_message(et, GlobalMetadata(version="0.0.4"))
            node_id_counter = 0

            # If the NPU is part of any group, create the collective node(s).
            if npu_id in npu_to_groups and npu_to_groups[npu_id]:
                for group_id in npu_to_groups[npu_id]:
                    members = groups[group_id]
                    # Generate collectives in sequence
                    for coll_idx, (coll_type, size_multiplier, coll_name) in enumerate(collectives_list):
                        comm_size = base_comm_size * size_multiplier
                        node = ChakraNode(
                            id=node_id_counter,
                            name=f"{coll_name}_Collective_{coll_idx+1}_in_pg_{group_id}",
                            type=COMM_COLL_NODE,
                        )
                        # Add dependency to previous node if not the first collective
                        if node_id_counter > 0:
                            node.data_deps.append(node_id_counter - 1)
                        
                        node.attr.extend([
                            ChakraAttr(name="comm_type", int64_val=coll_type),
                            ChakraAttr(name="comm_size", uint64_val=comm_size),
                            ChakraAttr(name="pg_name", string_val=str(group_id)),
                            ChakraAttr(name="involved_dim", bool_list=BoolList(values=[i in members for i in range(npus_count)]))
                        ])
                        encode_message(et, node)
                        node_id_counter += 1
            else:
                # If the NPU is not in any group, create a dummy compute node.
                dummy_node = ChakraNode(
                    id=node_id_counter,
                    name="dummy_node",
                    type=COMP_NODE
                )
                dummy_node.attr.extend([
                    ChakraAttr(name="num_ops", int64_val=10000),
                    ChakraAttr(name="tensor_size", int64_val=10000)
                ])
                encode_message(et, dummy_node)

# --- Main Execution ---

def main():
    """
    Main function to generate all workload variations.
    """
    # Single group with all 16 NPUs
    groups = {
        1: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    }

    total_workloads = len(WORKLOAD_CONFIGS)
    print(f"Starting workload generation for {NPUS_COUNT} NPUs.")
    print(f"Total workloads to generate: {total_workloads}")
    print("-" * 40)

    for idx, (workload_name, collectives_list) in enumerate(WORKLOAD_CONFIGS, 1):
        # Create workload description for logging
        workload_desc = "_".join([f"{coll_name}" for _, _, coll_name in collectives_list])
        sizes_desc = "_".join([str(COMM_SIZE * mult) for _, mult, _ in collectives_list])
        print(f"({idx}/{total_workloads}) Generating '{workload_name}': {workload_desc}, sizes={sizes_desc}")

        # Create output directory using the workload name
        output_dir = os.path.join(BASE_OUTPUT_DIR, workload_name)
        os.makedirs(output_dir, exist_ok=True)

        # Generate communicator group JSON file
        json_path = os.path.join(output_dir, "workload.json")
        generate_comm_group_json(json_path, groups)

        # Generate the actual trace files
        generate_et_files(NPUS_COUNT, COMM_SIZE, groups, output_dir, collectives_list, "workload")

    print("-" * 40)
    print("Workload generation complete.")
    print(f"All files are located in the '{BASE_OUTPUT_DIR}' directory.")

if __name__ == "__main__":
    main()