import json
import sys
import os
from typing import List, Dict

# --- Chakra Imports ---
# Ensure chakra-tools is installed and accessible in your PYTHONPATH
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
    print("Error: Could not import Chakra. Please ensure 'chakra-tools' is installed (`pip install chakra-tools`)")
    print("and that PYTHONPATH is set correctly if needed.")
    sys.exit(1)

# --- Configuration ---

# Total number of NPUs in the system
NPUS_COUNT = 128

# Base directory to store all generated workloads
BASE_OUTPUT_DIR = "/app/astra-sim/upc/comparing_networks/workload/multiple_collectives_128"

# Collectives to generate traces for
COLLECTIVES_TO_GENERATE = [
    "all_reduce",
    "all_gather",
]

# Communication sizes in bytes
COMM_SIZES_TO_GENERATE = [
    6*1024*1024,
    32*1024*1024,  # 32 MB
    128*1024*1024, # 128 MB
]

# --- Chakra Collective Mapping ---
COLLECTIVE_MAP = {
    "all_gather": ALL_GATHER,
    "all_reduce": ALL_REDUCE,
    "all_to_all": ALL_TO_ALL,
    "reduce_scatter": REDUCE_SCATTER,
}

# --- Group Generation Functions ---

def generate_contiguous_groups(num_groups: int) -> Dict[str, List[int]]:
    """
    Generates N contiguous groups of NPUs.
    Example: 16 NPUs, 2 groups -> {0: [0-7], 1: [8-15]}
    """
    if NPUS_COUNT % num_groups != 0:
        raise ValueError(f"NPUS_COUNT ({NPUS_COUNT}) must be divisible by num_groups ({num_groups})")
    group_size = NPUS_COUNT // num_groups
    groups = {}
    for i in range(num_groups):
        start_npu = i * group_size
        groups[i+1] = list(range(start_npu, start_npu + group_size))
    return groups

def generate_strided_groups(stride: int) -> Dict[str, List[int]]:
    """
    Generates groups with a given stride.
    Example: 16 NPUs, stride 8 -> {0: [0, 8], 1: [1, 9], ... 7: [7, 15]}
    """
    if NPUS_COUNT % stride != 0:
         raise ValueError(f"NPUS_COUNT ({NPUS_COUNT}) must be divisible by stride ({stride})")
    num_groups = stride
    group_size = NPUS_COUNT // num_groups
    groups = {}
    for i in range(num_groups):
        groups[i+1] = [i + j * stride for j in range(group_size)]
    return groups

def generate_custom_strided_groups(num_groups: int, stride: int) -> Dict[str, List[int]]:
    """
    Generates groups of NPUs with a custom stride between elements.
    Example: 16 NPUs, 8 groups, stride 4 -> {0:[0,4], 1:[1,5]... 4:[8,12]...}
    """
    group_size = NPUS_COUNT // num_groups
    groups = {}
    for i in range(num_groups):
        # This logic is a bit more complex to handle wrapping around
        # and creating the desired strided pairs.
        start_npu = (i % (NPUS_COUNT // stride)) + (i // (NPUS_COUNT // stride)) * stride * group_size
        groups[i+1] = [start_npu + j * stride for j in range(group_size)]
    return groups

def generate_chunked_even_odd_groups(chunk_id: int, chunk_size: int) -> Dict[str, List[int]]:
    """
    Creates groups of even and odd indexed NPUs within a specific chunk.
    Example: 16 NPUs, chunk_size 8, chunk_id 0 -> {1: [0,2,4,6], 2: [1,3,5,7]}
    Example: 16 NPUs, chunk_size 8, chunk_id 1 -> {1: [8,10,12,14], 2: [9,11,13,15]}
    """
    if NPUS_COUNT % chunk_size != 0:
        raise ValueError(f"NPUS_COUNT ({NPUS_COUNT}) must be divisible by chunk_size ({chunk_size})")
    if chunk_id < 0 or chunk_id >= (NPUS_COUNT // chunk_size):
        raise ValueError(f"Invalid chunk_id ({chunk_id}) for the given NPUS_COUNT and chunk_size.")

    groups = {}
    chunk_start_npu = chunk_id * chunk_size
    even_group = []
    odd_group = []

    for i in range(chunk_size):
        npu = chunk_start_npu + i
        if i % 2 == 0:
            even_group.append(npu)
        else:
            odd_group.append(npu)

    groups[1] = even_group
    groups[2] = odd_group
    return groups

# --- Trace Generation Logic ---

def generate_comm_group_json(json_path: str, groups: dict):
    """Generates the communicator groups JSON file."""
    with open(json_path, "w") as f:
        json.dump(groups, f, indent=4)

def generate_et_files(npus_count: int, comm_size: int, groups: dict, output_dir: str, coll_type: int, coll_name: str):
    """Generates the Execution Trace (.et) files for each NPU."""
    # Map each NPU to the list of groups it belongs to.
    npu_to_groups = {i: [] for i in range(npus_count)}
    for group_id, members in groups.items():
        for npu_id in members:
            if npu_id in npu_to_groups:
                npu_to_groups[npu_id].append(group_id)

    # Iterate through all NPUs in the system to create a trace file for each.
    for npu_id in range(npus_count):
        output_filename = os.path.join(output_dir, f"{coll_name}.{npu_id}.et")
        with open(output_filename, "wb") as et:
            encode_message(et, GlobalMetadata(version="0.0.4"))
            node_id_counter = 0

            # If the NPU is part of any group, create the collective node(s).
            if npu_id in npu_to_groups and npu_to_groups[npu_id]:
                for group_id in npu_to_groups[npu_id]:
                    members = groups[group_id]
                    node = ChakraNode(
                        id=node_id_counter,
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
    # Hardcoded group definitions with explicit NPU numbers
    group_definitions = {
        "1": {
            "groups": {
                1: [i for i in range(0, 128, 2)], 2: [i for i in range(1, 128, 2)]
            }, "comm_pattern": "X1_COMM"
        },
        "2": {
            "groups": {
                1: [i for i in range(0, 128, 4)], 2: [i for i in range(1, 128, 4)],
                3: [i for i in range(2, 128, 4)], 4: [i for i in range(3, 128, 4)]
            }, "comm_pattern": "X1_COMM"
        },
        "4": {
            "groups": {
                1: [i for i in range(0, 128, 16)], 2: [i for i in range(1, 128, 16)],
                3: [i for i in range(2, 128, 16)], 4: [i for i in range(3, 128, 16)],
                5: [i for i in range(4, 128, 16)], 6: [i for i in range(5, 128, 16)],
                7: [i for i in range(6, 128, 16)], 8: [i for i in range(7, 128, 16)],
                9: [i for i in range(8, 128, 16)], 10: [i for i in range(9, 128, 16)],
                11: [i for i in range(10, 128, 16)], 12: [i for i in range(11, 128, 16)],
                13: [i for i in range(12, 128, 16)], 14: [i for i in range(13, 128, 16)],
                15: [i for i in range(14, 128, 16)], 16: [i for i in range(15, 128, 16)]
            }, "comm_pattern": "X1_COMM"
        },
        "5": {
            "groups": {
                1: [i for i in range(0, 128, 8)], 2: [i for i in range(1, 128, 8)],
                3: [i for i in range(2, 128, 8)], 4: [i for i in range(3, 128, 8)],
                5: [i for i in range(4, 128, 8)], 6: [i for i in range(5, 128, 8)],
                7: [i for i in range(6, 128, 8)], 8: [i for i in range(7, 128, 8)]
            }, "comm_pattern": "X1_COMM"
        },
        "7": {
            "groups": {
                1: [i for i in range(0, 128, 32)], 2: [i for i in range(1, 128, 32)],
                3: [i for i in range(2, 128, 32)], 4: [i for i in range(3, 128, 32)],
                5: [i for i in range(4, 128, 32)], 6: [i for i in range(5, 128, 32)],
                7: [i for i in range(6, 128, 32)], 8: [i for i in range(7, 128, 32)],
                9: [i for i in range(8, 128, 32)], 10: [i for i in range(9, 128, 32)],
                11: [i for i in range(10, 128, 32)], 12: [i for i in range(11, 128, 32)],
                13: [i for i in range(12, 128, 32)], 14: [i for i in range(13, 128, 32)],
                15: [i for i in range(14, 128, 32)], 16: [i for i in range(15, 128, 32)],
                17: [i for i in range(16, 128, 32)], 18: [i for i in range(17, 128, 32)],
                19: [i for i in range(18, 128, 32)], 20: [i for i in range(19, 128, 32)],
                21: [i for i in range(20, 128, 32)], 22: [i for i in range(21, 128, 32)],
                23: [i for i in range(22, 128, 32)], 24: [i for i in range(23, 128, 32)],
                25: [i for i in range(24, 128, 32)], 26: [i for i in range(25, 128, 32)],
                27: [i for i in range(26, 128, 32)], 28: [i for i in range(27, 128, 32)],
                29: [i for i in range(28, 128, 32)], 30: [i for i in range(29, 128, 32)],
                31: [i for i in range(30, 128, 32)], 32: [i for i in range(31, 128, 32)]
            }, "comm_pattern": "X1_COMM"
        },
        "9": {
            "groups": {
                **{i+1: [j for j in range(i, 64, 8)] for i in range(8)},
                **{i+9: [j for j in range(i+64, 128, 8)] for i in range(8)}
            }, "comm_pattern": "X1_COMM"
        },
        "10": {
            "groups": {
                **{i+1: [j for j in range(i, 32, 8)] for i in range(8)},
                **{i+9: [j for j in range(i+32, 64, 8)] for i in range(8)},
                **{i+17: [j for j in range(i+64, 96, 8)] for i in range(8)},
                **{i+25: [j for j in range(i+96, 128, 8)] for i in range(8)}
            }, "comm_pattern": "X1_COMM"
        },
        "11": {
            "groups": {
                **{i+1: [j for j in range(i, 64, 4)] for i in range(4)},
                **{i+5: [j for j in range(i+64, 128, 4)] for i in range(4)}
            }, "comm_pattern": "X1_COMM"
        },
        "12": {
            "groups": {
                i*8+j+1: [k for k in range(i*16 + j, i*16 + j + 9, 8)] for i in range(8) for j in range(8)
            }, "comm_pattern": "X1_COMM"
        }
    }

    total_workloads = len(COLLECTIVES_TO_GENERATE) * len(COMM_SIZES_TO_GENERATE) * len(group_definitions)
    print(f"Starting workload generation for {NPUS_COUNT} NPUs.")
    print(f"Total workloads to generate: {total_workloads}")
    print("-" * 40)

    generated_count = 0
    for coll_name in COLLECTIVES_TO_GENERATE:
        for comm_size in COMM_SIZES_TO_GENERATE:
            for group_name, group_info in group_definitions.items():
                groups = group_info["groups"]
                generated_count += 1
                # The print statement now shows the group name for clarity
                print(f"({generated_count}/{total_workloads}) Generating: {coll_name}, size={comm_size}, group={group_name}")

                # Create a new directory name using the collective, size, and group name.
                scenario_name = f"{coll_name}_size_{comm_size}_{group_name}"
                output_dir = os.path.join(
                    BASE_OUTPUT_DIR,
                    scenario_name
                )
                os.makedirs(output_dir, exist_ok=True)

                # Get Chakra collective type
                coll_type = COLLECTIVE_MAP[coll_name]

                # Generate communicator group JSON file
                json_path = os.path.join(output_dir, f"{coll_name}.json")
                generate_comm_group_json(json_path, groups)

                # Generate the actual trace files
                generate_et_files(NPUS_COUNT, comm_size, groups, output_dir, coll_type, coll_name)

    print("-" * 40)
    print("Workload generation complete.")
    print(f"All files are located in the '{BASE_OUTPUT_DIR}' directory.")

if __name__ == "__main__":
    main()