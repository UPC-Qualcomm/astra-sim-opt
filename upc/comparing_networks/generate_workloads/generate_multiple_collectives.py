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
NPUS_COUNT = 16

# Base directory to store all generated workloads
BASE_OUTPUT_DIR = "/app/astra-sim/upc/comparing_networks/workload/multiple_collectives"

# Collectives to generate traces for
COLLECTIVES_TO_GENERATE = [
    "all_reduce",
    "all_gather",
    "reduce_scatter",
    "all_to_all"
]

# Communication sizes in bytes
COMM_SIZES_TO_GENERATE = [
    32*1024*1024,  # 32 MB
    1024*1024*1024, # 1 GB
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
    # Hardcoded group definitions
    group_definitions = {
        0: { # "1_contiguous_groups_of_16"
            1: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        },
        # 1: { # "2_contiguous_groups_of_8"
        #     1: [0, 1, 2, 3, 4, 5, 6, 7],
        #     2: [8, 9, 10, 11, 12, 13, 14, 15]
        # },
        2: { # "8_strided_groups_of_2_stride_8"
            1: [0, 8], 2: [1, 9], 3: [2, 10], 4: [3, 11],
            5: [4, 12], 6: [5, 13], 7: [6, 14], 8: [7, 15]
        },
        3: { # "8_strided_groups_of_2_stride_4"
            1: [0, 4], 2: [1, 5], 3: [2, 6], 4: [3, 7],
            5: [8, 12], 6: [9, 13], 7: [10, 14], 8: [11, 15]
        },
        4: { # "2_even_odd_groups_of_4_in_chunk_0"
            1: [0, 2, 4, 6],
            2: [1, 3, 5, 7]
        },
        5: { # "2_even_odd_groups_of_4_in_chunk_1"
            1: [8, 10, 12, 14],
            2: [9, 11, 13, 15]
        },
        # 6: { # "8_contiguous_groups_of_2"
        #     1: [0, 1], 2: [2, 3], 3: [4, 5], 4: [6, 7],
        #     5: [8, 9], 6: [10, 11], 7: [12, 13], 8: [14, 15]
        # },
        7: { # "4_strided_groups_of_4_stride_4"
            1: [0, 4, 8, 12], 2: [1, 5, 9, 13],
            3: [2, 6, 10, 14], 4: [3, 7, 11, 15]
        },
        # 8: { # "4_contiguous_groups_of_4"
        #     1: [0, 1, 2, 3], 2: [4, 5, 6, 7],
        #     3: [8, 9, 10, 11], 4: [12, 13, 14, 15]
        # },
        9: { # "2_strided_groups_of_8_stride_2"
            1: [0, 2, 4, 6, 8, 10, 12, 14],
            2: [1, 3, 5, 7, 9, 11, 13, 15]
        }
    }

    total_workloads = len(COLLECTIVES_TO_GENERATE) * len(COMM_SIZES_TO_GENERATE) * len(group_definitions)
    print(f"Starting workload generation for {NPUS_COUNT} NPUs.")
    print(f"Total workloads to generate: {total_workloads}")
    print("-" * 40)

    generated_count = 0
    for coll_name in COLLECTIVES_TO_GENERATE:
        for comm_size in COMM_SIZES_TO_GENERATE:
            for group_index, groups in group_definitions.items():
                generated_count += 1
                # The print statement now shows the group index for clarity
                print(f"({generated_count}/{total_workloads}) Generating: {coll_name}, size={comm_size}, group_key={group_index}")

                # Create a new directory name using the collective, size, and group index.
                scenario_name = f"{coll_name}_size_{comm_size}_group_{group_index}"
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