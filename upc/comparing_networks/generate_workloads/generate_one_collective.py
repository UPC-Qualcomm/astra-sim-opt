import os
import json
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

def generate_comm_group_json(json_path: str, groups: dict):
    with open(json_path, "w") as f:
        json.dump(groups, f)
    print(f"Generated communicator group config at {json_path}")

def generate_congested_et_files(npus_count: int, comm_size: int, groups: dict, output_dir: str, coll_type: int, coll_name: str):
    # Create traces for each communicator group in the given folder
    for group_id, members in groups.items():
        for npu_id in members:
            output_filename = f"{output_dir}/{coll_name}.{npu_id}.et"
            with open(output_filename, "wb") as et:
                encode_message(et, GlobalMetadata(version="0.0.4"))
                node_id = 1
                node = ChakraNode()
                node.id = node_id
                node.name = f"{coll_name}_Collective_in_pg_{group_id}"
                node.type = COMM_COLL_NODE
                node.attr.append(ChakraAttr(name="comm_type", int64_val=coll_type))
                node.attr.append(ChakraAttr(name="comm_size", uint64_val=comm_size))
                node.attr.append(ChakraAttr(name="pg_name", string_val=str(group_id)))
                involved_dims = [i in members for i in range(npus_count)]
                node.attr.append(ChakraAttr(name="involved_dim", bool_list=BoolList(values=involved_dims)))
                encode_message(et, node)
            print(f"Generated ET for NPU {npu_id} in group {group_id} at {output_dir}")

def main():
    npus_count = 8
    comm_size = 128 * 1024 * 1024  # 128 MB

    collectives = [
        ("all_gather", ALL_GATHER),
        ("all_reduce", ALL_REDUCE),
        ("all_to_all", ALL_TO_ALL),
        ("reduce_scatter", REDUCE_SCATTER),
    ]

    # Example: overlapping groups for real congestion
    groups = {
        "1": [0, 1, 2, 3, 4, 5, 6, 7],
    }

    base_dir = os.path.join(os.path.dirname(__file__), "..", "workload")
    for coll_name, coll_type in collectives:
        output_dir = os.path.join(base_dir, f"toy_{coll_name}_one_collective")
        os.makedirs(output_dir, exist_ok=True)
        json_path = os.path.join(output_dir, f"{coll_name}.json")
        generate_comm_group_json(json_path, groups)
        generate_congested_et_files(npus_count, comm_size, groups, output_dir, coll_type, coll_name)

if __name__ == "__main__":
    main()
