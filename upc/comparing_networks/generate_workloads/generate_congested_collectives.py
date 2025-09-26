import os
from chakra.src.third_party.utils.protolib import encodeMessage as encode_message
from chakra.schema.protobuf.et_def_pb2 import (
    Node as ChakraNode,
    GlobalMetadata,
    AttributeProto as ChakraAttr,
    COMM_COLL_NODE,
    ALL_REDUCE,
    ALL_TO_ALL,
    ALL_GATHER,
    REDUCE_SCATTER,
    BoolList,
)

def generate_congested_collective_workloads(npus_count: int, comm_size: int, num_repeats: int) -> None:
    """
    Generates traces with multiple, concurrent collective operations for various collective types
    to create network congestion.
    """
    collectives = [
        ("all_gather", ALL_GATHER),
        ("all_reduce", ALL_REDUCE),
        ("all_to_all", ALL_TO_ALL),
        ("reduce_scatter", REDUCE_SCATTER),
    ]

    for coll_name, coll_type in collectives:
        dir_path = os.path.join(os.path.dirname(__file__), "..", "workload", f"toy_{coll_name}_congested")
        os.makedirs(dir_path, exist_ok=True)

        for npu_id in range(npus_count):
            output_filename = f"{dir_path}/congested_{coll_name}.{npu_id}.et"
            with open(output_filename, "wb") as et:
                # Chakra Metadata
                encode_message(et, GlobalMetadata(version="0.0.4"))

                for i in range(num_repeats):
                    # Create a Chakra Node for each collective operation
                    node = ChakraNode()
                    node.id = i + 1
                    node_name_formatted = "".join(word.capitalize() for word in coll_name.split('_'))
                    node.name = f"{node_name_formatted}_{i+1}"
                    node.type = COMM_COLL_NODE

                    # No data dependencies are added, making the nodes concurrent.

                    # Assign attributes for the collective
                    node.attr.append(ChakraAttr(name="comm_type", int64_val=coll_type))
                    node.attr.append(ChakraAttr(name="comm_size", uint64_val=comm_size))

                    # All NPUs are involved in each collective
                    involved_dims = [True] * npus_count
                    node.attr.append(ChakraAttr(name="involved_dim", bool_list=BoolList(values=involved_dims)))

                    # Store Chakra ET file
                    encode_message(et, node)
        
        print(f"Generated congested {coll_name} traces in {dir_path}")


def main() -> None:
    npus_count = 8
    comm_size = 128 * 1024 * 1024  # 128 MB
    num_repeats = 4  # Number of concurrent collective calls

    generate_congested_collective_workloads(npus_count, comm_size, num_repeats)

if __name__ == "__main__":
    main()