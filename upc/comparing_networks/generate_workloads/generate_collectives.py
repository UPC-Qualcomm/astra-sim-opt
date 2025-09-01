import os
from chakra.src.third_party.utils.protolib import encodeMessage as encode_message
from chakra.schema.protobuf.et_def_pb2 import (
    Node as ChakraNode,
    BoolList,
    GlobalMetadata,
    AttributeProto as ChakraAttr,
    COMM_COLL_NODE,
    ALL_REDUCE,
    ALL_GATHER,
    ALL_TO_ALL,
    REDUCE_SCATTER,
)


def generate_single_collective_workloads(npus_count: int, comm_size: int) -> None:
    """
    Generates a trace for each NPU with a single collective operation.
    These are simple, non-congested workloads for basic testing.
    """
    collectives = [
        ("all_gather", ALL_GATHER),
        ("all_reduce", ALL_REDUCE),
        ("all_to_all", ALL_TO_ALL),
        ("reduce_scatter", REDUCE_SCATTER),
    ]

    for coll_name, coll_type in collectives:
        dir_path = os.path.join(os.path.dirname(__file__), "..", "workload", f"toy_{coll_name}_single")
        os.makedirs(dir_path, exist_ok=True)

        for npu_id in range(npus_count):
            output_filename = f"{dir_path}/{coll_name}.{npu_id}.et"
            with open(output_filename, "wb") as et:
                # Chakra Metadata
                encode_message(et, GlobalMetadata(version="0.0.4"))

                # Create a single Chakra Node
                node = ChakraNode()
                node.id = 1
                node.name = coll_name.replace("_", " ").title().replace(" ", "")
                node.type = COMM_COLL_NODE

                # Assign attributes
                node.attr.append(ChakraAttr(name="comm_type", int64_val=coll_type))
                node.attr.append(ChakraAttr(name="comm_size", uint64_val=comm_size))

                # All NPUs are involved in the collective
                involved_dims = [True] * npus_count
                node.attr.append(ChakraAttr(name="involved_dim", bool_list=BoolList(values=involved_dims)))

                # Store Chakra ET file
                encode_message(et, node)
        print(f"Generated single {coll_name} traces in {dir_path}")


def main() -> None:
    npus_count = 8
    comm_size = 128 * 1024 * 1024  # 128 MB

    generate_single_collective_workloads(npus_count, comm_size)


if __name__ == "__main__":
    main()