import os
from chakra.src.third_party.utils.protolib import encodeMessage as encode_message
from chakra.schema.protobuf.et_def_pb2 import (
    Node as ChakraNode,
    GlobalMetadata,
    AttributeProto as ChakraAttr,
    COMM_SEND_NODE,
    COMM_RECV_NODE,
)

def generate_naive_many_to_one_workload(npus_count: int, comm_size: int, num_repeats: int) -> None:
    """
    Simulates a many-to-one pattern where multiple NPUs send to a single receiver (rank 0).
    Each sender sends multiple messages simultaneously to induce congestion.
    """
    dir_path = os.path.join(os.path.dirname(__file__), "..", "workload", "toy_naive_many_to_one")
    os.makedirs(dir_path, exist_ok=True)

    receiver_npu_id = 0

    # Generate trace for the receiver (NPU 0)
    output_filename = f"{dir_path}/many_to_one.{receiver_npu_id}.et"
    with open(output_filename, "wb") as et:
        encode_message(et, GlobalMetadata(version="0.0.4"))
        node_id = 1
        for repeat in range(num_repeats):
            for sender_id in range(1, npus_count):
                comm_tag = (sender_id * num_repeats) + repeat
                node = ChakraNode()
                node.id = node_id
                node.name = f"Recv_from_{sender_id}_repeat_{repeat}"
                node.type = COMM_RECV_NODE
                node.attr.append(ChakraAttr(name="comm_size", uint64_val=comm_size))
                node.attr.append(ChakraAttr(name="comm_src", int64_val=sender_id))
                node.attr.append(ChakraAttr(name="comm_tag", uint64_val=comm_tag))
                encode_message(et, node)
                node_id += 1

    # Generate traces for the senders (NPU 1 to N-1)
    for sender_id in range(1, npus_count):
        output_filename = f"{dir_path}/many_to_one.{sender_id}.et"
        with open(output_filename, "wb") as et:
            encode_message(et, GlobalMetadata(version="0.0.4"))
            for repeat in range(num_repeats):
                comm_tag = (sender_id * num_repeats) + repeat
                node = ChakraNode()
                node.id = repeat + 1
                node.name = f"Send_to_{receiver_npu_id}_repeat_{repeat}"
                node.type = COMM_SEND_NODE
                node.attr.append(ChakraAttr(name="comm_size", uint64_val=comm_size))
                node.attr.append(ChakraAttr(name="comm_dst", int64_val=receiver_npu_id))
                node.attr.append(ChakraAttr(name="comm_tag", uint64_val=comm_tag))
                encode_message(et, node)
    
    print(f"Generated naive many-to-one traces in {dir_path}")

def main() -> None:
    npus_count = 8
    comm_size = 128 * 1024 * 1024  # 128 MB
    num_repeats = 4  # Each sender sends 4 messages for stronger congestion

    generate_naive_many_to_one_workload(npus_count, comm_size, num_repeats)

if __name__ == "__main__":
    main()