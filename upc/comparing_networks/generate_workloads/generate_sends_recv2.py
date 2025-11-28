import os
from chakra.src.third_party.utils.protolib import encodeMessage as encode_message
from chakra.schema.protobuf.et_def_pb2 import (
    Node as ChakraNode,
    GlobalMetadata,
    AttributeProto as ChakraAttr,
    COMM_SEND_NODE,
    COMM_RECV_NODE,
    COMP_NODE
)

def generate_naive_all_to_all(npus_count: int, message_pairs: list, num_concurrent_sends: int) -> None:
    """
    Generates a trace for each NPU that performs sends and receive,
    issuing many concurrent point-to-point sends and recvs.
    This is designed to bypass the system-level collective algorithms
    and create maximum network congestion.
    """
    dir_path = os.path.join(os.path.dirname(__file__), "..", "workload", "sends_recv_easy")
    os.makedirs(dir_path, exist_ok=True)

    involved_npus = set(npu for pair in message_pairs for npu in pair[:2])

    for src_npu in range(npus_count):
        output_filename = f"{dir_path}/sends_recv.{src_npu}.et"
        with open(output_filename, "wb") as et:
            # Chakra Metadata
            encode_message(et, GlobalMetadata(version="0.0.4"))

            node_id_counter = 1

            if src_npu not in involved_npus:
                # Add a dummy node for uninvolved NPUs
                dummy_node = ChakraNode()
                dummy_node.id = node_id_counter
                dummy_node.name = "dummy_node"
                dummy_node.type = COMP_NODE
                dummy_node.attr.append(ChakraAttr(name="num_ops", int64_val=10000))
                dummy_node.attr.append(ChakraAttr(name="tensor_size", int64_val=10240))
                encode_message(et, dummy_node)
                node_id_counter += 1
                continue

            for src, dst, comm_size in message_pairs:
                comm_tag = src * npus_count + dst  # unique tag per pair

                # SEND node (only in src's file)
                if src_npu == src:
                    send_node = ChakraNode()
                    send_node.id = node_id_counter
                    send_node.name = f"SEND_NPU{src}_to_NPU{dst}"
                    send_node.type = COMM_SEND_NODE
                    send_node.attr.append(ChakraAttr(name="comm_tag", int64_val=comm_tag))
                    send_node.attr.append(ChakraAttr(name="comm_size", uint64_val=comm_size))
                    send_node.attr.append(ChakraAttr(name="comm_src", int64_val=src))
                    send_node.attr.append(ChakraAttr(name="comm_dst", int64_val=dst))
                    encode_message(et, send_node)
                    node_id_counter += 1

                # RECV node (only in dst's file)
                if src_npu == dst:
                    recv_node = ChakraNode()
                    recv_node.id = node_id_counter
                    recv_node.name = f"RECV_NPU{dst}_from_NPU{src}"
                    recv_node.type = COMM_RECV_NODE
                    recv_node.attr.append(ChakraAttr(name="comm_tag", int64_val=comm_tag))
                    recv_node.attr.append(ChakraAttr(name="comm_size", uint64_val=comm_size))
                    recv_node.attr.append(ChakraAttr(name="comm_src", int64_val=src))
                    recv_node.attr.append(ChakraAttr(name="comm_dst", int64_val=dst))
                    encode_message(et, recv_node)
                    node_id_counter += 1


def main() -> None:
    npus_count = 16
    # message_pairs = [(0, 8), (4, 10)]
    # message_pairs = [(1, 2), (2, 1), (0, 2)]
    # Define message pairs as (src, dst, comm_size_in_bytes)
    # message_pairs = [
    #     (0, 2, 3000000),
    #     (1, 2, 3000000),
    # ]
    # message_pairs = [
    #     (0, 2, 15000000),
    #     (1, 2, 15000000),
    #     (3, 2, 15000000),
    # ]
    message_pairs = [
        (0, 2, 15000000),
        (1, 2, 15000000),
        (2, 1, 15000000),
    ]
    # Number of concurrent sends to issue.
    # This, combined with active-chunks-per-dimension, controls concurrency.
    num_concurrent_sends = 1

    generate_naive_all_to_all(npus_count, message_pairs, num_concurrent_sends)

if __name__ == "__main__":
    main()