import os
from chakra.src.third_party.utils.protolib import encodeMessage as encode_message
from chakra.schema.protobuf.et_def_pb2 import (
    Node as ChakraNode,
    GlobalMetadata,
    AttributeProto as ChakraAttr,
    COMM_SEND_NODE,
    COMM_RECV_NODE
)

def generate_naive_all_to_all_workload(npus_count: int, comm_size: int) -> None:
    """
    Generates a trace for each NPU that performs a naive all-to-all,
    issuing concurrent point-to-point sends and recvs.
    This bypasses collective algorithms to create direct network traffic.
    """
    dir_path = os.path.join(os.path.dirname(__file__), "..", "workload", "toy_naive_all_to_all")
    os.makedirs(dir_path, exist_ok=True)

    for npu_id in range(npus_count):
        output_filename = f"{dir_path}/naive_all_to_all.{npu_id}.et"
        with open(output_filename, "wb") as et:
            encode_message(et, GlobalMetadata(version="0.0.4"))
            node_id_counter = 1

            # Each NPU sends to all other NPUs and receives from all other NPUs
            for peer_npu_id in range(npus_count):
                if npu_id == peer_npu_id:
                    continue

                # SEND node to peer
                send_node = ChakraNode()
                send_node.id = node_id_counter
                send_node.name = f"SEND_to_{peer_npu_id}"
                send_node.type = COMM_SEND_NODE
                send_node.attr.append(ChakraAttr(name="comm_tag", int64_val=npu_id)) # Tag by source
                send_node.attr.append(ChakraAttr(name="comm_size", uint64_val=comm_size))
                send_node.attr.append(ChakraAttr(name="comm_dst", int64_val=peer_npu_id))
                encode_message(et, send_node)
                node_id_counter += 1

                # RECV node from peer
                recv_node = ChakraNode()
                recv_node.id = node_id_counter
                recv_node.name = f"RECV_from_{peer_npu_id}"
                recv_node.type = COMM_RECV_NODE
                recv_node.attr.append(ChakraAttr(name="comm_tag", int64_val=peer_npu_id)) # Tag by source
                recv_node.attr.append(ChakraAttr(name="comm_size", uint64_val=comm_size))
                recv_node.attr.append(ChakraAttr(name="comm_src", int64_val=peer_npu_id))
                encode_message(et, recv_node)
                node_id_counter += 1
    
    print(f"Generated naive All-to-All traces in {dir_path}")

def main() -> None:
    npus_count = 8
    comm_size = 128 * 1024 * 1024  # 128 MB
    generate_naive_all_to_all_workload(npus_count, comm_size)

if __name__ == "__main__":
    main()