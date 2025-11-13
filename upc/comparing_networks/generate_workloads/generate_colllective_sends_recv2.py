import os
import json
from chakra.src.third_party.utils.protolib import encodeMessage as encode_message
from chakra.schema.protobuf.et_def_pb2 import (
    Node as ChakraNode,
    GlobalMetadata,
    AttributeProto as ChakraAttr,
    COMP_NODE,
    COMM_COLL_NODE,
    COMM_SEND_NODE,
    COMM_RECV_NODE,
    ALL_REDUCE,
    BoolList
)

def generate_comm_group_json(json_path: str, groups: dict):
    with open(json_path, "w") as f:
        json.dump(groups, f)
    print(f"Generated communicator group config at {json_path}")

def generate_concurrent_allreduce(workload_name: str, npus_count: int, allreduce_groups: list, send_pairs: list) -> None:
    """
    Generates traces with concurrent AllReduce operations and point-to-point sends.
    
    AllReduce operations and Send/Recv CAN run concurrently in AstraSim.
    This will create real network congestion when operations share links.
    
    Args:
        workload_name: Name for the workload, used for the output directory.
        npus_count: Total number of NPUs
        allreduce_groups: List of (group_npus, tensor_size) tuples
                         e.g., ([0, 1, 2], 15000000) means AllReduce on NPUs 0,1,2 with 15MB data
        send_pairs: List of (src_npu, dst_npu, tensor_size) tuples
                   e.g., (5, 1, 15000000) means NPU 5 sends 15MB to NPU 1
    """
    dir_path = os.path.join(os.path.dirname(__file__), "..", "workload", workload_name)
    os.makedirs(dir_path, exist_ok=True)
    json_path = f"{dir_path}/{workload_name}.json"
    groups = {}
    for i, el in enumerate(allreduce_groups):
        groups[str(i+1)] = el[0]
    generate_comm_group_json(json_path, groups)

    # Track which NPUs are involved in which groups
    npu_to_groups = {i: [] for i in range(npus_count)}
    for group_id, (group_npus, _) in enumerate(allreduce_groups):
        for npu in group_npus:
            npu_to_groups[npu].append(group_id)
    
    # Track which NPUs are involved in send/recv operations
    npu_sends = {i: [] for i in range(npus_count)}  # NPU -> list of (dst, size)
    npu_recvs = {i: [] for i in range(npus_count)}  # NPU -> list of (src, size)
    for src, dst, size in send_pairs:
        npu_sends[src].append((dst, size))
        npu_recvs[dst].append((src, size))

    for npu_id in range(npus_count):
        output_filename = f"{dir_path}/{workload_name}.{npu_id}.et"
        with open(output_filename, "wb") as et:
            # Chakra Metadata
            encode_message(et, GlobalMetadata(version="0.0.4"))

            has_operations = npu_to_groups[npu_id] or npu_sends[npu_id] or npu_recvs[npu_id]
            
            # If NPU not involved in any operation, add dummy node
            if not has_operations:
                dummy_node = ChakraNode()
                dummy_node.id = npu_id
                dummy_node.name = "dummy_node"
                dummy_node.type = COMP_NODE
                dummy_node.attr.append(ChakraAttr(name="num_ops", int64_val=10000))
                dummy_node.attr.append(ChakraAttr(name="tensor_size", int64_val=10240))
                encode_message(et, dummy_node)
                continue

            node_id = 0
            
            # Create AllReduce nodes for each group this NPU participates in
            for group_id in npu_to_groups[npu_id]:
                group_npus, tensor_size = allreduce_groups[group_id]
                
                allreduce_node = ChakraNode()
                allreduce_node.id = node_id
                node_id += 1
                allreduce_node.name = f"ALLREDUCE_group{group_id}"
                allreduce_node.type = COMM_COLL_NODE
                
                # AllReduce attributes
                allreduce_node.attr.append(ChakraAttr(name="comm_size", uint64_val=tensor_size))
                allreduce_node.attr.append(ChakraAttr(name="comm_type", int64_val=ALL_REDUCE))
                
                # Involved dimensions - boolean array indicating which NPUs participate
                involved_dims = [i in group_npus for i in range(npus_count)]
                allreduce_node.attr.append(ChakraAttr(
                    name="involved_dim",
                    bool_list=BoolList(values=involved_dims)
                ))
                
                allreduce_node.attr.append(ChakraAttr(name="pg_name", string_val=str(group_id+1)))
                
                encode_message(et, allreduce_node)
            
            # Create SEND nodes for outgoing point-to-point communications
            for dst, size in npu_sends[npu_id]:
                send_node = ChakraNode()
                send_node.id = node_id
                node_id += 1
                send_node.name = f"SEND_to_{dst}"
                send_node.type = COMM_SEND_NODE
                
                send_node.attr.append(ChakraAttr(name="comm_size", uint64_val=size))
                send_node.attr.append(ChakraAttr(name="comm_src", uint64_val=npu_id))
                send_node.attr.append(ChakraAttr(name="comm_dst", uint64_val=dst))
                send_node.attr.append(ChakraAttr(name="comm_tag", uint64_val=0))
                
                encode_message(et, send_node)
            
            # Create RECV nodes for incoming point-to-point communications
            for src, size in npu_recvs[npu_id]:
                recv_node = ChakraNode()
                recv_node.id = node_id
                node_id += 1
                recv_node.name = f"RECV_from_{src}"
                recv_node.type = COMM_RECV_NODE
                
                recv_node.attr.append(ChakraAttr(name="comm_size", uint64_val=size))
                recv_node.attr.append(ChakraAttr(name="comm_src", uint64_val=src))
                recv_node.attr.append(ChakraAttr(name="comm_dst", uint64_val=npu_id))
                recv_node.attr.append(ChakraAttr(name="comm_tag", uint64_val=0))
                
                encode_message(et, recv_node)


def main() -> None:
    npus_count = 16
    
    # Define multiple workloads. Each inner list defines a new workload with its own set of AllReduce groups.
    workloads_allreduce_groups = [
        # Workload 1
        [
            ([0, 7], 15000000),
            ([1, 5], 15000000),
        ],
        # Workload 2
        [
            ([0, 1, 2, 3], 15000000),
            ([4, 5, 6, 7], 15000000),
            ([8, 9, 10, 11], 15000000),
            ([12, 13, 14, 15], 15000000),
        ],
        # Workload 3
        [
            ([0, 15], 15000000),
        ]
    ]
    
    # send_pairs are skipped for now as requested
    send_pairs = []
    
    for i, allreduce_groups in enumerate(workloads_allreduce_groups):
        workload_name = f"workload_{i+1}"
        print(f"--- Generating {workload_name} ---")
        
        generate_concurrent_allreduce(workload_name, npus_count, allreduce_groups, send_pairs)
        
        print(f"Generated concurrent communication workload in 'workload/{workload_name}':")
        print(f"  {len(allreduce_groups)} AllReduce groups")
        for j, (npus, size) in enumerate(allreduce_groups):
            print(f"    Group {j}: NPUs {npus} - {size / 1e6:.2f} MB AllReduce")
        print("-" * (20 + len(workload_name)))
        print()


if __name__ == "__main__":
    main()