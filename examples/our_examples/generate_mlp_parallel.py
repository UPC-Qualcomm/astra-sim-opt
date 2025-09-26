import os
import argparse
import chakra.src.generator.generator
from chakra.src.third_party.utils.protolib import encodeMessage as encode_message
from chakra.schema.protobuf.et_def_pb2 import (
    Node as ChakraNode,
    GlobalMetadata,
    AttributeProto as ChakraAttr,
    NodeType,
    ALL_GATHER,
    ALL_REDUCE,
    ALL_TO_ALL,
    BARRIER,
    BROADCAST,
    COMM_COLL_NODE,
    COMM_RECV_NODE,
    COMM_SEND_NODE,
    COMP_NODE,
    MEM_LOAD_NODE,
    MEM_STORE_NODE,
    METADATA_NODE,
    REDUCE_SCATTER,
    BoolList,
    BytesList,
    DoubleList,
    Fixed32List,
    Fixed64List,
    FloatList,
    Int32List,
    Int64List,
    Sfixed32List,
    Sfixed64List,
    Sint32List,
    Sint64List,
    StringList,
    Uint32List,
    Uint64List,
)


NODE_ID = 0


def get_node(node_name: str, node_type: NodeType) -> ChakraNode:
    """Generate a new ChakraNode with a unique ID."""
    global NODE_ID
    node = ChakraNode()
    node.id = NODE_ID
    node.name = node_name
    node.type = node_type
    NODE_ID += 1
    return node


def get_mem_load_node(name: str, tensor_size: int, data_deps: list[int] = []) -> ChakraNode:
    node = get_node(name, MEM_LOAD_NODE)
    node.attr.append(ChakraAttr(name="is_cpu_op", bool_val=False))
    node.attr.append(ChakraAttr(name="tensor_size", uint64_val=tensor_size))
    if data_deps:
        node.data_deps.extend(data_deps)
    
    return node

def get_mem_store_node(name: str, tensor_size: int, data_deps: list[int] = []) -> ChakraNode:
    node = get_node(name, MEM_STORE_NODE)
    node.attr.append(ChakraAttr(name="is_cpu_op", bool_val=False))
    node.attr.append(ChakraAttr(name="tensor_size", uint64_val=tensor_size))
    if data_deps:
        node.data_deps.extend(data_deps)
    
    return node

def get_comp_node(name: str, runtime: int, input_size: int=0, output_size: int=0, data_deps: list[int] = []) -> ChakraNode:
    node = get_node(name, COMP_NODE)
    node.attr.append(ChakraAttr(name="is_cpu_op", bool_val=False))
    node.attr.append(ChakraAttr(name="input_size", uint64_val=input_size))
    node.attr.append(ChakraAttr(name="output_size", uint64_val=input_size))
    node.duration_micros = runtime
    if data_deps:
        node.data_deps.extend(data_deps)
    
    return node


def get_comm_type(comm_type: str) -> int:
    if comm_type == "ALLREDUCE":
        return ALL_REDUCE
    elif comm_type == "ALLTOALL":
        return ALL_TO_ALL
    elif comm_type == "ALLGATHER":
        return ALL_GATHER
    elif comm_type == "REDUCESCATTER":
        return REDUCE_SCATTER
    return 0

def get_comm_coll_node(name: str, comm_type: str, comm_size: int) -> ChakraNode:
    node = get_node(name + '_' + comm_type, COMM_COLL_NODE)
    node.attr.append(ChakraAttr(name="comm_type", int64_val=get_comm_type(comm_type)))
    node.attr.append(ChakraAttr(name="comm_size", int64_val=comm_size))
    return node


# layer_sizes: the tensor size in each layer as an int
def mlp_parallel(num_npus: int, layers_tensor_size: list[int] = [1024, 512, 512], runtime: int = 6) -> None:
    num_layers = len(layers_tensor_size)
    for npu_id in range(num_npus):
        output_filename = f"mlp_L{num_layers}.{npu_id}.et"
        with open(output_filename, "wb") as et:
            attr = [
            ChakraAttr(name="schema", string_val="1.0.2-chakra.0.0.4"),
            ChakraAttr(name="input_file", string_val='Generated'),
            ]
            metadata = GlobalMetadata(attr=attr)
            encode_message(et, metadata)

           
            #For simplicity, we encode one iteration only.
            fwd_comm_node = None
            # forward pass
            for l in range(0, num_layers):
                fwd_comp_node = get_comp_node(f"COMP_NODE_COMP_MLP_L{l}_FWD", runtime=runtime)
                if l > 0:
                    fwd_comp_node.data_deps.append(fwd_comm_node.id)
                fwd_comm_node = get_comm_coll_node(f"COMM_COLL_NODE_COMM_MLP_L{l}_FWD", 'ALLGATHER', layers_tensor_size[l] * 4)
                fwd_comm_node.data_deps.append(fwd_comp_node.id)
                encode_message(et, fwd_comp_node)
                encode_message(et, fwd_comm_node)

            # backward pass
            for l in reversed(range(0, num_layers)):
                bwd_ig_comp_node = get_comp_node(f"COMP_NODE_COMP_MLP_L{l}_BWD_IG", runtime=runtime)
                if l == num_layers-1:
                    bwd_ig_comp_node.data_deps.append(fwd_comm_node.id)
                else:
                    bwd_ig_comp_node.data_deps.extend([bwd_wg_comp_node.id, bwd_ig_comm_node.id])
                
                if l != 0: # It is the input layer now, no need to all reduce as it is the last layer in backward pass
                    bwd_ig_comm_node = get_comm_coll_node(f"COMM_COLL_NODE_COMM_MLP_L{l}_BWD_IG", 'ALLREDUCE', layers_tensor_size[l] * 4)
                    bwd_ig_comm_node.data_deps.append(bwd_ig_comp_node.id)
                    encode_message(et, bwd_ig_comm_node)

                bwd_wg_comp_node = get_comp_node(f"COMP_NODE_COMP_MLP_L{l}_BWD_WG", runtime=runtime)
                bwd_wg_comp_node.data_deps.append(bwd_ig_comp_node.id)
                bwd_wg_comm_node = get_comm_coll_node(f"COMM_COLL_NODE_COMM_MLP_L{l}_BWD_WG", 'ALLREDUCE', layers_tensor_size[l] * 4)
                bwd_wg_comm_node.data_deps.append(bwd_wg_comp_node.id)


                
                encode_message(et, bwd_ig_comp_node)
                encode_message(et, bwd_wg_comp_node)
                encode_message(et, bwd_wg_comm_node)

def main() -> None:
    parser = argparse.ArgumentParser(description="Execution Trace Generator")
    parser.add_argument("--num_npus", type=int, default=64, help="Number of NPUs")
    parser.add_argument("--default_runtime", type=int, default=32291, help="Default runtime of compute nodes")
    parser.add_argument("--layers_tensor_size", type=int, default=1024, help="Default tensor size for each layer")
    parser.add_argument("--num_layers", type=int, default=3, help="Number of layers")
    args = parser.parse_args()

    mlp_parallel(args.num_npus, layers_tensor_size= [args.layers_tensor_size] * args.num_layers, runtime=args.default_runtime)

if __name__ == "__main__":
    main()

