import argparse
import os
import shutil
import sys
from tqdm import tqdm
from collections import OrderedDict

# Ensure we can import from extern
sys.path.append(os.getcwd())

try:
    from extern.graph_frontend.chakra.src.third_party.utils.protolib import decodeMessage, encodeMessage
    from extern.graph_frontend.chakra.schema.protobuf.et_def_pb2 import (
        Node as ChakraNode,
        GlobalMetadata,
        COMM_COLL_NODE,
        COMP_NODE,
        AttributeProto as ChakraAttr
    )
except ImportError:
    print("Error: Could not import Chakra libraries. Make sure you are running this script from the root of the workspace (e.g., /app/astra-sim).")
    sys.exit(1)

# Global state to track unique collectives across multiple workloads
# signature_to_colls[signature] = {
#   "coll_names": [name1, name2, ...],
#   "npu_nodes": {npu_id: node, ...},
#   "et_files": [et_file_names],
#   "comm_group_file": path_to_json,
#   "global_metadata": gm_object
# }
signature_to_colls = OrderedDict()
total_collectives_found = 0

def get_et_files(input_dir):
    files = [f for f in os.listdir(input_dir) if f.endswith(".et")]
    # Sort by NPU ID. Assuming format ending in .<id>.et
    # Example: 4_2_1_2_1.seq_32.batch_32.0.et
    def get_id(fname):
        try:
            # Split by dot, take second to last element
            return int(fname.split('.')[-2])
        except:
            return 0
    files.sort(key=get_id)
    return files

def process_workload(input_dir, output_dir):
    global total_collectives_found
    et_files = get_et_files(input_dir)
    if not et_files:
        print(f"No .et files found in {input_dir}, skipping.")
        return

    print(f"Processing {input_dir}...")
    print(f"Found {len(et_files)} ET files.")

    # Find the json comm group file
    json_files = [f for f in os.listdir(input_dir) if f.endswith(".json")]
    comm_group_file = None
    if json_files:
        comm_group_file = os.path.join(input_dir, json_files[0])
        print(f"Found communicator group file: {comm_group_file}")

    # Store collectives by name to handle non-uniform collectives across NPUs
    # collectives_by_name[coll_name] = {npu_id: node, ...}
    collectives_by_name = OrderedDict()
    global_metadata = None
    
    for npu_id, fname in enumerate(tqdm(et_files, desc="Parsing ET files")):
        fpath = os.path.join(input_dir, fname)
        with open(fpath, "rb") as f:
            # Read GlobalMetadata
            gm = GlobalMetadata()
            msg = decodeMessage(f, gm)
            if msg is not None:
                if global_metadata is None:
                    global_metadata = gm
            
            while True:
                node = ChakraNode()
                msg = decodeMessage(f, node)
                if msg is False:
                    break
                
                if node.type == COMM_COLL_NODE:
                    # Clear dependencies to make it runnable in isolation
                    del node.data_deps[:]
                    
                    if node.name not in collectives_by_name:
                        collectives_by_name[node.name] = {}
                    collectives_by_name[node.name][npu_id] = node

    if not collectives_by_name:
        print("No collectives found in this workload.")
        return

    num_collectives = len(collectives_by_name)
    total_collectives_found += num_collectives
    print(f"Found {num_collectives} unique collectives in this workload.")

    for coll_name, npu_nodes in tqdm(collectives_by_name.items(), desc="Grouping collectives"):
        if not npu_nodes:
            continue

        # Group NPUs by their communication group (pg_name)
        pg_to_npu_groups = {}
        for npu_id, node in npu_nodes.items():
            pg_name = None
            for attr in node.attr:
                if attr.name == "pg_name":
                    pg_name = attr.string_val
                    break
            
            if pg_name is not None:
                if pg_name not in pg_to_npu_groups:
                    pg_to_npu_groups[pg_name] = []
                pg_to_npu_groups[pg_name].append(npu_id)

        if not pg_to_npu_groups:
            continue

        # Create a canonical representation of the communication groups
        # A tuple of sorted tuples of NPU IDs
        comm_groups_tuple = tuple(sorted([tuple(sorted(v)) for v in pg_to_npu_groups.values()]))

        # Get attributes from the first available node
        first_node = next(iter(npu_nodes.values()))
        comm_type = None
        comm_size = None
        
        for attr in first_node.attr:
            if attr.name == "comm_type":
                comm_type = attr.int64_val
            elif attr.name == "comm_size":
                comm_size = attr.int64_val
        
        # The signature is the collection of communication groups, type, and size
        signature = (comm_groups_tuple, comm_type, comm_size)
        
        if signature not in signature_to_colls:
            signature_to_colls[signature] = {
                "coll_names": [],
                "npu_nodes": npu_nodes,
                "et_files": et_files,
                "comm_group_file": comm_group_file,
                "global_metadata": global_metadata
            }
        signature_to_colls[signature]["coll_names"].append(f"{os.path.basename(input_dir)}/{coll_name}")

def write_output_files(output_dir):
    if not signature_to_colls:
        print("No collectives found to write.")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    extracted_count = 0
    duplicates_log_path = os.path.join(output_dir, "duplicate_collectives.log")

    with open(duplicates_log_path, "w") as log_file:
        for signature, data in tqdm(signature_to_colls.items(), desc="Extracting collectives"):
            npu_nodes = data["npu_nodes"]
            coll_names = data["coll_names"]
            et_files = data["et_files"]
            comm_group_file = data["comm_group_file"]
            global_metadata = data["global_metadata"]
            
            # Use the first collective name for the folder, sanitized
            # The name now includes the workload folder for context
            base_name = coll_names[0].replace('/', '_')
            safe_name = "".join([c if c.isalnum() or c in ['_', '-'] else "_" for c in base_name])
            
            # Folder name: index_name
            folder_name = f"{extracted_count:04d}_{safe_name}"
            folder_path = os.path.join(output_dir, folder_name)
            
            os.makedirs(folder_path, exist_ok=True)

            # Write to duplicates log
            log_file.write(f"Workload: {folder_name}\n")
            log_file.write(f"  Signature: {signature}\n")
            log_file.write(f"  Grouped Collective Names ({len(coll_names)}):\n")
            for name in coll_names:
                log_file.write(f"    - {name}\n")
            log_file.write("\n")

            # Copy json file if exists
            if comm_group_file:
                shutil.copy(comm_group_file, folder_path)

            # Write ET files for each NPU
            for npu_id, fname in enumerate(et_files):
                out_fpath = os.path.join(folder_path, fname)
                
                with open(out_fpath, "wb") as f:
                    if global_metadata:
                        encodeMessage(f, global_metadata)
                    
                    if npu_id in npu_nodes:
                        # This NPU has the collective node
                        node = npu_nodes[npu_id]
                        encodeMessage(f, node)
                    else:
                        # This NPU does not participate, create a dummy compute node
                        dummy_node = ChakraNode()
                        dummy_node.id = 1 # A simple ID is fine as it's the only node
                        dummy_node.name = "dummy_comp_node"
                        dummy_node.type = COMP_NODE
                        # Add attributes to make it a valid, zero-op compute node
                        dummy_node.attr.append(ChakraAttr(name="num_ops", int64_val=100000))
                        dummy_node.attr.append(ChakraAttr(name="tensor_size", int64_val=100000))
                        encodeMessage(f, dummy_node)
            
            extracted_count += 1
        
    print(f"Successfully extracted {extracted_count} unique workloads (deduplicated from {total_collectives_found}) to {output_dir}")
    print(f"A log of grouped collectives has been saved to: {duplicates_log_path}")

def create_collectives_log(input_dir, log_path=None):
    if log_path is None:
        log_path = os.path.join(input_dir, "duplicate_collectives.log")
    
    et_files = get_et_files(input_dir)
    if not et_files:
        print(f"No .et files found in {input_dir}")
        return

    print(f"Found {len(et_files)} ET files in {input_dir}")

    # Store collectives by name to handle non-uniform collectives across NPUs
    # collectives_by_name[coll_name] = {npu_id: node, ...}
    collectives_by_name = OrderedDict()
    global_metadata = None
    
    print("Reading ET files and parsing nodes...")
    for npu_id, fname in enumerate(tqdm(et_files, desc="Parsing ET files")):
        fpath = os.path.join(input_dir, fname)
        with open(fpath, "rb") as f:
            # Read GlobalMetadata
            gm = GlobalMetadata()
            msg = decodeMessage(f, gm)
            if msg is not None:
                if global_metadata is None:
                    global_metadata = gm
            
            while True:
                node = ChakraNode()
                msg = decodeMessage(f, node)
                if msg is False:
                    break
                
                if node.type == COMM_COLL_NODE:
                    # Clear dependencies to make it runnable in isolation
                    del node.data_deps[:]
                    
                    if node.name not in collectives_by_name:
                        collectives_by_name[node.name] = {}
                    collectives_by_name[node.name][npu_id] = node

    if not collectives_by_name:
        print("No collectives found in traces.")
        return

    num_collectives = len(collectives_by_name)
    print(f"Found {num_collectives} unique collectives across all NPUs.")

    # Group collectives by signature
    # signature_to_colls[signature] = {
    #   "coll_names": [name1, name2, ...],
    #   "npu_nodes": {npu_id: node, ...}
    # }
    signature_to_colls = OrderedDict()

    print("Grouping collectives by signature...")
    for coll_name, npu_nodes in tqdm(collectives_by_name.items(), desc="Grouping collectives"):
        if not npu_nodes:
            continue

        # Group NPUs by their communication group (pg_name)
        pg_to_npu_groups = {}
        for npu_id, node in npu_nodes.items():
            pg_name = None
            for attr in node.attr:
                if attr.name == "pg_name":
                    pg_name = attr.string_val
                    break
            
            if pg_name is not None:
                if pg_name not in pg_to_npu_groups:
                    pg_to_npu_groups[pg_name] = []
                pg_to_npu_groups[pg_name].append(npu_id)

        if not pg_to_npu_groups:
            continue

        # Create a canonical representation of the communication groups
        # A tuple of sorted tuples of NPU IDs
        comm_groups_tuple = tuple(sorted([tuple(sorted(v)) for v in pg_to_npu_groups.values()]))

        # Get attributes from the first available node
        first_node = next(iter(npu_nodes.values()))
        comm_type = None
        comm_size = None
        
        for attr in first_node.attr:
            if attr.name == "comm_type":
                comm_type = attr.int64_val
            elif attr.name == "comm_size":
                comm_size = attr.int64_val
        
        # The signature is the collection of communication groups, type, and size
        signature = (comm_groups_tuple, comm_type, comm_size)
        
        if signature not in signature_to_colls:
            signature_to_colls[signature] = {
                "coll_names": [],
                "npu_nodes": npu_nodes
            }
        signature_to_colls[signature]["coll_names"].append(coll_name)

    with open(log_path, "w") as log_file:
        for signature, data in tqdm(signature_to_colls.items(), desc="Writing log"):
            coll_names = data["coll_names"]
            
            # Use the first collective name for the folder, sanitized
            safe_name = "".join([c if c.isalnum() or c in ['_', '-'] else "_" for c in coll_names[0]])
            
            # Folder name: index_name (but since no extraction, just for logging)
            folder_name = f"0000_{safe_name}"  # Placeholder, as no actual extraction
            
            # Write to log
            log_file.write(f"Workload: {folder_name}\n")
            log_file.write(f"  Signature: {signature}\n")
            log_file.write(f"  Grouped Collective Names ({len(coll_names)}):\n")
            for name in coll_names:
                log_file.write(f"    - {name}\n")
            log_file.write("\n")
        
    print(f"Log created at: {log_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract individual collective communications from Chakra traces.")
    parser.add_argument("--input_dir", required=True, help="Directory containing ET files, or a directory of directories with ET files.")
    parser.add_argument("--output_dir", required=True, help="Directory to save the extracted collectives")
    args = parser.parse_args()

    # Determine if input_dir is a directory of workloads or a single workload
    has_et_files = any(f.endswith(".et") for f in os.listdir(args.input_dir))
    
    if has_et_files:
        # Single workload directory
        workload_dirs = [args.input_dir]
    else:
        # Directory of workload directories
        workload_dirs = [os.path.join(args.input_dir, d) for d in os.listdir(args.input_dir) if os.path.isdir(os.path.join(args.input_dir, d))]

    if not workload_dirs:
        print(f"No workloads found in {args.input_dir}")
        sys.exit(1)

    for workload_dir in workload_dirs:
        process_workload(workload_dir, args.output_dir)

    write_output_files(args.output_dir)
