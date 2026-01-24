#include "astra-sim/workload/Synchronizer.hh"

#include <sstream>
#include <stdexcept>
#include <fstream>
#include <sys/stat.h>

#include "astra-sim/common/Logging.hh"
#include "astra-sim/system/CommunicatorGroup.hh"
#include "astra-sim/system/Sys.hh"
#include "astra-sim/workload/Workload.hh"

using namespace AstraSim;

std::shared_ptr<Synchronizer> Synchronizer::get_instance(Workload* workload) {
    static std::shared_ptr<Synchronizer> instance = 
        std::make_shared<Synchronizer>();

    instance->sys_workload_map[workload->sys->id] = workload;
    return instance;
}

std::string Synchronizer::get_coll_comm_node_identifier(
    std::shared_ptr<Chakra::FeederV3::ETFeederNode> node) {
    // Each NPU coll_comm is identified by "node_id"_"pg_name"
    return  std::to_string(node->id()) + "_" + node->pg_name<std::string>();
}

void Synchronizer::sync_coll_comm(
    std::shared_ptr<Chakra::FeederV3::ETFeederNode> node,
    uint64_t sys_id,
    CommunicatorGroup* comm_group) {
    auto logger = LoggerFactory::get_logger("Synchronizer");
    // Get the node identifier
    std::string node_identifier = get_coll_comm_node_identifier(node);
    //logger->debug("[SYNC] sys_id={} node_id={} node_identifier={} entering sync",
    //              sys_id, node->id(), node_identifier);
    
    // Store the node in coll_comm_nodes_sync_map for later reference
    if (coll_comm_nodes.find(node_identifier) == coll_comm_nodes.end()) {
        coll_comm_nodes[node_identifier] = node;
    }
    else {
        // Node already exists, log a warning
        auto logger = LoggerFactory::get_logger("Synchronizer");
        //logger->warn("Node identifier {} already exists in synchronizer map coll_comm_nodes", node_identifier);
    }
    
    // Store the communicator group for this node identifier
    if (comm_group != nullptr && coll_comm_groups.find(node_identifier) == coll_comm_groups.end()) {
        coll_comm_groups[node_identifier] = comm_group;
    }
    else if (comm_group != nullptr) {
        // Communicator group already exists, log a warning
        auto logger = LoggerFactory::get_logger("Synchronizer");
        //logger->warn("Node identifier {} already exists in synchronizer map coll_comm_groups", node_identifier);
    }
    
    // Increment count for this node identifier (creates entry with 0 if doesn't exist)
    pending_nodes[node_identifier]++;
    
    //// Record the collective communication data
    //CollCommRecord record;
    //record.identifier = node_identifier;
    //record.sys_id = sys_id;
    //record.node_id = node->id();
    //record.pg_name = node->pg_name<std::string>();
    //
    //// Build involved NPUs string
    //if (comm_group != nullptr) {
    //    std::ostringstream npus_ss;
    //    for (size_t i = 0; i < comm_group->involved_NPUs.size(); ++i) {
    //        npus_ss << comm_group->involved_NPUs[i];
    //        if (i < comm_group->involved_NPUs.size() - 1) npus_ss << ";";
    //    }
    //    record.involved_npus = npus_ss.str();
    //} else {
    //    record.involved_npus = "ALL";
    //}
    //
    //record.comm_tag = node->comm_tag<uint32_t>(0u);
    //record.node_name = node->name();
    //record.comm_type = node->comm_type<uint64_t>();
    //record.comm_size = node->comm_size<uint64_t>();
    //
    //coll_comm_records.push_back(record);
    
    // Try to issue this specific collective communication if ready
    Workload* workload = sys_workload_map[sys_id];
    if (workload != nullptr) {
        issue_single_coll_comm(node_identifier, workload);
    }
    else{
        auto logger = LoggerFactory::get_logger("Synchronizer");
        logger->error("No workload found for sys_id {}", sys_id);
    }
}

void Synchronizer::issue_single_coll_comm(const std::string& node_identifier, Workload* workload) {
    // Check if this specific node can be issued
    auto pending_it = pending_nodes.find(node_identifier);
    if (pending_it == pending_nodes.end()) {
        return;  // Not in pending nodes
    }
    
    // Get the node object
    auto node_it = coll_comm_nodes.find(node_identifier);
    if (node_it == coll_comm_nodes.end()) {
        return;  // Node not found
    }
    auto node = node_it->second;
    
    // Get communicator group
    CommunicatorGroup* comm_group = nullptr;
    auto comm_group_it = coll_comm_groups.find(node_identifier);
    if (comm_group_it != coll_comm_groups.end()) {
        comm_group = comm_group_it->second;
    }
    
    // Get involved NPUs count
    int involved_NPUs_count = 0;
    std::vector<int> involved_npus;
    if (comm_group != nullptr) {
        involved_npus = comm_group->involved_NPUs;
        involved_NPUs_count = involved_npus.size();
    }
    
    // Check if ready to issue
    if (can_issue(node_identifier, involved_NPUs_count)) {
        //auto logger = LoggerFactory::get_logger("Synchronizer");
        //logger->debug("[DISPATCH] node_id={} node_identifier={} ready to dispatch to {} NPUs",
        //              node->id(), node_identifier, involved_NPUs_count);
        // Dispatch to all involved NPUs
        if (comm_group != nullptr && !involved_npus.empty()) {
            for (int sys_id : involved_npus) {
                auto workload_it = sys_workload_map.find(sys_id);
                if (workload_it != sys_workload_map.end()) {
                    //logger->debug("[DISPATCH] Calling dispatch_coll_comm for sys_id={} node_id={}",
                    //              sys_id, node->id());
                    workload_it->second->dispatch_coll_comm(node);
                }
            }
        } else {
            // Dispatch to all NPUs
            for (auto& sys_workload_pair : sys_workload_map) {
                sys_workload_pair.second->dispatch_coll_comm(node);
            }
        }
        
        // Remove from ALL maps immediately after dispatch
        //logger->debug("[CLEANUP] node_id={} node_identifier={} removing from all maps after dispatch",
        //              node->id(), node_identifier);
        pending_nodes.erase(node_identifier);
        coll_comm_nodes.erase(node_identifier);
        coll_comm_groups.erase(node_identifier);
    }
}

void Synchronizer::issue_coll_comm(Workload *workload) {
    // Iterate over the pending_nodes 
    std::vector<std::string> to_remove;
    
    for (auto& pending_node : pending_nodes) {
        // Get the node identifier (first element of the pair)
        std::string node_identifier = pending_node.first;
        uint64_t ready_count = pending_node.second;


        // Get the node object from coll_comm_nodes
        std::shared_ptr<Chakra::FeederV3::ETFeederNode> node = nullptr;
        auto coll_comm_node_it = coll_comm_nodes.find(node_identifier);
        if (coll_comm_node_it != coll_comm_nodes.end()) {
            node = coll_comm_node_it->second;
        }

        // Get communicator group information from coll_comm_groups_map if available
        CommunicatorGroup* comm_group = nullptr;
        
        // Try to find comm_group from stored information
        auto comm_group_it = coll_comm_groups.find(node_identifier);
        if (comm_group_it != coll_comm_groups.end()) {
            comm_group = comm_group_it->second;
        }
        

        // Get all involved NPUs from the communicator group
        int involved_NPUs_count = 0;
        std::vector<int> involved_npus;
        if (comm_group != nullptr) {
            involved_npus = comm_group->involved_NPUs;
            involved_NPUs_count = involved_npus.size();
        }
        
        // Check if the identifier can be issued
        if (can_issue(node_identifier, involved_NPUs_count)) {
            // Get the involved NPUs from the communicator group
            if (comm_group != nullptr && !involved_npus.empty()) {
                // Iterate over involved NPUs and dispatch for each workload
                for (int sys_id : involved_npus) {
                    auto workload_it = sys_workload_map.find(sys_id);
                    if (workload_it != sys_workload_map.end()) {
                        Workload* target_workload = workload_it->second;
                        // Call dispatch_coll_comm on the target workload
                        target_workload->dispatch_coll_comm(node);
                    }
                }
            } else {
                // If no communicator group, dispatch to all NPUs in the system
                for (auto& sys_workload_pair : sys_workload_map) {
                    Workload* target_workload = sys_workload_pair.second;
                    target_workload->dispatch_coll_comm(node);
                }
            }
            
            // Mark for removal after issuing
            to_remove.push_back(pending_node.first);
            break;
        }
    }
    
    // Erase nodes that were issued - remove from ALL maps
    for (auto& key : to_remove) {
        pending_nodes.erase(key);
        coll_comm_nodes.erase(key);
        coll_comm_groups.erase(key);
    }
}

bool Synchronizer::can_issue(std::string node_identifier, int involved_NPUs_count) {
    // Get the node object from coll_comm_nodes_sync_map
    auto logger = LoggerFactory::get_logger("Synchronizer");
    auto node_it = coll_comm_nodes.find(node_identifier);
    if (node_it == coll_comm_nodes.end()) {
        // Node not found in sync map
        logger->error("Node identifier {} not found in synchronizer map coll_comm_nodes", node_identifier);
        return false;
    }
    
    // 1. Check if all nodes are ready
    bool all_nodes_ready = false;
    
    // Get the current ready count for this node_identifier
    auto pending_it = pending_nodes.find(node_identifier);
    if (pending_it == pending_nodes.end()) {
        // No ready nodes found
        logger->error("Node identifier {} not found in synchronizer map pending_nodes", node_identifier);
        return false;
    }
    uint64_t ready_count = pending_it->second;
    
    // 1.1 & 1.2: Compare ready count with involved NPUs count
    bool is_all_involved = false;
    if (involved_NPUs_count == 0) {
        // 1.2: If number of involved NPUs is null, compare with total NPUs in the system
        is_all_involved = true;
        involved_NPUs_count = sys_workload_map.size();
    }
    
    // 1.1: Check if ready count equals involved NPUs count
    if (ready_count == involved_NPUs_count) {
        all_nodes_ready = true;
    } else if (ready_count > involved_NPUs_count) {
        // Error: more nodes ready than expected
        logger->error(
            "Node identifier {} has {} ready nodes but expected {} involved NPUs",
            node_identifier,
            ready_count,
            involved_NPUs_count);
        return false;
    }

    // 2. Check if all hardware resources are available
    bool all_resources_available = true;
    
    // 2.1: For each sys in the involved NPUs, check hardware resource availability
    auto node = node_it->second;
    
    // Determine which NPUs to check
    std::vector<int> npus_to_check;
    if (is_all_involved) {
        // Check all NPUs in the system
        for (const auto& sys_workload_pair : sys_workload_map) {
            npus_to_check.push_back(sys_workload_pair.first);
        }
    } else {
        // Check only involved NPUs from communicator group
        npus_to_check = coll_comm_groups[node_identifier]->involved_NPUs;
    }
    
    // Check hardware resource availability for all relevant NPUs
    for (int sys_id : npus_to_check) {
        auto workload_it = sys_workload_map.find(sys_id);
        if (workload_it != sys_workload_map.end()) {
            Workload* workload = workload_it->second;
            if (workload != nullptr && workload->hw_resource != nullptr) {
                // 2.2: If any hardware resource is not available, break
                if (!workload->hw_resource->is_available(node)) {
                    all_resources_available = false;
                    break;
                }
            }
        }
    }
    // 2.3: all_resources_available flag is now set
    
    // 3. Return true if both flags are true
    return all_nodes_ready && all_resources_available;
}

Synchronizer::~Synchronizer() {
    auto logger = LoggerFactory::get_logger("Synchronizer");
    
    // Write collective communication records to CSV file
    //if (!coll_comm_records.empty()) {
    //    std::string output_dir = "/media/mohammad/extension/experiments/astra-sim/upc";
    //    std::string output_file = output_dir + "/coll_comm.csv";
    //    
    //    // Create directory if it doesn't exist
    //    struct stat st = {0};
    //    if (stat(output_dir.c_str(), &st) == -1) {
    //        mkdir(output_dir.c_str(), 0755);
    //    }
    //    
    //    std::ofstream csv_file(output_file);
    //    if (csv_file.is_open()) {
    //        // Write CSV header
    //        csv_file << "identifier,sys_id,node_id,pg_name,involved_npus,comm_tag,node_name,comm_type,comm_size\n";
    //        
    //        // Write each record
    //        for (const auto& record : coll_comm_records) {
    //            csv_file << record.identifier << ","
    //                    << record.sys_id << ","
    //                    << record.node_id << ","
    //                    << record.pg_name << ","
    //                    << record.involved_npus << ","
    //                    << record.comm_tag << ","
    //                    << record.node_name << ","
    //                    << record.comm_type << ","
    //                    << record.comm_size << "\n";
    //        }
    //        
    //        csv_file.close();
    //        logger->info("Wrote {} collective communication records to {}", 
    //                    coll_comm_records.size(), output_file);
    //    } else {
    //        logger->error("Failed to open file for writing: {}", output_file);
    //    }
    //}
    
    if (!pending_nodes.empty()) {
        logger->critical("Synchronizer destroyed with {} pending collective communication(s)", 
                        pending_nodes.size());
        
        for (const auto& pending_node : pending_nodes) {
            std::ostringstream ss;
            ss << "Pending node identifier: " << pending_node.first
               << " | Ready count: " << pending_node.second;
            
            // Get node details if available
            auto node_it = coll_comm_nodes.find(pending_node.first);
            if (node_it != coll_comm_nodes.end()) {
                auto node = node_it->second;
                ss << " | Node ID: " << node->id()
                   << " | PG name: " << node->pg_name<std::string>();
            }
            
            // Get communicator group details if available
            auto comm_group_it = coll_comm_groups.find(pending_node.first);
            if (comm_group_it != coll_comm_groups.end()) {
                auto comm_group = comm_group_it->second;
                ss << " | Expected NPUs: " << comm_group->involved_NPUs.size()
                   << " (";
                for (size_t i = 0; i < comm_group->involved_NPUs.size(); ++i) {
                    ss << comm_group->involved_NPUs[i];
                    if (i < comm_group->involved_NPUs.size() - 1) ss << ", ";
                }
                ss << ")";
            }
            
            logger->critical(ss.str());
        }
    }
}
