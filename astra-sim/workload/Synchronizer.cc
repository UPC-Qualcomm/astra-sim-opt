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
    
    if (!instance->logger) {
        instance->logger = LoggerFactory::get_logger("Synchronizer");
    }
    
    return instance;
}

std::string Synchronizer::get_coll_comm_node_identifier(
    std::shared_ptr<Chakra::FeederV3::ETFeederNode> node) {
    // Use string concatenation for node identifier
    std::string node_identifier = std::to_string(node->id()) + "_" + node->pg_name<std::string>();
    return node_identifier;
}

void Synchronizer::sync_coll_comm(
    std::shared_ptr<Chakra::FeederV3::ETFeederNode> node,
    uint64_t sys_id,
    CommunicatorGroup* comm_group) {
    // Get the node identifier
    std::string node_identifier = get_coll_comm_node_identifier(node);
    
    auto [it, inserted] = sync_data.try_emplace(node_identifier);
    
    if (inserted) {
        // First time seeing this node_identifier
        it->second.node = std::move(node);
        it->second.comm_group = comm_group;
        if (comm_group != nullptr) {
            it->second.expected_count = comm_group->involved_NPUs.size();
        } else {
            it->second.expected_count = 0;
        }
        it->second.ready_count = 1;
        it->second.is_ready = (it->second.expected_count > 0 && 
                                it->second.ready_count == static_cast<uint64_t>(it->second.expected_count));
    } else {
        // Increment ready count
        it->second.ready_count++;
        it->second.is_ready = (it->second.expected_count > 0 && 
                                it->second.ready_count == static_cast<uint64_t>(it->second.expected_count));
    }
    
    // Try to issue this specific collective communication if ready
    Workload* workload = sys_workload_map[sys_id];
    if (workload != nullptr) {
        issue_single_coll_comm(node_identifier, workload);
    }
    else{
        logger->error("No workload found for sys_id {}", sys_id);
    }
}

void Synchronizer::issue_single_coll_comm(const std::string& node_identifier, Workload* workload) {
    auto sync_it = sync_data.find(node_identifier);
    if (sync_it == sync_data.end()) {
        return;  // Not in sync_data
    }
    
    const auto& data = sync_it->second;
    std::shared_ptr<Chakra::FeederV3::ETFeederNode> node = data.node;
    CommunicatorGroup* comm_group = data.comm_group;
    
    int involved_NPUs_count = data.expected_count;
    
    const std::vector<int>* involved_npus = nullptr;
    if (comm_group != nullptr) {
        involved_npus = &(comm_group->involved_NPUs);
    }
    
    // Check if ready to issue
    if (can_issue(node_identifier, involved_NPUs_count)) {
        if (comm_group != nullptr && involved_npus != nullptr && !involved_npus->empty()) {
            for (int sys_id : *involved_npus) {
                auto workload_it = sys_workload_map.find(sys_id);
                if (workload_it != sys_workload_map.end()) {
                    workload_it->second->dispatch_coll_comm(node);
                }
            }
        } else {
            // Dispatch to all NPUs
            for (auto& sys_workload_pair : sys_workload_map) {
                sys_workload_pair.second->dispatch_coll_comm(node);
            }
        }
        sync_data.erase(node_identifier);
    }
}

void Synchronizer::issue_coll_comm(Workload *workload) {
    std::vector<std::string> to_remove;
    to_remove.reserve(sync_data.size());
    
    for (const auto& [node_identifier, data] : sync_data) {
        if (!data.is_ready) {
            continue;
        }
        
        std::shared_ptr<Chakra::FeederV3::ETFeederNode> node = data.node;
        CommunicatorGroup* comm_group = data.comm_group;
        

        int involved_NPUs_count = data.expected_count;
        const std::vector<int>* involved_npus = nullptr;
        if (comm_group != nullptr) {
            involved_npus = &(comm_group->involved_NPUs);
            involved_NPUs_count = involved_npus->size();
        }
        
        // Check if the identifier can be issued
        if (can_issue(node_identifier, involved_NPUs_count)) {
            // Get the involved NPUs from the communicator group
            if (comm_group != nullptr && involved_npus != nullptr && !involved_npus->empty()) {
                // Iterate over involved NPUs and dispatch for each workload
                for (int sys_id : *involved_npus) {
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
            to_remove.emplace_back(node_identifier);
            //break;
        }
    }
    for (const auto& key : to_remove) {
        sync_data.erase(key);
    }
}

bool Synchronizer::can_issue(const std::string& node_identifier, int involved_NPUs_count) noexcept {
    auto sync_it = sync_data.find(node_identifier);
    if (sync_it == sync_data.end()) {
        return false;
    }
    
    const auto& data = sync_it->second;
    uint64_t ready_count = data.ready_count;
    std::shared_ptr<Chakra::FeederV3::ETFeederNode> node = data.node;
    
    bool is_all_involved = false;
    if (involved_NPUs_count == 0) {
        // If number of involved NPUs is null, compare with total NPUs in the system
        is_all_involved = true;
        involved_NPUs_count = sys_workload_map.size();
    }
    
    // Check if ready count equals involved NPUs count
    if (ready_count != involved_NPUs_count) {
        if (ready_count > involved_NPUs_count) {
            logger->error(
                "Node identifier {} has {} ready nodes but expected {} involved NPUs",
                node_identifier,
                ready_count,
                involved_NPUs_count);
        }
        return false;
    }

    // Check if all hardware resources are available
    if (is_all_involved) {
        for (const auto& sys_workload_pair : sys_workload_map) {
            Workload* workload = sys_workload_pair.second;
            if (workload != nullptr && workload->hw_resource != nullptr) {
                if (!workload->hw_resource->is_available(node)) {
                    return false;
                }
            }
        }
    } else {
        // Check only involved NPUs from communicator group
        CommunicatorGroup* comm_group = data.comm_group;
        if (comm_group != nullptr) {
            const auto& involved_npus = comm_group->involved_NPUs;
            for (int sys_id : involved_npus) {
                auto workload_it = sys_workload_map.find(sys_id);
                if (workload_it != sys_workload_map.end()) {
                    Workload* workload = workload_it->second;
                    if (workload != nullptr && workload->hw_resource != nullptr) {
                        if (!workload->hw_resource->is_available(node)) {
                            return false;
                        }
                    }
                }
            }
        }
    }
    
    return true;
}

Synchronizer::~Synchronizer() {
    auto logger = LoggerFactory::get_logger("Synchronizer");
    
    if (!sync_data.empty()) {
        logger->critical("Synchronizer destroyed with {} pending collective communication(s)", 
                        sync_data.size());
        
        for (const auto& [node_identifier, data] : sync_data) {
            std::ostringstream ss;
            ss << "Pending node identifier: " << node_identifier
               << " | Ready count: " << data.ready_count
               << " | Expected count: " << data.expected_count;
            
            // Get node details if available
            if (data.node != nullptr) {
                ss << " | Node ID: " << data.node->id()
                   << " | PG name: " << data.node->template pg_name<std::string>();
            }
            
            // Get communicator group details if available
            if (data.comm_group != nullptr) {
                ss << " | Expected NPUs: " << data.comm_group->involved_NPUs.size()
                   << " (";
                for (size_t i = 0; i < data.comm_group->involved_NPUs.size(); ++i) {
                    ss << data.comm_group->involved_NPUs[i];
                    if (i < data.comm_group->involved_NPUs.size() - 1) ss << ", ";
                }
                ss << ")";
            }
            
            logger->critical(ss.str());
        }
    }
}
