/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "astra-sim/workload/CollectiveOrderEnforcer.hh"
#include "astra-sim/workload/Workload.hh"
#include "astra-sim/system/CommunicatorGroup.hh"
#include "astra-sim/system/Sys.hh"
#include "astra-sim/common/Logging.hh"

#include <algorithm>
#include <queue>

using namespace AstraSim;
using namespace Chakra::FeederV3;

// Initialize static members
std::unordered_map<int, Workload*> CollectiveOrderEnforcer::sys_workload_map;
std::unordered_map<int, uint64_t> CollectiveOrderEnforcer::last_comp_node;
std::unordered_map<int, uint64_t> CollectiveOrderEnforcer::last_comm_node;

CollectiveOrderEnforcer::CollectiveOrderEnforcer(Workload* workload)
    : workload(workload) {
  // Register this workload in the static map
  sys_workload_map[workload->sys->id] = workload;
}

void CollectiveOrderEnforcer::update_last_issued(
    int npu_id,
    uint64_t node_id,
    ChakraProtoMsg::NodeType node_type) {
  
  if (node_type == ChakraProtoMsg::NodeType::COMP_NODE) {
    last_comp_node[npu_id] = node_id;
  } else {
    // Track collective and send, but NOT receive (as user specified)
    last_comm_node[npu_id] = node_id;
  }
}

std::string CollectiveOrderEnforcer::get_pg_name(
    std::shared_ptr<ETFeederNode> node,
    CommunicatorGroup* comm_group) {
  // First try to get pg_name from the node
  std::string pg_name = node->pg_name<std::string>("0");
  
  // If empty or "0", use default
  if (pg_name.empty() || pg_name == "0") {
    return "default_all";
  }
  
  return pg_name;
}

std::vector<int> CollectiveOrderEnforcer::get_involved_npus(
    CommunicatorGroup* comm_group) {
  if (comm_group != nullptr) {
    return comm_group->involved_NPUs;
  }
  
  // Default: all NPUs in the system
  std::vector<int> all_npus;
  for (const auto& pair : sys_workload_map) {
    all_npus.push_back(pair.first);
  }
  return all_npus;
}

uint64_t CollectiveOrderEnforcer::find_next_collective(
    Workload* target_workload,
    uint64_t current_node_id) {
  auto& dep_resolver = target_workload->et_feeder->getDependancyResolver();
  auto& enabled_layer = dep_resolver.get_enabled_dependancy();
  
  // BFS to find next collective communication node
  std::queue<uint64_t> to_visit;
  std::unordered_set<uint64_t> visited;
  
  // Start from children of current node
  try {
    const auto& children = enabled_layer.get_children(current_node_id);
    for (const auto& child_id : children) {
      to_visit.push(child_id);
    }
  } catch (const std::exception& e) {
    // Current node not in graph yet
    return 0;
  }
  
  while (!to_visit.empty()) {
    uint64_t node_id = to_visit.front();
    to_visit.pop();
    
    if (visited.count(node_id)) continue;
    visited.insert(node_id);
    
    // Try to lookup this node
    try {
      auto node = target_workload->et_feeder->lookupNode(node_id);
      
      // Check if it's a collective communication node
      if (node->type() == ChakraProtoMsg::NodeType::COMM_COLL_NODE) {
        return node_id;  // Found next collective!
      }
      
      // Otherwise, continue searching through its children
      const auto& children = enabled_layer.get_children(node_id);
      for (const auto& child_id : children) {
        to_visit.push(child_id);
      }
    } catch (const std::exception& e) {
      // Node not accessible, skip it
      continue;
    }
  }
  
  return 0;  // No next collective found
}

bool CollectiveOrderEnforcer::inject_dependency(
    Workload* target_workload,
    uint64_t child_node_id,
    uint64_t parent_node_id,
    int target_npu_id,
    int source_npu_id) {
  
  auto logger = LoggerFactory::get_logger("workload");
  
  if (child_node_id == 0) {
    return false;  // No valid child node
  }
  
  auto& dep_resolver = target_workload->et_feeder->getDependancyResolver();
  auto& enabled_layer = dep_resolver.get_enabled_dependancy_mut();
  auto& data_layer = dep_resolver.get_data_dependancy_mut();
  
  try {
    // Check if dependency already exists
    const auto& parents = enabled_layer.get_parents(child_node_id);
    if (parents.find(parent_node_id) != parents.end()) {
      return false;  // Dependency already exists
    }
    
    // Add dependency to both layers
    std::unordered_set<uint64_t> children = {child_node_id};
    data_layer.add_node_children(parent_node_id, children);
    enabled_layer.add_node_children(parent_node_id, children);
    
    logger->debug(
        "[ORDER_ENFORCE] Injected: npu={} node={} depends on npu={} node={}",
        target_npu_id, child_node_id, source_npu_id, parent_node_id);
    
    return true;
  } catch (const std::exception& e) {
    logger->warn(
        "[ORDER_ENFORCE] Failed to inject dependency for npu={}: {}",
        target_npu_id, e.what());
    return false;
  }
}

void CollectiveOrderEnforcer::enforce_ordering(
    std::shared_ptr<ETFeederNode> node,
    int npu_id,
    CommunicatorGroup* comm_group) {
  
  auto logger = LoggerFactory::get_logger("workload");
  
  // Get pg_name (unique per group)
  std::string pg_name = get_pg_name(node, comm_group);
  
  // Get involved NPUs (cache if not present)
  if (group_npus_cache.find(pg_name) == group_npus_cache.end()) {
    group_npus_cache[pg_name] = get_involved_npus(comm_group);
  }
  const auto& involved_npus = group_npus_cache[pg_name];
  
  uint64_t current_node_id = node->id();
  
  // Check if this is the first NPU to reach this collective
  auto it = last_collective_per_group.find(pg_name);
  
  if (it == last_collective_per_group.end()) {
    // This is the FIRST NPU to reach this collective
          
    // Inject dependencies to OTHER NPUs: their next_coll depends on their current_coll
    for (int other_npu_id : involved_npus) {
      if (other_npu_id == npu_id) continue;  // Skip self
      
      auto workload_it = sys_workload_map.find(other_npu_id);
      if (workload_it == sys_workload_map.end()) continue;
      
      Workload* other_workload = workload_it->second;
      
      // Find next collectives starting from the other NPU's last issued nodes
      uint64_t next_coll_after_comp = 0;
      uint64_t next_coll_after_comm = 0;
      
      bool start_injection = false;
      auto comp_it = last_comp_node.find(other_npu_id);
      if (comp_it != last_comp_node.end()) {
        next_coll_after_comp = find_next_collective(other_workload, comp_it->second);
        start_injection &= true;
      }
      
      auto comm_it = last_comm_node.find(other_npu_id);
      if (comm_it != last_comm_node.end()) {
        next_coll_after_comm = find_next_collective(other_workload, comm_it->second);
        start_injection &= true;
      }
      
      // Inject dependencies for both possible next collectives
      if (start_injection) {
        inject_dependency(other_workload, next_coll_after_comp, current_node_id,
                        other_npu_id, npu_id);
        inject_dependency(other_workload, next_coll_after_comm, current_node_id,
                        other_npu_id, npu_id);
      }
    }
  }
  
  // Update last collective for this group
  last_collective_per_group[pg_name] = current_node_id;
}
