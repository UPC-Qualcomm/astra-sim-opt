/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef __COLLECTIVE_ORDER_ENFORCER_HH__
#define __COLLECTIVE_ORDER_ENFORCER_HH__

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>

#include "extern/graph_frontend/chakra/src/feeder_v3/et_feeder.h"

namespace AstraSim {

class CommunicatorGroup;
class Workload;

/**
 * Lightweight collective ordering enforcer using dynamic dependency injection.
 * Instead of barrier synchronization, enforces same ordering across NPUs by
 * injecting dependencies when first NPU reaches a new collective.
 */
class CollectiveOrderEnforcer {
 public:
  explicit CollectiveOrderEnforcer(Workload* workload);
  ~CollectiveOrderEnforcer() = default;

  /**
   * Enforce ordering for a collective communication node.
   * If this NPU is first to reach this collective in its group,
   * it sets the order for other NPUs by injecting dependencies.
   * 
   * @param node The collective communication node
   * @param npu_id The ID of the NPU issuing this collective
   * @param comm_group The communicator group (nullptr = all NPUs)
   */
  /**
   * Update the last issued node for a given NPU.
   * Called from Workload::issue() after occupying hardware resources.
   * 
   * @param npu_id The NPU ID
   * @param node_id The node ID being issued
   * @param node_type The type of the node
   */
  static void update_last_issued(
      int npu_id,
      uint64_t node_id,
      ChakraProtoMsg::NodeType node_type);

    void enforce_ordering(
      std::shared_ptr<Chakra::FeederV3::ETFeederNode> node,
      int npu_id,
      CommunicatorGroup* comm_group);
 private:
  Workload* workload;

  // Static map: sys_id -> Workload* for cross-NPU access
  static std::unordered_map<int, Workload*> sys_workload_map;

  // Track last issued compute node per NPU: sys_id -> node_id
  static std::unordered_map<int, uint64_t> last_comp_node;

  // Track last issued comm node per NPU: sys_id -> node_id
  static std::unordered_map<int, uint64_t> last_comm_node;

  // Track last collective per pg_name: pg_name -> node_id
  std::unordered_map<std::string, uint64_t> last_collective_per_group;

  // Cache pg_name -> involved NPUs
  std::unordered_map<std::string, std::vector<int>> group_npus_cache;

  // Find next collective communication node in the dependency graph
  uint64_t find_next_collective(
      Workload* target_workload,
      uint64_t current_node_id);

  // Get pg_name from node or comm_group
  std::string get_pg_name(
      std::shared_ptr<Chakra::FeederV3::ETFeederNode> node,
      CommunicatorGroup* comm_group);

  // Get involved NPUs from comm_group or default
  std::vector<int> get_involved_npus(CommunicatorGroup* comm_group);

  // Helper function to inject dependency between nodes
  bool inject_dependency(
      Workload* target_workload,
      uint64_t child_node_id,
      uint64_t parent_node_id,
      int target_npu_id,
      int source_npu_id);
};

}  // namespace AstraSim

#endif /* __COLLECTIVE_ORDER_ENFORCER_HH__ */
