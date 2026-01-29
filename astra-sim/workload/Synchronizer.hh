#ifndef SYNCHRONIZER_HH_
#define SYNCHRONIZER_HH_

#include "astra-sim/system/CommunicatorGroup.hh"
#include "extern/graph_frontend/chakra/src/feeder_v3/et_feeder.h"
#include <memory>
#include <stdint.h>
#include <unordered_map>
#include <vector>
#include <string>
#include <spdlog/spdlog.h>
namespace AstraSim {

class Sys;
class Workload;

class Synchronizer {  
  struct NodeSyncData {
      std::shared_ptr<Chakra::FeederV3::ETFeederNode> node;
      uint64_t ready_count;
      CommunicatorGroup* comm_group;
      int expected_count;
      bool is_ready;
  };
  
  public:
    static std::shared_ptr<Synchronizer> get_instance(Workload* workload);
    uint64_t get_coll_comm_node_identifier(
        std::shared_ptr<Chakra::FeederV3::ETFeederNode> node) noexcept;
    
    void sync_coll_comm(std::shared_ptr<Chakra::FeederV3::ETFeederNode> node,
                         uint64_t rank,
                         CommunicatorGroup* comm_group);
    
    void issue_coll_comm(Workload* workload);
    void issue_single_coll_comm(uint64_t node_identifier, Workload* workload);

    bool can_issue(uint64_t node_identifier, int involved_NPUs_count) noexcept;
    
    ~Synchronizer();
    
    std::unordered_map<uint64_t, NodeSyncData> sync_data;
    std::unordered_map<uint64_t, Workload*> sys_workload_map;
    std::shared_ptr<spdlog::logger> logger;
};
}

#endif
