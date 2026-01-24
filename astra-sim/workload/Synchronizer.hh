#ifndef SYNCHRONIZER_HH_
#define SYNCHRONIZER_HH_

#include "astra-sim/system/CommunicatorGroup.hh"
#include "extern/graph_frontend/chakra/src/feeder_v3/et_feeder.h"
#include <memory>
#include <stdint.h>
#include <unordered_map>
#include <vector>
#include <string>

namespace AstraSim {

class Sys;
class Workload;

class Synchronizer {
  //public:
  //  struct CollCommRecord {
  //      std::string identifier;
  //      uint64_t sys_id;
  //      uint64_t node_id;
  //      std::string pg_name;
  //      std::string involved_npus;
  //      uint32_t comm_tag;
  //      std::string node_name;
  //      uint64_t comm_type;
  //      uint64_t comm_size;
  //  };
//
  //public:
  public:
    static std::shared_ptr<Synchronizer> get_instance(Workload* workload);
    
    std::string get_coll_comm_node_identifier(
        std::shared_ptr<Chakra::FeederV3::ETFeederNode> node);
    
    void sync_coll_comm(std::shared_ptr<Chakra::FeederV3::ETFeederNode> node,
                         uint64_t rank,
                         CommunicatorGroup* comm_group);
    
    void issue_coll_comm(Workload* workload);
    void issue_single_coll_comm(const std::string& node_identifier, Workload* workload);

    bool can_issue(std::string node_identifier, int involved_NPUs_count);
    
    ~Synchronizer();
    
    std::unordered_map<std::string,
                           std::shared_ptr<Chakra::FeederV3::ETFeederNode>>
            coll_comm_nodes;
    std::unordered_map<std::string, uint64_t> pending_nodes;
    std::unordered_map<std::string, CommunicatorGroup*> coll_comm_groups;
    std::unordered_map<uint64_t, Workload*> sys_workload_map;
    //std::vector<CollCommRecord> coll_comm_records;
};
}  // namespace AstraSim

#endif
