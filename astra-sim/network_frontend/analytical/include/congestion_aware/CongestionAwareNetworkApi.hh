/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#pragma once

#include "common/CommonNetworkApi.hh"
#include <astra-network-analytical/congestion_aware/Topology.h>

using namespace AstraSim;
using namespace AstraSimAnalytical;
using namespace NetworkAnalytical;
using namespace NetworkAnalyticalCongestionAware;

namespace AstraSimAnalyticalCongestionAware {

class CongestionAwareNetworkApi final : public CommonNetworkApi {
  public:
    /**
     * Set the topology to be used.
     *
     * @param topology_ptr pointer to the topology
     */
    static void set_topology(std::shared_ptr<Topology> topology_ptr) noexcept;

    /**
     * Constructor.
     *
     * @param rank id of the API
     */
    explicit CongestionAwareNetworkApi(int rank) noexcept;

    /**
     * Implement sim_send of AstraNetworkAPI.
     */
    int sim_send(void* buffer,
                 uint64_t count,
                 int type,
                 int dst,
                 int tag,
                 uint64_t workload_node_id,
                 sim_request* request,
                 void (*msg_handler)(void* fun_arg),
                 void* fun_arg) override;

    /**
     * Process the arrival of a chunk.
     *
     * @param args arguments for processing the chunk
     */
    static void process_chunk_arrival(void* args) noexcept;

  private:
    /// topology
    static std::shared_ptr<Topology> topology;
};

}  // namespace AstraSimAnalyticalCongestionAware
