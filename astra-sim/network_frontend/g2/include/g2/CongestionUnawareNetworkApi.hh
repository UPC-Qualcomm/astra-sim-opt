/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#pragma once

#include "common/CommonNetworkApi.hh"
#include "network.h"
#include <astra-network-g2/common/Type.h>
#include <astra-network-g2/g2/Topology.h>
#include <string>
#include <vector>
#include <tuple>

using namespace AstraSim;
using namespace AstraSimAnalytical;
using namespace NetworkAnalytical;
using namespace NetworkAnalyticalCongestionUnaware;

namespace AstraSimAnalyticalCongestionUnaware {

/**
 * CongestionUnawareNetworkApi is a AstraNetworkAPI
 * implemented for congestion_unaware analytical network backend.
 */
class CongestionUnawareNetworkApi final : public CommonNetworkApi {
  public:
    /**
     * Set the topology to be used.
     *
     * @param topology_ptr pointer to the to
     */
    static void set_topology(std::shared_ptr<Topology> topology_ptr) noexcept;

    /**
     * Set the network object to be used.
     *
     * @param network_ptr pointer to the network object
     */
    static void set_network(Network* network_ptr) noexcept;
    
    /**
     * Internal callback to handle network updates.
     *
     * @param args arguments of the callback function
     */

    static void handle_network_update(void* args) noexcept;
    /**
     * Update network congestion and schedule message arrivals.
     * Should be called periodically from the simulation loop.
     */

    static int update_network_congestion();

    /**
     * Constructor.
     *
     * @param rank id of the API
     */
    explicit CongestionUnawareNetworkApi(int rank) noexcept;

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

    void log_network(std::string str);
    void init_logger(std::string str, bool enable_network_logger);

  private:
    /// topology
    static std::shared_ptr<Topology> topology;

    /// network
    static Network* network;

    /// An instance of the API to call non-static methods from static ones.
    static CongestionUnawareNetworkApi* api_instance;
};

}  // namespace AstraSimAnalyticalCongestionUnaware
