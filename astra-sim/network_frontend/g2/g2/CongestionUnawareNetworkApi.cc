/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "g2/CongestionUnawareNetworkApi.hh"
#include "astra-sim/common/Logging.hh"
#include "astra-sim/system/Sys.hh"
#include "network.h"
#include <cassert>

using namespace AstraSim;
using namespace AstraSimAnalyticalCongestionUnaware;
using namespace NetworkAnalytical;
using namespace NetworkAnalyticalCongestionUnaware;

std::shared_ptr<Topology> CongestionUnawareNetworkApi::topology;
Network* CongestionUnawareNetworkApi::network = nullptr;
CongestionUnawareNetworkApi* CongestionUnawareNetworkApi::api_instance = nullptr;

void CongestionUnawareNetworkApi::set_topology(
    std::shared_ptr<Topology> topology_ptr) noexcept {
    assert(topology_ptr != nullptr);

    // move topology
    CongestionUnawareNetworkApi::topology = std::move(topology_ptr);

    // set topology-related values
    CongestionUnawareNetworkApi::dims_count =
        CongestionUnawareNetworkApi::topology->get_dims_count();
    CongestionUnawareNetworkApi::bandwidth_per_dim =
        CongestionUnawareNetworkApi::topology->get_bandwidth_per_dim();
}

void CongestionUnawareNetworkApi::set_network(Network* network_ptr) noexcept {
    assert(network_ptr != nullptr);
    CongestionUnawareNetworkApi::network = network_ptr;
}

void CongestionUnawareNetworkApi::handle_network_update(void* args) noexcept {

    assert(args != nullptr);
    const auto last_time_calculated = *static_cast<double*>(args);
    delete static_cast<double*>(args);

    if (network->isEarlierThanUpdate(last_time_calculated)) {
        return;
    }

    auto fastest_flows = network->removeMessages(last_time_calculated);


    for (const auto& flow : fastest_flows) {
        auto [tag, src, dst, count, chunk_id] = flow;

        // create chunk
        auto chunk_arrival_arg = std::make_tuple(tag, src, dst, count, chunk_id);
        auto arg = std::make_unique<decltype(chunk_arrival_arg)>(chunk_arrival_arg);
        const auto arg_ptr = static_cast<void*>(arg.release());
        CongestionUnawareNetworkApi::process_chunk_arrival(arg_ptr);
    }
}

CongestionUnawareNetworkApi::CongestionUnawareNetworkApi(
    const int rank) noexcept
    : CommonNetworkApi(rank) {
    assert(rank >= 0);
    if (rank == 0) {
        CongestionUnawareNetworkApi::api_instance = this;
    }
}

int CongestionUnawareNetworkApi::sim_send(void* const buffer,
                                          const uint64_t count,
                                          const int type,
                                          const int dst,
                                          const int tag,
                                          uint64_t workload_node_id,
                                          sim_request* const request,
                                          void (*msg_handler)(void*),
                                          void* const fun_arg) {
    // query chunk id
    const auto src = sim_comm_get_rank();
    const auto chunk_id =
        CongestionUnawareNetworkApi::chunk_id_generator.create_send_chunk_id(
            tag, src, dst, count);

    // search tracker
    const auto entry =
        callback_tracker.search_entry(tag, src, dst, count, chunk_id);
    if (entry.has_value()) {
        // recv operation already issued.
        // add send event handler to the tracker
        entry.value()->register_send_callback(msg_handler, fun_arg);
    } else {
        // recv operation not issued yet
        // create new entry and insert send callback
        auto* const new_entry =
            callback_tracker.create_new_entry(tag, src, dst, count, chunk_id);
        new_entry->register_send_callback(msg_handler, fun_arg);
    }


    // create log data
    std::map<std::string, std::string> log_data;
    if (AstraNetworkAPI::network_enabled_log) {
        log_data["tag"] = std::to_string(tag);
    }

    // add route for network-level simulation
    if (CongestionUnawareNetworkApi::network != nullptr) {
        CongestionUnawareNetworkApi::network->addRoute(tag, src, dst, count, chunk_id);
    }

    // The scheduling of process_chunk_arrival is now handled by update_network_congestion
    // to account for network congestion.

    // return
    return 0;
}

int CongestionUnawareNetworkApi::update_network_congestion() {
    if (CongestionUnawareNetworkApi::network == nullptr) {
        return 0;
    }

    if (!CongestionUnawareNetworkApi::network->len_network()) {
        return 0;
    }

    const auto current_time = event_queue->get_current_time();
    const auto current_time_seconds = static_cast<double>(current_time) / 1'000'000'000.0;

    double scheduled_time_seconds = CongestionUnawareNetworkApi::network->getNextMessages(current_time_seconds);

    const auto delay = static_cast<double>((scheduled_time_seconds - current_time_seconds) * 1'000'000'000.0); // s to ns
    const auto delta = timespec_t({NS, delay});

    assert(api_instance != nullptr); // Ensure the instance is set
    auto* arg = new double(current_time_seconds);
    api_instance->sim_schedule(delta, CongestionUnawareNetworkApi::handle_network_update, static_cast<void*>(arg));

    return 0;
}


void CongestionUnawareNetworkApi::log_network(std::string str) {

    if (this->enable_network_logger) {
        NetworkLogger::getInstance().write(str);
    }
}

void CongestionUnawareNetworkApi::init_logger(std::string str,
                                              bool enable_network_logger) {
    NetworkLogger::getInstance().init(str, enable_network_logger);
}
