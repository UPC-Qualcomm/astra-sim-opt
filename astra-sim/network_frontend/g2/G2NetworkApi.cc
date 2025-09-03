/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "G2NetworkApi.hh"
#include "astra-sim/common/Logging.hh"
#include "astra-sim/system/Sys.hh"
#include "network.h"
#include <cassert>

using namespace AstraSim;
using namespace AstraSimG2;
using namespace NetworkAnalytical;

Network* G2NetworkApi::network = nullptr;
G2NetworkApi* G2NetworkApi::api_instance = nullptr;


void G2NetworkApi::set_network(Network* network_ptr) noexcept {
    assert(network_ptr != nullptr);
    G2NetworkApi::network = network_ptr;
}

void G2NetworkApi::handle_network_update(void* args) noexcept {

    assert(args != nullptr);
    const auto last_time_calculated = *static_cast<double*>(args);
    delete static_cast<double*>(args);

    if (network->isEarlierThanUpdate(last_time_calculated)) {
        return;
    }

    auto fastest_flows = network->removeMessages(last_time_calculated);


    for (const auto& flow : fastest_flows) {
        auto [tag, src, dst, count, chunk_id, workload_node_id, times, rates] = flow;

        if (AstraNetworkAPI::network_enabled_log && workload_node_id != -1) {
            for (size_t i = 0; i < times.size(); ++i) {
                double time = times[i];
                double rate = rates[i];
                LoggerFactory::get_network_logger()->info(
                    ",update,{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
                    src, dst, src, dst, count, tag, workload_node_id,
                    chunk_id, time, 0, 0, 0, 0, 0,
                    0, rate);
            }
        }

        // create chunk
        auto chunk_arrival_arg = std::make_tuple(tag, src, dst, count, chunk_id);
        auto arg = std::make_unique<decltype(chunk_arrival_arg)>(chunk_arrival_arg);
        const auto arg_ptr = static_cast<void*>(arg.release());
        G2NetworkApi::process_chunk_arrival(arg_ptr);
    }
}

G2NetworkApi::G2NetworkApi(
    const int rank) noexcept
    : CommonNetworkApi(rank) {
    assert(rank >= 0);
    if (rank == 0) {
        G2NetworkApi::api_instance = this;
    }
}

int G2NetworkApi::sim_send(void* const buffer,
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
        G2NetworkApi::chunk_id_generator.create_send_chunk_id(
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
    if (G2NetworkApi::network != nullptr) {
        G2NetworkApi::network->addRoute(tag, src, dst, count, chunk_id, workload_node_id);
    }

    // The scheduling of process_chunk_arrival is now handled by update_network_congestion
    // to account for network congestion.

    // return
    return 0;
}

int G2NetworkApi::update_network_congestion() {
    if (G2NetworkApi::network == nullptr) {
        return 0;
    }

    if (!G2NetworkApi::network->len_network()) {
        return 0;
    }

    const auto current_time = event_queue->get_current_time();
    const auto current_time_seconds = static_cast<double>(current_time) / 1'000'000'000.0;

    double scheduled_time_seconds = G2NetworkApi::network->getNextMessages(current_time_seconds);

    const auto delay = static_cast<double>((scheduled_time_seconds - current_time_seconds) * 1'000'000'000.0); // s to ns
    const auto delta = timespec_t({NS, delay});

    assert(api_instance != nullptr); // Ensure the instance is set
    auto* arg = new double(current_time_seconds);
    api_instance->sim_schedule(delta, G2NetworkApi::handle_network_update, static_cast<void*>(arg));

    return 0;
}


void G2NetworkApi::log_network(std::string str) {

    if (this->enable_network_logger) {
        NetworkLogger::getInstance().write(str);
    }
}

void G2NetworkApi::init_logger(std::string str,
                                              bool enable_network_logger) {
    NetworkLogger::getInstance().init(str, enable_network_logger);
}
