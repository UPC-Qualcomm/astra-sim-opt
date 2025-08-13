/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "congestion_unaware/CongestionUnawareNetworkApi.hh"
#include "astra-sim/common/Logging.hh"
#include "astra-sim/system/Sys.hh"
#include <cassert>

using namespace AstraSim;
using namespace AstraSimAnalyticalCongestionUnaware;
using namespace NetworkAnalytical;
using namespace NetworkAnalyticalCongestionUnaware;

std::shared_ptr<Topology> CongestionUnawareNetworkApi::topology;

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

CongestionUnawareNetworkApi::CongestionUnawareNetworkApi(
    const int rank) noexcept
    : CommonNetworkApi(rank) {
    assert(rank >= 0);
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

    // create chunk
    auto chunk_arrival_arg = std::tuple(tag, src, dst, count, chunk_id);
    auto arg = std::make_unique<decltype(chunk_arrival_arg)>(chunk_arrival_arg);
    const auto arg_ptr = static_cast<void*>(arg.release());

    // create log data
    std::map<std::string, std::string> log_data;
    if (AstraNetworkAPI::network_enabled_log) {
        log_data["tag"] = std::to_string(tag);
    }

    // compute send communication delay (in AstraSim format)
    const auto send_delay_ns = topology->send(src, dst, count, AstraNetworkAPI::network_enabled_log ? &log_data : nullptr);
    const auto send_delay = static_cast<double>(send_delay_ns);
    const auto delta = timespec_t({NS, send_delay});

    if (AstraNetworkAPI::network_enabled_log && workload_node_id != -1) {
        LoggerFactory::get_network_logger()->info(
            ",send,{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}", src, dst, src, dst,
            count, tag, workload_node_id, chunk_id, Sys::boostedTick(),
            log_data["bandwidth"], log_data["dims_count"], log_data["topology"],
            log_data["hops"], log_data["latency"], log_data["delay"]);
    }


    // Log Network Info
    // LogNetwork::getInstance().write(std::to_string(src) + ","+
    // std::to_string(dst) + ",");

    // register chunk arrival event after send communication delay
    sim_schedule(delta, CongestionUnawareNetworkApi::process_chunk_arrival,
                 arg_ptr);

    // return
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
