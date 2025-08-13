/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "congestion_aware/CongestionAwareNetworkApi.hh"
#include "astra-sim/common/Logging.hh"
#include "astra-sim/system/Sys.hh"
#include <astra-network-analytical/congestion_aware/Chunk.h>
#include <cassert>

using namespace AstraSim;
using namespace AstraSimAnalyticalCongestionAware;
using namespace NetworkAnalytical;
using namespace NetworkAnalyticalCongestionAware;

std::shared_ptr<Topology> CongestionAwareNetworkApi::topology;
struct ChunkArrivalLogArg {
    std::tuple<int, int, int, uint64_t, int>
        tuple;  // tag, src, dest, count, chunk_id
    NetworkAnalyticalCongestionAware::Chunk* chunk_ptr;
    uint64_t workload_node_id;
};

void CongestionAwareNetworkApi::set_topology(
    std::shared_ptr<Topology> topology_ptr) noexcept {
    assert(topology_ptr != nullptr);

    // move topology
    CongestionAwareNetworkApi::topology = std::move(topology_ptr);

    // set topology-related values
    CongestionAwareNetworkApi::dims_count =
        CongestionAwareNetworkApi::topology->get_dims_count();
    CongestionAwareNetworkApi::bandwidth_per_dim =
        CongestionAwareNetworkApi::topology->get_bandwidth_per_dim();
}

CongestionAwareNetworkApi::CongestionAwareNetworkApi(const int rank) noexcept
    : CommonNetworkApi(rank) {
    assert(rank >= 0);
}

int CongestionAwareNetworkApi::sim_send(void* const buffer,
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
        CongestionAwareNetworkApi::chunk_id_generator.create_send_chunk_id(
            tag, src, dst, count);

    // search tracker
    const auto entry =
        callback_tracker.search_entry(tag, src, dst, count, chunk_id);
    if (entry.has_value()) {
        // recv operation already issued.
        // register send callback
        entry.value()->register_send_callback(msg_handler, fun_arg);
    } else {
        // recv operation not issued yet
        // create new entry and insert callback
        auto* const new_entry =
            callback_tracker.create_new_entry(tag, src, dst, count, chunk_id);
        new_entry->register_send_callback(msg_handler, fun_arg);
    }

    auto chunk_arrival_tuple = std::tuple(tag, src, dst, count, chunk_id);
    NetworkAnalyticalCongestionAware::Chunk* chunk_ptr =
        nullptr;  // will set after chunk is constructed

    // Create the chunk first, then set the pointer in the struct
    auto chunk = std::make_unique<Chunk>(
        count, topology->route(src, dst),
        CongestionAwareNetworkApi::process_chunk_arrival, nullptr);

    chunk_ptr = chunk.get();

    auto log_arg = std::make_unique<ChunkArrivalLogArg>(
        ChunkArrivalLogArg{chunk_arrival_tuple, chunk_ptr, workload_node_id});
    const auto arg_ptr = static_cast<void*>(log_arg.release());

    // Now set the callback_arg in the chunk
    chunk->set_callback_arg(arg_ptr);

    // initiate transmission from src -> dst.

    if (AstraNetworkAPI::network_enabled_log && workload_node_id != -1) {
        LoggerFactory::get_network_logger()->info(
            ",issue,{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}", src, dst,
            src, dst, count, tag, workload_node_id, chunk_id,
            Sys::boostedTick(), 0, 0, 0, 0, 0, 0);
    }

    topology->send(std::move(chunk));

    // return
    return 0;
}

void CongestionAwareNetworkApi::process_chunk_arrival(void* args) noexcept {
    assert(args != nullptr);

    auto* log_arg = static_cast<ChunkArrivalLogArg*>(args);
    const auto [tag, src, dest, count, chunk_id] = log_arg->tuple;
    auto* chunk = log_arg->chunk_ptr;
    auto workload_node_id = log_arg->workload_node_id;

    // Get the transmission log entries
    const auto& log = chunk->get_transmission_log();
    if (!log.empty()) {
        for (const auto& entry : log) {
            auto src_log = entry.current_device;
            auto dst_log = entry.next_device;
            auto bandwidth = entry.bandwidth;
            auto latency = entry.latency;
            auto delay = entry.end_time - entry.start_time;

            if (AstraNetworkAPI::network_enabled_log &&
                workload_node_id != -1) {
                LoggerFactory::get_network_logger()->info(
                    ",send_mini,{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
                    src, dest, src_log, dst_log, count, tag, workload_node_id,
                    chunk_id, entry.start_time, bandwidth, 0, 0, 0, latency,
                    delay);
            }
        }
    }

    assert(args != nullptr);

    // search tracker
    auto& tracker = CommonNetworkApi::get_callback_tracker();
    const auto entry = tracker.search_entry(tag, src, dest, count, chunk_id);
    assert(entry.has_value());  // entry must exist

    // if both callbacks are registered, invoke both callbacks
    if (entry.value()->both_callbacks_registered()) {
        entry.value()->invoke_send_handler();
        entry.value()->invoke_recv_handler();

        // remove entry
        tracker.pop_entry(tag, src, dest, count, chunk_id);
    } else {
        // run only send callback, as recv is not ready yet.
        entry.value()->invoke_send_handler();

        // mark the transmission as finished
        // so that recv callback will be invoked immediately
        // when sim_recv() is called
        entry.value()->set_transmission_finished();
    }
    delete log_arg;  // Clean up
}
