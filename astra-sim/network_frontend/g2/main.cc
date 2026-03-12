/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "astra-sim/common/Logging.hh"
#include "common/CmdLineParser.hh"
#include "G2NetworkApi.hh"
#include <common/EventQueue.h>
#include <common/NetworkParser.h>
#include <remote_memory_backend/analytical/AnalyticalRemoteMemory.hh>
#include "network.h"

using namespace AstraSim;
using namespace Analytical;
using namespace AstraSimAnalytical;
using namespace AstraSimG2;
using namespace NetworkAnalytical;

int main(int argc, char* argv[]) {
    // Parse command line arguments
    auto cmd_line_parser = CmdLineParser(argv[0]);
    cmd_line_parser.parse(argc, argv);

    // Get command line arguments
    const auto workload_configuration =
        cmd_line_parser.get<std::string>("workload-configuration");
    const auto comm_group_configuration =
        cmd_line_parser.get<std::string>("comm-group-configuration");
    const auto system_configuration =
        cmd_line_parser.get<std::string>("system-configuration");
    const auto remote_memory_configuration =
        cmd_line_parser.get<std::string>("remote-memory-configuration");
    const auto network_configuration =
        cmd_line_parser.get<std::string>("network-configuration");
    const auto logging_configuration =
        cmd_line_parser.get<std::string>("logging-configuration");
    
    const auto logging_folder =
        cmd_line_parser.get<std::string>("logging-folder");    
    const auto num_queues_per_dim =
        cmd_line_parser.get<int>("num-queues-per-dim");
    const auto comm_scale = cmd_line_parser.get<double>("comm-scale");
    const auto injection_scale = cmd_line_parser.get<double>("injection-scale");
    const auto rendezvous_protocol =
        cmd_line_parser.get<bool>("rendezvous-protocol");
    // Log Networking information
    const auto network_log = cmd_line_parser.get<std::string>("network-log");

    AstraSim::LoggerFactory::init(logging_configuration, logging_folder);

    // Instantiate event queue
    const auto event_queue = std::make_shared<EventQueue>();

    // Generate topology
    const auto network_parser = NetworkParser(network_configuration);

    // Get topology information
    const auto npus_count_per_dim = network_parser.get_npus_counts_per_dim();
    const auto dims_count = network_parser.get_dims_count();
    const auto topology_file = network_parser.get_topology_file();
    const auto packet_size = network_parser.get_packet_size(); // in bytes
    const auto header_size = network_parser.get_header_size(); // in bytes
    const auto bandwidth_unit = network_parser.get_bandwidth_unit(); // bandwidth unit string
    const auto ecmp_seed = network_parser.get_ecmp_seed(); // ECMP random seed for path selection
    const auto routing_mode = network_parser.get_routing_mode(); // routing mode string
    
    // Get total number of NPUs
    auto npus_count = 1;
    for (const auto& count : npus_count_per_dim) {
        npus_count *= count;
    }

    // Set up Network API
    G2NetworkApi::set_event_queue(event_queue);

    Network net(topology_file, workload_configuration, packet_size, header_size, bandwidth_unit, logging_folder, ecmp_seed, routing_mode);
    G2NetworkApi::set_network(&net);

    // Create ASTRA-sim related resources
    auto network_apis =
        std::vector<std::unique_ptr<G2NetworkApi>>();
    const auto memory_api =
        std::make_unique<AnalyticalRemoteMemory>(remote_memory_configuration);
    auto systems = std::vector<Sys*>();

    auto queues_per_dim = std::vector<int>();
    for (auto i = 0; i < dims_count; i++) {
        queues_per_dim.push_back(num_queues_per_dim);
    }

    for (int i = 0; i < npus_count; i++) {
        // create network and system
        auto network_api = std::make_unique<G2NetworkApi>(i);
        auto* const system =
            new Sys(i, workload_configuration, comm_group_configuration,
                    system_configuration, memory_api.get(), network_api.get(),
                    npus_count_per_dim, queues_per_dim, injection_scale,
                    comm_scale, rendezvous_protocol);

        // push back network and system
        network_apis.push_back(std::move(network_api));
        systems.push_back(system);
    }
    // systems[0]->comm_NI->init_logger(network_log);
    //  Initiate simulation
    for (int i = 0; i < npus_count; i++) {
        systems[i]->comm_NI->init_logger(
            network_log, systems[i]->comm_NI->enable_network_logger);
        if (systems[i]->network_logger_enabled) {
            AstraNetworkAPI::network_enabled_log = true;
        }

        systems[i]->workload->fire();
    }

    // run simulation
    while (!event_queue->finished()) {
        event_queue->proceed();
        G2NetworkApi::update_network_congestion();
    }

    for (auto it : systems) {
        delete it;
    }
    systems.clear();

    // terminate simulation
    AstraSim::LoggerFactory::shutdown();
    return 0;
}
