#include "astra-sim/common/Logging.hh"

namespace AstraSim {

std::unordered_set<spdlog::sink_ptr> LoggerFactory::default_sinks;
std::shared_ptr<spdlog::logger> LoggerFactory::memory_logger = nullptr;
// std::shared_ptr<spdlog::logger> LoggerFactory::system_logger = nullptr;
// std::shared_ptr<spdlog::logger> LoggerFactory::workload_logger = nullptr;

std::shared_ptr<spdlog::logger> LoggerFactory::get_logger(
    const std::string& logger_name) {
    constexpr bool ENABLE_DEFAULT_SINK_FOR_OTHER_LOGGERS = true;
    auto logger = spdlog::get(logger_name);
    if (logger == nullptr) {
        logger = spdlog::create_async<spdlog::sinks::null_sink_mt>(logger_name);
        logger->set_level(spdlog::level::trace);
        logger->flush_on(spdlog::level::info);
    }
    if constexpr (!ENABLE_DEFAULT_SINK_FOR_OTHER_LOGGERS) {
        return logger;
    }
    auto& logger_sinks = logger->sinks();
    for (auto sink : default_sinks) {
        if (std::find(logger_sinks.begin(), logger_sinks.end(), sink) ==
            logger_sinks.end()) {
            logger_sinks.push_back(sink);
        }
    }
    return logger;
}

std::shared_ptr<spdlog::logger> LoggerFactory::get_memory_logger() {
    return memory_logger;
}

/*std::shared_ptr<spdlog::logger> LoggerFactory::get_system_logger() {
    return system_logger;
}*/

/*std::shared_ptr<spdlog::logger> LoggerFactory::get_workload_logger() {
    return workload_logger;
}*/

void LoggerFactory::init(const std::string& log_config_path) {
    if (log_config_path != "empty") {
        // spdlog_setup::from_file(log_config_path);
        init_default_components(log_config_path);
    } else {
        init_default_components("log/log");
    }
}

void LoggerFactory::shutdown(void) {
    default_sinks.clear();
    spdlog::drop_all();
    spdlog::shutdown();
}

void LoggerFactory::init_default_components(
    const std::string& log_config_path) {
    auto sink_color_console =
        std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    sink_color_console->set_level(spdlog::level::info);
    default_sinks.insert(sink_color_console);

    auto sink_rotate_out =
        std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
            log_config_path + ".log", 1024 * 1024 * 10, 10);
    sink_rotate_out->set_level(spdlog::level::debug);
    default_sinks.insert(sink_rotate_out);

    auto sink_rotate_err =
        std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
            log_config_path + ".err", 1024 * 1024 * 10, 10);
    sink_rotate_err->set_level(spdlog::level::err);
    default_sinks.insert(sink_rotate_err);

    // Initialize memory logger
    auto memory_sink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
        log_config_path + "_memory.csv", 1024 * 1024 * 10 * 10, 10);
    memory_sink->set_level(spdlog::level::info);
    memory_logger = std::make_shared<spdlog::logger>("memory", memory_sink);
    spdlog::register_logger(memory_logger);
    // Set the header
    memory_logger->info(", Sys, Time, Node ID, Node Name, Node Type ,Total "
                        "Memory,  Activation, Gradient, "
                        "Parameter, Optimizer");

    // Initialize system logger
    /*auto system_sink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
        log_config_path + "_system_trace.csv", 1024 * 1024 * 10, 10);
    system_sink->set_level(spdlog::level::info);
    system_logger = std::make_shared<spdlog::logger>("system", system_sink);
    spdlog::register_logger(system_logger);
    // Set the header
    system_logger->info(",action, sys_id, tick, node_id, node_name,
    node_type");*/

    // Initialize workload logger
    /*auto workload_sink =
    std::make_shared<spdlog::sinks::rotating_file_sink_mt>( log_config_path +
    "_workload.csv", 1024 * 1024 * 10, 10);
    workload_sink->set_level(spdlog::level::info);
    workload_logger =
        std::make_shared<spdlog::logger>("workload", workload_sink);
    spdlog::register_logger(workload_logger);*/

    spdlog::init_thread_pool(8192, 1);
    spdlog::set_pattern("[%Y-%m-%dT%T%z] [%L] <%n>: %v");
}

}  // namespace AstraSim
