#ifndef __COMMON_LOGGING_HH__
#define __COMMON_LOGGING_HH__

#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/spdlog.h"
#include "spdlog_setup/conf.h"
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace AstraSim {

class LoggerFactory {
  public:
    LoggerFactory() = delete;
    static std::shared_ptr<spdlog::logger> get_logger(
        const std::string& logger_name);
    static std::shared_ptr<spdlog::logger> get_memory_logger();
    static std::shared_ptr<spdlog::logger> get_trace_logger();
    static std::shared_ptr<spdlog::logger> get_roofline_logger();
    //static std::shared_ptr<spdlog::logger> get_system_logger();
    //static std::shared_ptr<spdlog::logger> get_workload_logger();
    static void init(const std::string& log_conf_path = "empty");
    static void shutdown(void);

  private:
    static void init_default_components(const std::string& log_config_path);
    static std::unordered_set<spdlog::sink_ptr> default_sinks;
    static std::shared_ptr<spdlog::logger> memory_logger;
    static std::shared_ptr<spdlog::logger> trace_logger;
    static std::shared_ptr<spdlog::logger> roofline_logger;
    //static std::shared_ptr<spdlog::logger> system_logger;
    //static std::shared_ptr<spdlog::logger> workload_logger;
};

}  // namespace AstraSim

#endif
