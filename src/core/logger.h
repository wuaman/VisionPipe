#pragma once

#include <memory>
#include <string>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/ostream_sink.h>
#include <spdlog/sinks/basic_file_sink.h>

namespace visionpipe {

/// @brief 日志格式类型
enum class LogFormat {
    Text,  ///< 纯文本格式（默认）
    Json   ///< JSON 结构化格式
};

/// @brief 日志系统初始化器
///
/// 提供全局日志配置，支持 text/json 两种输出格式。
/// 基于 spdlog 实现，所有模块共享同一 logger 实例。
///
/// @example
/// ```cpp
/// // 初始化为 JSON 格式
/// Logger::init(spdlog::level::info, LogFormat::Json);
///
/// // 输出日志
/// VP_LOG_INFO("Pipeline started with id={}", pipeline_id);
/// ```
class Logger {
public:
    /// @brief 初始化全局日志系统
    ///
    /// @param level 日志级别（trace/debug/info/warn/error/critical）
    /// @param format 输出格式（Text 或 Json）
    /// @note 多次调用会重新配置，通常在 main 开头调用一次
    static void init(spdlog::level::level_enum level = spdlog::level::info,
                     LogFormat format = LogFormat::Text);

    /// @brief 使用自定义 sink 初始化（用于测试）
    ///
    /// @param sink 自定义 spdlog sink
    /// @param level 日志级别
    /// @param format 输出格式
    static void init_with_sink(std::shared_ptr<spdlog::sinks::sink> sink,
                               spdlog::level::level_enum level = spdlog::level::info,
                               LogFormat format = LogFormat::Text);

    /// @brief 初始化文件日志
    ///
    /// @param filename 日志文件路径
    /// @param level 日志级别
    /// @param format 输出格式
    static void init_with_file(const std::string& filename,
                               spdlog::level::level_enum level = spdlog::level::info,
                               LogFormat format = LogFormat::Text);

    /// @brief 获取全局 logger 实例
    ///
    /// @return spdlog logger 指针
    /// @note 若未初始化，返回默认控制台 logger
    static std::shared_ptr<spdlog::logger> get();

    /// @brief 关闭日志系统
    static void shutdown();

    /// @brief 设置日志级别
    static void set_level(spdlog::level::level_enum level);

    /// @brief 刷新所有日志
    static void flush();

private:
    static std::shared_ptr<spdlog::logger> logger_;
    static bool initialized_;
};

}  // namespace visionpipe

// ============================================================================
// 日志宏定义
// ============================================================================

#define VP_LOG_TRACE(...)    SPDLOG_LOGGER_CALL(visionpipe::Logger::get(), spdlog::level::trace, __VA_ARGS__)
#define VP_LOG_DEBUG(...)    SPDLOG_LOGGER_CALL(visionpipe::Logger::get(), spdlog::level::debug, __VA_ARGS__)
#define VP_LOG_INFO(...)     SPDLOG_LOGGER_CALL(visionpipe::Logger::get(), spdlog::level::info, __VA_ARGS__)
#define VP_LOG_WARN(...)     SPDLOG_LOGGER_CALL(visionpipe::Logger::get(), spdlog::level::warn, __VA_ARGS__)
#define VP_LOG_ERROR(...)    SPDLOG_LOGGER_CALL(visionpipe::Logger::get(), spdlog::level::err, __VA_ARGS__)
#define VP_LOG_CRITICAL(...) SPDLOG_LOGGER_CALL(visionpipe::Logger::get(), spdlog::level::critical, __VA_ARGS__)
