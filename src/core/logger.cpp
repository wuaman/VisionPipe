#include "logger.h"

#include <spdlog/pattern_formatter.h>
#include <spdlog/fmt/fmt.h>
#include <mutex>
#include <ctime>
#include <iomanip>
#include <sstream>

namespace visionpipe {

// 静态成员初始化
std::shared_ptr<spdlog::logger> Logger::logger_ = nullptr;
bool Logger::initialized_ = false;

// JSON 格式化器
class JsonFormatter : public spdlog::custom_flag_formatter {
public:
    void format(const spdlog::details::log_msg& msg,
                const std::tm&,
                spdlog::memory_buf_t& dest) override {
        // 构建简单的 JSON 格式
        // 格式: {"timestamp":"...", "level":"...", "logger":"...", "message":"..."}
        auto time_t_val = std::chrono::system_clock::to_time_t(msg.time);
        std::tm tm_val = *std::localtime(&time_t_val);

        // 获取毫秒部分
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            msg.time.time_since_epoch()) % 1000;

        std::ostringstream oss;
        oss << std::put_time(&tm_val, "%Y-%m-%dT%H:%M:%S")
            << "." << std::setfill('0') << std::setw(3) << ms.count() << "Z";
        std::string timestamp = oss.str();

        std::string level;
        switch (msg.level) {
            case spdlog::level::trace:    level = "trace"; break;
            case spdlog::level::debug:    level = "debug"; break;
            case spdlog::level::info:     level = "info"; break;
            case spdlog::level::warn:     level = "warn"; break;
            case spdlog::level::err:      level = "error"; break;
            case spdlog::level::critical: level = "critical"; break;
            case spdlog::level::off:      level = "off"; break;
        }

        // 获取消息内容
        std::string message(msg.payload.begin(), msg.payload.end());

        // 转义 JSON 字符串中的特殊字符
        auto escape_json = [](const std::string& s) -> std::string {
            std::string result;
            result.reserve(s.size());
            for (char c : s) {
                switch (c) {
                    case '"':  result += "\\\""; break;
                    case '\\': result += "\\\\"; break;
                    case '\b': result += "\\b"; break;
                    case '\f': result += "\\f"; break;
                    case '\n': result += "\\n"; break;
                    case '\r': result += "\\r"; break;
                    case '\t': result += "\\t"; break;
                    default:
                        if (static_cast<unsigned char>(c) < 0x20) {
                            result += fmt::format("\\u{:04x}", static_cast<unsigned char>(c));
                        } else {
                            result += c;
                        }
                }
            }
            return result;
        };

        std::string logger_name(msg.logger_name.begin(), msg.logger_name.end());

        std::string json = fmt::format(
            "{{\"timestamp\":\"{}\",\"level\":\"{}\",\"logger\":\"{}\",\"message\":\"{}\"}}\n",
            timestamp, level, escape_json(logger_name), escape_json(message));

        dest.append(json.data(), json.data() + json.size());
    }

    std::unique_ptr<custom_flag_formatter> clone() const override {
        return std::make_unique<JsonFormatter>();
    }
};

void Logger::init(spdlog::level::level_enum level, LogFormat format) {
    auto sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    init_with_sink(sink, level, format);
}

void Logger::init_with_sink(std::shared_ptr<spdlog::sinks::sink> sink,
                            spdlog::level::level_enum level,
                            LogFormat format) {
    logger_ = std::make_shared<spdlog::logger>("visionpipe", sink);
    logger_->set_level(level);

    // 设置格式
    if (format == LogFormat::Json) {
        auto formatter = std::make_unique<spdlog::pattern_formatter>();
        formatter->add_flag<JsonFormatter>('*');
        formatter->set_pattern("%*");
        logger_->set_formatter(std::move(formatter));
    } else {
        // 文本格式: [时间戳] [级别] [logger] 消息
        logger_->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] [%n] %v");
    }

    // 注册为默认 logger
    spdlog::register_logger(logger_);
    spdlog::set_default_logger(logger_);

    initialized_ = true;
}

void Logger::init_with_file(const std::string& filename,
                            spdlog::level::level_enum level,
                            LogFormat format) {
    auto sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(filename, true);
    init_with_sink(sink, level, format);
}

std::shared_ptr<spdlog::logger> Logger::get() {
    if (!initialized_ || !logger_) {
        // 返回默认 logger（首次调用时 spdlog 会自动创建）
        return spdlog::default_logger();
    }
    return logger_;
}

void Logger::shutdown() {
    if (initialized_) {
        spdlog::shutdown();
        logger_ = nullptr;
        initialized_ = false;
    }
}

void Logger::set_level(spdlog::level::level_enum level) {
    if (logger_) {
        logger_->set_level(level);
    }
}

void Logger::flush() {
    if (logger_) {
        logger_->flush();
    }
}

}  // namespace visionpipe
