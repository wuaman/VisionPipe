#include <gtest/gtest.h>
#include <sstream>
#include <thread>
#include <chrono>
#include "core/logger.h"
#include <nlohmann/json.hpp>

using json = nlohmann::json;
using namespace visionpipe;

class LoggerTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 每个测试前重置日志系统
        Logger::shutdown();
    }

    void TearDown() override {
        Logger::shutdown();
    }
};

TEST_F(LoggerTest, DefaultInitialization) {
    // 默认初始化应该成功
    EXPECT_NO_THROW(Logger::init());

    auto logger = Logger::get();
    EXPECT_NE(logger, nullptr);
    EXPECT_EQ(logger->name(), "visionpipe");
}

TEST_F(LoggerTest, TextFormatOutput) {
    std::stringstream ss;
    auto sink = std::make_shared<spdlog::sinks::ostream_sink_mt>(ss);

    Logger::init_with_sink(sink, spdlog::level::info, LogFormat::Text);

    VP_LOG_INFO("test message");

    std::string output = ss.str();
    // 文本格式应包含时间戳、级别和消息
    EXPECT_TRUE(output.find("[info]") != std::string::npos ||
                output.find("[INFO]") != std::string::npos);
    EXPECT_TRUE(output.find("test message") != std::string::npos);
}

TEST_F(LoggerTest, JsonFormatParsable) {
    std::stringstream ss;
    auto sink = std::make_shared<spdlog::sinks::ostream_sink_mt>(ss);

    Logger::init_with_sink(sink, spdlog::level::info, LogFormat::Json);

    VP_LOG_INFO("test message {}", 42);

    std::string output = ss.str();

    // JSON 应该可以解析
    EXPECT_NO_THROW({
        auto j = json::parse(output);
        EXPECT_EQ(j["level"], "info");
        EXPECT_TRUE(j["message"].get<std::string>().find("test message 42") != std::string::npos);
        EXPECT_TRUE(j.contains("timestamp"));
        EXPECT_EQ(j["logger"], "visionpipe");
    });
}

TEST_F(LoggerTest, JsonFormatSpecialCharacters) {
    std::stringstream ss;
    auto sink = std::make_shared<spdlog::sinks::ostream_sink_mt>(ss);

    Logger::init_with_sink(sink, spdlog::level::info, LogFormat::Json);

    VP_LOG_INFO("message with \"quotes\" and \\backslash\\");

    std::string output = ss.str();

    EXPECT_NO_THROW({
        auto j = json::parse(output);
        // JSON 应该正确转义特殊字符
        EXPECT_TRUE(j["message"].get<std::string>().find("quotes") != std::string::npos);
    });
}

TEST_F(LoggerTest, LogLevelFiltering) {
    std::stringstream ss;
    auto sink = std::make_shared<spdlog::sinks::ostream_sink_mt>(ss);

    // 设置为 WARN 级别，INFO 应该被过滤
    Logger::init_with_sink(sink, spdlog::level::warn, LogFormat::Text);

    VP_LOG_INFO("this should be filtered");
    VP_LOG_WARN("this should appear");

    std::string output = ss.str();
    EXPECT_TRUE(output.find("filtered") == std::string::npos);
    EXPECT_TRUE(output.find("appear") != std::string::npos);
}

TEST_F(LoggerTest, MultipleLogLevels) {
    std::stringstream ss;
    auto sink = std::make_shared<spdlog::sinks::ostream_sink_mt>(ss);

    Logger::init_with_sink(sink, spdlog::level::trace, LogFormat::Json);

    VP_LOG_TRACE("trace msg");
    VP_LOG_DEBUG("debug msg");
    VP_LOG_INFO("info msg");
    VP_LOG_WARN("warn msg");
    VP_LOG_ERROR("error msg");

    std::string output = ss.str();
    std::istringstream iss(output);
    std::string line;
    int count = 0;

    while (std::getline(iss, line)) {
        if (line.empty()) continue;
        auto j = json::parse(line);
        std::string level = j["level"];
        EXPECT_TRUE(level == "trace" || level == "debug" || level == "info" ||
                    level == "warn" || level == "error");
        count++;
    }

    EXPECT_EQ(count, 5);
}

TEST_F(LoggerTest, DynamicLevelChange) {
    std::stringstream ss;
    auto sink = std::make_shared<spdlog::sinks::ostream_sink_mt>(ss);

    Logger::init_with_sink(sink, spdlog::level::info, LogFormat::Text);

    VP_LOG_DEBUG("debug should not appear");
    EXPECT_TRUE(ss.str().empty());

    // 动态修改级别
    Logger::set_level(spdlog::level::debug);

    VP_LOG_DEBUG("debug should now appear");
    EXPECT_TRUE(ss.str().find("appear") != std::string::npos);
}

TEST_F(LoggerTest, ThreadSafety) {
    std::stringstream ss;
    auto sink = std::make_shared<spdlog::sinks::ostream_sink_mt>(ss);

    Logger::init_with_sink(sink, spdlog::level::info, LogFormat::Json);

    // 多线程并发写日志
    const int num_threads = 4;
    const int msgs_per_thread = 100;
    std::vector<std::thread> threads;

    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([t]() {
            for (int i = 0; i < msgs_per_thread; ++i) {
                VP_LOG_INFO("thread {} message {}", t, i);
            }
        });
    }

    for (auto& th : threads) {
        th.join();
    }

    // 验证所有消息都被写入
    std::string output = ss.str();
    std::istringstream iss(output);
    std::string line;
    int count = 0;

    while (std::getline(iss, line)) {
        if (line.empty()) continue;
        EXPECT_NO_THROW(json::parse(line));
        count++;
    }

    EXPECT_EQ(count, num_threads * msgs_per_thread);
}

TEST_F(LoggerTest, LoggerShutdown) {
    Logger::init(spdlog::level::info, LogFormat::Text);

    EXPECT_NO_THROW(Logger::shutdown());

    // 关闭后 logger_ 被置空，get() 返回默认 logger（如果存在）
    // 注意：spdlog::shutdown() 会清空所有 logger，包括默认 logger
    // 所以这里只是验证 shutdown 不抛异常
}
