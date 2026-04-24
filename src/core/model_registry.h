#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>

#include "hal/imodel_engine.h"

namespace visionpipe {

std::string sha256_file(const std::string& path);

class ModelRegistry {
public:
    using EngineFactory = std::function<std::shared_ptr<IModelEngine>(const std::string& path)>;

    static ModelRegistry& instance();

    ModelRegistry();
    ~ModelRegistry();

    ModelRegistry(const ModelRegistry&) = delete;
    ModelRegistry& operator=(const ModelRegistry&) = delete;

    std::shared_ptr<IModelEngine> acquire(const std::string& path);
    void release(const std::string& path);

    void set_engine_factory(EngineFactory factory);
    void set_ttl(std::chrono::milliseconds ttl);
    std::chrono::milliseconds ttl() const;

    size_t ref_count(const std::string& path) const;
    bool contains(const std::string& path) const;
    void clear();
    void gc_once();

private:
    struct RegistryEntry {
        std::shared_ptr<IModelEngine> engine;
        size_t ref_count = 0;
        std::chrono::steady_clock::time_point expires_at = std::chrono::steady_clock::time_point::max();
    };

    std::string resolve_key_for_release(const std::string& path) const;
    void gc_loop();

    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::unordered_map<std::string, RegistryEntry> entries_;
    std::unordered_map<std::string, std::string> path_to_key_;
    EngineFactory engine_factory_;
    std::chrono::milliseconds ttl_{std::chrono::minutes(5)};
    std::chrono::milliseconds gc_interval_{std::chrono::milliseconds(10)};
    std::thread gc_thread_;
    std::atomic<bool> stop_gc_{false};
};

}  // namespace visionpipe
