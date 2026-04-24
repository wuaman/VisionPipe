#include "core/model_registry.h"

#include <array>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

#include "core/error.h"

namespace visionpipe {
namespace {

constexpr std::array<uint32_t, 64> kSha256Constants = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
    0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
    0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
    0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
    0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
    0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
    0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
    0xc67178f2,
};

constexpr std::array<uint32_t, 8> kSha256InitialState = {
    0x6a09e667,
    0xbb67ae85,
    0x3c6ef372,
    0xa54ff53a,
    0x510e527f,
    0x9b05688c,
    0x1f83d9ab,
    0x5be0cd19,
};

uint32_t rotr(uint32_t value, uint32_t shift) {
    return (value >> shift) | (value << (32U - shift));
}

std::vector<uint8_t> read_file_bytes(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw ModelLoadError(path, "cannot open file");
    }

    return std::vector<uint8_t>(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}

std::string sha256_bytes(const std::vector<uint8_t>& input) {
    std::vector<uint8_t> data = input;
    const uint64_t bit_length = static_cast<uint64_t>(data.size()) * 8U;

    data.push_back(0x80U);
    while ((data.size() % 64U) != 56U) {
        data.push_back(0x00U);
    }

    for (int shift = 56; shift >= 0; shift -= 8) {
        data.push_back(static_cast<uint8_t>((bit_length >> shift) & 0xffU));
    }

    auto hash = kSha256InitialState;

    for (size_t chunk_offset = 0; chunk_offset < data.size(); chunk_offset += 64U) {
        std::array<uint32_t, 64> schedule{};
        for (size_t i = 0; i < 16U; ++i) {
            const size_t index = chunk_offset + (i * 4U);
            schedule[i] = (static_cast<uint32_t>(data[index]) << 24U) |
                          (static_cast<uint32_t>(data[index + 1U]) << 16U) |
                          (static_cast<uint32_t>(data[index + 2U]) << 8U) |
                          static_cast<uint32_t>(data[index + 3U]);
        }

        for (size_t i = 16U; i < schedule.size(); ++i) {
            const uint32_t s0 = rotr(schedule[i - 15U], 7U) ^ rotr(schedule[i - 15U], 18U) ^
                                (schedule[i - 15U] >> 3U);
            const uint32_t s1 = rotr(schedule[i - 2U], 17U) ^ rotr(schedule[i - 2U], 19U) ^
                                (schedule[i - 2U] >> 10U);
            schedule[i] = schedule[i - 16U] + s0 + schedule[i - 7U] + s1;
        }

        uint32_t a = hash[0];
        uint32_t b = hash[1];
        uint32_t c = hash[2];
        uint32_t d = hash[3];
        uint32_t e = hash[4];
        uint32_t f = hash[5];
        uint32_t g = hash[6];
        uint32_t h = hash[7];

        for (size_t i = 0; i < schedule.size(); ++i) {
            const uint32_t sigma1 = rotr(e, 6U) ^ rotr(e, 11U) ^ rotr(e, 25U);
            const uint32_t choice = (e & f) ^ ((~e) & g);
            const uint32_t temp1 = h + sigma1 + choice + kSha256Constants[i] + schedule[i];
            const uint32_t sigma0 = rotr(a, 2U) ^ rotr(a, 13U) ^ rotr(a, 22U);
            const uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
            const uint32_t temp2 = sigma0 + majority;

            h = g;
            g = f;
            f = e;
            e = d + temp1;
            d = c;
            c = b;
            b = a;
            a = temp1 + temp2;
        }

        hash[0] += a;
        hash[1] += b;
        hash[2] += c;
        hash[3] += d;
        hash[4] += e;
        hash[5] += f;
        hash[6] += g;
        hash[7] += h;
    }

    std::ostringstream output;
    output << std::hex << std::setfill('0');
    for (uint32_t value : hash) {
        output << std::setw(8) << value;
    }
    return output.str();
}

}  // namespace

std::string sha256_file(const std::string& path) {
    return sha256_bytes(read_file_bytes(path));
}

ModelRegistry& ModelRegistry::instance() {
    static ModelRegistry registry;
    return registry;
}

ModelRegistry::ModelRegistry()
    : engine_factory_([](const std::string& path) {
          throw ModelLoadError(path, "engine factory not configured");
          return std::shared_ptr<IModelEngine>{};
      })
    , gc_thread_(&ModelRegistry::gc_loop, this) {}

ModelRegistry::~ModelRegistry() {
    stop_gc_.store(true);
    cv_.notify_all();
    if (gc_thread_.joinable()) {
        gc_thread_.join();
    }
}

std::shared_ptr<IModelEngine> ModelRegistry::acquire(const std::string& path) {
    const std::string key = sha256_file(path);

    std::lock_guard<std::mutex> lock(mutex_);
    auto it = entries_.find(key);
    if (it != entries_.end()) {
        ++it->second.ref_count;
        it->second.expires_at = std::chrono::steady_clock::time_point::max();
        path_to_key_[path] = key;
        return it->second.engine;
    }

    std::shared_ptr<IModelEngine> engine;
    try {
        engine = engine_factory_(path);
    } catch (const ModelLoadError&) {
        throw;
    } catch (const std::exception& error) {
        throw ModelLoadError(path, error.what());
    }

    if (!engine) {
        throw ModelLoadError(path, "engine factory returned null");
    }

    auto [inserted_it, inserted] = entries_.emplace(key, RegistryEntry{});
    inserted_it->second.engine = engine;
    inserted_it->second.ref_count = 1;
    inserted_it->second.expires_at = std::chrono::steady_clock::time_point::max();
    path_to_key_[path] = key;
    return inserted_it->second.engine;
}

void ModelRegistry::release(const std::string& path) {
    std::lock_guard<std::mutex> lock(mutex_);
    const std::string key = resolve_key_for_release(path);
    auto it = entries_.find(key);
    if (it == entries_.end()) {
        throw NotFoundError("Model '" + path + "' not found");
    }
    if (it->second.ref_count == 0) {
        throw ConfigError("Model '" + path + "' has already been fully released");
    }

    --it->second.ref_count;
    if (it->second.ref_count == 0) {
        it->second.expires_at = std::chrono::steady_clock::now() + ttl_;
        cv_.notify_all();
    }
}

void ModelRegistry::set_engine_factory(EngineFactory factory) {
    std::lock_guard<std::mutex> lock(mutex_);
    engine_factory_ = std::move(factory);
}

void ModelRegistry::set_ttl(std::chrono::milliseconds ttl) {
    std::lock_guard<std::mutex> lock(mutex_);
    ttl_ = ttl;
    cv_.notify_all();
}

std::chrono::milliseconds ModelRegistry::ttl() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return ttl_;
}

size_t ModelRegistry::ref_count(const std::string& path) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const std::string key = resolve_key_for_release(path);
    auto it = entries_.find(key);
    if (it == entries_.end()) {
        return 0;
    }
    return it->second.ref_count;
}

bool ModelRegistry::contains(const std::string& path) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto path_it = path_to_key_.find(path);
    if (path_it != path_to_key_.end()) {
        return entries_.count(path_it->second) > 0;
    }
    return false;
}

void ModelRegistry::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    entries_.clear();
    path_to_key_.clear();
}

void ModelRegistry::gc_once() {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto now = std::chrono::steady_clock::now();

    for (auto it = entries_.begin(); it != entries_.end();) {
        if (it->second.ref_count == 0 && now >= it->second.expires_at) {
            const std::string key = it->first;
            for (auto path_it = path_to_key_.begin(); path_it != path_to_key_.end();) {
                if (path_it->second == key) {
                    path_it = path_to_key_.erase(path_it);
                } else {
                    ++path_it;
                }
            }
            it = entries_.erase(it);
        } else {
            ++it;
        }
    }
}

std::string ModelRegistry::resolve_key_for_release(const std::string& path) const {
    auto path_it = path_to_key_.find(path);
    if (path_it != path_to_key_.end()) {
        return path_it->second;
    }
    return sha256_file(path);
}

void ModelRegistry::gc_loop() {
    std::unique_lock<std::mutex> lock(mutex_);
    while (!stop_gc_.load()) {
        cv_.wait_for(lock, gc_interval_, [this] { return stop_gc_.load(); });
        if (stop_gc_.load()) {
            break;
        }

        const auto now = std::chrono::steady_clock::now();
        for (auto it = entries_.begin(); it != entries_.end();) {
            if (it->second.ref_count == 0 && now >= it->second.expires_at) {
                const std::string key = it->first;
                for (auto path_it = path_to_key_.begin(); path_it != path_to_key_.end();) {
                    if (path_it->second == key) {
                        path_it = path_to_key_.erase(path_it);
                    } else {
                        ++path_it;
                    }
                }
                it = entries_.erase(it);
            } else {
                ++it;
            }
        }
    }
}

}  // namespace visionpipe
