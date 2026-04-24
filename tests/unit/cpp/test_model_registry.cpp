#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <unistd.h>

#include "core/error.h"
#include "core/model_registry.h"
#include "hal/imodel_engine.h"

namespace visionpipe {
namespace {

using namespace std::chrono_literals;

class CountingModelEngine final : public IModelEngine {
public:
    explicit CountingModelEngine(std::string path)
        : path_(std::move(path)) {
        ++constructed_;
    }

    ~CountingModelEngine() override { ++destroyed_; }

    std::unique_ptr<IExecContext> create_context() override {
        return std::make_unique<MockExecContext>();
    }

    size_t device_memory_bytes() const override { return path_.size(); }

    static void reset_counts() {
        constructed_.store(0);
        destroyed_.store(0);
    }

    static int constructed() { return constructed_.load(); }
    static int destroyed() { return destroyed_.load(); }

private:
    std::string path_;

    inline static std::atomic<int> constructed_{0};
    inline static std::atomic<int> destroyed_{0};
};

class ModelRegistryTest : public ::testing::Test {
protected:
    void SetUp() override {
        CountingModelEngine::reset_counts();
        registry_.clear();
        registry_.set_ttl(100ms);
        registry_.set_engine_factory([](const std::string& path) {
            return std::make_shared<CountingModelEngine>(path);
        });
    }

    void TearDown() override {
        registry_.clear();
        for (const auto& path : temp_files_) {
            std::error_code ec;
            std::filesystem::remove(path, ec);
        }
    }

    std::string create_temp_file(const std::string& contents) {
        char path_template[] = "/tmp/visionpipe_model_registry_XXXXXX";
        const int fd = ::mkstemp(path_template);
        if (fd == -1) {
            throw std::runtime_error("mkstemp failed");
        }
        ::close(fd);

        std::ofstream os(path_template, std::ios::binary);
        if (!os) {
            throw std::runtime_error("failed to open temp file for writing");
        }
        os << contents;
        os.close();
        if (!os) {
            throw std::runtime_error("failed to write temp file");
        }

        temp_files_.emplace_back(path_template);
        return std::string(path_template);
    }

    bool wait_until(const std::function<bool()>& predicate,
                    std::chrono::milliseconds timeout = 500ms) {
        const auto deadline = std::chrono::steady_clock::now() + timeout;
        while (std::chrono::steady_clock::now() < deadline) {
            if (predicate()) {
                return true;
            }
            std::this_thread::sleep_for(5ms);
        }
        return predicate();
    }

    ModelRegistry registry_;
    std::vector<std::filesystem::path> temp_files_;
};

TEST_F(ModelRegistryTest, Sha256FileIsStableAndContentSensitive) {
    const std::string file_a = create_temp_file("model-a-contents");
    const std::string file_b = create_temp_file("model-b-contents");

    const std::string hash_a1 = sha256_file(file_a);
    const std::string hash_a2 = sha256_file(file_a);
    const std::string hash_b = sha256_file(file_b);

    EXPECT_FALSE(hash_a1.empty());
    EXPECT_EQ(hash_a1, hash_a2);
    EXPECT_NE(hash_a1, hash_b);
}

TEST_F(ModelRegistryTest, AcquireSameFileTwiceReusesEngineAndIncrementsRefCount) {
    const std::string path = create_temp_file("shared-model");

    auto engine1 = registry_.acquire(path);
    auto engine2 = registry_.acquire(path);

    ASSERT_NE(engine1, nullptr);
    ASSERT_NE(engine2, nullptr);
    EXPECT_EQ(engine1.get(), engine2.get());
    EXPECT_EQ(CountingModelEngine::constructed(), 1);
    EXPECT_EQ(CountingModelEngine::destroyed(), 0);
    EXPECT_EQ(registry_.ref_count(path), 2u);
    EXPECT_TRUE(registry_.contains(path));
}

TEST_F(ModelRegistryTest, ReleaseToZeroWaitsForTtlBeforeGcLoopDestroysEngine) {
    const std::string path = create_temp_file("ttl-model");

    auto engine1 = registry_.acquire(path);
    auto engine2 = registry_.acquire(path);

    registry_.release(path);
    EXPECT_EQ(registry_.ref_count(path), 1u);
    EXPECT_EQ(CountingModelEngine::destroyed(), 0);

    registry_.release(path);
    EXPECT_EQ(registry_.ref_count(path), 0u);
    EXPECT_EQ(CountingModelEngine::destroyed(), 0);

    engine1.reset();
    engine2.reset();

    std::this_thread::sleep_for(40ms);
    EXPECT_EQ(CountingModelEngine::destroyed(), 0);

    const bool destroyed_after_ttl = wait_until(
        [] { return CountingModelEngine::destroyed() == 1; }, 500ms);
    EXPECT_TRUE(destroyed_after_ttl);
}

TEST_F(ModelRegistryTest, AcquireDifferentFilesCreatesIndependentInstances) {
    const std::string path_a = create_temp_file("model-a");
    const std::string path_b = create_temp_file("model-b");

    auto engine_a = registry_.acquire(path_a);
    auto engine_b = registry_.acquire(path_b);

    ASSERT_NE(engine_a, nullptr);
    ASSERT_NE(engine_b, nullptr);
    EXPECT_NE(engine_a.get(), engine_b.get());
    EXPECT_EQ(CountingModelEngine::constructed(), 2);
    EXPECT_EQ(registry_.ref_count(path_a), 1u);
    EXPECT_EQ(registry_.ref_count(path_b), 1u);

    registry_.release(path_a);
    EXPECT_EQ(registry_.ref_count(path_a), 0u);
    EXPECT_EQ(registry_.ref_count(path_b), 1u);

    registry_.release(path_b);
    EXPECT_EQ(registry_.ref_count(path_b), 0u);
}

TEST_F(ModelRegistryTest, AcquirePropagatesFactoryErrorsWithoutLeakingRegistryState) {
    const std::string path = create_temp_file("broken-model");

    registry_.set_engine_factory([](const std::string& model_path) -> std::shared_ptr<IModelEngine> {
        throw ModelLoadError(model_path, "factory failure");
    });

    EXPECT_THROW(registry_.acquire(path), ModelLoadError);
    EXPECT_EQ(CountingModelEngine::constructed(), 0);
    EXPECT_EQ(CountingModelEngine::destroyed(), 0);
    EXPECT_EQ(registry_.ref_count(path), 0u);
    EXPECT_FALSE(registry_.contains(path));
}

}  // namespace
}  // namespace visionpipe
