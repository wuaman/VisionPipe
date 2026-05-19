#include <chrono>
#include <cmath>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "core/frame.h"
#include "core/tensor.h"
#include "hal/imodel_engine.h"
#include "hal/nvidia/cuda_allocator.h"
#include "nodes/infer/classifier_node.h"

namespace visionpipe {
namespace {

namespace fs = std::filesystem;

// ============================================================================
// Mock Model Engine for Testing
// ============================================================================

class MockClassifierExecContext : public IExecContext {
public:
    explicit MockClassifierExecContext(int num_classes = 1000)
        : num_classes_(num_classes)
        , infer_count_(0) {}

    void infer(const Tensor& input, Tensor& output) override {
        std::lock_guard<std::mutex> lock(mutex_);

        ASSERT_TRUE(input.valid());
        ASSERT_GE(input.shape.size(), 2u);

        int64_t batch_size = input.shape[0];

        if (!output.valid() || output.shape[0] != batch_size || output.shape[1] != num_classes_) {
            static CpuAllocator allocator;
            output = Tensor({batch_size, num_classes_}, DataType::FLOAT32, &allocator);
        }

        float* out_data = static_cast<float*>(output.data);
        for (int64_t b = 0; b < batch_size; ++b) {
            int target_class = static_cast<int>(b % num_classes_);
            float sum = 0.0f;

            for (int c = 0; c < num_classes_; ++c) {
                float logit = (c == target_class) ? 10.0f : 0.1f;
                out_data[b * num_classes_ + c] = std::exp(logit);
                sum += out_data[b * num_classes_ + c];
            }

            for (int c = 0; c < num_classes_; ++c) {
                out_data[b * num_classes_ + c] /= sum;
            }
        }

        ++infer_count_;
    }

    int infer_count() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return infer_count_;
    }

private:
    int num_classes_;
    mutable std::mutex mutex_;
    int infer_count_;
};

class MockClassifierEngine : public IModelEngine {
public:
    explicit MockClassifierEngine(int num_classes = 1000)
        : num_classes_(num_classes) {}

    std::unique_ptr<IExecContext> create_context() override {
        return std::make_unique<MockClassifierExecContext>(num_classes_);
    }

    size_t device_memory_bytes() const override { return 0; }

    int num_classes() const { return num_classes_; }

private:
    int num_classes_;
};

// ============================================================================
// Test Utilities
// ============================================================================

Frame create_test_frame(int width = 640, int height = 480, int frame_id = 0) {
    Frame frame;
    frame.stream_id = 1;
    frame.frame_id = frame_id;
    frame.pts_us = frame_id * 33333;

    cv::Mat cpu_image(height, width, CV_8UC3);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            cpu_image.at<cv::Vec3b>(y, x) = cv::Vec3b(
                static_cast<uint8_t>((x * 255 / width) % 256),
                static_cast<uint8_t>((y * 255 / height) % 256),
                static_cast<uint8_t>(128));
        }
    }

    cv::cuda::GpuMat gpu_image;
    gpu_image.upload(cpu_image);

    static auto allocator = std::make_shared<CudaAllocator>();
    frame.image = Tensor({height, width, 3}, DataType::UINT8, allocator.get());
    cudaMemcpy(frame.image.data, gpu_image.data, frame.image.nbytes, cudaMemcpyDeviceToDevice);

    return frame;
}

Frame create_frame_with_detections(
    int width, int height, int frame_id,
    const std::vector<std::array<float, 4>>& bboxes,
    int class_id = 0) {

    Frame frame = create_test_frame(width, height, frame_id);

    for (size_t i = 0; i < bboxes.size(); ++i) {
        Detection det;
        det.bbox[0] = bboxes[i][0];
        det.bbox[1] = bboxes[i][1];
        det.bbox[2] = bboxes[i][2];
        det.bbox[3] = bboxes[i][3];
        det.class_id = class_id;
        det.confidence = 0.9f;
        det.track_id = -1;
        frame.detections.push_back(det);
    }

    return frame;
}

// ============================================================================
// ClassifierNode Construction Tests
// ============================================================================

class ClassifierNodeConstructionTest : public ::testing::Test {
protected:
    void SetUp() override {
        mock_engine_ = std::make_shared<MockClassifierEngine>();
    }

    std::shared_ptr<MockClassifierEngine> mock_engine_;
};

TEST_F(ClassifierNodeConstructionTest, CreateWithDefaultConfig) {
    ClassifierNode node(mock_engine_);
    EXPECT_EQ(node.name(), "classifier");
    EXPECT_EQ(node.config().input_width, 224);
    EXPECT_EQ(node.config().input_height, 224);
    EXPECT_EQ(node.config().max_batch_size, 32);
    EXPECT_EQ(node.config().workers, 1u);
    EXPECT_TRUE(node.config().normalize_mean_std);
    EXPECT_TRUE(node.config().target_classes.empty());
}

TEST_F(ClassifierNodeConstructionTest, CreateWithTargetClasses) {
    ClassifierConfig config;
    config.target_classes = {0, 1, 2};
    ClassifierNode node(mock_engine_, config, "crop_classifier");
    EXPECT_EQ(node.config().target_classes.size(), 3u);
    EXPECT_EQ(node.config().target_classes[0], 0);
    EXPECT_EQ(node.config().target_classes[2], 2);
}

TEST_F(ClassifierNodeConstructionTest, CreateWithCustomConfig) {
    ClassifierConfig config;
    config.input_width = 299;
    config.input_height = 299;
    config.max_batch_size = 64;
    config.workers = 4;
    config.normalize_mean_std = false;

    ClassifierNode node(mock_engine_, config, "custom_classifier");
    EXPECT_EQ(node.name(), "custom_classifier");
    EXPECT_EQ(node.config().input_width, 299);
    EXPECT_EQ(node.config().input_height, 299);
    EXPECT_EQ(node.config().max_batch_size, 64);
    EXPECT_EQ(node.config().workers, 4u);
    EXPECT_FALSE(node.config().normalize_mean_std);
}

TEST_F(ClassifierNodeConstructionTest, CreateWithSimpleName) {
    ClassifierNode node(mock_engine_, "simple_classifier");
    EXPECT_EQ(node.name(), "simple_classifier");
    EXPECT_EQ(node.config().input_width, 224);
}

TEST_F(ClassifierNodeConstructionTest, InitialStateIsInit) {
    ClassifierNode node(mock_engine_);
    EXPECT_EQ(node.state(), NodeState::INIT);
}

TEST_F(ClassifierNodeConstructionTest, WorkerCountMatchesConfig) {
    ClassifierConfig config;
    config.workers = 8;
    ClassifierNode node(mock_engine_, config);
    EXPECT_EQ(node.worker_count(), 8u);
}

TEST_F(ClassifierNodeConstructionTest, NullEngineThrows) {
    std::shared_ptr<IModelEngine> null_engine;
    EXPECT_THROW(ClassifierNode node(null_engine), ConfigError);
}

// ============================================================================
// ClassifierNode State Transition Tests
// ============================================================================

class ClassifierNodeStateTest : public ::testing::Test {
protected:
    void SetUp() override {
        mock_engine_ = std::make_shared<MockClassifierEngine>();
        config_.workers = 1;
    }

    std::shared_ptr<MockClassifierEngine> mock_engine_;
    ClassifierConfig config_;
};

TEST_F(ClassifierNodeStateTest, InitToRunning) {
    ClassifierNode node(mock_engine_, config_);
    node.create_output_queue(16, OverflowPolicy::BLOCK);

    EXPECT_EQ(node.state(), NodeState::INIT);

    node.start();
    EXPECT_EQ(node.state(), NodeState::RUNNING);

    node.stop(true);
    node.wait_stop();
    EXPECT_EQ(node.state(), NodeState::STOPPED);
}

TEST_F(ClassifierNodeStateTest, StopWithoutDrain) {
    ClassifierNode node(mock_engine_, config_);
    node.create_output_queue(16, OverflowPolicy::BLOCK);
    node.start();

    node.stop(false);
    node.wait_stop();
    EXPECT_EQ(node.state(), NodeState::STOPPED);
}

TEST_F(ClassifierNodeStateTest, MultipleStartStopCycles) {
    ClassifierNode node(mock_engine_, config_);
    node.create_output_queue(16, OverflowPolicy::BLOCK);

    for (int i = 0; i < 3; ++i) {
        node.start();
        EXPECT_EQ(node.state(), NodeState::RUNNING);

        node.stop(true);
        node.wait_stop();
        EXPECT_EQ(node.state(), NodeState::STOPPED);
    }
}

TEST_F(ClassifierNodeStateTest, StopIsIdempotent) {
    ClassifierNode node(mock_engine_, config_);
    node.create_output_queue(16, OverflowPolicy::BLOCK);
    node.start();
    node.stop(true);
    node.wait_stop();

    EXPECT_NO_THROW(node.stop(true));
    EXPECT_NO_THROW(node.wait_stop());
    EXPECT_EQ(node.state(), NodeState::STOPPED);
}

// ============================================================================
// Mode 2: Whole-Image Classification (target_classes empty)
// ============================================================================

class ClassifierNodeWholeImageTest : public ::testing::Test {
protected:
    void SetUp() override {
        mock_engine_ = std::make_shared<MockClassifierEngine>(10);
        config_.workers = 1;
        // target_classes empty → whole-image mode
    }

    std::shared_ptr<MockClassifierEngine> mock_engine_;
    ClassifierConfig config_;
};

TEST_F(ClassifierNodeWholeImageTest, WholeImageProducesClassification) {
    ClassifierNode node(mock_engine_, config_);
    node.create_output_queue(16, OverflowPolicy::BLOCK);
    node.start();

    Frame frame = create_test_frame(640, 480, 0);
    node.process(frame);

    ASSERT_EQ(frame.classifications.size(), 1u);
    EXPECT_EQ(frame.classifications[0].detection_index, -1);
    EXPECT_GE(frame.classifications[0].confidence, 0.0f);
    EXPECT_LE(frame.classifications[0].confidence, 1.0f);

    node.stop(true);
    node.wait_stop();
}

TEST_F(ClassifierNodeWholeImageTest, WholeImageIgnoresDetections) {
    ClassifierNode node(mock_engine_, config_);
    node.create_output_queue(16, OverflowPolicy::BLOCK);
    node.start();

    // Frame has detections, but whole-image mode ignores them
    std::vector<std::array<float, 4>> bboxes = {{0.1f, 0.1f, 0.5f, 0.5f}};
    Frame frame = create_frame_with_detections(640, 480, 0, bboxes);
    node.process(frame);

    // Should still produce whole-image classification
    ASSERT_EQ(frame.classifications.size(), 1u);
    EXPECT_EQ(frame.classifications[0].detection_index, -1);
    // Detections should be untouched
    ASSERT_EQ(frame.detections.size(), 1u);

    node.stop(true);
    node.wait_stop();
}

TEST_F(ClassifierNodeWholeImageTest, WholeImageNoImageThrows) {
    ClassifierNode node(mock_engine_, config_);
    node.create_output_queue(16, OverflowPolicy::BLOCK);
    node.start();

    Frame frame;
    frame.frame_id = 0;
    EXPECT_THROW(node.process(frame), InferError);

    node.stop(true);
    node.wait_stop();
}

TEST_F(ClassifierNodeWholeImageTest, WholeImagePreservesFrameMetadata) {
    ClassifierNode node(mock_engine_, config_);
    node.create_output_queue(16, OverflowPolicy::BLOCK);
    node.start();

    Frame frame = create_test_frame(640, 480, 42);
    frame.stream_id = 7;
    frame.pts_us = 123456;

    node.process(frame);

    EXPECT_EQ(frame.frame_id, 42);
    EXPECT_EQ(frame.stream_id, 7);
    EXPECT_EQ(frame.pts_us, 123456);

    node.stop(true);
    node.wait_stop();
}

// ============================================================================
// Mode 1: Crop-Based Secondary Classification (target_classes non-empty)
// ============================================================================

class ClassifierNodeCropTest : public ::testing::Test {
protected:
    void SetUp() override {
        mock_engine_ = std::make_shared<MockClassifierEngine>(10);
        config_.workers = 1;
        config_.target_classes = {0, 1, 2};
    }

    std::shared_ptr<MockClassifierEngine> mock_engine_;
    ClassifierConfig config_;
};

TEST_F(ClassifierNodeCropTest, MatchingDetectionsProduceClassifications) {
    ClassifierNode node(mock_engine_, config_);
    node.create_output_queue(16, OverflowPolicy::BLOCK);
    node.start();

    // class_id=0 matches target_classes {0,1,2}
    std::vector<std::array<float, 4>> bboxes = {
        {0.1f, 0.1f, 0.5f, 0.5f},
        {0.5f, 0.5f, 0.9f, 0.9f}
    };
    Frame frame = create_frame_with_detections(640, 480, 0, bboxes, 0);

    node.process(frame);

    ASSERT_EQ(frame.classifications.size(), 2u);
    EXPECT_EQ(frame.classifications[0].detection_index, 0);
    EXPECT_EQ(frame.classifications[1].detection_index, 1);
    EXPECT_GE(frame.classifications[0].confidence, 0.0f);
    EXPECT_GE(frame.classifications[1].confidence, 0.0f);
    // Detections should be untouched
    ASSERT_EQ(frame.detections.size(), 2u);
    EXPECT_EQ(frame.detections[0].class_id, 0);
    EXPECT_EQ(frame.detections[1].class_id, 0);

    node.stop(true);
    node.wait_stop();
}

TEST_F(ClassifierNodeCropTest, NonMatchingDetectionsPassthrough) {
    ClassifierNode node(mock_engine_, config_);
    node.create_output_queue(16, OverflowPolicy::BLOCK);
    node.start();

    // class_id=99 does NOT match target_classes {0,1,2}
    std::vector<std::array<float, 4>> bboxes = {{0.1f, 0.1f, 0.5f, 0.5f}};
    Frame frame = create_frame_with_detections(640, 480, 0, bboxes, 99);

    node.process(frame);

    // No matching detections → no classifications, transparent passthrough
    EXPECT_TRUE(frame.classifications.empty());
    ASSERT_EQ(frame.detections.size(), 1u);
    EXPECT_EQ(frame.detections[0].class_id, 99);

    node.stop(true);
    node.wait_stop();
}

TEST_F(ClassifierNodeCropTest, EmptyDetectionsPassthrough) {
    ClassifierNode node(mock_engine_, config_);
    node.create_output_queue(16, OverflowPolicy::BLOCK);
    node.start();

    Frame frame = create_test_frame(640, 480, 0);
    EXPECT_TRUE(frame.detections.empty());

    node.process(frame);

    EXPECT_TRUE(frame.classifications.empty());
    EXPECT_TRUE(frame.has_image());

    node.stop(true);
    node.wait_stop();
}

TEST_F(ClassifierNodeCropTest, MixedTargetClassFiltering) {
    ClassifierNode node(mock_engine_, config_);
    node.create_output_queue(16, OverflowPolicy::BLOCK);
    node.start();

    Frame frame = create_test_frame(640, 480, 0);
    // Add detections with mixed class IDs
    Detection d1; d1.bbox[0] = 0.1f; d1.bbox[1] = 0.1f; d1.bbox[2] = 0.4f; d1.bbox[3] = 0.4f;
    d1.class_id = 0; d1.confidence = 0.9f;  // matches
    Detection d2; d2.bbox[0] = 0.5f; d2.bbox[1] = 0.1f; d2.bbox[2] = 0.8f; d2.bbox[3] = 0.4f;
    d2.class_id = 99; d2.confidence = 0.9f; // doesn't match
    Detection d3; d3.bbox[0] = 0.1f; d3.bbox[1] = 0.5f; d3.bbox[2] = 0.4f; d3.bbox[3] = 0.8f;
    d3.class_id = 2; d3.confidence = 0.9f;  // matches
    frame.detections.push_back(d1);
    frame.detections.push_back(d2);
    frame.detections.push_back(d3);

    node.process(frame);

    // Only 2 matching detections → 2 classifications
    ASSERT_EQ(frame.classifications.size(), 2u);
    EXPECT_EQ(frame.classifications[0].detection_index, 0);  // d1
    EXPECT_EQ(frame.classifications[1].detection_index, 2);  // d3
    // Detections unchanged
    ASSERT_EQ(frame.detections.size(), 3u);

    node.stop(true);
    node.wait_stop();
}

TEST_F(ClassifierNodeCropTest, DoesNotOverwriteDetections) {
    ClassifierNode node(mock_engine_, config_);
    node.create_output_queue(16, OverflowPolicy::BLOCK);
    node.start();

    std::vector<std::array<float, 4>> bboxes = {{0.1f, 0.1f, 0.5f, 0.5f}};
    Frame frame = create_frame_with_detections(640, 480, 0, bboxes, 1);
    float original_conf = frame.detections[0].confidence;
    int original_class = frame.detections[0].class_id;

    node.process(frame);

    // Detection fields must remain unchanged
    EXPECT_EQ(frame.detections[0].class_id, original_class);
    EXPECT_FLOAT_EQ(frame.detections[0].confidence, original_conf);
    // Results go to classifications
    ASSERT_GE(frame.classifications.size(), 1u);

    node.stop(true);
    node.wait_stop();
}

// ============================================================================
// ClassifierNode Bbox Boundary Tests (crop mode)
// ============================================================================

class ClassifierNodeBboxBoundaryTest : public ::testing::Test {
protected:
    void SetUp() override {
        mock_engine_ = std::make_shared<MockClassifierEngine>(1000);
        config_.workers = 1;
        config_.target_classes = {0};  // crop mode
    }

    std::shared_ptr<MockClassifierEngine> mock_engine_;
    ClassifierConfig config_;
};

TEST_F(ClassifierNodeBboxBoundaryTest, BboxAtImageEdge) {
    ClassifierNode node(mock_engine_, config_);
    node.create_output_queue(16, OverflowPolicy::BLOCK);
    node.start();

    std::vector<std::array<float, 4>> bboxes = {{0.0f, 0.0f, 1.0f, 1.0f}};
    Frame frame = create_frame_with_detections(640, 480, 0, bboxes, 0);

    EXPECT_NO_THROW(node.process(frame));
    ASSERT_EQ(frame.detections.size(), 1u);

    node.stop(true);
    node.wait_stop();
}

TEST_F(ClassifierNodeBboxBoundaryTest, BboxExceedsImageBoundary) {
    ClassifierNode node(mock_engine_, config_);
    node.create_output_queue(16, OverflowPolicy::BLOCK);
    node.start();

    std::vector<std::array<float, 4>> bboxes = {{-0.1f, -0.1f, 1.1f, 1.1f}};
    Frame frame = create_frame_with_detections(640, 480, 0, bboxes, 0);

    EXPECT_NO_THROW(node.process(frame));
    ASSERT_EQ(frame.detections.size(), 1u);

    node.stop(true);
    node.wait_stop();
}

TEST_F(ClassifierNodeBboxBoundaryTest, BboxZeroArea) {
    ClassifierNode node(mock_engine_, config_);
    node.create_output_queue(16, OverflowPolicy::BLOCK);
    node.start();

    std::vector<std::array<float, 4>> bboxes = {{0.5f, 0.5f, 0.5f, 0.5f}};
    Frame frame = create_frame_with_detections(640, 480, 0, bboxes, 0);

    EXPECT_NO_THROW(node.process(frame));

    node.stop(true);
    node.wait_stop();
}

TEST_F(ClassifierNodeBboxBoundaryTest, SmallBbox) {
    ClassifierNode node(mock_engine_, config_);
    node.create_output_queue(16, OverflowPolicy::BLOCK);
    node.start();

    std::vector<std::array<float, 4>> bboxes = {{0.49f, 0.49f, 0.51f, 0.51f}};
    Frame frame = create_frame_with_detections(640, 480, 0, bboxes, 0);

    EXPECT_NO_THROW(node.process(frame));
    ASSERT_EQ(frame.detections.size(), 1u);

    node.stop(true);
    node.wait_stop();
}

// ============================================================================
// ClassifierNode Error Handling Tests
// ============================================================================

class ClassifierNodeErrorTest : public ::testing::Test {
protected:
    void SetUp() override {
        mock_engine_ = std::make_shared<MockClassifierEngine>();
        config_.workers = 1;
        config_.target_classes = {0};
    }

    std::shared_ptr<MockClassifierEngine> mock_engine_;
    ClassifierConfig config_;
};

TEST_F(ClassifierNodeErrorTest, FrameWithoutImage) {
    ClassifierNode node(mock_engine_, config_);
    node.create_output_queue(16, OverflowPolicy::BLOCK);
    node.start();

    Frame frame;
    frame.frame_id = 0;
    Detection det;
    det.bbox[0] = 0.0f; det.bbox[1] = 0.0f;
    det.bbox[2] = 0.5f; det.bbox[3] = 0.5f;
    det.class_id = 0;
    frame.detections.push_back(det);

    EXPECT_THROW(node.process(frame), InferError);

    node.stop(true);
    node.wait_stop();
}

TEST_F(ClassifierNodeErrorTest, FrameWithInvalidImageTensor) {
    ClassifierNode node(mock_engine_, config_);
    node.create_output_queue(16, OverflowPolicy::BLOCK);
    node.start();

    Frame frame;
    frame.frame_id = 0;
    frame.image = Tensor();
    Detection det;
    det.class_id = 0;
    frame.detections.push_back(det);

    EXPECT_THROW(node.process(frame), InferError);

    node.stop(true);
    node.wait_stop();
}

// ============================================================================
// ClassifierNode Statistics Tests
// ============================================================================

class ClassifierNodeStatsTest : public ::testing::Test {
protected:
    void SetUp() override {
        mock_engine_ = std::make_shared<MockClassifierEngine>();
        config_.workers = 1;
    }

    std::shared_ptr<MockClassifierEngine> mock_engine_;
    ClassifierConfig config_;
};

TEST_F(ClassifierNodeStatsTest, ProcessedCountUpdates) {
    ClassifierNode node(mock_engine_, config_);
    node.create_output_queue(100, OverflowPolicy::BLOCK);
    node.start();

    constexpr int kFrameCount = 10;
    for (int i = 0; i < kFrameCount; ++i) {
        Frame frame = create_test_frame(640, 480, i);
        node.input_queue()->push(std::move(frame));
    }

    node.stop(true);
    node.wait_stop();

    auto output_queue = node.output_queue();
    int processed_count = 0;
    while (auto frame_opt = output_queue->pop_for(std::chrono::milliseconds(100))) {
        processed_count++;
    }
    EXPECT_EQ(processed_count, kFrameCount);
}

TEST_F(ClassifierNodeStatsTest, ErrorCountZeroForValidFrames) {
    ClassifierNode node(mock_engine_, config_);
    node.create_output_queue(100, OverflowPolicy::BLOCK);
    node.start();

    // Whole-image mode (default config)
    Frame frame = create_test_frame(640, 480, 0);
    node.process(frame);

    auto stats = node.stats();
    EXPECT_EQ(stats.error_count, 0u);

    node.stop(true);
    node.wait_stop();
}

// ============================================================================
// ClassifierNode Integration Tests (real models — disabled by default)
// ============================================================================

class ClassifierNodeIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        auto test_data_dir = fs::current_path() / "tests" / "models";
        resources_available_ = fs::exists(test_data_dir);

        if (resources_available_) {
            resnet_model_path_ = test_data_dir / "resnet50_fp16.engine";
            efficientnet_model_path_ = test_data_dir / "efficientnet_b0_fp16.engine";
            shufflenet_model_path_ = test_data_dir / "shufflenetv2_fp16.engine";

            resources_available_ = fs::exists(resnet_model_path_) ||
                                   fs::exists(efficientnet_model_path_) ||
                                   fs::exists(shufflenet_model_path_);
        }
    }

    bool resources_available_ = false;
    fs::path resnet_model_path_;
    fs::path efficientnet_model_path_;
    fs::path shufflenet_model_path_;
};

TEST_F(ClassifierNodeIntegrationTest, DISABLED_ResNet50Classification) {
    if (!fs::exists(resnet_model_path_)) {
        GTEST_SKIP() << "ResNet50 model not found at " << resnet_model_path_;
    }
}

TEST_F(ClassifierNodeIntegrationTest, DISABLED_EfficientNetB0Classification) {
    if (!fs::exists(efficientnet_model_path_)) {
        GTEST_SKIP() << "EfficientNet-B0 model not found at " << efficientnet_model_path_;
    }
}

TEST_F(ClassifierNodeIntegrationTest, DISABLED_ShuffleNetV2Classification) {
    if (!fs::exists(shufflenet_model_path_)) {
        GTEST_SKIP() << "ShuffleNetV2 model not found at " << shufflenet_model_path_;
    }
}

}  // namespace
}  // namespace visionpipe
