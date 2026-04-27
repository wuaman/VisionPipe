#include <chrono>
#include <filesystem>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "core/frame.h"
#include "core/tensor.h"
#include "hal/nvidia/cuda_allocator.h"
#include "hal/nvidia/trt_model_engine.h"
#include "nodes/infer/detector_node.h"
#include "nodes/infer/post/detection_decoder.h"
#include "nodes/infer/pre/letterbox_resize.h"

namespace visionpipe {
namespace {

namespace fs = std::filesystem;

// 获取测试数据目录（从 build 目录或源目录运行）
fs::path get_test_data_dir() {
    // 尝试多个可能的路径
    std::vector<fs::path> candidates = {
        fs::current_path() / "tests" / "models",
        fs::current_path() / ".." / "tests" / "models",
        fs::path(__FILE__).parent_path().parent_path().parent_path().parent_path() / "tests" / "models"
    };

    for (const auto& path : candidates) {
        if (fs::exists(path)) {
            return path;
        }
    }

    return fs::path{};
}

// 测试模型路径（动态获取）
fs::path get_test_model_path() {
    return get_test_data_dir() / "yolov8n_fp16.engine";
}

// 测试视频路径
fs::path get_test_video_path() {
    return get_test_data_dir().parent_path() / "data" / "test_video_100frames.mp4";
}

// 检查测试资源是否存在
bool test_resources_available() {
    auto model_path = get_test_model_path();
    return fs::exists(model_path);
}

// 创建测试图像帧
Frame create_test_frame(int width = 640, int height = 480) {
    Frame frame;
    frame.stream_id = 1;
    frame.frame_id = 0;
    frame.pts_us = 0;

    // 创建随机图像数据
    cv::Mat cpu_image(height, width, CV_8UC3);
    cpu_image.setTo(cv::Scalar(128, 128, 128));

    // 上传到 GPU
    cv::cuda::GpuMat gpu_image;
    gpu_image.upload(cpu_image);

    // 创建 tensor
    static auto allocator = std::make_shared<CudaAllocator>();
    frame.image = Tensor({height, width, 3}, DataType::UINT8, allocator.get());
    cudaMemcpy(frame.image.data, gpu_image.data, frame.image.nbytes, cudaMemcpyDeviceToDevice);

    return frame;
}

class DetectorNodeTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!test_resources_available()) {
            GTEST_SKIP() << "Test resources not available, skipping tests. "
                         << "Run tests/data/download_test_assets.sh first.";
        }
    }
};

// LetterboxResize 测试
class LetterboxResizeTest : public ::testing::Test {};

TEST_F(LetterboxResizeTest, ComputeParamsSquare) {
    auto params = LetterboxResize::compute_params(640, 640, 640, 640);
    EXPECT_FLOAT_EQ(params.scale, 1.0f);
    EXPECT_EQ(params.pad_x, 0);
    EXPECT_EQ(params.pad_y, 0);
}

TEST_F(LetterboxResizeTest, ComputeParamsLandscape) {
    // 1920x1080 -> 640x640
    auto params = LetterboxResize::compute_params(1920, 1080, 640, 640);

    // scale = min(640/1920, 640/1080) = min(0.333, 0.593) = 0.333
    EXPECT_NEAR(params.scale, 640.0f / 1920.0f, 0.001f);

    // 缩放后: 640 x 360
    // pad_x = (640 - 640) / 2 = 0
    // pad_y = (640 - 360) / 2 = 140
    EXPECT_EQ(params.pad_x, 0);
    EXPECT_EQ(params.pad_y, 140);
}

TEST_F(LetterboxResizeTest, ComputeParamsPortrait) {
    // 1080x1920 -> 640x640
    auto params = LetterboxResize::compute_params(1080, 1920, 640, 640);

    // scale = min(640/1080, 640/1920) = min(0.593, 0.333) = 0.333
    EXPECT_NEAR(params.scale, 640.0f / 1920.0f, 0.001f);

    // 缩放后: 360 x 640
    // pad_x = (640 - 360) / 2 = 140
    // pad_y = (640 - 640) / 2 = 0
    EXPECT_EQ(params.pad_x, 140);
    EXPECT_EQ(params.pad_y, 0);
}

TEST_F(LetterboxResizeTest, MapBboxBack) {
    auto params = LetterboxResize::compute_params(1920, 1080, 640, 640);

    // letterbox 后的坐标
    float bbox[4] = {100.0f, 200.0f, 300.0f, 400.0f};
    LetterboxResize::map_bbox_back(bbox, params);

    // 坐标应该被映射回原图空间
    // x1' = (100 - 0) / 0.333 = 300
    // y1' = (200 - 140) / 0.333 = 180
    // x2' = (300 - 0) / 0.333 = 900
    // y2' = (400 - 140) / 0.333 = 780
    EXPECT_NEAR(bbox[0], 300.0f, 1.0f);
    EXPECT_NEAR(bbox[1], 180.0f, 1.0f);
    EXPECT_NEAR(bbox[2], 900.0f, 1.0f);
    EXPECT_NEAR(bbox[3], 780.0f, 1.0f);
}

// DetectionDecoder 测试
class DetectionDecoderTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 创建模拟的 YOLOv8 输出 tensor
        // 格式: [1, 84, 8400]
        // 84 = 4 (bbox) + 80 (classes)
        allocator_ = std::make_shared<CpuAllocator>();
    }

    std::shared_ptr<CpuAllocator> allocator_;
};

TEST_F(DetectionDecoderTest, NmsFiltersOverlappingBoxes) {
    std::vector<Detection> detections;

    // 添加三个重叠的检测框（同一类别）
    // d1 和 d2 高度重叠（IoU > 0.5），d3 与 d1 重叠度较低
    Detection d1;
    d1.bbox[0] = 0.0f; d1.bbox[1] = 0.0f; d1.bbox[2] = 100.0f; d1.bbox[3] = 100.0f;
    d1.class_id = 0;
    d1.confidence = 0.9f;
    detections.push_back(d1);

    Detection d2;
    d2.bbox[0] = 10.0f; d2.bbox[1] = 10.0f; d2.bbox[2] = 110.0f; d2.bbox[3] = 110.0f;
    d2.class_id = 0;
    d2.confidence = 0.8f;
    detections.push_back(d2);

    Detection d3;
    d3.bbox[0] = 50.0f; d3.bbox[1] = 50.0f; d3.bbox[2] = 150.0f; d3.bbox[3] = 150.0f;
    d3.class_id = 0;
    d3.confidence = 0.7f;
    detections.push_back(d3);

    // 执行 NMS
    DetectionDecoder::nms(detections, 0.5f);

    // d1 和 d2 的 IoU ≈ 0.68 > 0.5，d2 被抑制
    // d1 和 d3 的 IoU ≈ 0.14 < 0.5，d3 不被抑制
    // 最终保留 d1 和 d3
    EXPECT_EQ(detections.size(), 2u);
    EXPECT_FLOAT_EQ(detections[0].confidence, 0.9f);  // d1
    EXPECT_FLOAT_EQ(detections[1].confidence, 0.7f);  // d3
}

TEST_F(DetectionDecoderTest, NmsKeepsDifferentClasses) {
    std::vector<Detection> detections;

    // 添加两个完全重叠的检测框（不同类别）
    Detection d1;
    d1.bbox[0] = 0.0f; d1.bbox[1] = 0.0f; d1.bbox[2] = 100.0f; d1.bbox[3] = 100.0f;
    d1.class_id = 0;
    d1.confidence = 0.9f;
    detections.push_back(d1);

    Detection d2;
    d2.bbox[0] = 0.0f; d2.bbox[1] = 0.0f; d2.bbox[2] = 100.0f; d2.bbox[3] = 100.0f;
    d2.class_id = 1;
    d2.confidence = 0.8f;
    detections.push_back(d2);

    // 执行 NMS
    DetectionDecoder::nms(detections, 0.5f);

    // 不同类别应该都被保留
    EXPECT_EQ(detections.size(), 2u);
}

// DetectorNode 集成测试
TEST_F(DetectorNodeTest, CreateAndStart) {
    auto model_path = get_test_model_path();
    auto engine = std::make_shared<TrtModelEngine>(model_path.string());
    ASSERT_NE(engine, nullptr);

    DetectorConfig config;
    config.input_width = 640;
    config.input_height = 640;
    config.workers = 1;

    DetectorNode node(engine, config, "test_detector");
    EXPECT_EQ(node.name(), "test_detector");
    EXPECT_EQ(node.worker_count(), 1u);

    node.create_output_queue(16, OverflowPolicy::BLOCK);
    node.start();
    EXPECT_EQ(node.state(), NodeState::RUNNING);

    node.stop(true);
    node.wait_stop();
    EXPECT_EQ(node.state(), NodeState::STOPPED);
}

TEST_F(DetectorNodeTest, DetectSingleFrame) {
    auto model_path = get_test_model_path();
    auto engine = std::make_shared<TrtModelEngine>(model_path.string());

    DetectorConfig config;
    config.input_width = 640;
    config.input_height = 640;
    config.score_threshold = 0.25f;
    config.nms_threshold = 0.45f;
    config.workers = 1;

    DetectorNode node(engine, config);
    node.create_output_queue(16, OverflowPolicy::BLOCK);
    node.start();

    // 创建测试帧
    Frame frame = create_test_frame(640, 480);

    // 处理帧（同步方式）
    node.process(frame);

    // 验证检测结果
    // 注意：由于测试图像是纯灰色，可能没有检测结果
    // 这个测试主要验证流程能正常执行

    node.stop(true);
    node.wait_stop();
}

TEST_F(DetectorNodeTest, ParallelWorkers) {
    auto model_path = get_test_model_path();
    auto engine = std::make_shared<TrtModelEngine>(model_path.string());

    DetectorConfig config;
    config.workers = 3;

    DetectorNode node(engine, config);
    node.create_output_queue(100, OverflowPolicy::BLOCK);
    node.start();

    // 推送多帧
    constexpr int kFrameCount = 10;
    for (int i = 0; i < kFrameCount; ++i) {
        Frame frame = create_test_frame(640, 480);
        frame.frame_id = i;
        node.input_queue()->push(std::move(frame));
    }

    node.stop(true);
    node.wait_stop();

    // 验证所有帧都被处理
    auto output_queue = node.output_queue();
    int processed_count = 0;
    while (auto frame_opt = output_queue->pop_for(std::chrono::milliseconds(100))) {
        processed_count++;
    }
    EXPECT_EQ(processed_count, kFrameCount);
}

TEST_F(DetectorNodeTest, SetParamThreshold) {
    auto model_path = get_test_model_path();
    auto engine = std::make_shared<TrtModelEngine>(model_path.string());
    DetectorNode node(engine);

    // 测试设置 score_threshold
    EXPECT_TRUE(node.set_param("score_threshold", 0.5f));
    EXPECT_FLOAT_EQ(node.config().score_threshold, 0.5f);

    // 测试设置 nms_threshold
    EXPECT_TRUE(node.set_param("nms_threshold", 0.3f));
    EXPECT_FLOAT_EQ(node.config().nms_threshold, 0.3f);

    // 测试设置 max_detections
    EXPECT_TRUE(node.set_param("max_detections", 100));
    EXPECT_EQ(node.config().max_detections, 100);
}

TEST_F(DetectorNodeTest, SetRoi) {
    auto model_path = get_test_model_path();
    auto engine = std::make_shared<TrtModelEngine>(model_path.string());
    DetectorNode node(engine);

    // 设置 ROI（三角形区域）
    std::vector<std::vector<float>> polygons = {
        {0.0f, 0.0f, 1.0f, 0.0f, 0.5f, 1.0f}  // 三角形
    };
    node.set_roi(polygons);

    // 清除 ROI
    node.clear_roi();

    // 通过 set_param 设置 ROI
    std::vector<float> roi_coords = {0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f};
    EXPECT_TRUE(node.set_param("roi", roi_coords));
}

TEST_F(DetectorNodeTest, OutputFrameOrderPreserved) {
    auto model_path = get_test_model_path();
    auto engine = std::make_shared<TrtModelEngine>(model_path.string());

    DetectorConfig config;
    config.workers = 3;

    DetectorNode node(engine, config);
    node.create_output_queue(100, OverflowPolicy::BLOCK);
    node.start();

    // 推送帧
    constexpr int kFrameCount = 20;
    for (int i = 0; i < kFrameCount; ++i) {
        Frame frame = create_test_frame(640, 480);
        frame.frame_id = i;
        node.input_queue()->push(std::move(frame));
    }

    node.stop(true);
    node.wait_stop();

    // 验证输出帧顺序
    auto output_queue = node.output_queue();
    std::vector<int64_t> frame_ids;
    while (auto frame_opt = output_queue->pop_for(std::chrono::milliseconds(100))) {
        frame_ids.push_back(frame_opt->frame_id);
    }

    // 验证帧 ID 严格递增
    for (size_t i = 1; i < frame_ids.size(); ++i) {
        EXPECT_EQ(frame_ids[i], frame_ids[i-1] + 1)
            << "Frame order changed at index " << i;
    }
}

// 性能测试
TEST_F(DetectorNodeTest, InferenceLatency) {
    auto model_path = get_test_model_path();
    auto engine = std::make_shared<TrtModelEngine>(model_path.string());

    DetectorConfig config;
    config.workers = 1;

    DetectorNode node(engine, config);
    node.start();

    Frame frame = create_test_frame(1920, 1080);

    // 预热
    for (int i = 0; i < 3; ++i) {
        Frame warmup_frame = create_test_frame(1920, 1080);
        node.process(warmup_frame);
    }

    // 测量延迟
    constexpr int kIterations = 10;
    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < kIterations; ++i) {
        Frame test_frame = create_test_frame(1920, 1080);
        node.process(test_frame);
    }

    auto end = std::chrono::steady_clock::now();
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    double avg_latency_ms = static_cast<double>(elapsed_ms) / kIterations;

    std::cout << "Average inference latency: " << avg_latency_ms << " ms" << std::endl;

    // 验证延迟目标：< 20ms（RTX 3090）
    // 注意：这个测试可能在不同的 GPU 上有不同的结果
    // EXPECT_LT(avg_latency_ms, 20.0);

    node.stop(true);
    node.wait_stop();
}

}  // namespace
}  // namespace visionpipe
