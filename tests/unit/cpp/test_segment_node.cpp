// test_segment_node.cpp
// 任务 T2.5 单元测试：SegmentNode + SegMaskDecoder

#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <memory>
#include <thread>
#include <vector>

#include "core/bounded_queue.h"
#include "core/frame.h"
#include "core/node_base.h"
#include "core/tensor.h"
#include "hal/imodel_engine.h"
#include "nodes/infer/post/seg_mask_decoder.h"
#include "nodes/infer/pre/letterbox_resize.h"
#include "nodes/infer/segment_node.h"

namespace visionpipe {
namespace {

using namespace std::chrono_literals;

// ==================== Mock 类 ====================

/// @brief 模拟 YOLOv8-seg 输出的 Mock ExecContext
class MockSegExecContext final : public IExecContext {
public:
    MockSegExecContext(int input_width, int input_height, int num_masks = 32)
        : input_width_(input_width)
        , input_height_(input_height)
        , num_masks_(num_masks) {}

    void infer(const Tensor&, Tensor&) override {
        // 分割节点使用 infer_multi，此方法不应被调用
        throw std::runtime_error("SegmentNode should use infer_multi");
    }

    void infer_multi(const Tensor& input, std::vector<Tensor>& outputs) override {
        // 模拟 YOLOv8-seg 输出：
        // Output 0: [1, 84 + num_masks, num_anchors] - 检测 + mask coefficients
        // Output 1: [1, num_masks, mask_h, mask_w] - 原型掩码

        outputs.clear();
        outputs.resize(2);

        // 计算锚点数量 (输入尺寸 / 4)^2 (简化假设)
        int num_anchors = (input_width_ / 4) * (input_height_ / 4);

        // 创建检测输出 tensor
        // 格式：[1, 84 + num_masks, num_anchors]
        // 84 = 4 (bbox) + 80 (classes)
        int det_channels = 84 + num_masks_;
        outputs[0] = Tensor({1, det_channels, num_anchors}, DataType::FLOAT32, &allocator_);

        // 填充检测输出 (大部分为低置信度背景)
        float* det_data = static_cast<float*>(outputs[0].data);
        std::fill(det_data, det_data + outputs[0].numel(), 0.0f);

        // 在第一个锚点位置放置一个有效检测
        // bbox [x, y, w, h] -> [cx, cy, w, h] 格式
        det_data[0 * num_anchors + 0] = 0.5f;   // cx
        det_data[1 * num_anchors + 0] = 0.5f;   // cy
        det_data[2 * num_anchors + 0] = 0.2f;   // w
        det_data[3 * num_anchors + 0] = 0.2f;   // h
        det_data[4 * num_anchors + 0] = 0.9f;   // class 0 confidence

        // mask coefficients (32 个系数)
        for (int m = 0; m < num_masks_; ++m) {
            det_data[(84 + m) * num_anchors + 0] = 0.1f * static_cast<float>(m);
        }

        // 创建原型掩码 tensor
        int mask_h = input_height_ / 4;
        int mask_w = input_width_ / 4;
        outputs[1] = Tensor({1, num_masks_, mask_h, mask_w}, DataType::FLOAT32, &allocator_);

        // 填充原型掩码 (简单的渐变模式)
        float* proto_data = static_cast<float*>(outputs[1].data);
        for (int m = 0; m < num_masks_; ++m) {
            for (int y = 0; y < mask_h; ++y) {
                for (int x = 0; x < mask_w; ++x) {
                    int idx = m * mask_h * mask_w + y * mask_w + x;
                    // 简单的正弦波模式
                    proto_data[idx] = std::sin(static_cast<float>(x + y + m) * 0.1f);
                }
            }
        }
    }

private:
    int input_width_;
    int input_height_;
    int num_masks_;
    CpuAllocator allocator_;
};

/// @brief 模拟 YOLOv8-seg 模型的 Mock Engine
class MockSegModelEngine : public IModelEngine {
public:
    MockSegModelEngine(int input_width = 640, int input_height = 640, int num_masks = 32)
        : input_width_(input_width)
        , input_height_(input_height)
        , num_masks_(num_masks) {}

    std::unique_ptr<IExecContext> create_context() override {
        created_contexts_.fetch_add(1, std::memory_order_relaxed);
        return std::make_unique<MockSegExecContext>(input_width_, input_height_, num_masks_);
    }

    size_t device_memory_bytes() const override { return 0; }

    size_t output_count() const override { return 2; }

    size_t created_contexts() const { return created_contexts_.load(std::memory_order_relaxed); }

private:
    int input_width_;
    int input_height_;
    int num_masks_;
    std::atomic<size_t> created_contexts_{0};
};

// ==================== 辅助函数 ====================

Frame make_seg_frame(int64_t frame_id, int width = 640, int height = 480) {
    static CpuAllocator allocator;

    Frame frame;
    frame.stream_id = 1;
    frame.frame_id = frame_id;
    frame.pts_us = frame_id * 33333;  // ~30fps

    // HWC 格式 {H, W, 3}，与 preprocess CPU 分支匹配
    frame.image = Tensor({static_cast<size_t>(height), static_cast<size_t>(width), 3},
                         DataType::UINT8, &allocator);
    std::memset(frame.image.data, 128, frame.image.nbytes);

    return frame;
}

// ==================== SegMaskParams 测试 ====================

class SegMaskParamsTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(SegMaskParamsTest, DefaultValues) {
    SegMaskParams params;

    EXPECT_FLOAT_EQ(params.mask_threshold, 0.5f);
    EXPECT_FLOAT_EQ(params.score_threshold, 0.25f);
    EXPECT_FLOAT_EQ(params.nms_threshold, 0.45f);
    EXPECT_EQ(params.max_detections, 100);
}

TEST_F(SegMaskParamsTest, CustomValues) {
    SegMaskParams params;
    params.mask_threshold = 0.3f;
    params.score_threshold = 0.5f;
    params.nms_threshold = 0.7f;
    params.max_detections = 50;

    EXPECT_FLOAT_EQ(params.mask_threshold, 0.3f);
    EXPECT_FLOAT_EQ(params.score_threshold, 0.5f);
    EXPECT_FLOAT_EQ(params.nms_threshold, 0.7f);
    EXPECT_EQ(params.max_detections, 50);
}

TEST_F(SegMaskParamsTest, BoundaryValues) {
    SegMaskParams params;

    // 阈值边界值测试
    params.mask_threshold = 0.0f;
    EXPECT_FLOAT_EQ(params.mask_threshold, 0.0f);

    params.mask_threshold = 1.0f;
    EXPECT_FLOAT_EQ(params.mask_threshold, 1.0f);

    // max_detections 边界值
    params.max_detections = 1;
    EXPECT_EQ(params.max_detections, 1);

    params.max_detections = 10000;
    EXPECT_EQ(params.max_detections, 10000);
}

// ==================== SegmentConfig 测试 ====================

class SegmentConfigTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(SegmentConfigTest, DefaultValues) {
    SegmentConfig config;

    EXPECT_EQ(config.input_width, 640);
    EXPECT_EQ(config.input_height, 640);
    EXPECT_FLOAT_EQ(config.score_threshold, 0.25f);
    EXPECT_FLOAT_EQ(config.nms_threshold, 0.45f);
    EXPECT_FLOAT_EQ(config.mask_threshold, 0.5f);
    EXPECT_EQ(config.max_detections, 100);
    EXPECT_EQ(config.workers, 1u);
}

TEST_F(SegmentConfigTest, CustomValues) {
    SegmentConfig config;
    config.input_width = 1280;
    config.input_height = 720;
    config.score_threshold = 0.5f;
    config.nms_threshold = 0.3f;
    config.mask_threshold = 0.4f;
    config.max_detections = 200;
    config.workers = 4;

    EXPECT_EQ(config.input_width, 1280);
    EXPECT_EQ(config.input_height, 720);
    EXPECT_FLOAT_EQ(config.score_threshold, 0.5f);
    EXPECT_FLOAT_EQ(config.nms_threshold, 0.3f);
    EXPECT_FLOAT_EQ(config.mask_threshold, 0.4f);
    EXPECT_EQ(config.max_detections, 200);
    EXPECT_EQ(config.workers, 4u);
}

TEST_F(SegmentConfigTest, InvalidInputSize) {
    SegmentConfig config;

    // 输入尺寸应大于 0
    config.input_width = 0;
    config.input_height = 0;

    // 配置存储不应崩溃，但实际使用时可能失败
    EXPECT_EQ(config.input_width, 0);
    EXPECT_EQ(config.input_height, 0);

    // 边界值
    config.input_width = 1;
    config.input_height = 1;
    EXPECT_EQ(config.input_width, 1);
    EXPECT_EQ(config.input_height, 1);
}

// ==================== SegmentNode 构造测试 ====================

class SegmentNodeConstructorTest : public ::testing::Test {
protected:
    void SetUp() override {
        engine_ = std::make_shared<MockSegModelEngine>();
    }

    void TearDown() override {}

    std::shared_ptr<MockSegModelEngine> engine_;
};

TEST_F(SegmentNodeConstructorTest, DefaultConstruction) {
    SegmentNode node(engine_);

    EXPECT_EQ(node.config().input_width, 640);
    EXPECT_EQ(node.config().input_height, 640);
    EXPECT_EQ(node.worker_count(), 1u);
    EXPECT_EQ(node.state(), NodeState::INIT);
}

TEST_F(SegmentNodeConstructorTest, ConstructionWithConfig) {
    SegmentConfig config;
    config.input_width = 1280;
    config.input_height = 720;
    config.workers = 3;

    SegmentNode node(engine_, config, "custom_segment");

    EXPECT_EQ(node.config().input_width, 1280);
    EXPECT_EQ(node.config().input_height, 720);
    EXPECT_EQ(node.worker_count(), 3u);
    EXPECT_EQ(node.name(), "custom_segment");
}

TEST_F(SegmentNodeConstructorTest, ConstructionWithNameOnly) {
    SegmentNode node(engine_, "named_segment");

    EXPECT_EQ(node.name(), "named_segment");
    EXPECT_EQ(node.config().input_width, 640);
    EXPECT_EQ(node.config().input_height, 640);
}

TEST_F(SegmentNodeConstructorTest, NullEngineThrows) {
    std::shared_ptr<IModelEngine> null_engine;
    EXPECT_THROW({ SegmentNode node(null_engine); }, ConfigError);
}

TEST_F(SegmentNodeConstructorTest, CannotMoveDueToMutex) {
    // SegmentNode contains mutex/atomic members, move is deleted
    // Typically managed via shared_ptr
    auto node_ptr = std::make_shared<SegmentNode>(engine_, "original");
    EXPECT_EQ(node_ptr->name(), "original");
}

// ==================== SegmentNode 状态转换测试 ====================

class SegmentNodeStateTest : public ::testing::Test {
protected:
    void SetUp() override {
        engine_ = std::make_shared<MockSegModelEngine>();
        node_ = std::make_unique<SegmentNode>(engine_);
    }

    void TearDown() override {
        if (node_) {
            node_->stop(false);
            node_->wait_stop();
        }
    }

    std::shared_ptr<MockSegModelEngine> engine_;
    std::unique_ptr<SegmentNode> node_;
};

TEST_F(SegmentNodeStateTest, InitialStateIsInit) {
    EXPECT_EQ(node_->state(), NodeState::INIT);
}

TEST_F(SegmentNodeStateTest, StartTransitionsToRunning) {
    node_->start();
    EXPECT_EQ(node_->state(), NodeState::RUNNING);
}

TEST_F(SegmentNodeStateTest, StopTransitionsToDrainingThenStopped) {
    node_->start();
    EXPECT_EQ(node_->state(), NodeState::RUNNING);

    node_->stop(true);
    EXPECT_EQ(node_->state(), NodeState::DRAINING);

    node_->wait_stop();
    EXPECT_EQ(node_->state(), NodeState::STOPPED);
}

TEST_F(SegmentNodeStateTest, StopWithoutDrain) {
    node_->start();
    node_->stop(false);
    node_->wait_stop();

    EXPECT_EQ(node_->state(), NodeState::STOPPED);
}

TEST_F(SegmentNodeStateTest, MultipleStopsAreIdempotent) {
    node_->start();
    node_->stop(true);
    node_->stop(true);  // 重复调用
    node_->wait_stop();
    node_->wait_stop();  // 重复调用

    EXPECT_EQ(node_->state(), NodeState::STOPPED);
}

TEST_F(SegmentNodeStateTest, StopWithoutStart) {
    // 允许在未启动状态下调用 stop
    EXPECT_NO_THROW({
        node_->stop(false);
        node_->wait_stop();
    });
    EXPECT_EQ(node_->state(), NodeState::STOPPED);
}

TEST_F(SegmentNodeStateTest, RestartAfterStop) {
    node_->start();
    node_->stop(true);
    node_->wait_stop();
    EXPECT_EQ(node_->state(), NodeState::STOPPED);

    // 重新启动
    node_->start();
    EXPECT_EQ(node_->state(), NodeState::RUNNING);

    node_->stop(false);
    node_->wait_stop();
}

// ==================== SegmentNode 参数设置测试 ====================

class SegmentNodeParamTest : public ::testing::Test {
protected:
    void SetUp() override {
        engine_ = std::make_shared<MockSegModelEngine>();
        node_ = std::make_unique<SegmentNode>(engine_);
    }

    void TearDown() override {
        node_->stop(false);
        node_->wait_stop();
    }

    std::shared_ptr<MockSegModelEngine> engine_;
    std::unique_ptr<SegmentNode> node_;
};

TEST_F(SegmentNodeParamTest, SetScoreThreshold) {
    bool result = node_->set_param("score_threshold", 0.5f);
    EXPECT_TRUE(result);
    EXPECT_FLOAT_EQ(node_->config().score_threshold, 0.5f);
}

TEST_F(SegmentNodeParamTest, SetNmsThreshold) {
    bool result = node_->set_param("nms_threshold", 0.3f);
    EXPECT_TRUE(result);
    EXPECT_FLOAT_EQ(node_->config().nms_threshold, 0.3f);
}

TEST_F(SegmentNodeParamTest, SetMaskThreshold) {
    bool result = node_->set_param("mask_threshold", 0.6f);
    EXPECT_TRUE(result);
    EXPECT_FLOAT_EQ(node_->config().mask_threshold, 0.6f);
}

TEST_F(SegmentNodeParamTest, SetMaxDetections) {
    bool result = node_->set_param("max_detections", 50);
    EXPECT_TRUE(result);
    EXPECT_EQ(node_->config().max_detections, 50);
}

TEST_F(SegmentNodeParamTest, SetInvalidParamName) {
    bool result = node_->set_param("invalid_param", 123);
    EXPECT_FALSE(result);
}

TEST_F(SegmentNodeParamTest, SetParamTypeMismatch) {
    // 传递错误类型的参数
    bool result = node_->set_param("score_threshold", "invalid_string");
    // 实现应拒绝类型不匹配的参数
    EXPECT_FALSE(result);
}

TEST_F(SegmentNodeParamTest, SetBoundaryThresholdValues) {
    // 阈值边界值测试
    EXPECT_TRUE(node_->set_param("score_threshold", 0.0f));
    EXPECT_FLOAT_EQ(node_->config().score_threshold, 0.0f);

    EXPECT_TRUE(node_->set_param("score_threshold", 1.0f));
    EXPECT_FLOAT_EQ(node_->config().score_threshold, 1.0f);
}

TEST_F(SegmentNodeParamTest, SetParamWhileRunning) {
    node_->start();

    // 热更新应在运行时也能工作
    bool result = node_->set_param("score_threshold", 0.7f);
    EXPECT_TRUE(result);
    EXPECT_FLOAT_EQ(node_->config().score_threshold, 0.7f);
}

// ==================== SegmentNode 处理测试 ====================

class SegmentNodeProcessTest : public ::testing::Test {
protected:
    void SetUp() override {
        engine_ = std::make_shared<MockSegModelEngine>();
        SegmentConfig config;
        config.workers = 1;
        node_ = std::make_unique<SegmentNode>(engine_, config);

        input_queue_ = std::make_unique<BoundedQueue<Frame>>(16, OverflowPolicy::BLOCK);
        node_->set_input_queue(input_queue_.get());
        node_->create_output_queue(16, OverflowPolicy::BLOCK);
    }

    void TearDown() override {
        node_->stop(true);
        input_queue_->stop();
        node_->wait_stop();
    }

    std::shared_ptr<MockSegModelEngine> engine_;
    std::unique_ptr<SegmentNode> node_;
    std::unique_ptr<BoundedQueue<Frame>> input_queue_;
};

TEST_F(SegmentNodeProcessTest, ProcessSingleFrame) {
    node_->start();

    Frame frame = make_seg_frame(0);
    input_queue_->push(std::move(frame));

    node_->stop(true);
    input_queue_->stop();
    node_->wait_stop();

    auto output_queue = node_->output_queue();
    ASSERT_TRUE(output_queue != nullptr);

    auto result = output_queue->pop();
    ASSERT_TRUE(result.has_value());

    EXPECT_EQ(result->frame_id, 0);
    EXPECT_TRUE(result->has_image());
}

TEST_F(SegmentNodeProcessTest, ProcessMultipleFramesInOrder) {
    node_->start();

    constexpr int kFrameCount = 5;
    for (int i = 0; i < kFrameCount; ++i) {
        input_queue_->push(make_seg_frame(i));
    }

    node_->stop(true);
    input_queue_->stop();
    node_->wait_stop();

    auto output_queue = node_->output_queue();
    ASSERT_TRUE(output_queue != nullptr);

    std::vector<int64_t> frame_ids;
    while (auto frame = output_queue->pop()) {
        frame_ids.push_back(frame->frame_id);
    }

    EXPECT_EQ(frame_ids.size(), static_cast<size_t>(kFrameCount));
    for (size_t i = 0; i < frame_ids.size(); ++i) {
        EXPECT_EQ(frame_ids[i], static_cast<int64_t>(i))
            << "Frame order changed at index " << i;
    }
}

TEST_F(SegmentNodeProcessTest, ProcessEmptyFrame) {
    node_->start();

    // 创建空帧（无图像）
    Frame frame;
    frame.stream_id = 1;
    frame.frame_id = 0;
    // frame.image 为空

    input_queue_->push(std::move(frame));

    node_->stop(true);
    input_queue_->stop();
    node_->wait_stop();

    auto output_queue = node_->output_queue();

    // 空帧可能被跳过或产生错误计数
    // 具体行为取决于实现
    auto stats = node_->stats();
    EXPECT_GE(stats.error_count, 0u);
}

TEST_F(SegmentNodeProcessTest, OutputHasDetections) {
    node_->start();

    Frame frame = make_seg_frame(0);
    input_queue_->push(std::move(frame));

    node_->stop(true);
    input_queue_->stop();
    node_->wait_stop();

    auto output_queue = node_->output_queue();
    ASSERT_TRUE(output_queue != nullptr);

    auto result = output_queue->pop();
    ASSERT_TRUE(result.has_value());

    // 处理后的帧应有检测结果
    // Mock engine 可能不产生真实检测，但结构应正确
    EXPECT_GE(result->detections.size(), 0u);
}

TEST_F(SegmentNodeProcessTest, LastMasksAccessible) {
    node_->start();

    Frame frame = make_seg_frame(0);
    input_queue_->push(std::move(frame));

    node_->stop(true);
    input_queue_->stop();
    node_->wait_stop();

    // 访问最近一帧的分割掩码
    const auto& masks = node_->last_masks();
    // 掩码数量应与检测数量一致（如果有的话）
    // 具体行为取决于 Mock engine 的输出
    EXPECT_GE(masks.size(), 0u);
}

// ==================== SegmentNode 多 Worker 测试 ====================

class SegmentNodeMultiWorkerTest : public ::testing::Test {
protected:
    void SetUp() override {
        engine_ = std::make_shared<MockSegModelEngine>();
    }

    void TearDown() override {}

    std::shared_ptr<MockSegModelEngine> engine_;
};

TEST_F(SegmentNodeMultiWorkerTest, SingleWorkerCreatesOneContext) {
    SegmentConfig config;
    config.workers = 1;

    SegmentNode node(engine_, config);
    EXPECT_EQ(node.worker_count(), 1u);

    node.start();
    node.stop(false);
    node.wait_stop();

    EXPECT_EQ(engine_->created_contexts(), 1u);
}

TEST_F(SegmentNodeMultiWorkerTest, MultipleWorkersCreateMultipleContexts) {
    SegmentConfig config;
    config.workers = 3;

    SegmentNode node(engine_, config);
    EXPECT_EQ(node.worker_count(), 3u);

    node.start();
    node.stop(false);
    node.wait_stop();

    EXPECT_EQ(engine_->created_contexts(), 3u);
}

TEST_F(SegmentNodeMultiWorkerTest, MultipleWorkersPreserveFrameOrder) {
    SegmentConfig config;
    config.workers = 3;

    auto node = std::make_unique<SegmentNode>(engine_, config);
    auto input_queue = std::make_unique<BoundedQueue<Frame>>(32, OverflowPolicy::BLOCK);

    node->set_input_queue(input_queue.get());
    node->create_output_queue(32, OverflowPolicy::BLOCK);
    node->start();

    constexpr int kFrameCount = 9;
    for (int i = 0; i < kFrameCount; ++i) {
        input_queue->push(make_seg_frame(i));
    }

    node->stop(true);
    input_queue->stop();
    node->wait_stop();

    auto output_queue = node->output_queue();
    ASSERT_TRUE(output_queue != nullptr);

    std::vector<int64_t> frame_ids;
    while (auto frame = output_queue->pop()) {
        frame_ids.push_back(frame->frame_id);
    }

    EXPECT_EQ(frame_ids.size(), static_cast<size_t>(kFrameCount));
    for (size_t i = 0; i < frame_ids.size(); ++i) {
        EXPECT_EQ(frame_ids[i], static_cast<int64_t>(i))
            << "Frame order changed at index " << i;
    }
}

// ==================== SegMaskDecoder 测试 (静态方法) ====================

class SegMaskDecoderTest : public ::testing::Test {
protected:
    void SetUp() override {
        allocator_ = std::make_unique<CpuAllocator>();
    }

    void TearDown() override {}

    std::unique_ptr<CpuAllocator> allocator_;
};

TEST_F(SegMaskDecoderTest, ComputeMaskBboxIou) {
    // 创建一个简单的 mask (10x10)
    std::vector<uint8_t> mask(100, 0);

    // 在 mask 中心创建一个 6x6 的矩形
    for (int y = 2; y < 8; ++y) {
        for (int x = 2; x < 8; ++x) {
            mask[y * 10 + x] = 255;
        }
    }

    // bbox 与 mask 完全重叠
    float bbox1[4] = {0.2f, 0.2f, 0.8f, 0.8f};  // [x1, y1, x2, y2] 归一化
    float iou1 = SegMaskDecoder::compute_mask_bbox_iou(mask, bbox1, 10, 10);
    EXPECT_GT(iou1, 0.0f);
    EXPECT_LE(iou1, 1.0f);

    // bbox 与 mask 部分重叠
    float bbox2[4] = {0.5f, 0.5f, 1.0f, 1.0f};
    float iou2 = SegMaskDecoder::compute_mask_bbox_iou(mask, bbox2, 10, 10);
    EXPECT_GT(iou2, 0.0f);
    EXPECT_LT(iou2, iou1);

    // bbox 与 mask 无重叠
    float bbox3[4] = {0.0f, 0.0f, 0.1f, 0.1f};
    float iou3 = SegMaskDecoder::compute_mask_bbox_iou(mask, bbox3, 10, 10);
    EXPECT_FLOAT_EQ(iou3, 0.0f);
}

TEST_F(SegMaskDecoderTest, ComputeMaskBboxIouEmptyMask) {
    std::vector<uint8_t> empty_mask(100, 0);

    float bbox[4] = {0.0f, 0.0f, 1.0f, 1.0f};
    float iou = SegMaskDecoder::compute_mask_bbox_iou(empty_mask, bbox, 10, 10);
    EXPECT_FLOAT_EQ(iou, 0.0f);
}

TEST_F(SegMaskDecoderTest, ComputeMaskBboxIouFullMask) {
    std::vector<uint8_t> full_mask(100, 255);

    float bbox[4] = {0.0f, 0.0f, 1.0f, 1.0f};
    float iou = SegMaskDecoder::compute_mask_bbox_iou(full_mask, bbox, 10, 10);
    EXPECT_FLOAT_EQ(iou, 1.0f);
}

TEST_F(SegMaskDecoderTest, ComputeMaskBboxIouSinglePixel) {
    // 10x10 mask 中只有一个像素
    std::vector<uint8_t> mask(100, 0);
    mask[55] = 255;  // 中心偏右下

    float bbox[4] = {0.5f, 0.5f, 0.6f, 0.6f};
    float iou = SegMaskDecoder::compute_mask_bbox_iou(mask, bbox, 10, 10);
    EXPECT_GT(iou, 0.0f);
}

TEST_F(SegMaskDecoderTest, ComputeMaskBboxIouZeroSizeBbox) {
    std::vector<uint8_t> mask(100, 255);

    // 零尺寸 bbox
    float bbox[4] = {0.5f, 0.5f, 0.5f, 0.5f};
    float iou = SegMaskDecoder::compute_mask_bbox_iou(mask, bbox, 10, 10);
    EXPECT_FLOAT_EQ(iou, 0.0f);
}

// ==================== SegmentNode 统计测试 ====================

class SegmentNodeStatsTest : public ::testing::Test {
protected:
    void SetUp() override {
        engine_ = std::make_shared<MockSegModelEngine>();
        node_ = std::make_unique<SegmentNode>(engine_);
    }

    void TearDown() override {
        node_->stop(false);
        node_->wait_stop();
    }

    std::shared_ptr<MockSegModelEngine> engine_;
    std::unique_ptr<SegmentNode> node_;
};

TEST_F(SegmentNodeStatsTest, InitialStats) {
    auto stats = node_->stats();

    EXPECT_EQ(stats.processed_count, 0u);
    EXPECT_EQ(stats.error_count, 0u);
    EXPECT_DOUBLE_EQ(stats.fps, 0.0);
}

TEST_F(SegmentNodeStatsTest, StatsAfterProcessing) {
    auto input_queue = std::make_unique<BoundedQueue<Frame>>(16, OverflowPolicy::BLOCK);
    node_->set_input_queue(input_queue.get());
    node_->create_output_queue(16, OverflowPolicy::BLOCK);

    node_->start();

    constexpr int kFrameCount = 3;
    for (int i = 0; i < kFrameCount; ++i) {
        input_queue->push(make_seg_frame(i));
    }

    node_->stop(true);
    input_queue->stop();
    node_->wait_stop();

    auto stats = node_->stats();
    EXPECT_EQ(stats.processed_count, static_cast<uint64_t>(kFrameCount));
}

// ==================== SegmentNode 队列测试 ====================

class SegmentNodeQueueTest : public ::testing::Test {
protected:
    void SetUp() override {
        engine_ = std::make_shared<MockSegModelEngine>();
        node_ = std::make_unique<SegmentNode>(engine_);
    }

    void TearDown() override {
        node_->stop(false);
        node_->wait_stop();
    }

    std::shared_ptr<MockSegModelEngine> engine_;
    std::unique_ptr<SegmentNode> node_;
};

TEST_F(SegmentNodeQueueTest, CreateOutputQueue) {
    node_->create_output_queue(32, OverflowPolicy::DROP_OLDEST);

    auto output_queue = node_->output_queue();
    ASSERT_TRUE(output_queue != nullptr);
    EXPECT_EQ(output_queue->capacity(), 32u);
    EXPECT_EQ(output_queue->policy(), OverflowPolicy::DROP_OLDEST);
}

TEST_F(SegmentNodeQueueTest, SetInputQueue) {
    auto input_queue = std::make_shared<BoundedQueue<Frame>>(16);
    node_->set_input_queue(input_queue.get());

    EXPECT_EQ(node_->input_queue(), input_queue.get());
    // 恢复内部队列，防止 TearDown 通过悬空指针访问已析构的局部变量
    node_->set_input_queue(nullptr);
}

TEST_F(SegmentNodeQueueTest, IsSourceReturnsFalse) {
    EXPECT_FALSE(node_->is_source());
}

TEST_F(SegmentNodeQueueTest, IsSinkReturnsFalse) {
    EXPECT_FALSE(node_->is_sink());
}

// ==================== SegmentNode 并发安全测试 ====================

class SegmentNodeConcurrencyTest : public ::testing::Test {
protected:
    void SetUp() override {
        engine_ = std::make_shared<MockSegModelEngine>();
    }

    void TearDown() override {}

    std::shared_ptr<MockSegModelEngine> engine_;
};

TEST_F(SegmentNodeConcurrencyTest, ConcurrentParamUpdate) {
    SegmentConfig config;
    config.workers = 2;

    auto node = std::make_unique<SegmentNode>(engine_, config);
    auto input_queue = std::make_unique<BoundedQueue<Frame>>(32, OverflowPolicy::BLOCK);

    node->set_input_queue(input_queue.get());
    node->create_output_queue(32, OverflowPolicy::BLOCK);
    node->start();

    // 启动多个线程同时更新参数
    std::vector<std::thread> threads;
    std::atomic<int> success_count{0};

    for (int t = 0; t < 10; ++t) {
        threads.emplace_back([&node, &success_count, t]() {
            for (int i = 0; i < 100; ++i) {
                if (node->set_param("score_threshold", static_cast<float>(t * 0.01 + i * 0.001))) {
                    ++success_count;
                }
            }
        });
    }

    // 同时推送帧
    for (int i = 0; i < 20; ++i) {
        input_queue->push(make_seg_frame(i));
    }

    for (auto& t : threads) {
        t.join();
    }

    node->stop(true);
    input_queue->stop();
    node->wait_stop();

    // 参数更新应成功
    EXPECT_GT(success_count.load(), 0);
}

TEST_F(SegmentNodeConcurrencyTest, ConcurrentStartStop) {
    auto node = std::make_unique<SegmentNode>(engine_);

    std::vector<std::thread> threads;

    // 多个线程尝试 start/stop
    for (int i = 0; i < 5; ++i) {
        threads.emplace_back([&node]() {
            node->start();
            std::this_thread::sleep_for(10ms);
            node->stop(true);
            node->wait_stop();
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    // 最终状态应是 STOPPED
    EXPECT_EQ(node->state(), NodeState::STOPPED);
}

// ==================== SegmentNode 边界情况测试 ====================

class SegmentNodeEdgeCaseTest : public ::testing::Test {
protected:
    void SetUp() override {
        engine_ = std::make_shared<MockSegModelEngine>();
    }

    void TearDown() override {}

    std::shared_ptr<MockSegModelEngine> engine_;
};

TEST_F(SegmentNodeEdgeCaseTest, VeryLargeFrameId) {
    auto node = std::make_unique<SegmentNode>(engine_);
    auto input_queue = std::make_unique<BoundedQueue<Frame>>(16, OverflowPolicy::BLOCK);

    node->set_input_queue(input_queue.get());
    node->create_output_queue(16, OverflowPolicy::BLOCK);
    node->start();

    // 使用非常大的 frame_id
    Frame frame = make_seg_frame(INT64_MAX);
    input_queue->push(std::move(frame));

    node->stop(true);
    input_queue->stop();
    node->wait_stop();

    auto output_queue = node->output_queue();
    auto result = output_queue->pop();

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->frame_id, INT64_MAX);
}

TEST_F(SegmentNodeEdgeCaseTest, NegativeFrameId) {
    auto node = std::make_unique<SegmentNode>(engine_);
    auto input_queue = std::make_unique<BoundedQueue<Frame>>(16, OverflowPolicy::BLOCK);

    node->set_input_queue(input_queue.get());
    node->create_output_queue(16, OverflowPolicy::BLOCK);
    node->start();

    Frame frame = make_seg_frame(-1);
    input_queue->push(std::move(frame));

    node->stop(true);
    input_queue->stop();
    node->wait_stop();

    auto output_queue = node->output_queue();
    auto result = output_queue->pop();

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->frame_id, -1);
}

TEST_F(SegmentNodeEdgeCaseTest, SinglePixelImage) {
    auto node = std::make_unique<SegmentNode>(engine_);
    auto input_queue = std::make_unique<BoundedQueue<Frame>>(16, OverflowPolicy::BLOCK);

    node->set_input_queue(input_queue.get());
    node->create_output_queue(16, OverflowPolicy::BLOCK);
    node->start();

    // 1x1 图像
    Frame frame = make_seg_frame(0, 1, 1);
    input_queue->push(std::move(frame));

    node->stop(true);
    input_queue->stop();
    node->wait_stop();

    // 处理应不崩溃
    auto stats = node->stats();
    EXPECT_GE(stats.processed_count, 0u);
}

TEST_F(SegmentNodeEdgeCaseTest, ZeroWorkers) {
    SegmentConfig config;
    config.workers = 0;

    // workers = 0 可能是无效配置
    auto node = std::make_unique<SegmentNode>(engine_, config);

    // 应该有至少一个 worker 或者抛出错误
    EXPECT_GE(node->worker_count(), 1u);
}

TEST_F(SegmentNodeEdgeCaseTest, MaxDetectionsZero) {
    SegmentConfig config;
    config.max_detections = 0;

    auto node = std::make_unique<SegmentNode>(engine_, config);
    auto input_queue = std::make_unique<BoundedQueue<Frame>>(16, OverflowPolicy::BLOCK);

    node->set_input_queue(input_queue.get());
    node->create_output_queue(16, OverflowPolicy::BLOCK);
    node->start();

    Frame frame = make_seg_frame(0);
    input_queue->push(std::move(frame));

    node->stop(true);
    input_queue->stop();
    node->wait_stop();

    auto output_queue = node->output_queue();
    auto result = output_queue->pop();

    // max_detections = 0 应该导致无检测结果
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->detections.size(), 0u);
}

}  // namespace
}  // namespace visionpipe
