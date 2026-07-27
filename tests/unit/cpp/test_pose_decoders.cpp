// 关键点解码器单元测试（纯 CPU，无需 GPU 推理）
#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "core/tensor.h"
#include "nodes/infer/post/simcc_decoder.h"
#include "nodes/infer/post/yolo_pose_decoder.h"
#include "nodes/infer/pre/letterbox_resize.h"

namespace visionpipe {

// ==================== SimccDecoder ====================

class SimccDecoderTest : public ::testing::Test {
protected:
    // 构造 K 个关键点的 simcc 向量，argmax 放在指定 bin
    static std::vector<float> make_simcc(int num_kpts, int bins,
                                         const std::vector<int>& peak_bins,
                                         float peak_val) {
        std::vector<float> data(static_cast<size_t>(num_kpts) * bins, 0.01f);
        for (int k = 0; k < num_kpts; ++k) {
            data[static_cast<size_t>(k) * bins + peak_bins[k]] = peak_val;
        }
        return data;
    }
};

TEST_F(SimccDecoderTest, DecodesArgmaxDividedBySplitRatio) {
    const int K = 3, x_bins = 384, y_bins = 512;
    auto sx = make_simcc(K, x_bins, {100, 200, 300}, 0.9f);
    auto sy = make_simcc(K, y_bins, {50, 250, 450}, 0.7f);

    std::vector<Keypoint> kpts;
    SimccDecoder::decode(sx.data(), sy.data(), K, x_bins, y_bins, 2.0f, kpts);

    ASSERT_EQ(kpts.size(), 3u);
    EXPECT_FLOAT_EQ(kpts[0].x, 50.0f);    // 100 / 2
    EXPECT_FLOAT_EQ(kpts[0].y, 25.0f);    // 50 / 2
    EXPECT_FLOAT_EQ(kpts[1].x, 100.0f);
    EXPECT_FLOAT_EQ(kpts[1].y, 125.0f);
    EXPECT_FLOAT_EQ(kpts[2].x, 150.0f);
    EXPECT_FLOAT_EQ(kpts[2].y, 225.0f);
    // score = 0.5 * (max_x + max_y)
    EXPECT_NEAR(kpts[0].score, 0.5f * (0.9f + 0.7f), 1e-5f);
}

TEST_F(SimccDecoderTest, NegativeResponseClampedToZeroScore) {
    const int K = 1, bins = 16;
    std::vector<float> sx(bins, -1.0f);
    std::vector<float> sy(bins, -1.0f);
    sx[4] = -0.5f;
    sy[8] = -0.5f;

    std::vector<Keypoint> kpts;
    SimccDecoder::decode(sx.data(), sy.data(), K, bins, bins, 2.0f, kpts);

    ASSERT_EQ(kpts.size(), 1u);
    EXPECT_FLOAT_EQ(kpts[0].x, 2.0f);
    EXPECT_FLOAT_EQ(kpts[0].y, 4.0f);
    EXPECT_FLOAT_EQ(kpts[0].score, 0.0f);
}

// ==================== YoloPoseDecoder ====================

class YoloPoseDecoderTest : public ::testing::Test {
protected:
    CpuAllocator allocator_;

    // 构造 [1, 5+K*3, num_anchors] 输出，anchor 0 为有效检测
    Tensor make_output(int num_kpts, int num_anchors,
                       float cx, float cy, float w, float h, float conf,
                       float kpt_x, float kpt_y, float kpt_score) {
        const int channels = 5 + num_kpts * 3;
        Tensor t({1, channels, num_anchors}, DataType::FLOAT32, &allocator_);
        float* data = static_cast<float*>(t.data);
        std::fill(data, data + t.numel(), 0.0f);

        data[0 * num_anchors + 0] = cx;
        data[1 * num_anchors + 0] = cy;
        data[2 * num_anchors + 0] = w;
        data[3 * num_anchors + 0] = h;
        data[4 * num_anchors + 0] = conf;
        for (int k = 0; k < num_kpts; ++k) {
            data[(5 + k * 3 + 0) * num_anchors + 0] = kpt_x;
            data[(5 + k * 3 + 1) * num_anchors + 0] = kpt_y;
            data[(5 + k * 3 + 2) * num_anchors + 0] = kpt_score;
        }
        return t;
    }
};

TEST_F(YoloPoseDecoderTest, DecodesDetectionAndKeypointsWithLetterboxUnmap) {
    // 原图 1280x720 → letterbox 640x640: scale=0.5, pad_y=(640-360)/2=140
    const int orig_w = 1280, orig_h = 720;
    auto lb = LetterboxResize::compute_params(orig_w, orig_h, 640, 640);
    ASSERT_FLOAT_EQ(lb.scale, 0.5f);
    ASSERT_EQ(lb.pad_x, 0);
    ASSERT_EQ(lb.pad_y, 140);

    // letterbox 空间: bbox 中心 (320, 320) 尺寸 100x80; kpt (320, 320)
    auto out = make_output(17, 100, 320, 320, 100, 80, 0.9f, 320, 320, 0.8f);

    YoloPoseParams params;
    params.score_threshold = 0.5f;
    std::vector<Detection> dets;
    std::vector<PoseResult> poses;
    YoloPoseDecoder::decode(out, dets, poses, params, lb, orig_w, orig_h);

    ASSERT_EQ(dets.size(), 1u);
    ASSERT_EQ(poses.size(), 1u);
    EXPECT_EQ(poses[0].detection_index, 0);
    ASSERT_EQ(poses[0].keypoints.size(), 17u);

    // letterbox (320,320) → 原图 ((320-0)/0.5, (320-140)/0.5) = (640, 360) → 归一化 (0.5, 0.5)
    EXPECT_NEAR(poses[0].keypoints[0].x, 0.5f, 1e-4f);
    EXPECT_NEAR(poses[0].keypoints[0].y, 0.5f, 1e-4f);
    EXPECT_FLOAT_EQ(poses[0].keypoints[0].score, 0.8f);

    // bbox: cx=320 w=100 → letterbox [270,370] → 原图 x [540,740] → 归一化 [0.4219, 0.5781]
    EXPECT_NEAR(dets[0].bbox[0], 540.0f / 1280.0f, 1e-4f);
    EXPECT_NEAR(dets[0].bbox[2], 740.0f / 1280.0f, 1e-4f);
    EXPECT_FLOAT_EQ(dets[0].confidence, 0.9f);
    EXPECT_EQ(dets[0].class_id, 0);
}

TEST_F(YoloPoseDecoderTest, FiltersBelowScoreThreshold) {
    auto lb = LetterboxResize::compute_params(640, 640, 640, 640);
    auto out = make_output(17, 100, 320, 320, 100, 80, 0.2f, 320, 320, 0.8f);

    YoloPoseParams params;
    params.score_threshold = 0.5f;
    std::vector<Detection> dets;
    std::vector<PoseResult> poses;
    YoloPoseDecoder::decode(out, dets, poses, params, lb, 640, 640);

    EXPECT_TRUE(dets.empty());
    EXPECT_TRUE(poses.empty());
}

TEST_F(YoloPoseDecoderTest, RejectsUnexpectedChannelCount) {
    auto lb = LetterboxResize::compute_params(640, 640, 640, 640);
    // K=17 但 params.num_keypoints=13 → 通道数不匹配
    auto out = make_output(17, 100, 320, 320, 100, 80, 0.9f, 320, 320, 0.8f);

    YoloPoseParams params;
    params.num_keypoints = 13;
    std::vector<Detection> dets;
    std::vector<PoseResult> poses;
    YoloPoseDecoder::decode(out, dets, poses, params, lb, 640, 640);

    EXPECT_TRUE(dets.empty());
}

TEST_F(YoloPoseDecoderTest, NmsSuppressesOverlappingCandidates) {
    auto lb = LetterboxResize::compute_params(640, 640, 640, 640);

    const int K = 17, num_anchors = 100;
    const int channels = 5 + K * 3;
    Tensor t({1, channels, num_anchors}, DataType::FLOAT32, &allocator_);
    float* data = static_cast<float*>(t.data);
    std::fill(data, data + t.numel(), 0.0f);
    // 两个高度重叠的候选
    for (int i = 0; i < 2; ++i) {
        data[0 * num_anchors + i] = 320.0f + i;  // cx 几乎相同
        data[1 * num_anchors + i] = 320.0f;
        data[2 * num_anchors + i] = 100.0f;
        data[3 * num_anchors + i] = 100.0f;
        data[4 * num_anchors + i] = 0.9f - 0.1f * i;
    }

    YoloPoseParams params;
    params.score_threshold = 0.5f;
    params.nms_threshold = 0.45f;
    std::vector<Detection> dets;
    std::vector<PoseResult> poses;
    YoloPoseDecoder::decode(t, dets, poses, params, lb, 640, 640);

    EXPECT_EQ(dets.size(), 1u);
    EXPECT_FLOAT_EQ(dets[0].confidence, 0.9f);
}

}  // namespace visionpipe
