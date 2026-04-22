#pragma once

#include <any>
#include <cstdint>
#include <memory>
#include <vector>
#include "core/tensor.h"

namespace visionpipe {

/// @brief 检测结果
struct Detection {
    float bbox[4];       ///< 边界框 [x1, y1, x2, y2]，归一化坐标
    int class_id = 0;    ///< 类别 ID
    float confidence = 0.0f;  ///< 置信度
    int64_t track_id = -1;    ///< 关联的追踪 ID，-1 表示未追踪

    /// @brief 获取边界框宽度
    float width() const { return bbox[2] - bbox[0]; }

    /// @brief 获取边界框高度
    float height() const { return bbox[3] - bbox[1]; }

    /// @brief 获取边界框面积
    float area() const { return width() * height(); }
};

/// @brief 追踪目标
struct Track {
    int64_t track_id = 0;    ///< 追踪 ID
    int class_id = 0;        ///< 类别 ID
    float bbox[4];           ///< 边界框 [x1, y1, x2, y2]
    int age = 0;             ///< 追踪年龄（帧数）
    float confidence = 0.0f;  ///< 置信度
};

/// @brief 帧数据结构
struct Frame {
    int64_t stream_id = 0;           ///< 流标识符，同一 pipeline 内唯一
    int64_t frame_id = 0;            ///< 全局单调递增，用于重排序
    int64_t pts_us = 0;              ///< 时间戳（微秒）
    Tensor image;                    ///< GPU/CPU tensor
    std::vector<Detection> detections;  ///< 检测结果
    std::vector<Track> tracks;          ///< 追踪结果
    std::any user_data;                 ///< 用户附加数据

    /// @brief 默认构造函数
    Frame() = default;

    /// @brief 析构函数
    ~Frame() = default;

    // 允许移动
    Frame(Frame&&) = default;
    Frame& operator=(Frame&&) = default;

    // 禁止拷贝（避免意外复制大张量）
    Frame(const Frame&) = delete;
    Frame& operator=(const Frame&) = delete;

    /// @brief 清空帧数据
    void clear() {
        stream_id = 0;
        frame_id = 0;
        pts_us = 0;
        image = Tensor();
        detections.clear();
        tracks.clear();
        user_data.reset();
    }

    /// @brief 是否有图像数据
    bool has_image() const { return image.valid(); }
};

/// @brief Frame 智能指针类型
using FramePtr = std::unique_ptr<Frame>;

}  // namespace visionpipe
