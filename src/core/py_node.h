#pragma once

#include <functional>
#include <string>

#include "core/frame.h"
#include "core/node_base.h"

namespace visionpipe {

/// @brief Python 可扩展节点
///
/// 持有 process(Frame&) 回调，C++ 工作线程通过 GIL-safe 方式调用。
/// Python 端异常被捕获、记录日志，不会 crash C++ 线程。
class PyNode : public NodeBase {
public:
    /// @brief 回调函数类型：接受 Frame 引用，就地修改
    using ProcessFn = std::function<void(Frame&)>;

    /// @brief 构造函数
    /// @param process_fn process 回调
    /// @param name 节点名称
    explicit PyNode(ProcessFn process_fn, const std::string& name = "py_node");

    ~PyNode() override = default;

    void process(Frame& frame) override;

private:
    ProcessFn process_fn_;
};

}  // namespace visionpipe
