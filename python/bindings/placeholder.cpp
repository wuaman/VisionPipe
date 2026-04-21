// placeholder.cpp
// Python 绑定占位文件，任务 T0.1 验证 CMake 构建通过
// 后续任务 3.1 会添加真正的绑定代码

#include <nanobind/nanobind.h>

namespace nb = nanobind;

NB_MODULE(visionpipe_python, m) {
    // 占位模块定义
    m.doc() = "VisionPipe Python bindings (placeholder)";

    // 后续任务会在此添加：
    // - Pipeline 绑定
    // - PipelineManager 绑定
    // - Frame 绑定
    // - Detection/Track 绑定
    // - 各节点绑定
}
