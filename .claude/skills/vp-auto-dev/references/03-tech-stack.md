## 3. 技术选型

### 3.1 核心依赖

| 层次 | 组件 | 版本要求 | 用途 |
|---|---|---|---|
| 推理 | TensorRT | >=8.6 | 模型编译与 GPU 推理 |
| 编解码（一期） | `cv::cudacodec::VideoReader`（NVDEC 硬解） | OpenCV >=4.7（CUDA + NVCUVID build） | GPU 直接解码，零拷贝，快速验证全链路 |
| 编解码（二期） | ICodec HAL 抽象 + 平台专属实现 | 各平台 SDK | 跨平台硬件解码（NVDEC/DVPP/MPP） |
| 编解码（CPU fallback） | OpenCV `VideoCapture` (CPU 路径) | OpenCV >=4.7 | 无 GPU 解码器时的 CPU 软解码 fallback |
| 图像处理 | OpenCV | >=4.7（CUDA + NVCUVID build） | 预处理、可视化、一期视频解码 |
| GPU 计算 | CUDA | >=11.8 | CUDA stream 管理、内存分配 |
| 追踪算法 | ByteTrack（内置 C++ 实现） | — | 多目标追踪 |
| Python 绑定 | nanobind | >=1.9 | C++ ↔ Python 互调，含 `gil_scoped_release` |

### 3.2 构建与工程

| 维度 | 选型 | 说明 |
|---|---|---|
| C++ 标准 | C++17 | `optional`/`variant`/`filesystem`/结构化绑定 |
| 构建系统 | CMake 3.20+ | CUDA 原生支持，FetchContent 管理轻依赖 |
| 包管理 | 系统包 + FetchContent | 重依赖系统安装；spdlog/nlohmann-json/cpp-httplib 由 FetchContent 拉取 |
| Python 版本 | >=3.10 | match 语法，类型注解完善 |
| C++ 日志 | spdlog | 异步模式，支持 JSON 结构化输出 |
| Python 日志 | logging + structlog | 与 spdlog 统一输出格式 |
| 序列化 | nlohmann/json (C++) + pydantic v2 (Python) | 管理 API 数据格式、Pipeline YAML 导出 |
| WebRTC | libdatachannel | 轻量 C++ 原生，无 Chromium 依赖 |
| 管理 API | aiohttp（协程，同进程） | 暴露 Pipeline CRUD 和健康检查接口 |
| 测试（C++） | Google Test + Google Mock | 单元测试 / 集成测试 |
| 测试（Python） | pytest + pytest-asyncio | Python 层及 E2E 测试 |
| CI | GitHub Actions | Ubuntu 22.04，CUDA mock 环境单元测试，实机集成测试 |
| 目标 OS | Linux（Ubuntu 22.04/24.04） | 服务器端及 Jetson 均为 Linux |
| License | Apache 2.0 | 依赖均兼容（避免 GPL FFmpeg 选项） |

### 3.3 一期验证模型

| 任务 | 模型 | 优先级 | 交付物 |
|---|---|---|---|
| 目标检测 | YOLOv8 / YOLOv11 | P0 | ONNX→TRT 转换脚本 + 后处理插件 + benchmark |
| 图像分类 | ResNet50 / EfficientNet-B0 / ShuffleNetV2 | P0 | 同上 |
| 实例分割 | YOLOv8-seg | P1 | 同上 |
| 目标追踪 | ByteTrack | P1 | 内置 C++ 实现，无需 GPU |
| 语义分割 | — | 二期 | — |
| 姿态估计 | — | 二期 | — |

---
