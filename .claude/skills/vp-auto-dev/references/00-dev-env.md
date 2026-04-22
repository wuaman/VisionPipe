## 0. 开发环境

### 0.1 必需依赖

| 组件 | 版本要求 | 验证命令 |
|------|----------|----------|
| CUDA Toolkit | >=11.8 | `nvcc --version` |
| cuDNN | >=8.6 | `cat /usr/local/cuda/include/cudnn_version.h` |
| TensorRT | >=8.6 | `trtexec --version` |
| Python | >=3.10 | `python3 --version` |
| CMake | >=3.20 | `cmake --version` |
| GCC | >=9.0 | `g++ --version` |
| uv | 最新版 | `uv --version` |
| clang-format | >=14 | `clang-format --version` |

### 0.2 环境配置

**Python 包管理（uv）**

项目使用 [uv](https://github.com/astral-sh/uv) 管理 Python 依赖，配置文件为 `pyproject.toml`。

```bash
# 安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 创建虚拟环境并安装依赖
uv venv
uv pip install -e ".[dev]"
```

**C++ 包管理**

采用 **系统包管理器 + CMake FetchContent** 混合方案：

| 依赖类型 | 管理方式 | 示例 |
|----------|----------|------|
| 重依赖（CUDA、TensorRT、OpenCV） | 系统包管理器或源码编译 | OpenCV 需源码编译，开启 `-DWITH_CUDA=ON -DWITH_NVCUVID=ON`（系统 `libopencv-dev` 通常不含 CUDA 模块） |
| 轻依赖（spdlog、nlohmann-json、googletest、nanobind） | CMake FetchContent | 自动下载到 `build/_deps/` |

### 0.3 GPU 环境要求

⚠️ **重要**：本项目的所有测试（单元测试、集成测试、E2E 测试）均需在真实 GPU 环境运行。

- 开发机必须配备 NVIDIA GPU（推荐 RTX 3090 或更高）
- CUDA 驱动版本 >= 525.60.13
- 确保 `nvidia-smi` 正常输出
- 测试执行前必须通过 GPU 环境验证：`python scripts/check_gpu_env.py`

### 0.4 代码风格

| 语言 | 工具 | 配置文件 |
|------|------|----------|
| C++ | clang-format | `.clang-format`（基于 Google 风格） |
| Python | ruff + mypy | `pyproject.toml` |

```bash
# 格式化检查（提交前运行）
find src -name "*.h" -o -name "*.cpp" | xargs clang-format -i
uv run ruff format python/
uv run mypy python/
```

### 0.5 测试数据

| 资源 | 路径 | 说明 |
|------|------|------|
| YOLOv8 ONNX | `models/yolov8/yolov8n.onnx` | Ultralytics 官方下载 |
| YOLOv8 TRT Engine | `models/yolov8/yolov8n.engine` | 由 `convert.sh` 生成，不纳入 git |
| 测试视频 | `tests/data/sample_100frames.mp4` | 1080p H.264，100 帧 |
| COCO 子集 | `tests/data/coco_val_subset/` | 100 张验证集图片 |

资源获取脚本：`tests/data/download_test_assets.sh`（首次构建时自动运行）

---
