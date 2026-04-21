# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 在此仓库中工作时提供指导。

## 项目概述

VisionPipe-py 是一个视频 AI 推理框架，核心由 C++（CUDA/TensorRT）驱动，通过 Python 绑定提供灵活接口。采用基于 DAG 的管道架构进行 GPU 加速视频处理。

**技术栈：**
- C++17 + CUDA 11.8+ + TensorRT 8.6+
- Python 3.10+ + nanobind 绑定
- CMake 3.20+ 构建
- uv 管理 Python 包

## 构建命令

```bash
# 配置并构建 C++
cmake -B build && cmake --build build

# 构建 Python 扩展（nanobind）
cmake --build build --target visionpipe_python

# 运行 C++ 测试（需要 GPU）
ctest --test-dir build

# 运行指定 C++ 测试
ctest --test-dir build -R <test_name>
```

## Python 开发

```bash
# 创建虚拟环境并安装依赖
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"

# 运行 Python 测试（需要 GPU）
uv run pytest

# 运行指定测试文件
uv run pytest tests/unit/python/test_bindings.py -v
```

## 代码风格

```bash
# C++ 格式化
find src -name "*.h" -o -name "*.cpp" | xargs clang-format -i

# Python 代码检查和类型检查
uv run ruff check python/
uv run ruff format python/
uv run mypy python/
```

## 架构

```
┌─────────────────────────────────────────────────────────┐
│                    Python 层                              │
│  Pipeline DSL | 业务节点 | 管理 API (aiohttp)             │
├─────────────────────────────────────────────────────────┤
│                  nanobind 绑定层                          │
├─────────────────────────────────────────────────────────┤
│                    C++ 核心层                             │
│  PipelineManager | ModelRegistry | Pipeline (DAG)       │
│  BoundedQueue | NodeBase | InferNode                     │
├─────────────────────────────────────────────────────────┤
│                    HAL 硬件抽象层                         │
│  IModelEngine | IExecContext | IAllocator                │
│  实现: TrtEngine, CudaAllocator                           │
└─────────────────────────────────────────────────────────┘
```

**关键组件：**

| 组件 | 位置 | 用途 |
|------|------|------|
| `PipelineManager` | `src/core/` | Pipeline 的创建/启动/停止/销毁 |
| `ModelRegistry` | `src/core/` | 模型去重（SHA-256）、引用计数、TTL 清理 |
| `BoundedQueue<T>` | `src/core/` | 线程安全有界队列，支持 DROP_OLDEST/BLOCK 策略 |
| `InferNode` | `src/core/` | 推理节点，支持 parallel_workers 并行 |
| `IModelEngine` | `src/hal/` | HAL 接口，用于 GPU 引擎（TensorRT 为主） |

**异常层次：**
- `VisionPipeError`（基类）
  - `ConfigError`, `NotFoundError`
  - `CudaError`, `ModelLoadError`, `InferError`

## 包管理

- **Python**: uv (pyproject.toml) - `uv pip install -e ".[dev]"`
- **C++ 重依赖**（CUDA、TensorRT、FFmpeg、OpenCV）：系统包管理器安装
- **C++ 轻依赖**（spdlog、nlohmann-json、googletest、nanobind）：CMake FetchContent

## GPU 环境

⚠️ 所有测试（单元测试、集成测试、E2E 测试）均需要真实的 NVIDIA GPU。无 mock GPU 环境。

测试数据位置：`tests/data/`（通过 `tests/data/download_test_assets.sh` 下载）

## 阶段门禁

每个开发阶段必须满足：
1. 所有测试通过：`ctest --test-dir build` 和 `uv run pytest`
2. 核心模块覆盖率 >90%，整体 >80%
3. 代码风格检查通过
4. 无内存泄漏（Valgrind/ASAN）