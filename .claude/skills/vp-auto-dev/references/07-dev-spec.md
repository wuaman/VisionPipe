## 7. 开发规范

### 7.1 错误处理策略

#### C++ 层

| 错误类型 | 处理方式 | 示例 |
|----------|----------|------|
| 可恢复的业务错误 | 返回 `std::optional` 或 `Result<T>` | `pop()` 返回空 |
| 配置/参数错误 | 抛出自定义异常 | `ConfigError` |
| GPU/CUDA 错误 | 抛出 `CudaError`，携带错误码 | `cudaMalloc` 失败 |
| 模型加载错误 | 抛出 `ModelLoadError` | engine 文件损坏 |
| 推理错误 | 抛出 `InferError` | 输入 shape 不匹配 |

**异常基类层次**：

```cpp
class VisionPipeError : public std::runtime_error {
public:
    explicit VisionPipeError(const std::string& msg) : std::runtime_error(msg) {}
};

class ConfigError : public VisionPipeError { /* ... */ };
class NotFoundError : public VisionPipeError { /* ... */ };
class CudaError : public VisionPipeError { /* ... */ };
class ModelLoadError : public VisionPipeError { /* ... */ };
class InferError : public VisionPipeError { /* ... */ };
```

#### Python ↔ C++ 异常穿透

```
C++ 异常                      →  Python 异常
─────────────────────────────────────────────
VisionPipeError (基类)         →  visionpipe.VisionPipeError
ConfigError                   →  visionpipe.ConfigError
NotFoundError                 →  visionpipe.NotFoundError
CudaError                     →  visionpipe.CudaError
ModelLoadError                →  visionpipe.ModelLoadError
InferError                    →  visionpipe.InferError
std::exception                →  RuntimeError
未知异常                       →  RuntimeError (wrapped)
```

nanobind 绑定示例：

```cpp
// python/bindings/bind_exceptions.cpp
nb::exception<VisionPipeError>(m, "VisionPipeError", PyExc_RuntimeError);
nb::exception<ConfigError>(m, "ConfigError", visionpipe_error);
// ...
```

### 7.2 日志规范

#### 日志级别

| 级别 | 使用场景 | 示例 |
|------|----------|------|
| TRACE | 详细调试信息，生产环境关闭 | 每帧处理时间戳 |
| DEBUG | 开发调试信息 | 节点队列状态变化 |
| INFO | 正常运行事件 | Pipeline 启动/停止、模型加载 |
| WARN | 可恢复的异常情况 | 队列积压、帧丢帧 |
| ERROR | 错误但进程可继续 | 单节点异常、GPU OOM 缓解 |
| CRITICAL | 进程级故障 | 无法恢复的错误 |

#### 必须记录日志的操作

- Pipeline 生命周期：create/start/stop/destroy
- 模型加载：acquire/release（含路径、耗时、显存）
- GPU 错误：所有 CUDA 调用失败
- 节点异常：单个帧处理失败（记录 frame_id）
- 配置热更：set_param 调用（参数名、新值）

#### 结构化日志字段

```json
{
  "timestamp": "2026-04-20T10:30:45.123Z",
  "level": "info",
  "logger": "PipelineManager",
  "message": "Pipeline started",
  "pipeline_id": "abc123",
  "elapsed_ms": 42
}
```

### 7.3 代码风格

#### C++ (clang-format)

配置文件：`.clang-format`

```yaml
BasedOnStyle: Google
Language: Cpp
Standard: cpp17
IndentWidth: 4
ColumnLimit: 120
BreakBeforeBraces: Attach
AllowShortFunctionsOnASingleLine: Empty
IncludeBlocks: Preserve
```

检查命令：

```bash
find src -name "*.h" -o -name "*.cpp" | xargs clang-format --dry-run --Werror
```

#### Python (ruff + mypy)

配置文件：`pyproject.toml`

```toml
[tool.ruff]
line-length = 120
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W", "UP"]

[tool.mypy]
python_version = "3.10"
strict = true
```

检查命令：

```bash
uv run ruff check python/
uv run mypy python/
```

### 7.4 阶段门禁

每个 Phase 结束前必须满足：

1. **所有测试通过**：`ctest --test-dir build` 和 `uv run pytest` 全绿
2. **覆盖率达标**：核心模块 >90%，整体 >80%
3. **代码风格通过**：clang-format 和 ruff 无警告
4. **无内存泄漏**：Valgrind/ASAN 检查通过
5. **性能基准达标**：对应阶段性能指标满足

门禁检查脚本：`scripts/phase_gate.sh <phase_number>`
