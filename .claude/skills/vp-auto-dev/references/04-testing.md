## 4. 测试方案

### 4.1 测试理念

采用 **TDD（测试驱动开发）** 理念：先写测试用例定义接口契约，再实现功能。所有新功能 PR 必须附带对应测试，CI 强制 gate。

### 4.2 分层测试结构

```
tests/
├── unit/           # 单元测试：纯逻辑验证，需 GPU 环境（部分测试加载模型）
│   ├── cpp/        # Google Test
│   └── python/     # pytest
├── integration/    # 集成测试：节点间交互、Pipeline 生命周期，需真实 GPU
│   ├── cpp/
│   └── python/
└── e2e/            # 端到端测试：完整 pipeline 运行，含 WebRTC/REST API 验证
    └── python/
```

### 4.3 各层测试目标

#### 单元测试

| 测试目标 | 方法 |
|---|---|
| `BoundedQueue` 入队/出队/溢出策略 | 纯 C++ 逻辑，Google Test |
| `ModelRegistry` SHA-256 计算、引用计数、TTL 清理 | 加载真实 Mock 模型 |
| `PipelineManager` 状态机（INIT→RUNNING→DRAINING→STOPPED） | Mock Pipeline |
| `ControlChannel` ROI 坐标归一化/反归一化 | 纯数学逻辑 |
| Python DSL 节点图构建、YAML 导出/导入 | pytest，需加载 C++ 扩展 |
| pydantic 管理 API 数据模型校验 | pytest |

#### 集成测试（需 GPU）

| 测试目标 | 方法 |
|---|---|
| `TrtInferNode` 端到端推理（输入 tensor → 输出 bbox） | 加载真实 YOLOv8 TRT engine |
| 多 worker 并行推理结果一致性 | 对比 worker=1 和 worker=3 结果差异 <1e-4 |
| `FileSource`（GPU/CPU/AUTO 三种 DecodeMode）解码帧数与视频帧数匹配 | 固定测试视频文件（100帧）|
| Pipeline 优雅停止：DRAINING 期间不丢已入队帧 | `BLOCK` 策略下计帧 |
| ModelRegistry 跨 Pipeline 共享：两条 pipeline 共用同一 engine，VRAM 增量为零 | `nvml` 查询显存 |
| `set_param()` 热更 ROI 原子性：热更后第 N+1 帧生效，第 N 帧不受影响 | 帧级断言 |

#### E2E 测试

| 测试目标 | 方法 |
|---|---|
| REST API 创建 → 启动 → 查询 → 停止 pipeline 全流程 | pytest + httpx |
| WebRTC 流建立、收到视频帧 | puppeteer / playwright 无头浏览器 |
| WebSocket ROI 热更后检测结果区域变化 | 构造测试帧，断言输出 bbox 在 ROI 内 |
| 多 Pipeline 并发：物理+化学同时运行，结果互不污染 | 两路不同测试视频 + 断言类别集合不相交 |
| 进程收到 `SIGTERM` 后所有 pipeline 优雅退出 | subprocess + 计时断言 <2s |

### 4.4 性能基准测试（非 CI gate，定期运行）

| 指标 | 目标值 | 测试方法 |
|---|---|---|
| 单路 1080p YOLOv8 吞吐 | ≥25 FPS（RTX 3090） | 跑 300 帧取均值 |
| 16路 1080p 同卡总吞吐 | ≥200 FPS | 16 路并发 |
| Pipeline 启动耗时（模型已缓存） | <500ms | 计时 |
| 优雅停止耗时 | <500ms | 计时 |
| ROI 热更生效延迟 | ≤1 帧（@25fps = 40ms） | 帧级断言 |
| GPU 显存占用（16路，共享模型） | 对比不共享减少 ≥30% | nvml 采样 |

---
