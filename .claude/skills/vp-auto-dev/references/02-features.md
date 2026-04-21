## 2. 核心特点

| # | 特点 | 说明 |
|---|---|---|
| 1 | **Python DSL 编排** | 用 `>>` 运算符连接节点构图，可导出/导入 YAML 用于版本化和运维下发 |
| 2 | **C++ 热路径，零 GIL 干扰** | 推理、编解码、调度全在 C++ 线程池；Python 业务节点回调时短暂 acquire GIL |
| 3 | **同进程多 Pipeline** | PipelineManager 支持动态创建/销毁多条 pipeline，内嵌 REST 管理 API，无需 Docker 隔离 |
| 4 | **模型去重复用** | ModelRegistry 按引擎文件 SHA-256 去重，多条 pipeline 共享同一 `IModelEngine`，VRAM 不重复占用 |
| 5 | **优雅启停协议** | DRAINING → teardown → STOPPED 三段式退出，GPU 资源安全释放，典型耗时 <500ms |
| 6 | **节点并发扩展** | 瓶颈节点配置 `parallel_workers=N`，多个 worker 共享模型权重独立执行上下文，真 GPU 并行 |
| 7 | **有界队列 + 溢出策略** | 每节点有界输入队列；实时流默认 `DROP_OLDEST` 保低延迟，文件处理可选 `BLOCK` 不丢帧 |
| 8 | **ROI 实时热更** | 前端 canvas 框选 → WebSocket 归一化坐标 → C++ `set_param()` 原子写 → 下一帧生效 |
| 9 | **HAL 硬件抽象** | `IModelEngine` / `IExecContext` / `IAllocator` 三接口屏蔽厂商；扩展新硬件只需实现三个类 |
| 10 | **内置可观测性** | 每节点暴露队列占用率、丢帧计数、FPS；健康接口 `GET /pipelines/{id}/health`；spdlog 结构化日志 |

---
