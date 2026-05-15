# Spec Review Progress

## Status
Last reviewed: Phase 3
Next phase: Phase 4

## Confirmed Phases

### Phase 0: 工程骨架 + CI 基础（已确认）
- **Frame.user_data** → `map<string, any>`（多 PyNode 各自用 key 隔离，互不覆盖）
- **IAllocator** → 留在 `src/core/tensor.h`（避免 core→hal 循环依赖，HAL 层只放推理相关接口）
- **Tensor** → move-only RAII（析构自动释放内存，allocator 弱引用；allocator 是全局/pipeline 级别生命周期，不需要 shared_ptr）

#### 代码验证结果
- IAllocator 位置：匹配（tensor.h:28-44）
- Tensor RAII：匹配（析构释放 tensor.h:114-119，拷贝删除 122-123，move 实现 126-155）
- Frame.user_data：**不匹配** — 代码仍为 `std::any`（frame.h:46），需改为 `map<string, any>`。T0.2 已标记为未完成。

### Phase 1: C++ 核心调度框架（已确认）
- **1-A SourceNode 架构** → 保留继承 NodeBase，引入 SourceNode 中间抽象类。SourceConfig 增加 loop/skip_frames/max_retries/retry_interval_ms。图片=单帧源。StreamError 新异常类型。
- **1-B DAG 拓扑** → 支持合并（多对一，多 Source 共享下游 input_queue，BoundedQueue 多生产者并发 push，零拷贝）。不支持分叉（一对多，与 Frame move-only 冲突）。stream_id 区分帧来源。
- **1-C 停止机制** → 队列停止级联（Source 结束 → output_queue_.stop() → 下游检测 stopped_and_empty() → 自停 → 级联传播）。非 EOF 哨兵帧。

#### 代码验证结果
- SourceNode 中间抽象类：**不匹配** — FileSource/RtspSource 直接继承 NodeBase，无中间类
- SourceConfig 扩展 (loop/skip_frames/retry)：**不匹配** — 代码中无这些字段
- 合并支持 (多 Source → 共享 input_queue)：**不匹配** — 代码中无此功能
- 队列停止级联：匹配 ✓（node_base.cpp:130-134, 152-158）
- StreamError 异常：**不匹配** — 代码中无此类型
- T1.1 已标记为未完成

### Phase 2: NVIDIA 推理 + 编解码（已确认）
- **2-A HAL NVIDIA 实现** → TrtModelEngine/TrtExecContext（独立 CUDA stream）/CudaAllocator，承接 Phase 1 parallel_workers
- **2-B 视频解码策略** → 一期 cv::cudacodec 直接调用（Phase 2），二期 ICodec HAL（Phase 5）；三种 DecodeMode（AUTO/GPU/CPU）；FileSource/RtspSource 继承 SourceNode
- **2-C 推理节点继承** → DetectorNode/ClassifierNode/SegmentNode : InferNode；ByteTrackNode : NodeBase
- **Batch 推理** → InferNode 提供动态攒帧 + `process_batch` 虚函数 + `run_inference` 辅助方法，子类实现具体 batch 逻辑
- **Frame 字段** → detections/classifications/segments(masks)/tracks 各自独立字段，通过 detection_index 关联
- **ClassifierNode 双模式** → target_classes 非空=二级分类（筛选 detections crop），为空=整图分类

#### 代码验证结果
- T2.1 HAL NVIDIA：匹配 ✓
- T2.2a FileSource 继承：**不匹配** — 继承 NodeBase 而非 SourceNode；SourceConfig 缺少 Phase 1 字段
- InferNode batch 机制：**不匹配** — 当前为单帧 infer_frame 接口，无 process_batch
- Frame.classifications：**不匹配** — 代码中无此字段
- ClassifierNode 双模式：**不匹配** — 无 target_classes，结果覆盖 detections
- T2.2a/T2.3/T2.4/T2.5 已标记为未完成

### Phase 3: Python 绑定 + DSL（已确认）
- **3-A 绑定粒度** → 绑定所有核心类 + PipelineBuilder/PipelineConfig/AnnotatorNode/MockModelEngine 等额外类
- **3-B GIL 策略** → run()/stop() 释放 GIL；PyNode 回调获取 GIL；Frame 零拷贝引用（rv_policy::reference）
- **3-C DSL 设计** → `>>` 直接返回 Pipeline（去掉 .build()）；`[src1, src2] >> det` 合并语法；公开 API 为 `run(block=False)` + `stop()`
- **3-D CustomNode 子进程架构** → 用户面向 CustomNode 基类（on_frame + FrameView）；默认 subprocess 模式（独立进程，真并行）；C++ ProcessProxyNode + IPC（Unix Socket + CUDA IPC）；子进程崩溃自动重启
- **3-E YAML 序列化** → export_yaml/load_yaml + CustomNode 自动导入（module/class 字段）

#### 代码验证结果
- nanobind 绑定核心类：基础绑定已实现，但 API 不匹配新规范
- `>>` 返回 PipelineBuilder 而非 Pipeline：**不匹配**
- 合并语法 `[src1, src2] >> det`：**不匹配** — 不支持
- `run(block=False)` API：**不匹配** — 当前 run() 只是 start() 别名
- CustomNode / ProcessProxyNode / FrameView：**不匹配** — 不存在
- YAML CustomNode 自动导入：**不匹配** — 不支持
- T3.1/T3.2/T3.3 全部标记为未完成

## Cross-Phase Dependencies

- Phase 1 合并拓扑 + Phase 0 Frame move-only：兼容（多生产者各自 move 到共享队列，零拷贝）
- Phase 1 合并拓扑 + 队列停止：Pipeline 需追踪 source→queue 映射，所有 Source 结束后才 stop 共享队列
- Phase 1 不支持分叉 ← Phase 0 Frame move-only 约束
- Phase 2 InferNode process_batch ← Phase 1 InferNode parallel_workers（细化，非冲突）
- Phase 2 FileSource : SourceNode ← Phase 1 SourceNode 中间抽象类
- Phase 2 Frame.classifications 新字段 ← Phase 0 Frame 结构体扩展
- Phase 3 CustomNode subprocess ← Phase 0 Frame move-only（Frame 留主进程，只传 metadata 副本）
- Phase 3 CUDA IPC ← Phase 2 CudaAllocator（分配的显存支持 IPC handle）
- Phase 3 合并语法 ← Phase 1 合并拓扑（多 Source 共享 input_queue）
- Phase 3 run(block=False) ← Phase 1 停止机制（队列级联停止）

## Notes

- 2026-05-13: Phase 0 确认，发现 user_data 实现与方案不一致，T0.2 标记为 [ ]
- 2026-05-14: Phase 1 确认，发现 SourceNode 中间类/合并支持/SourceConfig 扩展/StreamError 均未实现，T1.1 标记为 [ ]
- 2026-05-14: Phase 2 确认，发现 InferNode batch/ClassifierNode 双模式/Frame.classifications/FileSource 继承均需 rework，T2.2a/T2.3/T2.4/T2.5 标记为 [ ]
- 2026-05-15: Phase 3 确认，新增 CustomNode 子进程架构（3-D）和 DSL 重构（3-C），T3.1/T3.2/T3.3 全部标记为 [ ]
