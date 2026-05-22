# Spec Review Progress

## Status
Last reviewed: Phase 5
Next phase: (all phases confirmed)

## Confirmed Phases

### Phase 0: 工程骨架 + CI 基础（已确认）
- **Frame.user_data** → `map<string, any>`（多 PyNode 各自用 key 隔离，互不覆盖）
- **IAllocator** → 留在 `src/core/tensor.h`（避免 core→hal 循环依赖，HAL 层只放推理相关接口）
- **Tensor** → move-only RAII（析构自动释放内存，allocator 弱引用；allocator 是全局/pipeline 级别生命周期，不需要 shared_ptr）

#### 代码验证结果
- IAllocator 位置：匹配（tensor.h:28-44）
- Tensor RAII：匹配（析构释放 tensor.h:114-119，拷贝删除 122-123，move 实现 126-155）
- Frame.user_data：已修复 ✓ — 改为 `unordered_map<string, any>`（frame.h:56）

### Phase 1: C++ 核心调度框架（已确认）
- **1-A SourceNode 架构** → 保留继承 NodeBase，引入 SourceNode 中间抽象类。SourceConfig 增加 loop/skip_frames/max_retries/retry_interval_ms。图片=单帧源。StreamError 新异常类型。
- **1-B DAG 拓扑** → 支持合并（多对一，多 Source 共享下游 input_queue，BoundedQueue 多生产者并发 push，零拷贝）。不支持分叉（一对多，与 Frame move-only 冲突）。stream_id 区分帧来源。
- **1-C 停止机制** → 队列停止级联（Source 结束 → output_queue_.stop() → 下游检测 stopped_and_empty() → 自停 → 级联传播）。非 EOF 哨兵帧。

#### 代码验证结果
- SourceNode 中间抽象类：已修复 ✓（src/core/source_node.h, source_node.cpp — Template Method 模式）
- SourceConfig 扩展 (loop/skip_frames/retry)：已修复 ✓（source_config.h 新增 4 字段）
- 合并支持 (多 Source → 共享 input_queue)：已修复 ✓（pipeline.cpp connect() 合并拓扑 + producer count 跟踪）
- 队列停止级联：匹配 ✓（node_base.cpp:130-134, 152-158）
- StreamError 异常：已修复 ✓（error.h 新增 StreamError : VisionPipeError）
- T1.1 已标记为完成 ✓

### Phase 2: NVIDIA 推理 + 编解码（已确认）
- **2-A HAL NVIDIA 实现** → TrtModelEngine/TrtExecContext（独立 CUDA stream）/CudaAllocator，承接 Phase 1 parallel_workers
- **2-B 视频解码策略** → 一期 cv::cudacodec 直接调用（Phase 2），二期 ICodec HAL（Phase 5）；三种 DecodeMode（AUTO/GPU/CPU）；FileSource/RtspSource 继承 SourceNode
- **2-C 推理节点继承** → DetectorNode/ClassifierNode/SegmentNode : InferNode；ByteTrackNode : NodeBase
- **Batch 推理** → InferNode 提供动态攒帧 + `process_batch` 虚函数 + `run_inference` 辅助方法，子类实现具体 batch 逻辑
- **Frame 字段** → detections/classifications/segments(masks)/tracks 各自独立字段，通过 detection_index 关联
- **ClassifierNode 双模式** → target_classes 非空=二级分类（筛选 detections crop），为空=整图分类

#### 代码验证结果
- T2.1 HAL NVIDIA：匹配 ✓
- T2.2a FileSource 继承：已修复 ✓ — FileSource/RtspSource 继承 SourceNode；SourceConfig 含 Phase 1 字段
- InferNode batch 机制：已修复 ✓ — 改为 process_batch(vector<Frame>&) + run_inference/run_inference_multi 辅助方法 + 动态攒帧（max_batch_size + batch_timeout）
- Frame.classifications：已修复 ✓ — 新增 Classification 结构体 + classifications 向量（frame.h:40-44, 53）
- ClassifierNode 双模式：已修复 ✓ — target_classes 非空=二级分类，空=整图分类；结果写入 frame.classifications
- T2.2a 视频源节点：已修复 ✓ — FileSource GPU 解码填充 Frame::image（BGRA→RGB + CudaAllocator）；RtspSource CPU 解码填充 Frame::image；GPU 模式不支持时抛 CudaError
- T2.3/T2.5 已标记为未完成

### Phase 3: Python 绑定 + DSL（已确认）
- **3-A 绑定粒度** → 绑定所有核心类 + PipelineBuilder/PipelineConfig/AnnotatorNode/MockModelEngine 等额外类
- **3-B GIL 策略** → run()/stop() 释放 GIL；PyNode 回调获取 GIL；Frame 零拷贝引用（rv_policy::reference）
- **3-C DSL 设计** → `>>` 直接返回 Pipeline（去掉 .build()）；`[src1, src2] >> det` 合并语法；公开 API 为 `run(block=False)` + `stop()`
- **3-D CustomNode 子进程架构** → 用户面向 CustomNode 基类（on_frame + FrameView）；默认 subprocess 模式（独立进程，真并行）；C++ ProcessProxyNode + IPC（Unix Socket + CUDA IPC）；子进程崩溃自动重启
- **3-E YAML 序列化** → export_yaml/load_yaml + CustomNode 自动导入（module/class 字段）

#### 代码验证结果
- nanobind 绑定核心类：基础绑定已实现，但 API 不匹配新规范
- `>>` 返回 Pipeline：已修复 ✓ — NodeBase.__rshift__ 直接返回 Pipeline，Pipeline.__rshift__ 链式追加
- 合并语法 `[src1, src2] >> det`：已修复 ✓ — NodeBase.__rrshift__ 处理 list/tuple 合并拓扑
- `run(block=False)` API：已修复 ✓ — run(block=False) 非阻塞，run(block=True) 阻塞至 source 结束
- CustomNode / ProcessProxyNode / FrameView：已修复 ✓ — ProcessProxyNode(C++ UDS IPC) + CustomNode(subprocess/inline) + FrameView(安全视图) + IPC protocol + worker
- YAML CustomNode 自动导入：已修复 ✓ — NodeSpec 新增 module/class_name/process_mode 字段，_import_custom_node 自动导入，from_yaml 构建 Pipeline
- T3.1/T3.2/T3.3 全部标记为未完成

### Phase 4: 管理 API + 前端交付（已确认）
- **4-A 管理 API** → aiohttp 同进程嵌入；分离生命周期（create/start/stop/delete）；新增 `GET /pipelines/{id}/nodes` 返回 per-node NodeStats
- **4-B WebRTC Sink** → libdatachannel + NVENC H.264 + Python signaling；单视角切换（Annotator enable/disable，不需要分叉）
- **4-C 控制通道** → 通用 WebSocket `/ws/{id}/control`（type+payload 格式），承载 ROI + 任意 set_param 转发
- **4-D JsonResultSink** → 独立 WebSocket `/ws/{id}/results`（与 control 分离，避免 backpressure 互影响）
- **4-D MjpegSink** → 保留作为调试/降级方案，默认 enabled=false，通过 set_param 运行时开启
- **NodeStats 补齐** → fps / latency_ms / frames_processed / errors / state（NodeState: INIT/RUNNING/STOPPED/ERROR）
- **SinkNode 基类** → 统一提供 `enabled` 属性（默认 true），MjpegSink 覆盖为 false

#### 代码验证结果
- 管理 REST API (aiohttp)：已修复 ✓ — 分离生命周期（POST start/stop 独立端点）+ GET /nodes 接口已实现
- WebRTC Sink (libdatachannel + NVENC)：匹配 ✓
- WebSocket 控制通道：已修复 ✓ — 新增 set_param 消息类型，转发到任意节点 `set_param(name, value)`，覆盖参数校验/节点查找/拒绝/异常路径
- JsonResultSink：匹配 ✓（独立 BoundedQueue）
- MjpegSink：已修复 ✓ — enabled 开关已通过 SinkNode 基类实现，默认 enabled=false
- NodeStats：已修复 ✓ — latency_ms (EMA α=0.1) 和 state (NodeState 枚举) 已补齐
- SinkNode 基类：已修复 ✓ — SinkNode : NodeBase 中间基类，统一 enabled 属性，三个 Sink 均已迁移
- Pipeline 生命周期分离 (create/start/stop/destroy)：匹配 ✓
- T4.1 已完成；T4.4 已完成；T4.3 已完成；T4.2 已完成 ✓

### Phase 5: 集成验证 + 收尾（已确认）
- **Phase 5 重新定位** → 从"集成测试 + 性能调优"改为"集成验证 + 收尾"，聚焦功能正确性而非性能指标
- **T5.1 多 Pipeline 并发** → 去掉 VRAM ≤10% 硬指标，改为功能性验证（互不干扰 + 模型共享生效 + 生命周期隔离）
- **T5.2 端到端验证测试（新增）** → 三层递进验证：节点级正确性 → 数据流完整性（Frame 累积语义）→ 控制面验证（REST/WebSocket/YAML 往返）。完整链路：FileSource → DetectorNode → ClassifierNode → TrackerNode → CustomNode(subprocess) → AnnotatorNode → JsonResultSink
- **旧 T5.2 性能 benchmark** → 完全移除，后续专项处理（Section 4.4 标注为参考指标）
- **T2.2b ICodec HAL** → 从正式排期移除，移至 DEV_SPEC「未来扩展」章节

#### 代码验证结果
- T5.1（标记 [x]）：test_multi_pipeline.py 已 rework — 修正测试资产路径 (48-3.mp4, tests/models/) + 重写 shared-engine 测试为功能性生命周期隔离 + 修复 Pipeline::connect 对 InferNode 自有 input_queue 的处理。3 个测试全部通过 ✓
- benchmarks/ 目录不存在（旧 T5.2 已移除，无需创建）
- examples/ 和 docs/ 目录不存在，但项目根目录有 demo_detect.py/demo_full_pipeline.py 等可作为 T5.3 基础素材
- 开发环境：RTX 3060 12GB + WSL2（16GB 内存），跑 2-3 路 1080p pipeline 可行

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
- Phase 4 SinkNode 基类 ← Phase 1 NodeBase（SinkNode 继承 NodeBase，新增 enabled 属性）
- Phase 4 NodeStats.state ← Phase 1 NodeBase 生命周期（INIT/RUNNING/STOPPED/ERROR 状态机）
- Phase 4 通用 set_param 转发 ← Phase 1 NodeBase::set_param（WebSocket 只是暴露已有接口）
- Phase 4 单视角切换 ← Phase 1 不支持分叉（通过 Annotator enable/disable 而非 DAG 分叉）
- Phase 4 REST create/start/stop/delete ← Phase 1 PipelineManager 生命周期（已分离）
- Phase 4 JsonResultSink 独立 WS ← Phase 0 Frame detections/tracks 字段（序列化为 JSON）
- Phase 5 T5.2 端到端验证 ← Phase 2 所有推理节点（DetectorNode/ClassifierNode/TrackerNode）
- Phase 5 T5.2 端到端验证 ← Phase 3 CustomNode subprocess + YAML 往返
- Phase 5 T5.2 端到端验证 ← Phase 4 REST API + WebSocket 控制通道
- Phase 5 T5.1 多 Pipeline ← Phase 1 PipelineManager + Phase 2 ModelRegistry 共享

## Notes

- 2026-05-13: Phase 0 确认，发现 user_data 实现与方案不一致，T0.2 标记为 [ ]
- 2026-05-14: Phase 1 确认，发现 SourceNode 中间类/合并支持/SourceConfig 扩展/StreamError 均未实现，T1.1 标记为 [ ]
- 2026-05-14: Phase 2 确认，发现 InferNode batch/ClassifierNode 双模式/Frame.classifications/FileSource 继承均需 rework，T2.2a/T2.3/T2.4/T2.5 标记为 [ ]
- 2026-05-15: Phase 3 确认，新增 CustomNode 子进程架构（3-D）和 DSL 重构（3-C），T3.1/T3.2/T3.3 全部标记为 [ ]
- 2026-05-15: Phase 4 确认，关键决策：分离生命周期、通用控制通道、JsonResult 独立 WS、MjpegSink 默认关闭、SinkNode 基类统一 enabled、NodeStats 补齐 latency_ms+state。T4.1/T4.2/T4.3/T4.4 标记为 [ ]
- 2026-05-19: Phase 5 确认，重新定位为「集成验证 + 收尾」。移除性能 benchmark（后续专项），新增 T5.2 端到端验证测试（三层递进），T2.2b ICodec HAL 移至未来扩展，T5.1 标记回 [ ]（测试资产路径需修复）
- 2026-05-20: T2.3 补齐完成 — InferNode 从 infer_frame 重构为 process_batch 接口，DetectorNode/ClassifierNode/SegmentNode 全部适配，parallel_workers 测试恢复启用并通过
- 2026-05-20: T3.1 补齐完成 — DSL 重构：>> 返回 Pipeline（非 PipelineBuilder）、合并语法 [src1,src2]>>det、run(block=False)/stop() API
- 2026-05-20: T3.2 完成 — CustomNode 子进程架构：C++ ProcessProxyNode(UDS JSON IPC) + Python CustomNode(subprocess/inline 双模式) + FrameView 安全视图 + IPC protocol + worker + 崩溃自动重启
- 2026-05-21: T4.3 补齐完成 — WebSocket 控制通道新增通用 set_param 消息类型，扁平格式 `{type, node_id, param_name, value}` 与 REST `/params` 一致；覆盖参数校验、节点查找、拒绝（返回 False）、异常路径；23 个新测试 + 18 个原有 ROI 测试全部通过
