# VisionPipe-py 示例索引

本目录包含 4 个 Python 示例,从最小冒烟到端到端业务链路,演示框架的不同能力。

| 示例 | 演示能力 | 关键节点 | 输出形式 |
|---|---|---|---|
| `quickstart.py` | 最小可运行 pipeline | Source + Detector + JsonResultSink | 控制台 JSON |
| `multi_pipeline_demo.py` | 多 Pipeline 并发 + 生命周期隔离 + ModelRegistry 显存复用 | 两条并行链 | 控制台 FPS / 类别统计 |
| `webrtc_demo.py` | 检测 + **二阶段分类** + 追踪 + 浏览器实时视频 | Detector + Classifier + ByteTrack + Annotator + WebRTCSink | 浏览器 (WebRTC) |
| `roi_hotupdate_demo.py` | 运行时 **ROI 热更新**, 管理通道 (REST/WS) + 实时观察 | Detector + Annotator + MjpegSink | 浏览器 (MJPEG) |

---

## 公共前置条件

```bash
# 1. C++ + Python 扩展构建 (含 CUDA / TensorRT)
cmake -B build -DUSE_CUDA=ON -DUSE_TENSORRT=ON
cmake --build build --target visionpipe_python

# 2. WebRTC demo 额外需要 libdatachannel + FFmpeg NVENC
cmake -B build -DVISIONPIPE_USE_WEBRTC=ON
cmake --build build --target visionpipe_python

# 3. Python 依赖
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"

# 4. 测试视频与 engine
ls tests/data/48-3.mp4
ls tests/models/yolov8n_dynamic.engine
ls tests/models/efficientnet_b0_fp16.engine   # 仅 webrtc_demo 需要
```

视频获取参考 `data/download_test_assets.sh`;TensorRT engine 由对应 ONNX 用 `trtexec` 离线转换。

---

## 示例 1: `quickstart.py`

冒烟脚本: 跑通 source→detector→sink,前 N 帧打印 JSON,5 s 后输出 FPS 与类别统计。

```bash
uv run python examples/quickstart.py
```

---

## 示例 2: `multi_pipeline_demo.py`

同进程跑两条 Pipeline 共享同一 TrtModelEngine,验证 ModelRegistry 显存复用与生命周期隔离 (一条提前 stop 不影响另一条)。

```bash
uv run python examples/multi_pipeline_demo.py
```

---

## 示例 3: `webrtc_demo.py` (检测 + 二阶段分类 + 追踪 + WebRTC)

启动后在浏览器打开 URL 即可看到带检测框 + 跟踪 ID 的视频流。

```bash
uv run python examples/webrtc_demo.py
# → 打开浏览器: http://localhost:8080/?pid=<打印出的 pipeline-id>
```

**二阶段分类**通过 `ClassifierConfig.target_classes = [0]` 触发,仅对 person 检测框做 crop + 二次精修推理 (结果写入 `frame.classifications`,不在画面上渲染)。

常用参数:

```bash
uv run python examples/webrtc_demo.py \
    --video <path> --port 9090 --fps 30 --bitrate-kbps 3000 \
    --no-classifier        # 关闭二阶段, 仅 detector+tracker
```

按 Ctrl+C 干净退出。

---

## 示例 4: `roi_hotupdate_demo.py` (ROI 热更新)

启动后通过 MJPEG 流观察 ROI 切换效果。脚本会自动每 5 s 循环切换 ROI (无 ROI → 左半屏 → 右半屏 → 中心三角形 → 循环)。

```bash
uv run python examples/roi_hotupdate_demo.py
# → 浏览器: http://localhost:8080/mjpeg/<打印出的 pipeline-id>
```

**手动触发** (任意时刻另开终端,把 `<pid>` 替换为实际值):

```bash
# WS 通道 (polygons = [[x,y],…] 点对列表, 归一化)
echo '{"type":"roi","polygons":[[0,0],[0.5,0],[0.5,1],[0,1]]}' \
    | websocat ws://localhost:8080/ws/<pid>/control
echo '{"type":"roi_clear"}' | websocat ws://localhost:8080/ws/<pid>/control

# REST 通道 (value = 扁平 [x1,y1,x2,y2,…])
curl -X POST http://localhost:8080/pipelines/<pid>/params \
     -H 'Content-Type: application/json' \
     -d '{"node_id":"detector","param_name":"roi",
          "value":[0.0,0.0,0.5,0.0,0.5,1.0,0.0,1.0]}'
```

教学模式 (关自动轮换,只起服务等手动触发):

```bash
uv run python examples/roi_hotupdate_demo.py --no-rotate
```

---

## 常见问题

| 现象 | 原因 / 排查 |
|---|---|
| WebRTC 浏览器永远 connecting | 未启用 `-DVISIONPIPE_USE_WEBRTC=ON`;脚本会在启动时检测 stub 并报错退出 |
| `InferError: ... fixed dimension mismatch` (classifier) | `efficientnet_b0_fp16.engine` 默认按 fixed batch=1 转换;保持 `--cls-max-batch=1` (默认) 或用 `DYNAMIC_BATCH=true ./models/efficientnet_b0/convert.sh` 重转 |
| MJPEG / WebRTC 卡顿、画面冻结 | 检查 GPU 占用、视频比特率、NVDEC/NVENC 并发上限 |
| ROI 切了但检测框没变化 | 坐标必须**归一化**到 `[0, 1]`;像素坐标会被全部过滤导致 "看似全失效" |
| ROI 设了但只过滤了部分 bbox | 命中判定是 **bbox 中心点** `pointPolygonTest`,边界目标会跳进跳出 |
| 跨主机/外网访问 WebRTC 失败 | 仓库默认只配 STUN (`stun.l.google.com:19302`),NAT 穿透需自配 TURN |
| 端口 8080 被占用 | 通过 `--port` 指定其他端口 |
| `EngineLoadError`/找不到 engine | 先用 `trtexec` 把 onnx 转 engine,或参考 `data/` 与 `tests/models/` 的转换脚本 |
