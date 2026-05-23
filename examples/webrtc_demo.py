"""VisionPipe-py WebRTC 端到端推理示例。

演示完整业务链路:

    FileSource → DetectorNode → ClassifierNode(二阶段) → ByteTrackNode
              → AnnotatorNode → WebRTCSink → 浏览器

其中:
- DetectorNode 输出 YOLOv8 检测框 (写入 frame.detections)
- ClassifierNode 以 target_classes=[0] 触发**二阶段模式**:仅对 person bbox
  做二次精修分类 (crop → batch → 写 frame.classifications)。注意此结果不会
  在 WebRTC 画面上渲染 (AnnotatorNode 当前只画 detections / tracks / masks),
  但保留在数据通路。如需查看可用 ``/ws/<pid>/results`` 配合 JsonResultSink。
- ByteTrackNode 给每个 detection 加 track_id
- AnnotatorNode 在 CPU BGR 帧上绘制检测框 + 跟踪 ID
- WebRTCSink 用 NVENC 编码 H.264 经 libdatachannel 推流

依赖资源 (默认从仓库内查找, 可用 CLI 参数覆盖):
- 视频:  tests/data/48-3.mp4
- 检测:  tests/models/yolov8n_dynamic.engine
- 分类:  tests/models/efficientnet_b0_fp16.engine

⚠️  Engine batch 约束
-------------------
``efficientnet_b0_fp16.engine`` 由 ``models/efficientnet_b0/convert.sh`` 默认配置
转换 (``BATCH_SIZE=1, DYNAMIC_BATCH=false``), 即 **fixed batch=1**。
本 demo 默认 ``--cls-max-batch=1`` 与之匹配; 改大会触发
``InferError: input tensor shape does not match TensorRT engine``。

如需更高吞吐, 用动态 batch 重转 engine:

    cd models/efficientnet_b0
    DYNAMIC_BATCH=true MIN_BATCH=1 OPT_BATCH=4 MAX_BATCH=8 ./convert.sh
    # 再加 --cls-max-batch 8 运行 demo

构建前置条件
-----------
本 demo 需要带 WebRTC 支持的构建:

    cmake -B build -DVISIONPIPE_USE_WEBRTC=ON
    cmake --build build --target visionpipe_python

否则 WebRTCSink 会退化为 no-op stub, 浏览器永远收不到 offer。

运行
----
    uv run python examples/webrtc_demo.py
    # → 打开浏览器访问 http://localhost:8080/?pid=<pipeline-id>

参数
----
    --video / --det-engine / --cls-engine 自定义资源路径
    --port               管理服务端口, 默认 8080
    --fps / --bitrate-kbps WebRTC 编码参数
    --no-classifier      关闭二阶段分类, 对比单 detector pipeline

按 Ctrl+C 退出, 进程将干净地停止 pipeline + 关闭 aiohttp。
"""

from __future__ import annotations

import argparse
import asyncio
import signal
import sys
import threading
import time
from pathlib import Path

from aiohttp import web

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

import visionpipe  # noqa: E402
from visionpipe.server import ManagementServer  # noqa: E402

DEFAULT_VIDEO = REPO_ROOT / "tests" / "data" / "48-3.mp4"
DEFAULT_DET_ENGINE = REPO_ROOT / "tests" / "models" / "yolov8n_dynamic.engine"
DEFAULT_CLS_ENGINE = REPO_ROOT / "tests" / "models" / "efficientnet_b0_fp16.engine"
VIEWER_HTML = Path(__file__).parent / "webrtc_viewer.html"

# COCO 80 类 (YOLOv8 默认顺序), 供 AnnotatorNode 渲染 bbox 标签
COCO_80 = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]


def _say(msg: str = "") -> None:
    print(msg, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VisionPipe-py WebRTC 端到端 demo (检测+二阶段分类+追踪+前端)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO,
                        help=f"输入视频路径 (默认: {DEFAULT_VIDEO})")
    parser.add_argument("--det-engine", type=Path, default=DEFAULT_DET_ENGINE,
                        help=f"检测 TensorRT engine (默认: {DEFAULT_DET_ENGINE})")
    parser.add_argument("--cls-engine", type=Path, default=DEFAULT_CLS_ENGINE,
                        help=f"分类 TensorRT engine (默认: {DEFAULT_CLS_ENGINE})")
    parser.add_argument("--host", default="0.0.0.0", help="管理服务 bind 地址 (默认: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8080, help="管理服务端口 (默认: 8080)")
    parser.add_argument("--fps", type=int, default=15, help="WebRTC 输出帧率 (默认: 15)")
    parser.add_argument("--bitrate-kbps", type=int, default=1500,
                        help="WebRTC 视频比特率 (默认: 1500 kbps)")
    parser.add_argument("--score-threshold", type=float, default=0.3,
                        help="检测分数阈值 (默认: 0.3)")
    parser.add_argument("--no-classifier", action="store_true",
                        help="关闭二阶段分类, 仅 detector+tracker+annotator+webrtc")
    parser.add_argument("--cls-max-batch", type=int, default=1,
                        help="二阶段分类单帧最多处理的 person 数 "
                             "(默认: 1, 受 efficientnet_b0_fp16.engine fixed batch=1 限制; "
                             "engine 用 --dynamic-batch 重转后可改大)")
    return parser.parse_args()


def check_assets(args: argparse.Namespace) -> None:
    required = [args.video, args.det_engine]
    if not args.no_classifier:
        required.append(args.cls_engine)
    required.append(VIEWER_HTML)

    missing = [p for p in required if not p.exists()]
    if not missing:
        return
    _say("缺少必要资源:")
    for p in missing:
        _say(f"  - {p}")
    _say("\n请先准备视频/engine, 或通过 CLI 参数指定。")
    sys.exit(2)


def check_webrtc_build() -> None:
    """启动前确认扩展带 WebRTC; stub 模式会导致浏览器收不到 offer。"""
    probe = visionpipe.WebRTCSink()
    peer_id = probe.create_peer()
    if not peer_id:
        _say("ERROR: WebRTCSink 是 stub 模式 (未启用 -DVISIONPIPE_USE_WEBRTC=ON)。")
        _say("请重新构建:")
        _say("    cmake -B build -DVISIONPIPE_USE_WEBRTC=ON")
        _say("    cmake --build build --target visionpipe_python")
        sys.exit(3)
    probe.remove_peer(peer_id)


def build_pipeline(args: argparse.Namespace) -> visionpipe.Pipeline:
    # Source
    src_cfg = visionpipe.SourceConfig(str(args.video))
    src_cfg.decode_mode = visionpipe.DecodeMode.AUTO
    src_cfg.loop = True
    src_cfg.queue_capacity = 8
    src_cfg.overflow_policy = visionpipe.OverflowPolicy.BLOCK  # InferNode 排序要求
    source = visionpipe.FileSource(src_cfg)

    # Detector
    det_engine = visionpipe.TrtModelEngine(str(args.det_engine))
    det_cfg = visionpipe.DetectorConfig()
    det_cfg.score_threshold = args.score_threshold
    detector = visionpipe.DetectorNode(det_engine, det_cfg, "detector")

    # Classifier (二阶段, target_classes=[0] 仅对 person 精修)
    # 注意: tests/models/efficientnet_b0_fp16.engine 默认按 fixed batch=1 构建
    # (见 models/efficientnet_b0/convert.sh 默认 BATCH_SIZE=1 / DYNAMIC_BATCH=false),
    # 因此这里必须 max_batch_size=1 否则会触发 InferError: shape mismatch。
    # 如需更高吞吐, 用 --dynamic-batch 重新转 engine 后改大此值。
    classifier = None
    if not args.no_classifier:
        cls_engine = visionpipe.TrtModelEngine(str(args.cls_engine))
        cls_cfg = visionpipe.ClassifierConfig()
        cls_cfg.input_width = 224
        cls_cfg.input_height = 224
        cls_cfg.max_batch_size = args.cls_max_batch
        cls_cfg.target_classes = [0]  # 仅对 person 走二阶段
        classifier = visionpipe.ClassifierNode(cls_engine, cls_cfg, "classifier")

    # Tracker
    trk_cfg = visionpipe.ByteTrackConfig()
    trk_cfg.track_thresh = 0.5
    trk_cfg.match_thresh = 0.8
    trk_cfg.frame_rate = args.fps
    tracker = visionpipe.ByteTrackNode(trk_cfg, "tracker")

    # Annotator
    ann_cfg = visionpipe.AnnotatorConfig()
    ann_cfg.draw_detections = True
    ann_cfg.draw_tracks = True
    ann_cfg.draw_masks = False
    ann_cfg.class_names = COCO_80
    annotator = visionpipe.AnnotatorNode(ann_cfg, "annotator")

    # WebRTC sink
    rtc_cfg = visionpipe.WebRTCSinkConfig()
    rtc_cfg.fps = args.fps
    rtc_cfg.video_bitrate_kbps = args.bitrate_kbps
    rtc_cfg.keyframe_interval = max(args.fps * 2, 30)
    rtc_cfg.use_nvenc = True
    rtc_cfg.stun_server = "stun:stun.l.google.com:19302"
    webrtc = visionpipe.WebRTCSink(rtc_cfg, "webrtc_sink")

    # DSL 链式构建
    if classifier is not None:
        return source >> detector >> classifier >> tracker >> annotator >> webrtc
    return source >> detector >> tracker >> annotator >> webrtc


async def _index_handler(request: web.Request) -> web.Response:
    text = VIEWER_HTML.read_text(encoding="utf-8")
    return web.Response(text=text, content_type="text/html", charset="utf-8")


def _serve_in_thread(server: ManagementServer, loop: asyncio.AbstractEventLoop,
                     ready: threading.Event) -> None:
    asyncio.set_event_loop(loop)
    loop.run_until_complete(server.start())
    ready.set()
    loop.run_forever()


def main() -> int:
    args = parse_args()
    check_assets(args)
    check_webrtc_build()

    _say(f"[webrtc-demo] 视频     : {args.video}")
    _say(f"[webrtc-demo] 检测     : {args.det_engine}")
    if not args.no_classifier:
        _say(f"[webrtc-demo] 分类    : {args.cls_engine}  (target_classes=[0]/person)")
    else:
        _say("[webrtc-demo] 分类    : (disabled)")
    _say(f"[webrtc-demo] FPS / 码率: {args.fps} / {args.bitrate_kbps} kbps")
    _say("")

    manager = visionpipe.PipelineManager()
    pipeline_cfg = visionpipe.PipelineConfig()
    pipeline_cfg.name = "webrtc-demo"

    pipeline = build_pipeline(args)
    pipeline_id = manager.create_pipeline(pipeline)
    manager.start(pipeline_id)
    _say(f"[webrtc-demo] Pipeline ID: {pipeline_id}")

    server = ManagementServer(manager, host=args.host, port=args.port)
    # 在 server.start() 之前插入静态 HTML 路由
    server._app.router.add_get("/", _index_handler)
    server._app.router.add_get("/viewer", _index_handler)

    server_loop = asyncio.new_event_loop()
    ready = threading.Event()
    server_thread = threading.Thread(
        target=_serve_in_thread, args=(server, server_loop, ready),
        name="aiohttp-server", daemon=True,
    )
    server_thread.start()
    if not ready.wait(timeout=10):
        _say("ERROR: ManagementServer 启动超时")
        return 1

    display_host = "localhost" if args.host in ("0.0.0.0", "127.0.0.1") else args.host
    viewer_url = f"http://{display_host}:{args.port}/?pid={pipeline_id}"
    _say("")
    _say("=" * 70)
    _say(f"  在浏览器打开: {viewer_url}")
    _say("=" * 70)
    _say("")
    _say("提示:")
    _say(f"  - 健康检查    : curl http://{display_host}:{args.port}/pipelines/{pipeline_id}/health")
    _say(f"  - JSON 结果流  : ws://{display_host}:{args.port}/ws/{pipeline_id}/results")
    _say("    (本 demo 未挂 JsonResultSink, 该 WS 会以 4004 关闭。如需观察分类结果")
    _say("    请添加 --with-json-sink 扇出, 或参考 examples/quickstart.py 改造)")
    _say("  - 跨主机访问需自配 TURN (当前仅 STUN: stun.l.google.com:19302)")
    _say("")
    _say("按 Ctrl+C 退出")

    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())

    try:
        while not stop.is_set():
            time.sleep(0.5)
    finally:
        _say("\n[webrtc-demo] 清理中 ...")

        async def _shutdown() -> None:
            await server.stop()

        try:
            asyncio.run_coroutine_threadsafe(_shutdown(), server_loop).result(timeout=10)
        except Exception as exc:
            _say(f"[webrtc-demo] server.stop 异常: {exc}")
        server_loop.call_soon_threadsafe(server_loop.stop)
        server_thread.join(timeout=5)
        try:
            server_loop.close()
        except Exception:
            pass

        try:
            if manager.status(pipeline_id) != visionpipe.PipelineStatus.STOPPED:
                manager.stop(pipeline_id)
        except Exception as exc:
            _say(f"[webrtc-demo] pipeline.stop 异常: {exc}")
        try:
            manager.destroy(pipeline_id)
        except Exception as exc:
            _say(f"[webrtc-demo] pipeline.destroy 异常: {exc}")

        _say("[webrtc-demo] 退出完毕")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
