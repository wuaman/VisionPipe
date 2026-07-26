"""VisionPipe-py 实例分割端到端示例。

演示 YOLOv8-seg 实例分割全链路:

    FileSource → YoloSegNode → ByteTrackNode → AnnotatorNode → WebRTCSink → 浏览器

其中:
- YoloSegNode 输出检测框 (frame.detections) + 实例掩码 (frame.masks) 双输出
- ByteTrackNode 给每个 detection 加 track_id
- AnnotatorNode 以 draw_masks=True 叠加半透明掩码 + 检测框 + 跟踪 ID
- WebRTCSink 用 NVENC 编码 H.264 经 libdatachannel 推流

依赖资源 (默认从仓库内查找, 可用 CLI 参数覆盖):
- 视频:  tests/data/48-3.mp4
- 分割:  tests/models/yolov8m-seg_fp16.engine

构建前置条件
-----------
本 demo 需要带 WebRTC 支持的构建:

    cmake -B build -DVISIONPIPE_USE_WEBRTC=ON
    cmake --build build --target visionpipe_python

运行
----
    uv run python examples/segment_demo.py
    # → 打开浏览器访问 http://localhost:8080/ (Dashboard)
    #   或 http://localhost:8080/viewer?pid=<pipeline-id> (独立 viewer)

参数
----
    --video / --seg-engine 自定义资源路径
    --port               管理服务端口, 默认 8080
    --fps / --bitrate-kbps WebRTC 编码参数
    --score-threshold / --mask-threshold 分割阈值

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
DEFAULT_SEG_ENGINE = REPO_ROOT / "tests" / "models" / "yolov8m-seg_fp16.engine"
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
        description="VisionPipe-py 实例分割端到端 demo (分割+追踪+掩码渲染+前端)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO,
                        help=f"输入视频路径 (默认: {DEFAULT_VIDEO})")
    parser.add_argument("--seg-engine", type=Path, default=DEFAULT_SEG_ENGINE,
                        help=f"分割 TensorRT engine (默认: {DEFAULT_SEG_ENGINE})")
    parser.add_argument("--host", default="0.0.0.0", help="管理服务 bind 地址 (默认: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8080, help="管理服务端口 (默认: 8080)")
    parser.add_argument("--fps", type=int, default=15, help="WebRTC 输出帧率 (默认: 15)")
    parser.add_argument("--bitrate-kbps", type=int, default=2000,
                        help="WebRTC 视频比特率 (默认: 2000 kbps, 掩码细节较多建议略高)")
    parser.add_argument("--score-threshold", type=float, default=0.3,
                        help="分割分数阈值 (默认: 0.3)")
    parser.add_argument("--mask-threshold", type=float, default=0.5,
                        help="掩码二值化阈值 (默认: 0.5)")
    return parser.parse_args()


def check_assets(args: argparse.Namespace) -> None:
    missing = [p for p in (args.video, args.seg_engine, VIEWER_HTML) if not p.exists()]
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

    # Segment (检测框 + 实例掩码双输出)
    seg_engine = visionpipe.TrtModelEngine(str(args.seg_engine))
    seg_cfg = visionpipe.YoloSegConfig()
    seg_cfg.input_width = 640
    seg_cfg.input_height = 640
    seg_cfg.score_threshold = args.score_threshold
    seg_cfg.mask_threshold = args.mask_threshold
    segment = visionpipe.YoloSegNode(seg_engine, seg_cfg, "segment")

    # Tracker
    trk_cfg = visionpipe.ByteTrackConfig()
    trk_cfg.track_thresh = 0.5
    trk_cfg.match_thresh = 0.8
    trk_cfg.frame_rate = args.fps
    tracker = visionpipe.ByteTrackNode(trk_cfg, "tracker")

    # Annotator (掩码渲染打开)
    ann_cfg = visionpipe.AnnotatorConfig()
    ann_cfg.draw_detections = True
    ann_cfg.draw_tracks = True
    ann_cfg.draw_masks = True
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

    return source >> segment >> tracker >> annotator >> webrtc


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

    _say(f"[segment-demo] 视频     : {args.video}")
    _say(f"[segment-demo] 分割     : {args.seg_engine}")
    _say(f"[segment-demo] FPS / 码率: {args.fps} / {args.bitrate_kbps} kbps")
    _say("")

    manager = visionpipe.PipelineManager()

    pipeline = build_pipeline(args)
    pipeline_id = manager.create_pipeline(pipeline)
    manager.start(pipeline_id)
    _say(f"[segment-demo] Pipeline ID: {pipeline_id}")

    server = ManagementServer(manager, host=args.host, port=args.port)
    # ManagementServer 的 / 保留给 dashboard；demo 专用 WebRTC viewer 挂到 /viewer。
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
    _say("")
    _say("=" * 70)
    _say(f"  Dashboard : http://{display_host}:{args.port}/")
    _say(f"  Viewer    : http://{display_host}:{args.port}/viewer?pid={pipeline_id}")
    _say("=" * 70)
    _say("")
    _say("按 Ctrl+C 退出")

    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())

    try:
        while not stop.is_set():
            time.sleep(0.5)
    finally:
        _say("\n[segment-demo] 清理中 ...")

        async def _shutdown() -> None:
            await server.stop()

        try:
            asyncio.run_coroutine_threadsafe(_shutdown(), server_loop).result(timeout=10)
        except Exception as exc:
            _say(f"[segment-demo] server.stop 异常: {exc}")
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
            _say(f"[segment-demo] pipeline.stop 异常: {exc}")
        try:
            manager.destroy(pipeline_id)
        except Exception as exc:
            _say(f"[segment-demo] pipeline.destroy 异常: {exc}")

        _say("[segment-demo] 退出完毕")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
