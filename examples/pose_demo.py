"""VisionPipe-py 关键点检测端到端示例。

支持两种姿态估计后端:

1. rtmpose (默认, top-down):
       FileSource → DetectorNode → RtmPoseNode → AnnotatorNode → WebRTCSink
   RTMPose-m 精度高, 延迟随人数增长, 依赖检测器提供人体框。

2. yolo-pose (单阶段):
       FileSource → YoloPoseNode → AnnotatorNode → WebRTCSink
   YOLOv8-pose 一次推理同时输出人体框+关键点, 人数多时延迟恒定。

AnnotatorNode 以 draw_keypoints=True 绘制 COCO-17 骨架。

依赖资源 (默认从仓库内查找, 可用 CLI 参数覆盖):
- 视频:      tests/data/48-3.mp4
- 检测:      tests/models/yolov8n_fp16.engine        (仅 rtmpose 后端)
- RTMPose:   tests/models/rtmpose-m_body7_fp16.engine
- YOLO-pose: tests/models/yolov8n-pose_fp16.engine   (仅 yolo-pose 后端)

模型转换见 models/rtmpose/convert.sh 与 models/yolov8_pose/convert.sh。

运行
----
    uv run python examples/pose_demo.py                      # RTMPose top-down
    uv run python examples/pose_demo.py --backend yolo-pose  # 单阶段
    # → 打开浏览器访问 http://localhost:8080/ (Dashboard)

按 Ctrl+C 退出。
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
DEFAULT_DET_ENGINE = REPO_ROOT / "tests" / "models" / "yolov8n_fp16.engine"
DEFAULT_RTMPOSE_ENGINE = REPO_ROOT / "tests" / "models" / "rtmpose-m_body7_fp16.engine"
DEFAULT_YOLOPOSE_ENGINE = REPO_ROOT / "tests" / "models" / "yolov8n-pose_fp16.engine"
VIEWER_HTML = Path(__file__).parent / "webrtc_viewer.html"


def _say(msg: str = "") -> None:
    print(msg, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VisionPipe-py 关键点检测端到端 demo (RTMPose top-down / YOLO-pose 单阶段)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--backend", choices=["rtmpose", "yolo-pose"], default="rtmpose",
                        help="姿态估计后端 (默认: rtmpose)")
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO,
                        help=f"输入视频路径 (默认: {DEFAULT_VIDEO})")
    parser.add_argument("--det-engine", type=Path, default=DEFAULT_DET_ENGINE,
                        help="检测 TensorRT engine (rtmpose 后端使用)")
    parser.add_argument("--pose-engine", type=Path, default=None,
                        help="姿态 TensorRT engine (默认按后端选择)")
    parser.add_argument("--host", default="0.0.0.0", help="管理服务 bind 地址 (默认: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8080, help="管理服务端口 (默认: 8080)")
    parser.add_argument("--fps", type=int, default=15, help="WebRTC 输出帧率 (默认: 15)")
    parser.add_argument("--bitrate-kbps", type=int, default=1500,
                        help="WebRTC 视频比特率 (默认: 1500 kbps)")
    parser.add_argument("--score-threshold", type=float, default=0.35,
                        help="人体检测置信度阈值 (默认: 0.35)")
    parser.add_argument("--kpt-threshold", type=float, default=0.3,
                        help="关键点渲染置信度阈值 (默认: 0.3)")
    args = parser.parse_args()
    if args.pose_engine is None:
        args.pose_engine = (DEFAULT_RTMPOSE_ENGINE if args.backend == "rtmpose"
                            else DEFAULT_YOLOPOSE_ENGINE)
    return args


def check_assets(args: argparse.Namespace) -> None:
    required = [args.video, args.pose_engine, VIEWER_HTML]
    if args.backend == "rtmpose":
        required.append(args.det_engine)
    missing = [p for p in required if not p.exists()]
    if not missing:
        return
    _say("缺少必要资源:")
    for p in missing:
        _say(f"  - {p}")
    _say("\n模型转换参考 models/rtmpose/convert.sh 与 models/yolov8_pose/convert.sh。")
    sys.exit(2)


def check_webrtc_build() -> None:
    probe = visionpipe.WebRTCSink()
    peer_id = probe.create_peer()
    if not peer_id:
        _say("ERROR: WebRTCSink 是 stub 模式 (未启用 -DVISIONPIPE_USE_WEBRTC=ON)。")
        sys.exit(3)
    probe.remove_peer(peer_id)


def build_pipeline(args: argparse.Namespace) -> visionpipe.Pipeline:
    src_cfg = visionpipe.SourceConfig(str(args.video))
    src_cfg.decode_mode = visionpipe.DecodeMode.AUTO
    src_cfg.loop = True
    src_cfg.queue_capacity = 8
    src_cfg.overflow_policy = visionpipe.OverflowPolicy.BLOCK
    source = visionpipe.FileSource(src_cfg)

    ann_cfg = visionpipe.AnnotatorConfig()
    ann_cfg.draw_detections = True
    ann_cfg.draw_tracks = False
    ann_cfg.draw_masks = False
    ann_cfg.draw_keypoints = True
    ann_cfg.kpt_score_threshold = args.kpt_threshold
    ann_cfg.class_names = ["person"]
    annotator = visionpipe.AnnotatorNode(ann_cfg, "annotator")

    rtc_cfg = visionpipe.WebRTCSinkConfig()
    rtc_cfg.fps = args.fps
    rtc_cfg.video_bitrate_kbps = args.bitrate_kbps
    rtc_cfg.keyframe_interval = max(args.fps * 2, 30)
    rtc_cfg.use_nvenc = True
    rtc_cfg.stun_server = "stun:stun.l.google.com:19302"
    webrtc = visionpipe.WebRTCSink(rtc_cfg, "webrtc_sink")

    if args.backend == "rtmpose":
        det_engine = visionpipe.TrtModelEngine(str(args.det_engine))
        det_cfg = visionpipe.DetectorConfig()
        det_cfg.score_threshold = args.score_threshold
        detector = visionpipe.DetectorNode(det_engine, det_cfg, "detector")

        pose_engine = visionpipe.TrtModelEngine(str(args.pose_engine))
        pose_cfg = visionpipe.RtmPoseConfig()
        pose_cfg.target_classes = [0]  # person
        pose = visionpipe.RtmPoseNode(pose_engine, pose_cfg, "rtmpose")

        return source >> detector >> pose >> annotator >> webrtc

    pose_engine = visionpipe.TrtModelEngine(str(args.pose_engine))
    pose_cfg = visionpipe.YoloPoseConfig()
    pose_cfg.score_threshold = args.score_threshold
    # 启用帧级 batch（需 models/yolov8_pose/convert.sh 导出的动态 batch engine）
    # pose_cfg.max_batch_size = 4
    pose = visionpipe.YoloPoseNode(pose_engine, pose_cfg, "yolo_pose")

    return source >> pose >> annotator >> webrtc


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

    _say(f"[pose-demo] 后端     : {args.backend}")
    _say(f"[pose-demo] 视频     : {args.video}")
    if args.backend == "rtmpose":
        _say(f"[pose-demo] 检测     : {args.det_engine}")
    _say(f"[pose-demo] 姿态     : {args.pose_engine}")
    _say("")

    manager = visionpipe.PipelineManager()
    pipeline = build_pipeline(args)
    pipeline_id = manager.create_pipeline(pipeline)
    manager.start(pipeline_id)
    _say(f"[pose-demo] Pipeline ID: {pipeline_id}")

    server = ManagementServer(manager, host=args.host, port=args.port)
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
        _say("\n[pose-demo] 清理中 ...")

        async def _shutdown() -> None:
            await server.stop()

        try:
            asyncio.run_coroutine_threadsafe(_shutdown(), server_loop).result(timeout=10)
        except Exception as exc:
            _say(f"[pose-demo] server.stop 异常: {exc}")
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
            _say(f"[pose-demo] pipeline.stop 异常: {exc}")
        try:
            manager.destroy(pipeline_id)
        except Exception as exc:
            _say(f"[pose-demo] pipeline.destroy 异常: {exc}")

        _say("[pose-demo] 退出完毕")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
