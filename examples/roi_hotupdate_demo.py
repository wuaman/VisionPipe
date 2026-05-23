"""VisionPipe-py ROI 热更新示例。

演示运行时通过**管理通道** (WebSocket `/ws/<id>/control`) 动态修改
DetectorNode 的检测 ROI 区域,效果通过浏览器实时观察。

预览方式两选一 (--sink, 默认 mjpeg):

    --sink mjpeg   FileSource → DetectorNode → AnnotatorNode → MjpegSink
                   浏览器走 multipart/x-mixed-replace, 无需 WebRTC 构建
    --sink webrtc  FileSource → DetectorNode → AnnotatorNode → WebRTCSink
                   浏览器走 RTP/H.264, 延迟更低; 需 -DVISIONPIPE_USE_WEBRTC=ON

ROI 切换序列 (默认 5 s 一档, 循环):

    阶段 0  →  clear           (无 ROI, 全图检测)
    阶段 1  →  左半屏 (矩形)
    阶段 2  →  右半屏 (矩形)
    阶段 3  →  中心三角形
    回到阶段 0

每次切换通过 WS 发送 ``{"type":"roi", "polygons":[[x,y],…]}`` (归一化坐标)。
浏览器观察检测框只在 ROI 范围内出现 (中心点判定)。

注意 AnnotatorNode 当前 **不绘制 ROI 多边形本身**, 只能间接观察
"检测框出现/消失" 验证 ROI 生效。

依赖资源:
- 视频:  tests/data/48-3.mp4
- 检测:  tests/models/yolov8n_dynamic.engine
- --sink webrtc 额外需要: examples/webrtc_viewer.html (随仓库提供)

运行
----
    uv run python examples/roi_hotupdate_demo.py                 # 默认 MJPEG
    uv run python examples/roi_hotupdate_demo.py --sink webrtc   # WebRTC 预览

启动后浏览器打开终端中打印的 viewer URL (内嵌 <img>/WebRTC <video>), 切换日志
会每 5 s 在终端输出。Ctrl+C 退出。

> 直接打 http://localhost:8080/mjpeg/<pid> 在某些 Chrome 版本下不渲染
> (`multipart/x-mixed-replace` 兼容性问题); 这正是默认提供 `/viewer`
> 内嵌 <img> 页面的原因。Firefox 直接访问通常正常。

手动触发 (任意时刻另开终端运行)
-------------------------------
通过 **WS 控制通道** (polygons 是 [x,y] 点对列表):

    echo '{"type":"roi","polygons":[[0,0],[0.5,0],[0.5,1],[0,1]],"coord":"normalized"}' \\
        | websocat ws://localhost:8080/ws/<pid>/control
    echo '{"type":"roi_clear"}' | websocat ws://localhost:8080/ws/<pid>/control

通过 **REST set_param** (value 是扁平 [x1,y1,x2,y2,…] 列表):

    curl -X POST http://localhost:8080/pipelines/<pid>/params \\
         -H 'Content-Type: application/json' \\
         -d '{"node_id":"detector","param_name":"roi",
              "value":[0.0,0.0,0.5,0.0,0.5,1.0,0.0,1.0]}'

参数
----
    --sink {mjpeg,webrtc} 预览方式 (默认: mjpeg)
    --no-rotate           关闭自动轮换 (教学模式: 只起服务等手动触发)
    --interval FLOAT      自动轮换间隔秒数, 默认 5.0
    --fps / --bitrate-kbps  仅 --sink webrtc 生效
"""

from __future__ import annotations

import argparse
import asyncio
import json
import signal
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import aiohttp
from aiohttp import web

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

import visionpipe  # noqa: E402
from visionpipe.server import ManagementServer  # noqa: E402

DEFAULT_VIDEO = REPO_ROOT / "tests" / "data" / "48-3.mp4"
DEFAULT_DET_ENGINE = REPO_ROOT / "tests" / "models" / "yolov8n_dynamic.engine"
WEBRTC_VIEWER_HTML = Path(__file__).parent / "webrtc_viewer.html"

MJPEG_VIEWER_HTML_TPL = """<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>VisionPipe-py ROI Hot-Update (MJPEG)</title>
<style>
  body {{ font-family: -apple-system,"Segoe UI",sans-serif; background:#1e1e1e; color:#ddd;
         margin:0; padding:16px; }}
  h1 {{ margin:0 0 10px 0; font-size:18px; }}
  .meta {{ font-size:13px; color:#999; margin-bottom:8px; }}
  .meta code {{ background:#2d2d2d; padding:2px 6px; border-radius:3px; color:#6cf; }}
  img {{ background:#000; border:1px solid #444; max-width:100%; height:auto; display:block; }}
</style></head><body>
<h1>VisionPipe-py · ROI Hot-Update Viewer (MJPEG)</h1>
<div class="meta">Pipeline ID: <code>{pid}</code> · Stream: <code>{stream_url}</code></div>
<img src="{stream_url}" alt="MJPEG stream">
<div class="meta" style="margin-top:8px">每 5 s 自动循环切换 ROI: clear → 左半屏 → 右半屏 → 中心三角形</div>
</body></html>
"""

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


@dataclass
class RoiStage:
    label: str
    payload: dict


# 归一化坐标 [0, 1], 内层每元素是 [x, y]
ROI_STAGES: list[RoiStage] = [
    RoiStage("clear (无 ROI, 全图检测)", {"type": "roi_clear"}),
    RoiStage(
        "left half (左半屏)",
        {
            "type": "roi",
            "polygons": [[0.0, 0.0], [0.5, 0.0], [0.5, 1.0], [0.0, 1.0]],
            "coord": "normalized",
        },
    ),
    RoiStage(
        "right half (右半屏)",
        {
            "type": "roi",
            "polygons": [[0.5, 0.0], [1.0, 0.0], [1.0, 1.0], [0.5, 1.0]],
            "coord": "normalized",
        },
    ),
    RoiStage(
        "center triangle (中心三角形)",
        {
            "type": "roi",
            "polygons": [[0.3, 0.2], [0.7, 0.2], [0.5, 0.85]],
            "coord": "normalized",
        },
    ),
]


def _say(msg: str = "") -> None:
    print(msg, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VisionPipe-py ROI 热更新 demo (管理通道 + MJPEG 浏览器)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--det-engine", type=Path, default=DEFAULT_DET_ENGINE)
    parser.add_argument("--host", default="0.0.0.0", help="管理服务 bind 地址 (默认: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--interval", type=float, default=5.0,
                        help="ROI 自动轮换间隔秒数 (默认: 5.0)")
    parser.add_argument("--score-threshold", type=float, default=0.25)
    parser.add_argument("--sink", choices=["mjpeg", "webrtc"], default="mjpeg",
                        help="预览方式: mjpeg (默认, 无需 WebRTC 构建) "
                             "或 webrtc (需 -DVISIONPIPE_USE_WEBRTC=ON)")
    parser.add_argument("--fps", type=int, default=15,
                        help="--sink webrtc 时的编码帧率 (默认: 15)")
    parser.add_argument("--bitrate-kbps", type=int, default=1500,
                        help="--sink webrtc 时的视频码率 (默认: 1500)")
    parser.add_argument("--no-rotate", action="store_true",
                        help="关闭自动轮换, 只启动服务等手动触发")
    return parser.parse_args()


def check_assets(args: argparse.Namespace) -> None:
    required = [args.video, args.det_engine]
    if args.sink == "webrtc":
        required.append(WEBRTC_VIEWER_HTML)
    missing = [p for p in required if not p.exists()]
    if missing:
        _say("缺少必要资源:")
        for p in missing:
            _say(f"  - {p}")
        sys.exit(2)


def check_webrtc_build() -> None:
    """启动前确认扩展带 WebRTC; stub 模式会让浏览器一直 connecting。"""
    probe = visionpipe.WebRTCSink()
    peer_id = probe.create_peer()
    if not peer_id:
        _say("ERROR: WebRTCSink 是 stub 模式 (未启用 -DVISIONPIPE_USE_WEBRTC=ON)。")
        _say("请重新构建后再用 --sink webrtc:")
        _say("    cmake -B build -DVISIONPIPE_USE_WEBRTC=ON")
        _say("    cmake --build build --target visionpipe_python")
        sys.exit(3)
    probe.remove_peer(peer_id)


def build_pipeline(args: argparse.Namespace) -> visionpipe.Pipeline:
    src_cfg = visionpipe.SourceConfig(str(args.video))
    src_cfg.decode_mode = visionpipe.DecodeMode.AUTO
    src_cfg.loop = True
    src_cfg.queue_capacity = 8
    src_cfg.overflow_policy = visionpipe.OverflowPolicy.BLOCK
    source = visionpipe.FileSource(src_cfg)

    det_engine = visionpipe.TrtModelEngine(str(args.det_engine))
    det_cfg = visionpipe.DetectorConfig()
    det_cfg.score_threshold = args.score_threshold
    # name 必须是 "detector", control_ws 通过 isinstance(DetectorNode) 自动查找,
    # 这里用显式 name 同时方便走 REST /params 路径 (node_id="detector")
    detector = visionpipe.DetectorNode(det_engine, det_cfg, "detector")

    ann_cfg = visionpipe.AnnotatorConfig()
    ann_cfg.draw_detections = True
    ann_cfg.draw_tracks = False
    ann_cfg.draw_masks = False
    ann_cfg.class_names = COCO_80
    annotator = visionpipe.AnnotatorNode(ann_cfg, "annotator")

    if args.sink == "webrtc":
        rtc_cfg = visionpipe.WebRTCSinkConfig()
        rtc_cfg.fps = args.fps
        rtc_cfg.video_bitrate_kbps = args.bitrate_kbps
        rtc_cfg.keyframe_interval = max(args.fps * 2, 30)
        rtc_cfg.use_nvenc = True
        rtc_cfg.stun_server = "stun:stun.l.google.com:19302"
        sink = visionpipe.WebRTCSink(rtc_cfg, "webrtc_sink")
    else:
        sink_cfg = visionpipe.MjpegSinkConfig()
        sink_cfg.jpeg_quality = 80
        sink_cfg.buffer_capacity = 4
        sink = visionpipe.MjpegSink(sink_cfg, "mjpeg")

    return source >> detector >> annotator >> sink


async def _roi_rotator(ws_url: str, interval: float, start_ts: float,
                       stop_event: asyncio.Event) -> None:
    """循环切换 ROI: stage 0 → 1 → 2 → 3 → 0 ..."""
    backoff = 1.0
    while not stop_event.is_set():
        try:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(ws_url, timeout=10) as ws:
                    _say(f"[roi-demo] 控制 WS 已连接: {ws_url}")
                    idx = 0
                    while not stop_event.is_set():
                        stage = ROI_STAGES[idx]
                        elapsed = time.monotonic() - start_ts
                        _say(f"[t={elapsed:5.1f}s] ROI → {stage.label} ... ", )
                        await ws.send_str(json.dumps(stage.payload))
                        try:
                            resp = await asyncio.wait_for(ws.receive_str(), timeout=3.0)
                            data = json.loads(resp)
                            if data.get("type") == "ack":
                                _say(f"           ack ref_type={data.get('ref_type')}")
                            else:
                                _say(f"           UNEXPECTED: {data}")
                        except asyncio.TimeoutError:
                            _say("           (ack 超时, 继续)")
                        idx = (idx + 1) % len(ROI_STAGES)
                        try:
                            await asyncio.wait_for(stop_event.wait(), timeout=interval)
                        except asyncio.TimeoutError:
                            pass
                    break  # stop_event set
        except (aiohttp.ClientError, OSError) as exc:
            _say(f"[roi-demo] 控制 WS 异常: {exc}, {backoff:.1f}s 后重试")
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=backoff)
            except asyncio.TimeoutError:
                pass
            backoff = min(backoff * 2, 10.0)


def _serve_in_thread(server: ManagementServer, loop: asyncio.AbstractEventLoop,
                     ready: threading.Event) -> None:
    asyncio.set_event_loop(loop)
    loop.run_until_complete(server.start())
    ready.set()
    loop.run_forever()


def main() -> int:
    args = parse_args()
    check_assets(args)
    if args.sink == "webrtc":
        check_webrtc_build()

    _say(f"[roi-demo] 视频     : {args.video}")
    _say(f"[roi-demo] 检测     : {args.det_engine}")
    _say(f"[roi-demo] 切换间隔 : {args.interval:.1f}s (--no-rotate: {args.no_rotate})")
    _say(f"[roi-demo] 预览方式 : {args.sink}")
    _say("")

    manager = visionpipe.PipelineManager()
    pipeline = build_pipeline(args)
    pipeline_id = manager.create_pipeline(pipeline)
    manager.start(pipeline_id)

    server = ManagementServer(manager, host=args.host, port=args.port)

    # 根据 sink 类型挂内嵌 viewer 页面 (规避浏览器对 multipart 的兼容问题
    # 以及 WebRTC 需要 HTML 信令 client)
    if args.sink == "mjpeg":
        stream_path = f"/mjpeg/{pipeline_id}"

        async def _mjpeg_index(_req: web.Request) -> web.Response:
            html = MJPEG_VIEWER_HTML_TPL.format(pid=pipeline_id, stream_url=stream_path)
            return web.Response(text=html, content_type="text/html", charset="utf-8")

        server._app.router.add_get("/", _mjpeg_index)
        server._app.router.add_get("/viewer", _mjpeg_index)
    else:
        async def _webrtc_index(_req: web.Request) -> web.Response:
            html = WEBRTC_VIEWER_HTML.read_text(encoding="utf-8")
            return web.Response(text=html, content_type="text/html", charset="utf-8")

        server._app.router.add_get("/", _webrtc_index)
        server._app.router.add_get("/viewer", _webrtc_index)

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
    base_http = f"http://{display_host}:{args.port}"
    base_ws = f"ws://{display_host}:{args.port}"
    if args.sink == "mjpeg":
        viewer_url = f"{base_http}/viewer"
        stream_url = f"{base_http}/mjpeg/{pipeline_id}"
        stream_label = "MJPEG 原始流"
    else:
        viewer_url = f"{base_http}/?pid={pipeline_id}"
        stream_url = f"{base_ws}/ws/{pipeline_id}/webrtc"
        stream_label = "WebRTC 信令"

    _say("=" * 70)
    _say(f"  Pipeline ID  : {pipeline_id}")
    _say(f"  浏览器 viewer : {viewer_url}")
    _say(f"  {stream_label:<12s} : {stream_url}")
    _say("=" * 70)
    _say("")
    _say("手动触发 ROI 切换 (任意时刻):")
    _say("  # 通过 WS (polygons = [[x,y],…] 点对)")
    _say("  echo '{\"type\":\"roi\",\"polygons\":[[0,0],[0.5,0],[0.5,1],[0,1]]}' \\")
    _say(f"      | websocat {base_ws}/ws/{pipeline_id}/control")
    _say(f"  echo '{{\"type\":\"roi_clear\"}}' | websocat {base_ws}/ws/{pipeline_id}/control")
    _say("")
    _say("  # 通过 REST (value = 扁平 [x1,y1,x2,y2,…])")
    _say(f"  curl -X POST {base_http}/pipelines/{pipeline_id}/params \\")
    _say("       -H 'Content-Type: application/json' \\")
    _say("       -d '{\"node_id\":\"detector\",\"param_name\":\"roi\",")
    _say("            \"value\":[0.0,0.0,0.5,0.0,0.5,1.0,0.0,1.0]}'")
    _say("")
    _say("按 Ctrl+C 退出")
    _say("")

    rotator_stop = asyncio.Event()
    rotator_future = None
    if not args.no_rotate:
        ws_url = f"{base_ws}/ws/{pipeline_id}/control"
        rotator_future = asyncio.run_coroutine_threadsafe(
            _roi_rotator(ws_url, args.interval, time.monotonic(), rotator_stop),
            server_loop,
        )

    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())

    try:
        while not stop.is_set():
            time.sleep(0.5)
    finally:
        _say("\n[roi-demo] 清理中 ...")

        if rotator_future is not None:
            server_loop.call_soon_threadsafe(rotator_stop.set)
            try:
                rotator_future.result(timeout=5)
            except Exception:
                pass

        async def _shutdown() -> None:
            await server.stop()

        try:
            asyncio.run_coroutine_threadsafe(_shutdown(), server_loop).result(timeout=10)
        except Exception as exc:
            _say(f"[roi-demo] server.stop 异常: {exc}")
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
            _say(f"[roi-demo] pipeline.stop 异常: {exc}")
        try:
            manager.destroy(pipeline_id)
        except Exception as exc:
            _say(f"[roi-demo] pipeline.destroy 异常: {exc}")

        _say("[roi-demo] 退出完毕")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
