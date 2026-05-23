"""VisionPipe-py 快速入门示例。

演示最小可运行 pipeline:

    FileSource -> DetectorNode -> JsonResultSink

依赖资源（默认从仓库内查找，可用 CLI 参数覆盖）:
- 视频:  tests/data/48-3.mp4
- 引擎:  tests/models/yolov8n_dynamic.engine

10 分钟内跑通流程:
1. 按 README 完成依赖安装与 cmake/uv 构建
2. 在仓库根目录执行:
       uv run python examples/quickstart.py
3. 控制台会打印前若干帧的 JSON 推理结果

运行约 5 秒后自动停止，并输出每秒处理帧数 (FPS) 与累计检测目标统计。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

import visionpipe  # noqa: E402

DEFAULT_VIDEO = REPO_ROOT / "tests" / "data" / "video_4k_test.mp4"
DEFAULT_ENGINE = REPO_ROOT / "tests" / "models" / "yolov8n_dynamic.engine"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VisionPipe-py quickstart demo")
    parser.add_argument(
        "--video",
        type=Path,
        default=DEFAULT_VIDEO,
        help=f"输入视频路径 (默认: {DEFAULT_VIDEO})",
    )
    parser.add_argument(
        "--engine",
        type=Path,
        default=DEFAULT_ENGINE,
        help=f"YOLOv8 TensorRT engine 路径 (默认: {DEFAULT_ENGINE})",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="运行时长 (秒)；到时间后调用 pipeline.stop() (默认: 5.0)",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.25,
        help="检测分数阈值 (默认: 0.25)",
    )
    parser.add_argument(
        "--print-frames",
        type=int,
        default=3,
        help="打印前 N 帧的完整 JSON 结果 (默认: 3)",
    )
    return parser.parse_args()


def check_assets(video: Path, engine: Path) -> None:
    missing = [p for p in (video, engine) if not p.exists()]
    if not missing:
        return

    print("缺少必要资源:", file=sys.stderr)
    for p in missing:
        print(f"  - {p}", file=sys.stderr)
    print(
        "\n请按 README 下载测试视频与转换 YOLOv8 engine，或通过 --video / --engine 显式指定路径。",
        file=sys.stderr,
    )
    sys.exit(2)


def build_pipeline(
    video: Path, engine_path: Path, score_threshold: float
) -> tuple[visionpipe.Pipeline, visionpipe.JsonResultSink, visionpipe.TrtModelEngine]:
    src_cfg = visionpipe.SourceConfig(str(video))
    src_cfg.decode_mode = visionpipe.DecodeMode.AUTO
    src_cfg.loop = True
    src_cfg.queue_capacity = 8
    src_cfg.overflow_policy = visionpipe.OverflowPolicy.BLOCK
    source = visionpipe.FileSource(src_cfg)

    engine = visionpipe.TrtModelEngine(str(engine_path))

    det_cfg = visionpipe.DetectorConfig()
    det_cfg.score_threshold = score_threshold
    detector = visionpipe.DetectorNode(engine, det_cfg, "detector")

    sink_cfg = visionpipe.JsonResultSinkConfig()
    sink_cfg.include_detections = True
    sink_cfg.include_tracks = False
    sink_cfg.buffer_capacity = 1024
    sink = visionpipe.JsonResultSink(sink_cfg, "sink")

    # DSL: >> 运算符直接构建 Pipeline，无需手动 add_node/connect
    pipeline = source >> detector >> sink
    return pipeline, sink, engine


def main() -> int:
    args = parse_args()
    check_assets(args.video, args.engine)

    print(f"[quickstart] 视频   : {args.video}")
    print(f"[quickstart] 引擎   : {args.engine}")
    print(f"[quickstart] 阈值   : {args.score_threshold}")
    print(f"[quickstart] 时长   : {args.duration} s\n")

    pipeline, sink, _engine = build_pipeline(args.video, args.engine, args.score_threshold)

    pipeline.run(block=False)
    start_ts = time.monotonic()
    deadline = start_ts + args.duration

    printed = 0
    total_frames = 0
    class_counter: Counter[int] = Counter()
    try:
        while time.monotonic() < deadline:
            payload = sink.pop_json(200)
            if payload is None:
                continue
            record = json.loads(payload)
            total_frames += 1
            for det in record.get("detections", []):
                class_counter[det["class_id"]] += 1
            if printed < args.print_frames:
                print(f"--- frame {record['frame_id']} ---")
                print(json.dumps(record, indent=2, ensure_ascii=False))
                printed += 1
    finally:
        pipeline.stop()

    elapsed = max(time.monotonic() - start_ts, 1e-6)
    fps = total_frames / elapsed
    print("\n[quickstart] 运行汇总")
    print(f"  累计帧数: {total_frames}")
    print(f"  实测 FPS: {fps:.2f}")
    if class_counter:
        top = ", ".join(f"class {cid}: {cnt}" for cid, cnt in class_counter.most_common(5))
        print(f"  Top 类别 (前 5): {top}")
    else:
        print("  未检测到目标 — 可尝试调低 --score-threshold")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
