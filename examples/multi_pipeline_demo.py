"""多 Pipeline 并发演示。

演示同进程内同时运行两条 Pipeline，验证 T5.1 设计目标:

- 两路 Pipeline 共享同一个 TrtModelEngine (ModelRegistry/显存复用)
- 各自独立 Source / Sink / 生命周期 (一路停止不影响另一路)
- 通过 PipelineManager 统一管理 create -> start -> stop -> destroy

场景设置 (基于 COCO class id, 消费端过滤):
- pipe-people : 只统计 person 系列 (person=0)
- pipe-animal : 只统计动物系列 (cat=15, dog=16, horse=17, sheep=18, cow=19,
                                 elephant=20, bear=21, zebra=22, giraffe=23)

两路 Pipeline 各自的 JsonResultSink 在独立线程中被消费,
每秒打印一次实测 FPS 与累计目标统计, 结束时确认类别集合不相交。

依赖资源 (默认从仓库内查找, 可用 CLI 参数覆盖):
- 视频:  tests/data/48-3.mp4
- 引擎:  tests/models/yolov8n_dynamic.engine

运行:
    uv run python examples/multi_pipeline_demo.py
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

import visionpipe  # noqa: E402

DEFAULT_VIDEO = REPO_ROOT / "tests" / "data" / "48-3.mp4"
DEFAULT_ENGINE = REPO_ROOT / "tests" / "models" / "yolov8n_dynamic.engine"

VEHICLE_CLASS_IDS = {15, 16, 17, 18, 19, 20, 21, 22, 23}  # COCO 动物类
PERSON_CLASS_IDS = {0}


def _say(msg: str = "") -> None:
    """统一 print + flush, 避免 stdout 缓冲与 spdlog 输出交错时丢失。"""
    print(msg, flush=True)


@dataclass
class PipelineRecord:
    name: str
    keep_classes: set[int]
    pipeline: object
    sink: object
    pid: str = ""
    frames: int = 0
    class_count: Counter = field(default_factory=Counter)
    stop_event: threading.Event = field(default_factory=threading.Event)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VisionPipe-py 多 Pipeline 并发演示")
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--engine", type=Path, default=DEFAULT_ENGINE)
    parser.add_argument(
        "--duration",
        type=float,
        default=6.0,
        help="同时运行时长 (秒)",
    )
    parser.add_argument(
        "--early-stop-traffic-at",
        type=float,
        default=3.0,
        help="提前在第 N 秒 stop 'pipe-people', 演示生命周期隔离 (设 0 表示禁用)",
    )
    return parser.parse_args()


def check_assets(video: Path, engine: Path) -> None:
    missing = [p for p in (video, engine) if not p.exists()]
    if missing:
        _say("缺少必要资源:")
        for p in missing:
            _say(f"  - {p}")
        _say("\n请按 README 准备视频与 engine。")
        sys.exit(2)


def _build_one_pipeline(
    video: Path,
    engine: visionpipe.TrtModelEngine,
    name: str,
) -> PipelineRecord:
    """构建一条 FileSource -> DetectorNode -> JsonResultSink pipeline。"""
    src_cfg = visionpipe.SourceConfig(str(video))
    src_cfg.decode_mode = visionpipe.DecodeMode.AUTO
    src_cfg.loop = True
    src_cfg.queue_capacity = 8
    # DetectorNode 是 InferNode, 输出按 frame_id 排序, 必须用 BLOCK 避免空洞
    src_cfg.overflow_policy = visionpipe.OverflowPolicy.BLOCK
    source = visionpipe.FileSource(src_cfg)

    detector = visionpipe.DetectorNode(engine, visionpipe.DetectorConfig(), f"det-{name}")

    sink_cfg = visionpipe.JsonResultSinkConfig()
    sink_cfg.include_detections = True
    sink_cfg.include_tracks = False
    sink_cfg.buffer_capacity = 1024
    sink = visionpipe.JsonResultSink(sink_cfg, f"sink-{name}")

    cfg = visionpipe.PipelineConfig()
    cfg.name = name
    pipeline = visionpipe.Pipeline(cfg)
    pipeline.add_node(source)
    pipeline.add_node(detector)
    pipeline.add_node(sink)
    pipeline.connect(source, detector)
    pipeline.connect(detector, sink)

    return PipelineRecord(
        name=name,
        keep_classes=set(),  # 由 caller 填入
        pipeline=pipeline,
        sink=sink,
    )


def _consume_loop(record: PipelineRecord) -> None:
    """从 sink 拉 JSON, 仅统计 record.keep_classes 内的 class_id。"""
    while not record.stop_event.is_set():
        payload = record.sink.pop_json(200)
        if payload is None:
            continue
        try:
            r = json.loads(payload)
        except json.JSONDecodeError:
            continue
        record.frames += 1
        for det in r.get("detections", []):
            cid = det["class_id"]
            if cid in record.keep_classes:
                record.class_count[cid] += 1


def _safe_stop_destroy(manager: visionpipe.PipelineManager, pid: str) -> None:
    try:
        if manager.status(pid) != visionpipe.PipelineStatus.STOPPED:
            manager.stop(pid)
    except Exception as exc:
        _say(f"[multi-demo] stop({pid}) 异常: {exc}")
    try:
        manager.destroy(pid)
    except Exception as exc:
        _say(f"[multi-demo] destroy({pid}) 异常: {exc}")


def main() -> int:
    args = parse_args()
    check_assets(args.video, args.engine)

    _say(f"[multi-demo] 视频   : {args.video}")
    _say(f"[multi-demo] 引擎   : {args.engine}")
    _say(f"[multi-demo] 时长   : {args.duration} s")
    if args.early_stop_traffic_at > 0:
        _say(f"[multi-demo] 第 {args.early_stop_traffic_at:.1f} s 提前停止 pipe-people 演示生命周期隔离")
    _say("")

    manager = visionpipe.PipelineManager()
    engine = visionpipe.TrtModelEngine(str(args.engine))

    pipes: list[PipelineRecord] = []
    rec_people = _build_one_pipeline(args.video, engine, "pipe-people")
    rec_people.keep_classes = set(PERSON_CLASS_IDS)
    pipes.append(rec_people)

    rec_animal = _build_one_pipeline(args.video, engine, "pipe-animal")
    rec_animal.keep_classes = set(VEHICLE_CLASS_IDS)
    pipes.append(rec_animal)

    consumer_threads: list[threading.Thread] = []
    started_pids: list[str] = []
    start_ts = time.monotonic()
    try:
        for rec in pipes:
            rec.pid = manager.create_pipeline(rec.pipeline)
            started_pids.append(rec.pid)
            manager.start(rec.pid)
            t = threading.Thread(target=_consume_loop, args=(rec,), daemon=True)
            t.start()
            consumer_threads.append(t)

        start_ts = time.monotonic()
        traffic_stopped = False
        last_print = start_ts
        while time.monotonic() - start_ts < args.duration:
            now = time.monotonic()
            if now - last_print >= 1.0:
                elapsed = now - start_ts
                status_parts = []
                for rec in pipes:
                    try:
                        st = manager.status(rec.pid).name
                    except Exception:
                        st = "?"
                    status_parts.append(
                        f"{rec.name}: frames={rec.frames:4d} ({rec.frames / max(elapsed, 1e-6):5.1f} fps), state={st}"
                    )
                _say(f"[t={elapsed:5.2f}s] " + " | ".join(status_parts))
                last_print = now

            if (
                args.early_stop_traffic_at > 0
                and not traffic_stopped
                and (now - start_ts) >= args.early_stop_traffic_at
            ):
                _say(f"\n[multi-demo] 主动停止 {pipes[0].name} (t={now - start_ts:.2f}s)")
                manager.stop(pipes[0].pid)
                pipes[0].stop_event.set()
                traffic_stopped = True
                try:
                    other_state = manager.status(pipes[1].pid).name
                except Exception:
                    other_state = "?"
                _say(f"[multi-demo] {pipes[0].name} 已停止; {pipes[1].name} 当前状态 = {other_state}\n")

            time.sleep(0.05)
    finally:
        _say("\n[multi-demo] 进入清理阶段 ...")
        for rec in pipes:
            rec.stop_event.set()
        for pid in started_pids:
            _safe_stop_destroy(manager, pid)
        for t in consumer_threads:
            t.join(timeout=2.0)
        _say("[multi-demo] 清理完成")

    _say("\n[multi-demo] 最终汇总")
    total_elapsed = max(time.monotonic() - start_ts, 1e-6)
    for rec in pipes:
        cls_summary = ", ".join(f"{cid}:{cnt}" for cid, cnt in sorted(rec.class_count.items()))
        _say(
            f"  {rec.name}: frames={rec.frames}, "
            f"avg_fps={rec.frames / total_elapsed:.2f}, "
            f"class_count={{{cls_summary}}}"
        )

    seen_people = set(pipes[0].class_count.keys())
    seen_animal = set(pipes[1].class_count.keys())
    intersection = seen_people & seen_animal
    _say(f"\n[multi-demo] 类别集合检验: people={seen_people} ∩ animal={seen_animal} = {intersection}")
    assert seen_people.issubset(PERSON_CLASS_IDS), f"pipe-people 出现非 person 类别: {seen_people - PERSON_CLASS_IDS}"
    assert seen_animal.issubset(VEHICLE_CLASS_IDS), f"pipe-animal 出现非 animal 类别: {seen_animal - VEHICLE_CLASS_IDS}"
    assert not intersection, f"两路类别不应重叠: {intersection}"
    _say("[multi-demo] OK — 类别隔离与生命周期隔离均符合预期。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
