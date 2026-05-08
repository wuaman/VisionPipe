"""
VisionPipe demo: YOLOv8n detection on test.mp4

Pipeline:  FileSource -> DetectorNode
Mode:      drain=True, then collect all output frames and render to video.
Output:    demo_output.mp4
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "python"))

import cv2
import numpy as np
import visionpipe as vp

ENGINE_PATH = "tests/models/yolov8n_dynamic.engine"
VIDEO_PATH  = "tests/data/48-3.mp4"
OUTPUT_PATH = "demo_output.mp4"
SCORE_THR   = 0.35

COCO_NAMES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck",
    "boat","traffic light","fire hydrant","stop sign","parking meter","bench",
    "bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe",
    "backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard",
    "sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
    "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl",
    "banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza",
    "donut","cake","chair","couch","potted plant","bed","dining table","toilet",
    "tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven",
    "toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear",
    "hair drier","toothbrush",
]

COLORS = [
    (255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29),
    (207, 210, 49), (72, 249, 10), (146, 204, 23), (61, 219, 134),
    (26, 147, 52), (0, 212, 187), (44, 153, 168), (0, 194, 255),
    (52, 69, 147), (100, 115, 255), (0, 24, 236), (132, 56, 255),
    (82, 0, 133), (203, 56, 255), (255, 149, 200), (255, 55, 199),
]


def draw_detections(bgr: np.ndarray, detections: list) -> None:
    img_h, img_w = bgr.shape[:2]
    for det in detections:
        x1 = int(det.bbox[0] * img_w)
        y1 = int(det.bbox[1] * img_h)
        x2 = int(det.bbox[2] * img_w)
        y2 = int(det.bbox[3] * img_h)
        cid = det.class_id
        color = COLORS[cid % len(COLORS)]
        label = f"{COCO_NAMES[cid] if cid < len(COCO_NAMES) else cid} {det.confidence:.2f}"
        cv2.rectangle(bgr, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(bgr, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(bgr, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


def main() -> None:
    print(f"Loading engine: {ENGINE_PATH}")
    engine = vp.TrtModelEngine(ENGINE_PATH)
    print(f"  output_count={engine.output_count()}  "
          f"device_mem={engine.device_memory_bytes() // 1024 // 1024} MiB")

    det_cfg = vp.DetectorConfig()
    det_cfg.score_threshold = SCORE_THR
    det_cfg.workers = 1

    source_cfg = vp.SourceConfig()
    source_cfg.uri = VIDEO_PATH
    source_cfg.decode_mode = vp.DecodeMode.CPU
    source_cfg.queue_capacity = 1024
    source_cfg.overflow_policy = vp.OverflowPolicy.BLOCK

    source = vp.FileSource(source_cfg)
    detector = vp.DetectorNode(engine, det_cfg, "detector")
    pipeline = (source >> detector).build()

    cap = cv2.VideoCapture(VIDEO_PATH)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    print(f"Video: {w}x{h} @{fps_src:.1f}fps  total={total_frames} frames")
    print("Processing all frames ...")

    pipeline.start()
    t0 = time.perf_counter()

    # Wait for pipeline to finish naturally (FileSource EOF triggers drain)
    pipeline.wait_stop()
    elapsed_infer = time.perf_counter() - t0

    # Collect all output frames after pipeline is done
    results = []
    while True:
        frame = detector.pop_frame(0)
        if frame is None:
            break
        if frame.has_image():
            rgb = np.array(frame.image_numpy(), copy=True)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        else:
            bgr = np.zeros((h, w, 3), dtype=np.uint8)
        results.append((frame.frame_id, bgr, list(frame.detections)))

    print(f"\nInference done: {len(results)} frames in {elapsed_infer:.2f}s  "
          f"({len(results)/elapsed_infer:.1f} fps)")

    # Render to video
    print(f"Rendering {OUTPUT_PATH} ...")
    writer = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*"mp4v"), fps_src, (w, h))
    total_dets = 0
    for fid, bgr, dets in results:
        draw_detections(bgr, dets)
        fps_str = f"{len(results)/elapsed_infer:.1f}"
        cv2.putText(bgr,
                    f"VisionPipe  FPS:{fps_str}  #{fid}  dets:{len(dets)}",
                    (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
        writer.write(bgr)
        total_dets += len(dets)
    writer.release()

    print(f"Total detections: {total_dets}  avg/frame: {total_dets/max(len(results),1):.1f}")
    print(f"Output saved to:  {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
