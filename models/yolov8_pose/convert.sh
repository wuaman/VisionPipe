#!/bin/bash
# YOLOv8-pose ONNX 导出 + TensorRT Engine 转换脚本
#
# 用法:
#   ./convert.sh [--variant yolov8n-pose] [--output PATH]
#
# 输出契约（YoloPoseNode 依赖）:
#   input   : [1, 3, 640, 640]
#   output0 : [1, 56, 8400]  (4 bbox cxcywh + 1 conf + 17*3 关键点)

set -e

VARIANT="yolov8n-pose"
OUTPUT=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --variant) VARIANT="$2"; shift 2 ;;
        --output)  OUTPUT="$2"; shift 2 ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

if [[ -z "$OUTPUT" ]]; then
    OUTPUT="../../tests/models/${VARIANT}_fp16.engine"
fi

TRTEXEC=${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}

# 1) ultralytics 导出 ONNX（首次运行自动下载 .pt 权重）
if [[ ! -f "${VARIANT}.onnx" ]]; then
    yolo export model="${VARIANT}.pt" format=onnx opset=12 imgsz=640
fi

# 2) trtexec 转 FP16 engine
"$TRTEXEC" \
    --onnx="${VARIANT}.onnx" \
    --saveEngine="$OUTPUT" \
    --fp16

echo "转换完成: $OUTPUT"
