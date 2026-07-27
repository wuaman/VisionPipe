#!/bin/bash
# YOLOv8-pose ONNX 导出 + TensorRT Engine 转换脚本
#
# 用法:
#   ./convert.sh [--variant yolov8n-pose] [--output PATH] [--max-batch N] [--opt-batch N]
#
# 输出契约（YoloPoseNode 依赖）:
#   input   : [batch, 3, 640, 640]  动态 batch
#   output0 : [batch, 56, 8400]  (4 bbox cxcywh + 1 conf + 17*3 关键点)
#
# 注：ultralytics 导出的 ONNX 输入张量名为 "images"（已验证），故 shape profile 使用 images。

set -e

VARIANT="yolov8n-pose"
OUTPUT=""
MAX_BATCH=16
OPT_BATCH=4

while [[ $# -gt 0 ]]; do
    case $1 in
        --variant)  VARIANT="$2"; shift 2 ;;
        --output)   OUTPUT="$2"; shift 2 ;;
        --max-batch) MAX_BATCH="$2"; shift 2 ;;
        --opt-batch) OPT_BATCH="$2"; shift 2 ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

if [[ -z "$OUTPUT" ]]; then
    OUTPUT="../../tests/models/${VARIANT}_fp16.engine"
fi

TRTEXEC=${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}

# 1) ultralytics 导出 ONNX（动态 batch，覆盖既有静态 onnx）
yolo export model="${VARIANT}.pt" format=onnx opset=12 imgsz=640 dynamic=True

# 2) trtexec 转 FP16 engine（动态 batch shape profile）
"$TRTEXEC" \
    --onnx="${VARIANT}.onnx" \
    --saveEngine="$OUTPUT" \
    --fp16 \
    --minShapes=images:1x3x640x640 \
    --optShapes=images:${OPT_BATCH}x3x640x640 \
    --maxShapes=images:${MAX_BATCH}x3x640x640

echo "转换完成: $OUTPUT"
