#!/bin/bash
# RTMPose ONNX → TensorRT Engine 转换脚本
#
# ONNX 来源: OpenMMLab 官方 SDK 包（mmdeploy 导出的 end2end.onnx, SimCC 双输出）
#   rtmpose-m 256x192 body7:
#   https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip
#
# 用法:
#   ./convert.sh [--model PATH] [--output PATH] [--max-batch N]
#
# 输出契约（RtmPoseNode 依赖）:
#   input   : [batch, 3, 256, 192]  动态 batch
#   simcc_x : [batch, K, 384]  (192 * split_ratio 2)
#   simcc_y : [batch, K, 512]  (256 * split_ratio 2)

set -e

MODEL="20230831/rtmpose_onnx/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504/end2end.onnx"
OUTPUT="../../tests/models/rtmpose-m_body7_fp16.engine"
MAX_BATCH=16
OPT_BATCH=4

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)     MODEL="$2"; shift 2 ;;
        --output)    OUTPUT="$2"; shift 2 ;;
        --max-batch) MAX_BATCH="$2"; shift 2 ;;
        --opt-batch) OPT_BATCH="$2"; shift 2 ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

TRTEXEC=${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}

"$TRTEXEC" \
    --onnx="$MODEL" \
    --saveEngine="$OUTPUT" \
    --fp16 \
    --minShapes=input:1x3x256x192 \
    --optShapes=input:${OPT_BATCH}x3x256x192 \
    --maxShapes=input:${MAX_BATCH}x3x256x192

echo "转换完成: $OUTPUT"
