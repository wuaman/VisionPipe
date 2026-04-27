#!/bin/bash
# YOLOv8 ONNX → TensorRT Engine 转换脚本
#
# 用法:
#   ./convert.sh [options]
#
# 选项:
#   --model PATH       ONNX 模型路径（默认: yolov8n.onnx）
#   --output PATH      输出 engine 路径（默认: yolov8n.engine）
#   --fp16             启用 FP16 精度（默认）
#   --fp32             使用 FP32 精度
#   --int8             启用 INT8 精度（需要校准数据）
#   --batch-size N     批量大小（默认: 1）
#   --dynamic-batch    启用动态批量
#   --workspace N      工作空间大小 MB（默认: 4096）
#   --verbose          显示详细日志
#
# 示例:
#   ./convert.sh --model yolov8n.onnx --fp16
#   ./convert.sh --model yolov8n.onnx --dynamic-batch --batch-size 4

set -e

# 默认参数
MODEL="yolov8n.onnx"
OUTPUT=""
PRECISION="fp16"
BATCH_SIZE=1
DYNAMIC_BATCH=false
WORKSPACE=4096
VERBOSE=false

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --output)
            OUTPUT="$2"
            shift 2
            ;;
        --fp16)
            PRECISION="fp16"
            shift
            ;;
        --fp32)
            PRECISION="fp32"
            shift
            ;;
        --int8)
            PRECISION="int8"
            shift
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --dynamic-batch)
            DYNAMIC_BATCH=true
            shift
            ;;
        --workspace)
            WORKSPACE="$2"
            shift 2
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            head -30 "$0" | tail -28
            exit 0
            ;;
        *)
            echo "未知选项: $1"
            exit 1
            ;;
    esac
done

# 检查模型文件
if [[ ! -f "$MODEL" ]]; then
    echo "错误: 模型文件不存在: $MODEL"
    exit 1
fi

# 设置输出路径
if [[ -z "$OUTPUT" ]]; then
    BASENAME=$(basename "$MODEL" .onnx)
    OUTPUT="${BASENAME}_${PRECISION}.engine"
fi

echo "=========================================="
echo "YOLOv8 ONNX → TensorRT 转换"
echo "=========================================="
echo "输入模型: $MODEL"
echo "输出引擎: $OUTPUT"
echo "精度: $PRECISION"
echo "批量大小: $BATCH_SIZE"
echo "动态批量: $DYNAMIC_BATCH"
echo "工作空间: ${WORKSPACE}MB"
echo "=========================================="

# 构建 trtexec 命令
TRT_CMD="trtexec"

# 基本参数
TRT_CMD+=" --onnx=$MODEL"
TRT_CMD+=" --saveEngine=$OUTPUT"
TRT_CMD+=" --workspace=$WORKSPACE"

# 精度设置
case $PRECISION in
    fp16)
        TRT_CMD+=" --fp16"
        ;;
    int8)
        TRT_CMD+=" --int8"
        # INT8 需要校准数据，这里暂不实现
        echo "警告: INT8 精度需要校准数据，请确保已准备好"
        ;;
    fp32)
        # FP32 是默认精度，无需额外参数
        ;;
esac

# 批量设置
if [[ "$DYNAMIC_BATCH" == "true" ]]; then
    # 动态批量：设置最小/最优/最大批量
    MIN_BATCH=1
    OPT_BATCH=$BATCH_SIZE
    MAX_BATCH=$((BATCH_SIZE * 2))
    TRT_CMD+=" --minShapes=input:1x3x640x640"
    TRT_CMD+=" --optShapes=input:${OPT_BATCH}x3x640x640"
    TRT_CMD+=" --maxShapes=input:${MAX_BATCH}x3x640x640"
else
    # 静态批量
    TRT_CMD+=" --shapes=input:${BATCH_SIZE}x3x640x640"
fi

# 详细日志
if [[ "$VERBOSE" == "true" ]]; then
    TRT_CMD+=" --verbose"
fi

echo "执行命令:"
echo "$TRT_CMD"
echo ""

# 执行转换
eval $TRT_CMD

# 检查输出
if [[ -f "$OUTPUT" ]]; then
    ENGINE_SIZE=$(du -h "$OUTPUT" | cut -f1)
    echo ""
    echo "=========================================="
    echo "转换成功!"
    echo "输出引擎: $OUTPUT"
    echo "引擎大小: $ENGINE_SIZE"
    echo "=========================================="
else
    echo "错误: 转换失败，未生成引擎文件"
    exit 1
fi
