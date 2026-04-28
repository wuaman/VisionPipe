#!/bin/bash
# ResNet50 ONNX → TensorRT Engine 转换脚本
#
# 用法:
#   ./convert.sh [options]
#
# 选项:
#   --model PATH       ONNX 模型路径（默认: resnet50.onnx）
#   --output PATH      输出 engine 路径（默认: resnet50_fp16.engine）
#   --fp16             启用 FP16 精度（默认）
#   --fp32             使用 FP32 精度
#   --int8             启用 INT8 精度（需要校准数据）
#   --batch-size N     批量大小（默认: 1）
#   --max-batch N      最大动态批量（默认: 32）
#   --dynamic-batch    启用动态批量
#   --workspace N      工作空间大小 MB（默认: 4096）
#   --input-size W H   输入尺寸（默认: 224 224）
#   --verbose          显示详细日志
#
# 示例:
#   ./convert.sh --model resnet50.onnx --fp16
#   ./convert.sh --model resnet50.onnx --dynamic-batch --batch-size 8 --max-batch 32

set -e

# 默认参数
MODEL="resnet50.onnx"
OUTPUT=""
PRECISION="fp16"
BATCH_SIZE=1
MAX_BATCH=32
DYNAMIC_BATCH=false
WORKSPACE=4096
INPUT_WIDTH=224
INPUT_HEIGHT=224
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
        --max-batch)
            MAX_BATCH="$2"
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
        --input-size)
            INPUT_WIDTH="$2"
            INPUT_HEIGHT="$3"
            shift 3
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            head -35 "$0" | tail -33
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
    echo "提示: 可从以下地址下载 ResNet50 ONNX 模型:"
    echo "  https://github.com/onnx/models/tree/main/vision/classification/resnet"
    exit 1
fi

# 设置输出路径
if [[ -z "$OUTPUT" ]]; then
    BASENAME=$(basename "$MODEL" .onnx)
    OUTPUT="${BASENAME}_${PRECISION}_b${BATCH_SIZE}.engine"
fi

echo "=========================================="
echo "ResNet50 ONNX → TensorRT 转换"
echo "=========================================="
echo "输入模型: $MODEL"
echo "输出引擎: $OUTPUT"
echo "精度: $PRECISION"
echo "输入尺寸: ${INPUT_WIDTH}x${INPUT_HEIGHT}"
echo "批量大小: $BATCH_SIZE"
echo "最大批量: $MAX_BATCH"
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
        echo "警告: INT8 精度需要校准数据，请确保已准备好"
        ;;
    fp32)
        # FP32 是默认精度
        ;;
esac

# 批量设置
if [[ "$DYNAMIC_BATCH" == "true" ]]; then
    MIN_BATCH=1
    OPT_BATCH=$BATCH_SIZE
    TRT_CMD+=" --minShapes=input:${MIN_BATCH}x3x${INPUT_HEIGHT}x${INPUT_WIDTH}"
    TRT_CMD+=" --optShapes=input:${OPT_BATCH}x3x${INPUT_HEIGHT}x${INPUT_WIDTH}"
    TRT_CMD+=" --maxShapes=input:${MAX_BATCH}x3x${INPUT_HEIGHT}x${INPUT_WIDTH}"
else
    TRT_CMD+=" --shapes=input:${BATCH_SIZE}x3x${INPUT_HEIGHT}x${INPUT_WIDTH}"
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