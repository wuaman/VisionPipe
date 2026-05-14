from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path

from visionpipe.py_node import PyNode

__version__ = "0.1.0"


def _load_extension():
    try:
        return import_module("visionpipe.visionpipe_python")
    except ImportError:
        pass

    try:
        return import_module("visionpipe_python")
    except ImportError:
        pass

    build_python = Path(__file__).resolve().parents[2] / "build" / "python"
    if build_python.exists():
        sys.path.insert(0, str(build_python))
        return import_module("visionpipe_python")

    return import_module("visionpipe_python")


_ext = _load_extension()

VisionPipeError = _ext.VisionPipeError
ConfigError = _ext.ConfigError
NotFoundError = _ext.NotFoundError
CudaError = _ext.CudaError
ModelLoadError = _ext.ModelLoadError
InferError = _ext.InferError

PipelineState = _ext.PipelineState
PipelineStatus = _ext.PipelineStatus
NodeState = _ext.NodeState
OverflowPolicy = _ext.OverflowPolicy
DecodeMode = _ext.DecodeMode

QueueStats = _ext.QueueStats
NodeStats = _ext.NodeStats
PipelineConfig = _ext.PipelineConfig
PipelineStats = _ext.PipelineStats
SourceConfig = _ext.SourceConfig
ByteTrackConfig = _ext.ByteTrackConfig
DetectorConfig = _ext.DetectorConfig
ClassifierConfig = _ext.ClassifierConfig
SegmentConfig = _ext.SegmentConfig
Detection = _ext.Detection
Track = _ext.Track
Frame = _ext.Frame

IModelEngine = _ext.IModelEngine
MockModelEngine = _ext.MockModelEngine
TrtModelEngine = _ext.TrtModelEngine
NodeBase = _ext.NodeBase
_PyNodeCpp = _ext.PyNode
FileSource = _ext.FileSource
RtspSource = _ext.RtspSource
ByteTrackNode = _ext.ByteTrackNode
DetectorNode = _ext.DetectorNode
ClassifierNode = _ext.ClassifierNode
SegmentNode = _ext.SegmentNode
Pipeline = _ext.Pipeline
PipelineBuilder = _ext.PipelineBuilder
PipelineManager = _ext.PipelineManager

JsonResultSinkConfig = _ext.JsonResultSinkConfig
JsonResultSink = _ext.JsonResultSink
MjpegSinkConfig = _ext.MjpegSinkConfig
MjpegSink = _ext.MjpegSink
WebRTCSinkConfig = _ext.WebRTCSinkConfig
WebRTCSink = _ext.WebRTCSink
AnnotatorConfig = _ext.AnnotatorConfig
AnnotatorNode = _ext.AnnotatorNode


def _node_rshift(self: NodeBase, other: NodeBase) -> PipelineBuilder:
    builder = PipelineBuilder()
    builder.__rshift__(self)
    return builder.__rshift__(other)


def _pipeline_run(self: Pipeline) -> Pipeline:
    self.start()
    return self


NodeBase.__rshift__ = _node_rshift
Pipeline.run = _pipeline_run

from visionpipe.serialization import (  # noqa: E402
    EdgeSpec,
    NodeSpec,
    PipelineSpec,
    _attach_to_pipeline,
)

_attach_to_pipeline()

__all__ = [
    "__version__",
    "PyNode",
    "VisionPipeError",
    "ConfigError",
    "NotFoundError",
    "CudaError",
    "ModelLoadError",
    "InferError",
    "PipelineState",
    "PipelineStatus",
    "NodeState",
    "OverflowPolicy",
    "DecodeMode",
    "QueueStats",
    "NodeStats",
    "PipelineConfig",
    "PipelineStats",
    "SourceConfig",
    "ByteTrackConfig",
    "DetectorConfig",
    "ClassifierConfig",
    "SegmentConfig",
    "Detection",
    "Track",
    "Frame",
    "IModelEngine",
    "MockModelEngine",
    "TrtModelEngine",
    "NodeBase",
    "FileSource",
    "RtspSource",
    "ByteTrackNode",
    "DetectorNode",
    "ClassifierNode",
    "SegmentNode",
    "Pipeline",
    "PipelineBuilder",
    "PipelineManager",
    "PipelineSpec",
    "NodeSpec",
    "EdgeSpec",
    "JsonResultSinkConfig",
    "JsonResultSink",
    "MjpegSinkConfig",
    "MjpegSink",
    "WebRTCSinkConfig",
    "WebRTCSink",
    "AnnotatorConfig",
    "AnnotatorNode",
]
