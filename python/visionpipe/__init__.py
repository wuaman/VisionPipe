from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path

from visionpipe.custom_node import CustomNode
from visionpipe.frame_view import FrameView
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

ProcessProxyNode = _ext.ProcessProxyNode
VisionPipeError = _ext.VisionPipeError
ConfigError = _ext.ConfigError
NotFoundError = _ext.NotFoundError
CudaError = _ext.CudaError
ModelLoadError = _ext.ModelLoadError
InferError = _ext.InferError
StreamError = _ext.StreamError

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
Classification = _ext.Classification
Track = _ext.Track
Frame = _ext.Frame

IModelEngine = _ext.IModelEngine
MockModelEngine = _ext.MockModelEngine
TrtModelEngine = _ext.TrtModelEngine
NodeBase = _ext.NodeBase
SourceNode = _ext.SourceNode
SinkNode = _ext.SinkNode
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


def _unwrap_node(obj):
    """Extract the C++ NodeBase from a PyNode/CustomNode wrapper or return as-is."""
    from visionpipe.custom_node import CustomNode as _CustomNodePy
    from visionpipe.py_node import PyNode as _PyNodePy
    if isinstance(obj, (_PyNodePy, _CustomNodePy)):
        return obj._cpp_node
    return obj


def _node_rshift(self, other):
    """NodeBase >> NodeBase → Pipeline (with _tail tracking)."""
    other_cpp = _unwrap_node(other)
    self_cpp = _unwrap_node(self)
    pipe = Pipeline()
    pipe.add_node(self_cpp)
    pipe.add_node(other_cpp)
    pipe.connect(self_cpp, other_cpp)
    pipe._tail = other_cpp
    return pipe


def _node_rrshift(self, other):
    """Handles [src1, src2] >> node → Pipeline with merge topology."""
    if isinstance(other, (list, tuple)):
        pipe = Pipeline()
        self_cpp = _unwrap_node(self)
        pipe.add_node(self_cpp)
        for src in other:
            src_cpp = _unwrap_node(src)
            pipe.add_node(src_cpp)
            pipe.connect(src_cpp, self_cpp)
        pipe._tail = self_cpp
        return pipe
    return NotImplemented


def _pipeline_rshift(self, other):
    """Pipeline >> NodeBase → Pipeline (chaining)."""
    other_cpp = _unwrap_node(other)
    self.add_node(other_cpp)
    tail = getattr(self, "_tail", None)
    if tail is not None:
        self.connect(tail, other_cpp)
    self._tail = other_cpp
    return self


def _pipeline_run(self, block: bool = False, **config) -> "Pipeline":
    self.start()
    if block:
        self.wait_stop()
    return self


NodeBase.__rshift__ = _node_rshift
NodeBase.__rrshift__ = _node_rrshift
Pipeline.__rshift__ = _pipeline_rshift
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
    "CustomNode",
    "FrameView",
    "ProcessProxyNode",
    "PyNode",
    "VisionPipeError",
    "ConfigError",
    "NotFoundError",
    "CudaError",
    "ModelLoadError",
    "InferError",
    "StreamError",
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
    "Classification",
    "Track",
    "Frame",
    "IModelEngine",
    "MockModelEngine",
    "TrtModelEngine",
    "NodeBase",
    "SourceNode",
    "SinkNode",
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
