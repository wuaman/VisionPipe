"""Pipeline YAML serialization / deserialization.

Defines pydantic models (PipelineSpec, NodeSpec, EdgeSpec) and two helpers
that are monkey-patched onto the C++ Pipeline class:

    Pipeline.export_yaml(path)   – write pipeline topology to YAML
    Pipeline.load_yaml(path)     – class-method, reconstruct Pipeline from YAML

Only the pipeline *topology* and node *configuration* are serialized.
Running state (frames in-flight, GPU resources) is not preserved.

Supported node types
--------------------
  file_source, rtsp_source, detector, classifier, segment, bytetrack, py_node
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, field_validator, model_validator

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

NodeType = Literal[
    "file_source",
    "rtsp_source",
    "detector",
    "classifier",
    "segment",
    "bytetrack",
    "py_node",
    "json_result_sink",
    "mjpeg_sink",
    "webrtc_sink",
]

_VALID_NODE_TYPES: set[str] = set(NodeType.__args__)  # type: ignore[attr-defined]


class NodeSpec(BaseModel):
    name: str
    type: NodeType
    params: dict[str, Any] = {}

    @field_validator("type", mode="before")
    @classmethod
    def _validate_type(cls, v: str) -> str:
        if v not in _VALID_NODE_TYPES:
            raise ValueError(f"Unknown node type '{v}'. Must be one of: {sorted(_VALID_NODE_TYPES)}")
        return v


class EdgeSpec(BaseModel):
    from_node: str
    to_node: str


class PipelineSpec(BaseModel):
    name: str = "pipeline"
    id: str = ""
    default_queue_capacity: int = 16
    default_overflow_policy: str = "DROP_OLDEST"
    nodes: list[NodeSpec]
    edges: list[EdgeSpec] = []

    @model_validator(mode="after")
    def _validate_edges(self) -> PipelineSpec:
        node_names = {n.name for n in self.nodes}
        for edge in self.edges:
            if edge.from_node not in node_names:
                raise ValueError(f"Edge references unknown source node '{edge.from_node}'")
            if edge.to_node not in node_names:
                raise ValueError(f"Edge references unknown target node '{edge.to_node}'")
        return self


# ---------------------------------------------------------------------------
# Node type → string mapping helpers
# ---------------------------------------------------------------------------


def _node_type_str(node: Any) -> str:
    """Infer the serializable type string from a C++ NodeBase subclass."""
    type_name = type(node).__name__
    _map = {
        "FileSource": "file_source",
        "RtspSource": "rtsp_source",
        "DetectorNode": "detector",
        "ClassifierNode": "classifier",
        "SegmentNode": "segment",
        "ByteTrackNode": "bytetrack",
        "PyNode": "py_node",
        "JsonResultSink": "json_result_sink",
        "MjpegSink": "mjpeg_sink",
        "WebRTCSink": "webrtc_sink",
    }
    result = _map.get(type_name)
    if result is None:
        # PyNode Python subclass wraps a C++ PyNode via _cpp_node
        from visionpipe.py_node import PyNode as _PyNode

        if isinstance(node, _PyNode):
            return "py_node"
        raise ValueError(f"Cannot serialize node of type '{type_name}'. Only built-in node types are supported.")
    return result


def _node_params(node: Any) -> dict[str, Any]:
    """Extract serializable params from a C++ config struct."""
    type_name = type(node).__name__

    if type_name == "FileSource":
        cfg = node.config()
        return {
            "uri": cfg.uri,
            "decode_mode": cfg.decode_mode.name,
            "gpu_device": cfg.gpu_device,
            "queue_capacity": cfg.queue_capacity,
            "stream_id": cfg.stream_id,
        }
    if type_name == "RtspSource":
        cfg = node.config()
        return {
            "uri": cfg.uri,
            "decode_mode": cfg.decode_mode.name,
            "gpu_device": cfg.gpu_device,
            "queue_capacity": cfg.queue_capacity,
            "stream_id": cfg.stream_id,
        }
    if type_name == "DetectorNode":
        cfg = node.config()
        return {
            "input_width": cfg.input_width,
            "input_height": cfg.input_height,
            "score_threshold": cfg.score_threshold,
            "nms_threshold": cfg.nms_threshold,
            "max_detections": cfg.max_detections,
            "workers": cfg.workers,
        }
    if type_name == "ClassifierNode":
        cfg = node.config()
        return {
            "input_width": cfg.input_width,
            "input_height": cfg.input_height,
            "max_batch_size": cfg.max_batch_size,
            "workers": cfg.workers,
        }
    if type_name == "SegmentNode":
        cfg = node.config()
        return {
            "input_width": cfg.input_width,
            "input_height": cfg.input_height,
            "score_threshold": cfg.score_threshold,
            "nms_threshold": cfg.nms_threshold,
            "mask_threshold": cfg.mask_threshold,
            "max_detections": cfg.max_detections,
            "workers": cfg.workers,
        }
    if type_name == "ByteTrackNode":
        cfg = node.config()
        return {
            "track_thresh": cfg.track_thresh,
            "track_buffer": cfg.track_buffer,
            "match_thresh": cfg.match_thresh,
            "frame_rate": cfg.frame_rate,
        }
    if type_name == "JsonResultSink":
        cfg = node.config()
        return {
            "buffer_capacity": cfg.buffer_capacity,
            "include_detections": cfg.include_detections,
            "include_tracks": cfg.include_tracks,
        }
    if type_name == "MjpegSink":
        cfg = node.config()
        return {
            "jpeg_quality": cfg.jpeg_quality,
            "buffer_capacity": cfg.buffer_capacity,
        }
    if type_name == "WebRTCSink":
        cfg = node.config()
        return {
            "video_bitrate_kbps": cfg.video_bitrate_kbps,
            "fps": cfg.fps,
            "keyframe_interval": cfg.keyframe_interval,
            "stun_server": cfg.stun_server,
            "use_nvenc": cfg.use_nvenc,
        }
    # py_node / unknown — no serializable params
    return {}


# ---------------------------------------------------------------------------
# Topology extraction
# ---------------------------------------------------------------------------


def _extract_edges(pipeline: Any) -> list[tuple[str, str]]:
    """Walk each node's output queue to reconstruct edges."""
    # The C++ Pipeline exposes nodes() as {name: NodeBase}.
    # To reconstruct edges we compare queue identities.
    nodes_map: dict[str, Any] = pipeline.nodes()

    # Build: queue_id -> node_name for output queues
    output_queue_to_node: dict[int, str] = {}
    for name, node in nodes_map.items():
        oq = node.output_queue() if hasattr(node, "output_queue") else None
        if oq is not None:
            output_queue_to_node[id(oq)] = name

    edges: list[tuple[str, str]] = []
    for name, node in nodes_map.items():
        iq = node.input_queue() if hasattr(node, "input_queue") else None
        if iq is None:
            continue
        upstream = output_queue_to_node.get(id(iq))
        if upstream is not None:
            edges.append((upstream, name))
    return edges


# ---------------------------------------------------------------------------
# Public API – attached to Pipeline
# ---------------------------------------------------------------------------


def export_yaml(self: Any, path: str | Path) -> None:
    """Export pipeline topology to a YAML file.

    Args:
        path: Destination file path.
    """

    nodes_map: dict[str, Any] = self.nodes()
    node_specs = []
    for node_name, node in nodes_map.items():
        node_type = _node_type_str(node)
        params = _node_params(node)
        node_specs.append(NodeSpec(name=node_name, type=node_type, params=params))

    raw_edges = _extract_edges(self)
    edge_specs = [EdgeSpec(from_node=f, to_node=t) for f, t in raw_edges]

    # default_overflow_policy comes from PipelineConfig; best-effort via stats
    policy_name = "DROP_OLDEST"

    spec = PipelineSpec(
        name=self.name(),
        id=self.id(),
        default_queue_capacity=16,
        default_overflow_policy=policy_name,
        nodes=node_specs,
        edges=edge_specs,
    )

    data = spec.model_dump()
    Path(path).write_text(yaml.dump(data, allow_unicode=True, sort_keys=False))


def load_yaml(path: str | Path) -> PipelineSpec:
    """Load and validate a pipeline YAML file.

    Returns a :class:`PipelineSpec` (pydantic model); does *not* instantiate
    C++ objects (GPU resources are needed for that).

    Args:
        path: Source YAML file path.

    Returns:
        Validated PipelineSpec.

    Raises:
        pydantic.ValidationError: If the YAML fails schema validation.
    """
    raw = yaml.safe_load(Path(path).read_text())
    return PipelineSpec.model_validate(raw)


def _rebuild_pipeline(spec: PipelineSpec, node_factory: dict[str, Any]) -> Any:
    """Reconstruct a Pipeline from a PipelineSpec using caller-supplied nodes.

    Args:
        spec: Validated pipeline spec.
        node_factory: Mapping of node_name → already-constructed NodeBase.
                      All nodes listed in spec.nodes must be present.

    Returns:
        A configured (but not yet started) Pipeline.

    Raises:
        KeyError: If a required node is missing from node_factory.
    """
    from visionpipe import OverflowPolicy, Pipeline, PipelineConfig

    policy_map = {
        "DROP_OLDEST": OverflowPolicy.DROP_OLDEST,
        "DROP_NEWEST": OverflowPolicy.DROP_NEWEST,
        "BLOCK": OverflowPolicy.BLOCK,
    }

    cfg = PipelineConfig()
    cfg.name = spec.name
    cfg.id = spec.id
    cfg.default_queue_capacity = spec.default_queue_capacity
    cfg.default_overflow_policy = policy_map.get(spec.default_overflow_policy, OverflowPolicy.DROP_OLDEST)

    pipeline = Pipeline(cfg)
    for node_spec in spec.nodes:
        node = node_factory[node_spec.name]
        pipeline.add_node(node)

    for edge in spec.edges:
        upstream = node_factory[edge.from_node]
        downstream = node_factory[edge.to_node]
        pipeline.connect(upstream, downstream)

    return pipeline


def _attach_to_pipeline() -> None:
    """Monkey-patch export_yaml / load_yaml onto the C++ Pipeline class."""
    from visionpipe import Pipeline

    Pipeline.export_yaml = export_yaml
    Pipeline.load_yaml = staticmethod(load_yaml)
    Pipeline.rebuild_from_spec = staticmethod(_rebuild_pipeline)
