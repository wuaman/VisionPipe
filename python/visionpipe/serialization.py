"""Pipeline YAML serialization / deserialization.

Defines pydantic models (PipelineSpec, NodeSpec, EdgeSpec) and two helpers
that are monkey-patched onto the C++ Pipeline class:

    Pipeline.export_yaml(path)   – write pipeline topology to YAML
    Pipeline.load_yaml(path)     – class-method, reconstruct Pipeline from YAML

Only the pipeline *topology* and node *configuration* are serialized.
Running state (frames in-flight, GPU resources) is not preserved.

Supported node types
--------------------
  file_source, rtsp_source, detector, classifier, segment, bytetrack,
  annotator, py_node, custom_node, json_result_sink, mjpeg_sink, webrtc_sink
"""

from __future__ import annotations

import importlib
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
    "rtmpose",
    "yolo_pose",
    "bytetrack",
    "annotator",
    "py_node",
    "custom_node",
    "json_result_sink",
    "mjpeg_sink",
    "webrtc_sink",
]

_VALID_NODE_TYPES: set[str] = set(NodeType.__args__)  # type: ignore[attr-defined]


class NodeSpec(BaseModel):
    name: str
    type: NodeType
    params: dict[str, Any] = {}
    module: str | None = None
    class_name: str | None = None
    process_mode: str | None = None

    @field_validator("type", mode="before")
    @classmethod
    def _validate_type(cls, v: str) -> str:
        if v not in _VALID_NODE_TYPES:
            raise ValueError(f"Unknown node type '{v}'. Must be one of: {sorted(_VALID_NODE_TYPES)}")
        return v

    @model_validator(mode="after")
    def _validate_custom_node_fields(self) -> NodeSpec:
        if self.type == "custom_node":
            if not self.module or not self.class_name:
                raise ValueError(
                    "custom_node requires 'module' and 'class_name' fields"
                )
            if self.process_mode and self.process_mode not in ("subprocess", "inline"):
                raise ValueError(
                    f"process_mode must be 'subprocess' or 'inline', got '{self.process_mode}'"
                )
        return self


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
    from visionpipe.custom_node import CustomNode as _CustomNode
    from visionpipe.py_node import PyNode as _PyNode

    if isinstance(node, _CustomNode):
        return "custom_node"
    if isinstance(node, _PyNode):
        return "py_node"

    type_name = type(node).__name__
    _map = {
        "FileSource": "file_source",
        "RtspSource": "rtsp_source",
        "DetectorNode": "detector",
        "ClassifierNode": "classifier",
        "YoloSegNode": "segment",
        "RtmPoseNode": "rtmpose",
        "YoloPoseNode": "yolo_pose",
        "ByteTrackNode": "bytetrack",
        "AnnotatorNode": "annotator",
        "PyNode": "py_node",
        "JsonResultSink": "json_result_sink",
        "MjpegSink": "mjpeg_sink",
        "WebRTCSink": "webrtc_sink",
    }
    result = _map.get(type_name)
    if result is None:
        raise ValueError(f"Cannot serialize node of type '{type_name}'. Only built-in node types are supported.")
    return result


def _node_params(node: Any) -> dict[str, Any]:
    """Extract serializable params from a C++ config struct."""
    from visionpipe.custom_node import CustomNode as _CustomNode

    if isinstance(node, _CustomNode):
        return node.get_config()

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
    if type_name == "YoloSegNode":
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
    if type_name == "RtmPoseNode":
        cfg = node.config()
        return {
            "input_width": cfg.input_width,
            "input_height": cfg.input_height,
            "target_classes": list(cfg.target_classes),
            "score_threshold": cfg.score_threshold,
            "bbox_padding": cfg.bbox_padding,
            "max_batch_size": cfg.max_batch_size,
            "workers": cfg.workers,
        }
    if type_name == "YoloPoseNode":
        cfg = node.config()
        return {
            "input_width": cfg.input_width,
            "input_height": cfg.input_height,
            "score_threshold": cfg.score_threshold,
            "nms_threshold": cfg.nms_threshold,
            "max_detections": cfg.max_detections,
            "workers": cfg.workers,
            "max_batch_size": cfg.max_batch_size,
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
    if type_name == "AnnotatorNode":
        cfg = node.config()
        return {
            "draw_detections": cfg.draw_detections,
            "draw_tracks": cfg.draw_tracks,
            "draw_masks": cfg.draw_masks,
            "mask_alpha": cfg.mask_alpha,
            "class_names": list(cfg.class_names),
        }
    # py_node / unknown — no serializable params
    return {}


# ---------------------------------------------------------------------------
# Topology extraction
# ---------------------------------------------------------------------------


def _extract_edges(pipeline: Any) -> list[tuple[str, str]]:
    """Walk each node's output queue to reconstruct edges.

    Uses ``output_queue_id()`` / ``input_queue_id()`` which return the C++
    pointer of the underlying BoundedQueue. ``id()`` on a freshly-cast Python
    wrapper is unstable, so the C++ pointer is the only reliable identity.
    """
    nodes_map: dict[str, Any] = pipeline.nodes()

    output_queue_to_node: dict[int, str] = {}
    for name, node in nodes_map.items():
        oqid = node.output_queue_id() if hasattr(node, "output_queue_id") else None
        if oqid is not None:
            output_queue_to_node[int(oqid)] = name

    edges: list[tuple[str, str]] = []
    for name, node in nodes_map.items():
        iqid = node.input_queue_id() if hasattr(node, "input_queue_id") else None
        if iqid is None:
            continue
        upstream = output_queue_to_node.get(int(iqid))
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
        extra: dict[str, Any] = {}
        if node_type == "custom_node":
            from visionpipe.custom_node import CustomNode as _CustomNode

            if isinstance(node, _CustomNode):
                real_cls = type(node)
                extra["module"] = real_cls.__module__
                extra["class_name"] = real_cls.__qualname__
                extra["process_mode"] = node._process_mode
        node_specs.append(
            NodeSpec(name=node_name, type=node_type, params=params, **extra)
        )

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


def _import_custom_node(spec: NodeSpec) -> Any:
    """Auto-import and instantiate a CustomNode from module/class_name.

    Args:
        spec: NodeSpec with type="custom_node", module, and class_name set.

    Returns:
        Instantiated CustomNode subclass.

    Raises:
        ImportError: If the module cannot be imported.
        AttributeError: If the class is not found in the module.
    """
    mod = importlib.import_module(spec.module)  # type: ignore[arg-type]
    cls = getattr(mod, spec.class_name)  # type: ignore[arg-type]
    return cls(
        name=spec.name,
        process_mode=spec.process_mode or "subprocess",
        **spec.params,
    )


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


def from_yaml(path: str | Path, node_overrides: dict[str, Any] | None = None) -> Any:
    """Load a YAML pipeline spec and build a Pipeline, auto-importing CustomNodes.

    For ``custom_node`` entries, the class is automatically imported from the
    specified ``module`` and ``class_name``.  Other node types must be provided
    via *node_overrides* (keyed by node name).

    Args:
        path: Source YAML file path.
        node_overrides: Pre-constructed nodes to use instead of auto-import.
                        Keys are node names matching the YAML spec.

    Returns:
        A configured (but not yet started) Pipeline.

    Raises:
        pydantic.ValidationError: If the YAML fails schema validation.
        KeyError: If a non-custom node is missing from *node_overrides*.
        ImportError: If a custom_node module cannot be imported.
    """
    spec = load_yaml(path)
    overrides = node_overrides or {}

    factory: dict[str, Any] = {}
    for node_spec in spec.nodes:
        if node_spec.name in overrides:
            factory[node_spec.name] = overrides[node_spec.name]
        elif node_spec.type == "custom_node":
            factory[node_spec.name] = _import_custom_node(node_spec)
        else:
            raise KeyError(
                f"Node '{node_spec.name}' (type={node_spec.type}) not in "
                f"node_overrides — only custom_node can be auto-instantiated"
            )

    return _rebuild_pipeline(spec, factory)


def _attach_to_pipeline() -> None:
    """Monkey-patch export_yaml / load_yaml / from_yaml onto the C++ Pipeline class."""
    from visionpipe import Pipeline

    Pipeline.export_yaml = export_yaml
    Pipeline.load_yaml = staticmethod(load_yaml)
    Pipeline.from_yaml = staticmethod(from_yaml)
    Pipeline.rebuild_from_spec = staticmethod(_rebuild_pipeline)
