"""Embedded management REST API server for VisionPipe.

Endpoints
---------
POST   /pipelines                  Create and start a pipeline from YAML/JSON spec
GET    /pipelines                  List all pipeline IDs and states
DELETE /pipelines/{id}             Stop and destroy a pipeline
GET    /pipelines/{id}/health      Return per-node QueueStats + FPS
POST   /pipelines/{id}/params      Set a runtime param on a node
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import yaml
from aiohttp import web

from visionpipe.serialization import PipelineSpec
from visionpipe.server.schemas import (
    CreatePipelineRequest,
    ErrorResponse,
    NodeHealthSchema,
    PipelineHealthResponse,
    PipelineInfo,
    QueueStatsSchema,
    SetParamRequest,
    SetParamResponse,
)

logger = logging.getLogger(__name__)


def _json(obj: Any, status: int = 200) -> web.Response:
    return web.Response(
        status=status,
        content_type="application/json",
        text=json.dumps(obj),
    )


def _err(msg: str, status: int) -> web.Response:
    return _json(ErrorResponse(error=msg).model_dump(), status)


def _pipeline_state_name(manager: Any, pipeline_id: str) -> str:
    try:
        status = manager.status(pipeline_id)
        return status.name
    except Exception:
        return "UNKNOWN"


class ManagementServer:
    """Async aiohttp-based management server.

    Parameters
    ----------
    manager:
        A ``visionpipe.PipelineManager`` instance (shared with application).
    host:
        Bind address (default ``"0.0.0.0"``).
    port:
        Bind port (default ``8080``).
    """

    def __init__(self, manager: Any, *, host: str = "0.0.0.0", port: int = 8080) -> None:
        self._manager = manager
        self._host = host
        self._port = port
        self._app = web.Application()
        self._runner: web.AppRunner | None = None
        self._setup_routes()

    # ------------------------------------------------------------------
    # Route setup
    # ------------------------------------------------------------------

    def _setup_routes(self) -> None:
        self._app.router.add_post("/pipelines", self._post_pipelines)
        self._app.router.add_get("/pipelines", self._get_pipelines)
        self._app.router.add_delete("/pipelines/{id}", self._delete_pipeline)
        self._app.router.add_get("/pipelines/{id}/health", self._get_health)
        self._app.router.add_post("/pipelines/{id}/params", self._post_params)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the HTTP server (non-blocking)."""
        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self._host, self._port)
        await site.start()
        logger.info("ManagementServer listening on %s:%s", self._host, self._port)

    async def stop(self) -> None:
        """Gracefully shut down the HTTP server."""
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    async def _post_pipelines(self, request: web.Request) -> web.Response:
        try:
            body = await request.json()
        except Exception:
            return _err("Invalid JSON body", 400)

        try:
            req = CreatePipelineRequest.model_validate(body)
        except Exception as exc:
            return _err(str(exc), 400)

        spec_data: dict[str, Any]
        if isinstance(req.spec, str):
            try:
                spec_data = yaml.safe_load(req.spec)
            except yaml.YAMLError as exc:
                return _err(f"Invalid YAML: {exc}", 400)
        else:
            spec_data = req.spec

        try:
            spec = PipelineSpec.model_validate(spec_data)
        except Exception as exc:
            return _err(f"Schema validation error: {exc}", 422)

        try:
            pipeline = await asyncio.get_event_loop().run_in_executor(None, self._build_and_register, spec)
        except Exception as exc:
            logger.exception("Failed to create pipeline")
            return _err(str(exc), 500)

        return _json({"id": pipeline}, status=201)

    async def _get_pipelines(self, request: web.Request) -> web.Response:
        ids: list[str] = self._manager.list()
        items = []
        for pid in ids:
            pipeline = self._manager.get(pid)
            state = _pipeline_state_name(self._manager, pid)
            items.append(PipelineInfo(id=pid, name=pipeline.name(), state=state).model_dump())
        return _json(items)

    async def _delete_pipeline(self, request: web.Request) -> web.Response:
        pid = request.match_info["id"]
        try:
            await asyncio.get_event_loop().run_in_executor(None, self._stop_and_destroy, pid)
        except Exception as exc:
            status = 404 if "not found" in str(exc).lower() else 500
            return _err(str(exc), status)
        return web.Response(status=204)

    async def _get_health(self, request: web.Request) -> web.Response:
        pid = request.match_info["id"]
        try:
            pipeline = self._manager.get(pid)
        except Exception as exc:
            return _err(str(exc), 404)

        stats = pipeline.stats()
        node_health = []
        for node_name, ns in stats.node_stats:
            qs = ns.input_queue_stats
            node_health.append(
                NodeHealthSchema(
                    name=node_name,
                    processed_count=ns.processed_count,
                    error_count=ns.error_count,
                    fps=ns.fps,
                    input_queue=QueueStatsSchema(
                        capacity=qs.capacity,
                        current_size=qs.current_size,
                        total_pushed=qs.total_pushed,
                        total_popped=qs.total_popped,
                        dropped_count=qs.dropped_count,
                    ),
                )
            )

        resp = PipelineHealthResponse(
            id=pid,
            state=_pipeline_state_name(self._manager, pid),
            total_frames_processed=stats.total_frames_processed,
            total_errors=stats.total_errors,
            nodes=node_health,
        )
        return _json(resp.model_dump())

    async def _post_params(self, request: web.Request) -> web.Response:
        pid = request.match_info["id"]
        try:
            body = await request.json()
        except Exception:
            return _err("Invalid JSON body", 400)

        try:
            req = SetParamRequest.model_validate(body)
        except Exception as exc:
            return _err(str(exc), 400)

        try:
            pipeline = self._manager.get(pid)
        except Exception as exc:
            return _err(str(exc), 404)

        nodes = pipeline.nodes()
        node = nodes.get(req.node_id)
        if node is None:
            return _err(f"Node '{req.node_id}' not found in pipeline '{pid}'", 404)

        ok = node.set_param(req.param_name, req.value)
        resp = SetParamResponse(
            ok=ok,
            message="ok" if ok else f"param '{req.param_name}' not accepted by node '{req.node_id}'",
        )
        return _json(resp.model_dump(), status=200 if ok else 422)

    # ------------------------------------------------------------------
    # Blocking helpers (run in executor)
    # ------------------------------------------------------------------

    def _build_and_register(self, spec: PipelineSpec) -> str:
        """Build a Pipeline from spec and register it with the manager. Returns pipeline id."""
        import visionpipe

        policy_map = {
            "DROP_OLDEST": visionpipe.OverflowPolicy.DROP_OLDEST,
            "DROP_NEWEST": visionpipe.OverflowPolicy.DROP_NEWEST,
            "BLOCK": visionpipe.OverflowPolicy.BLOCK,
        }

        node_map: dict[str, Any] = {}
        for ns in spec.nodes:
            p = ns.params
            if ns.type == "file_source":
                cfg = visionpipe.SourceConfig()
                cfg.uri = p.get("uri", "")
                mode_str = p.get("decode_mode", "CPU")
                cfg.decode_mode = visionpipe.DecodeMode[mode_str]
                cfg.gpu_device = p.get("gpu_device", 0)
                cfg.queue_capacity = p.get("queue_capacity", 16)
                cfg.stream_id = p.get("stream_id", ns.name)
                node_map[ns.name] = visionpipe.FileSource(cfg)
            elif ns.type == "rtsp_source":
                cfg = visionpipe.SourceConfig()
                cfg.uri = p.get("uri", "")
                mode_str = p.get("decode_mode", "CPU")
                cfg.decode_mode = visionpipe.DecodeMode[mode_str]
                cfg.gpu_device = p.get("gpu_device", 0)
                cfg.queue_capacity = p.get("queue_capacity", 16)
                cfg.stream_id = p.get("stream_id", ns.name)
                node_map[ns.name] = visionpipe.RtspSource(cfg)
            elif ns.type == "detector":
                engine_path = p.get("engine_path")
                if not engine_path:
                    raise ValueError(f"Node '{ns.name}' (detector) requires 'engine_path' in params")
                engine = visionpipe.TrtModelEngine(engine_path)
                cfg = visionpipe.DetectorConfig()
                cfg.input_width = p.get("input_width", 640)
                cfg.input_height = p.get("input_height", 640)
                cfg.score_threshold = p.get("score_threshold", 0.25)
                cfg.nms_threshold = p.get("nms_threshold", 0.45)
                cfg.max_detections = p.get("max_detections", 100)
                cfg.workers = p.get("workers", 1)
                node_map[ns.name] = visionpipe.DetectorNode(engine, cfg, ns.name)
            elif ns.type == "classifier":
                engine_path = p.get("engine_path")
                if not engine_path:
                    raise ValueError(f"Node '{ns.name}' (classifier) requires 'engine_path' in params")
                engine = visionpipe.TrtModelEngine(engine_path)
                cfg = visionpipe.ClassifierConfig()
                cfg.input_width = p.get("input_width", 224)
                cfg.input_height = p.get("input_height", 224)
                cfg.max_batch_size = p.get("max_batch_size", 8)
                cfg.workers = p.get("workers", 1)
                node_map[ns.name] = visionpipe.ClassifierNode(engine, cfg, ns.name)
            elif ns.type == "segment":
                engine_path = p.get("engine_path")
                if not engine_path:
                    raise ValueError(f"Node '{ns.name}' (segment) requires 'engine_path' in params")
                engine = visionpipe.TrtModelEngine(engine_path)
                cfg = visionpipe.SegmentConfig()
                cfg.input_width = p.get("input_width", 640)
                cfg.input_height = p.get("input_height", 640)
                cfg.score_threshold = p.get("score_threshold", 0.25)
                cfg.nms_threshold = p.get("nms_threshold", 0.45)
                cfg.mask_threshold = p.get("mask_threshold", 0.5)
                cfg.max_detections = p.get("max_detections", 100)
                cfg.workers = p.get("workers", 1)
                node_map[ns.name] = visionpipe.SegmentNode(engine, cfg, ns.name)
            elif ns.type == "bytetrack":
                cfg = visionpipe.ByteTrackConfig()
                cfg.track_thresh = p.get("track_thresh", 0.5)
                cfg.track_buffer = p.get("track_buffer", 30)
                cfg.match_thresh = p.get("match_thresh", 0.8)
                cfg.frame_rate = p.get("frame_rate", 30)
                node_map[ns.name] = visionpipe.ByteTrackNode(cfg, ns.name)
            elif ns.type == "py_node":
                node_map[ns.name] = visionpipe.PyNode(ns.name)
            else:
                raise ValueError(f"Unsupported node type: {ns.type}")

        pipeline_cfg = visionpipe.PipelineConfig()
        if spec.id:
            pipeline_cfg.id = spec.id
        pipeline_cfg.name = spec.name
        pipeline_cfg.default_queue_capacity = spec.default_queue_capacity
        pipeline_cfg.default_overflow_policy = policy_map.get(
            spec.default_overflow_policy, visionpipe.OverflowPolicy.DROP_OLDEST
        )

        pipeline = visionpipe.Pipeline(pipeline_cfg)
        for node in node_map.values():
            pipeline.add_node(node)
        for edge in spec.edges:
            pipeline.connect(node_map[edge.from_node], node_map[edge.to_node])

        pipeline_id: str = self._manager.create_pipeline(pipeline)
        # Only start if there is at least one real source node (FileSource/RtspSource).
        # Pipelines without a source node will be started on first real frame arrival
        # or left in INIT state for manual lifecycle control.
        has_real_source = any(ns.type in ("file_source", "rtsp_source") for ns in spec.nodes)
        if has_real_source:
            self._manager.start(pipeline_id)
        return pipeline_id

    def _stop_and_destroy(self, pipeline_id: str) -> None:
        import visionpipe

        try:
            status = self._manager.status(pipeline_id)
            if status.name not in ("STOPPED", "ERROR", "INIT"):
                self._manager.stop(pipeline_id)
        except visionpipe.NotFoundError:
            raise
        self._manager.destroy(pipeline_id)
