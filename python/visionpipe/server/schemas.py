"""Request / response schemas for the management REST API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class CreatePipelineRequest(BaseModel):
    """Body for POST /pipelines. Accepts either a YAML string or a JSON spec dict."""

    spec: dict[str, Any] | str


class PipelineInfo(BaseModel):
    id: str
    name: str
    state: str


class QueueStatsSchema(BaseModel):
    capacity: int
    current_size: int
    total_pushed: int
    total_popped: int
    dropped_count: int


class NodeHealthSchema(BaseModel):
    name: str
    processed_count: int
    error_count: int
    fps: float
    input_queue: QueueStatsSchema


class PipelineHealthResponse(BaseModel):
    id: str
    state: str
    total_frames_processed: int
    total_errors: int
    nodes: list[NodeHealthSchema]


class SetParamRequest(BaseModel):
    node_id: str
    param_name: str
    value: Any


class SetParamResponse(BaseModel):
    ok: bool
    message: str


class ErrorResponse(BaseModel):
    error: str


class NodeStatsSchema(BaseModel):
    name: str
    fps: float
    latency_ms: float
    frames_processed: int
    errors: int
    state: str
    input_queue: QueueStatsSchema


class TopologyResponse(BaseModel):
    nodes: list[str]
    edges: list[list[str]]
