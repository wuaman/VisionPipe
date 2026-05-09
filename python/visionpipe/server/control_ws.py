"""WebSocket control channel for VisionPipe pipelines.

Protocol (JSON messages from client)
-------------------------------------
ROI update:  {"type": "roi", "polygons": [[x,y], ...], "coord": "normalized"}
ROI clear:   {"type": "roi_clear"}
Ping:        {"type": "ping"}

Server responses
-----------------
{"type": "ack",   "ref_type": "<original_type>"}
{"type": "error", "message": "..."}
{"type": "pong"}

ROI polygons are lists of [x, y] normalized point pairs (values in [0, 1]).
At least 3 points are required to form a valid polygon.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from aiohttp import WSMsgType, web

logger = logging.getLogger(__name__)


def _find_detector(pipeline: Any) -> Any | None:
    """Return the first DetectorNode found in *pipeline*, or None."""
    import visionpipe

    return next(
        (n for n in pipeline.nodes().values() if isinstance(n, visionpipe.DetectorNode)),
        None,
    )


async def handle_control_ws(
    request: web.Request,
    pipeline: Any,
) -> web.WebSocketResponse:
    """Run the control WebSocket channel for one pipeline.

    Parameters
    ----------
    request:
        The aiohttp WebSocket upgrade request.
    pipeline:
        A ``visionpipe.Pipeline`` instance.
    """
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    logger.info("Control WS connected for pipeline '%s'", pipeline.name())

    try:
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                except json.JSONDecodeError as exc:
                    await ws.send_str(json.dumps({"type": "error", "message": f"Invalid JSON: {exc}"}))
                    continue

                await _dispatch(ws, pipeline, data)

            elif msg.type in (WSMsgType.ERROR, WSMsgType.CLOSE):
                break
    except Exception:
        logger.exception("Control WS error for pipeline '%s'", pipeline.name())
    finally:
        logger.info("Control WS disconnected from pipeline '%s'", pipeline.name())

    return ws


async def _dispatch(ws: web.WebSocketResponse, pipeline: Any, data: dict[str, Any]) -> None:
    msg_type = data.get("type")

    if msg_type == "ping":
        await ws.send_str(json.dumps({"type": "pong"}))
    elif msg_type == "roi":
        await _handle_roi(ws, pipeline, data)
    elif msg_type == "roi_clear":
        await _handle_roi_clear(ws, pipeline)
    else:
        await ws.send_str(json.dumps({"type": "error", "message": f"Unknown message type: {msg_type!r}"}))


async def _handle_roi(ws: web.WebSocketResponse, pipeline: Any, data: dict[str, Any]) -> None:
    detector = _find_detector(pipeline)
    if detector is None:
        await ws.send_str(json.dumps({"type": "error", "message": "Pipeline has no DetectorNode"}))
        return

    polygons_raw = data.get("polygons")
    if not isinstance(polygons_raw, list) or len(polygons_raw) < 3:
        await ws.send_str(json.dumps({
            "type": "error",
            "message": "polygons must be a list of at least 3 [x,y] points",
        }))
        return

    try:
        flat: list[float] = []
        for point in polygons_raw:
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                raise ValueError(f"Each polygon point must be [x, y], got {point!r}")
            flat.append(float(point[0]))
            flat.append(float(point[1]))
    except (TypeError, ValueError) as exc:
        await ws.send_str(json.dumps({"type": "error", "message": str(exc)}))
        return

    # C++ set_roi takes vector<vector<float>> (multiple polygons, each flat [x1,y1,...])
    detector.set_roi([flat])
    await ws.send_str(json.dumps({"type": "ack", "ref_type": "roi"}))
    logger.debug("ROI updated: %d points", len(polygons_raw))


async def _handle_roi_clear(ws: web.WebSocketResponse, pipeline: Any) -> None:
    detector = _find_detector(pipeline)
    if detector is None:
        await ws.send_str(json.dumps({"type": "error", "message": "Pipeline has no DetectorNode"}))
        return

    detector.clear_roi()
    await ws.send_str(json.dumps({"type": "ack", "ref_type": "roi_clear"}))
    logger.debug("ROI cleared")
