"""Integration tests for T4.3: WebSocket control channel + ROI hot-update.

Covers:
  - /ws/{id}/control endpoint routing (via ManagementServer)
  - ROI message dispatch: valid polygon, too few points, bad format, no DetectorNode
  - roi_clear message dispatch
  - ping/pong
  - Unknown message type → error
  - Concurrent ROI updates (atomicity check)
  - ManagementServer _ws_control returns 4004 for unknown pipeline id
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parents[3]
PYTHON_DIR = ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from visionpipe.server.control_ws import (
    _dispatch,
    _handle_roi,
    _handle_roi_clear,
    _handle_set_param,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_detector() -> MagicMock:
    node = MagicMock()
    node.set_roi = MagicMock()
    node.clear_roi = MagicMock()
    return node


def _fake_pipeline(detector: MagicMock | None = None) -> MagicMock:
    pipeline = MagicMock()
    nodes = {"detector": detector} if detector is not None else {}
    pipeline.nodes.return_value = nodes
    pipeline.name.return_value = "test"
    return pipeline


# ---------------------------------------------------------------------------
# ping / pong
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ping_returns_pong() -> None:
    ws = AsyncMock()
    await _dispatch(ws, _fake_pipeline(), {"type": "ping"})
    ws.send_str.assert_called_once_with(json.dumps({"type": "pong"}))


# ---------------------------------------------------------------------------
# Unknown message type
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_unknown_type_returns_error() -> None:
    ws = AsyncMock()
    await _dispatch(ws, _fake_pipeline(), {"type": "no_such_type"})
    resp = json.loads(ws.send_str.call_args[0][0])
    assert resp["type"] == "error"
    assert "no_such_type" in resp["message"]


@pytest.mark.asyncio
async def test_missing_type_returns_error() -> None:
    ws = AsyncMock()
    await _dispatch(ws, _fake_pipeline(), {})
    resp = json.loads(ws.send_str.call_args[0][0])
    assert resp["type"] == "error"


# ---------------------------------------------------------------------------
# ROI – happy path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_roi_triangle_calls_set_roi_with_flat_coords() -> None:
    ws = AsyncMock()
    det = _fake_detector()
    data = {"type": "roi", "polygons": [[0.1, 0.2], [0.8, 0.2], [0.5, 0.9]], "coord": "normalized"}
    with patch("visionpipe.server.control_ws._find_detector", return_value=det):
        await _handle_roi(ws, MagicMock(), data)
    det.set_roi.assert_called_once_with([[0.1, 0.2, 0.8, 0.2, 0.5, 0.9]])
    ack = json.loads(ws.send_str.call_args[0][0])
    assert ack == {"type": "ack", "ref_type": "roi"}


@pytest.mark.asyncio
async def test_roi_polygon_exactly_3_points_accepted() -> None:
    ws = AsyncMock()
    det = _fake_detector()
    data = {"type": "roi", "polygons": [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]], "coord": "normalized"}
    with patch("visionpipe.server.control_ws._find_detector", return_value=det):
        await _handle_roi(ws, MagicMock(), data)
    det.set_roi.assert_called_once()
    ack = json.loads(ws.send_str.call_args[0][0])
    assert ack["type"] == "ack"


@pytest.mark.asyncio
async def test_roi_boundary_coords_0_and_1_accepted() -> None:
    ws = AsyncMock()
    det = _fake_detector()
    data = {
        "type": "roi",
        "polygons": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        "coord": "normalized",
    }
    with patch("visionpipe.server.control_ws._find_detector", return_value=det):
        await _handle_roi(ws, MagicMock(), data)
    det.set_roi.assert_called_once_with([[0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]])


# ---------------------------------------------------------------------------
# ROI – error paths
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_roi_no_detector_returns_error() -> None:
    ws = AsyncMock()
    data = {"type": "roi", "polygons": [[0.1, 0.1], [0.9, 0.1], [0.5, 0.9]], "coord": "normalized"}
    with patch("visionpipe.server.control_ws._find_detector", return_value=None):
        await _handle_roi(ws, MagicMock(), data)
    err = json.loads(ws.send_str.call_args[0][0])
    assert err["type"] == "error"
    assert "DetectorNode" in err["message"]


@pytest.mark.asyncio
async def test_roi_fewer_than_3_points_returns_error() -> None:
    ws = AsyncMock()
    data = {"type": "roi", "polygons": [[0.1, 0.1], [0.9, 0.1]], "coord": "normalized"}
    with patch("visionpipe.server.control_ws._find_detector", return_value=_fake_detector()):
        await _handle_roi(ws, MagicMock(), data)
    err = json.loads(ws.send_str.call_args[0][0])
    assert err["type"] == "error"


@pytest.mark.asyncio
async def test_roi_empty_polygons_returns_error() -> None:
    ws = AsyncMock()
    data = {"type": "roi", "polygons": [], "coord": "normalized"}
    with patch("visionpipe.server.control_ws._find_detector", return_value=_fake_detector()):
        await _handle_roi(ws, MagicMock(), data)
    err = json.loads(ws.send_str.call_args[0][0])
    assert err["type"] == "error"


@pytest.mark.asyncio
async def test_roi_polygons_none_returns_error() -> None:
    ws = AsyncMock()
    data = {"type": "roi", "polygons": None, "coord": "normalized"}
    with patch("visionpipe.server.control_ws._find_detector", return_value=_fake_detector()):
        await _handle_roi(ws, MagicMock(), data)
    err = json.loads(ws.send_str.call_args[0][0])
    assert err["type"] == "error"


@pytest.mark.asyncio
async def test_roi_point_with_wrong_length_returns_error() -> None:
    ws = AsyncMock()
    # point has only 1 element
    data = {"type": "roi", "polygons": [[0.1], [0.9, 0.1], [0.5, 0.9]], "coord": "normalized"}
    with patch("visionpipe.server.control_ws._find_detector", return_value=_fake_detector()):
        await _handle_roi(ws, MagicMock(), data)
    err = json.loads(ws.send_str.call_args[0][0])
    assert err["type"] == "error"


@pytest.mark.asyncio
async def test_roi_point_is_string_returns_error() -> None:
    ws = AsyncMock()
    data = {"type": "roi", "polygons": [[0.1, 0.1], "bad", [0.5, 0.9]], "coord": "normalized"}
    with patch("visionpipe.server.control_ws._find_detector", return_value=_fake_detector()):
        await _handle_roi(ws, MagicMock(), data)
    err = json.loads(ws.send_str.call_args[0][0])
    assert err["type"] == "error"


@pytest.mark.asyncio
async def test_roi_point_is_scalar_returns_error() -> None:
    ws = AsyncMock()
    data = {"type": "roi", "polygons": [0.1, 0.2, 0.3], "coord": "normalized"}
    with patch("visionpipe.server.control_ws._find_detector", return_value=_fake_detector()):
        await _handle_roi(ws, MagicMock(), data)
    err = json.loads(ws.send_str.call_args[0][0])
    assert err["type"] == "error"


# ---------------------------------------------------------------------------
# roi_clear
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_roi_clear_calls_clear_roi_and_acks() -> None:
    ws = AsyncMock()
    det = _fake_detector()
    with patch("visionpipe.server.control_ws._find_detector", return_value=det):
        await _handle_roi_clear(ws, MagicMock())
    det.clear_roi.assert_called_once()
    ack = json.loads(ws.send_str.call_args[0][0])
    assert ack == {"type": "ack", "ref_type": "roi_clear"}


@pytest.mark.asyncio
async def test_roi_clear_no_detector_returns_error() -> None:
    ws = AsyncMock()
    with patch("visionpipe.server.control_ws._find_detector", return_value=None):
        await _handle_roi_clear(ws, MagicMock())
    err = json.loads(ws.send_str.call_args[0][0])
    assert err["type"] == "error"
    assert "DetectorNode" in err["message"]


# ---------------------------------------------------------------------------
# Concurrency: set_roi atomicity
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_concurrent_roi_updates_do_not_crash() -> None:
    """20 concurrent ROI updates must all succeed without exceptions."""
    det = _fake_detector()
    data = {"type": "roi", "polygons": [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]], "coord": "normalized"}

    async def one() -> None:
        ws = AsyncMock()
        with patch("visionpipe.server.control_ws._find_detector", return_value=det):
            await _handle_roi(ws, MagicMock(), data)
        resp = json.loads(ws.send_str.call_args[0][0])
        assert resp["type"] == "ack"

    await asyncio.gather(*[one() for _ in range(20)])
    assert det.set_roi.call_count == 20


@pytest.mark.asyncio
async def test_concurrent_roi_and_clear_do_not_crash() -> None:
    """Interleaved ROI set and clear must not raise."""
    det = _fake_detector()
    roi_data = {"type": "roi", "polygons": [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]], "coord": "normalized"}

    async def do_set() -> None:
        ws = AsyncMock()
        with patch("visionpipe.server.control_ws._find_detector", return_value=det):
            await _handle_roi(ws, MagicMock(), roi_data)

    async def do_clear() -> None:
        ws = AsyncMock()
        with patch("visionpipe.server.control_ws._find_detector", return_value=det):
            await _handle_roi_clear(ws, MagicMock())

    tasks = [do_set() if i % 2 == 0 else do_clear() for i in range(20)]
    await asyncio.gather(*tasks)


# ---------------------------------------------------------------------------
# ManagementServer routing: 4004 for unknown pipeline
# ---------------------------------------------------------------------------

def _has_gpu() -> bool:
    try:
        import visionpipe
        visionpipe.PipelineManager()
        return True
    except Exception:
        return False


requires_gpu = pytest.mark.skipif(not _has_gpu(), reason="GPU required")


@pytest.mark.asyncio
async def test_ws_control_unknown_pipeline_closes_with_4004() -> None:
    """_ws_control must close WebSocket with code 4004 if pipeline is not found."""
    import visionpipe
    from aiohttp.test_utils import TestClient, TestServer
    from visionpipe.server import ManagementServer

    mock_manager = MagicMock()
    mock_manager.get.side_effect = visionpipe.NotFoundError("not found")
    server = ManagementServer(mock_manager)

    async with TestClient(TestServer(server._app)) as client:
        ws = await client.ws_connect("/ws/no-such-id/control")
        msg = await ws.receive()
        # aiohttp delivers WSMsgType.CLOSE when the server closes the socket
        from aiohttp import WSMsgType
        assert msg.type == WSMsgType.CLOSE
        assert msg.data == 4004


# ===========================================================================
# T4.3 — set_param universal dispatch
# ===========================================================================
#
# Protocol (flat structure, fields at top level):
#   {"type": "set_param", "node_id": "<str>", "param_name": "<str>", "value": <any>}
# Supported value types (C++ ParamValue variant): int / float / str / list[float].
# Server responses:
#   success → {"type": "ack",   "ref_type": "set_param"}
#   failure → {"type": "error", "message": "..."}


def _fake_node(set_param_return: bool = True) -> MagicMock:
    """Generic mock node whose set_param(name, value) returns ``set_param_return``."""
    node = MagicMock()
    node.set_param = MagicMock(return_value=set_param_return)
    return node


def _fake_pipeline_with_nodes(nodes: dict[str, MagicMock]) -> MagicMock:
    """Pipeline mock whose .nodes() returns the given id→node dict."""
    pipeline = MagicMock()
    pipeline.nodes.return_value = nodes
    pipeline.name.return_value = "test"
    return pipeline


# ---------------------------------------------------------------------------
# set_param — happy path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_set_param_float_value_forwards_and_acks() -> None:
    ws = AsyncMock()
    node = _fake_node()
    pipeline = _fake_pipeline_with_nodes({"detector": node})
    data = {"node_id": "detector", "param_name": "conf_threshold", "value": 0.45}
    await _handle_set_param(ws, pipeline, data)
    node.set_param.assert_called_once_with("conf_threshold", 0.45)
    resp = json.loads(ws.send_str.call_args[0][0])
    assert resp == {"type": "ack", "ref_type": "set_param"}


@pytest.mark.asyncio
async def test_set_param_int_value_forwards_and_acks() -> None:
    ws = AsyncMock()
    node = _fake_node()
    pipeline = _fake_pipeline_with_nodes({"tracker": node})
    data = {"node_id": "tracker", "param_name": "max_age", "value": 30}
    await _handle_set_param(ws, pipeline, data)
    node.set_param.assert_called_once_with("max_age", 30)
    resp = json.loads(ws.send_str.call_args[0][0])
    assert resp == {"type": "ack", "ref_type": "set_param"}


@pytest.mark.asyncio
async def test_set_param_string_value_forwards_and_acks() -> None:
    ws = AsyncMock()
    node = _fake_node()
    pipeline = _fake_pipeline_with_nodes({"sink": node})
    data = {"node_id": "sink", "param_name": "label", "value": "front_camera"}
    await _handle_set_param(ws, pipeline, data)
    node.set_param.assert_called_once_with("label", "front_camera")
    resp = json.loads(ws.send_str.call_args[0][0])
    assert resp == {"type": "ack", "ref_type": "set_param"}


@pytest.mark.asyncio
async def test_set_param_list_float_value_forwards_and_acks() -> None:
    ws = AsyncMock()
    node = _fake_node()
    pipeline = _fake_pipeline_with_nodes({"detector": node})
    polygon_flat = [0.1, 0.2, 0.8, 0.2, 0.5, 0.9]
    data = {"node_id": "detector", "param_name": "roi_polygon", "value": polygon_flat}
    await _handle_set_param(ws, pipeline, data)
    node.set_param.assert_called_once_with("roi_polygon", polygon_flat)
    resp = json.loads(ws.send_str.call_args[0][0])
    assert resp == {"type": "ack", "ref_type": "set_param"}


# ---------------------------------------------------------------------------
# set_param — error paths
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_set_param_missing_node_id_returns_error() -> None:
    ws = AsyncMock()
    node = _fake_node()
    pipeline = _fake_pipeline_with_nodes({"detector": node})
    data = {"param_name": "conf_threshold", "value": 0.5}
    await _handle_set_param(ws, pipeline, data)
    resp = json.loads(ws.send_str.call_args[0][0])
    assert resp["type"] == "error"
    assert "node_id" in resp["message"]
    node.set_param.assert_not_called()


@pytest.mark.asyncio
async def test_set_param_empty_node_id_returns_error() -> None:
    ws = AsyncMock()
    node = _fake_node()
    pipeline = _fake_pipeline_with_nodes({"detector": node})
    data = {"node_id": "", "param_name": "conf_threshold", "value": 0.5}
    await _handle_set_param(ws, pipeline, data)
    resp = json.loads(ws.send_str.call_args[0][0])
    assert resp["type"] == "error"
    assert "node_id" in resp["message"]
    node.set_param.assert_not_called()


@pytest.mark.asyncio
async def test_set_param_non_string_node_id_returns_error() -> None:
    ws = AsyncMock()
    node = _fake_node()
    pipeline = _fake_pipeline_with_nodes({"detector": node})
    data = {"node_id": 123, "param_name": "conf_threshold", "value": 0.5}
    await _handle_set_param(ws, pipeline, data)
    resp = json.loads(ws.send_str.call_args[0][0])
    assert resp["type"] == "error"
    assert "node_id" in resp["message"]
    node.set_param.assert_not_called()


@pytest.mark.asyncio
async def test_set_param_missing_param_name_returns_error() -> None:
    ws = AsyncMock()
    node = _fake_node()
    pipeline = _fake_pipeline_with_nodes({"detector": node})
    data = {"node_id": "detector", "value": 0.5}
    await _handle_set_param(ws, pipeline, data)
    resp = json.loads(ws.send_str.call_args[0][0])
    assert resp["type"] == "error"
    assert "param_name" in resp["message"]
    node.set_param.assert_not_called()


@pytest.mark.asyncio
async def test_set_param_empty_param_name_returns_error() -> None:
    ws = AsyncMock()
    node = _fake_node()
    pipeline = _fake_pipeline_with_nodes({"detector": node})
    data = {"node_id": "detector", "param_name": "", "value": 0.5}
    await _handle_set_param(ws, pipeline, data)
    resp = json.loads(ws.send_str.call_args[0][0])
    assert resp["type"] == "error"
    assert "param_name" in resp["message"]
    node.set_param.assert_not_called()


@pytest.mark.asyncio
async def test_set_param_non_string_param_name_returns_error() -> None:
    ws = AsyncMock()
    node = _fake_node()
    pipeline = _fake_pipeline_with_nodes({"detector": node})
    data = {"node_id": "detector", "param_name": 99, "value": 0.5}
    await _handle_set_param(ws, pipeline, data)
    resp = json.loads(ws.send_str.call_args[0][0])
    assert resp["type"] == "error"
    assert "param_name" in resp["message"]
    node.set_param.assert_not_called()


@pytest.mark.asyncio
async def test_set_param_missing_value_field_returns_error() -> None:
    ws = AsyncMock()
    node = _fake_node()
    pipeline = _fake_pipeline_with_nodes({"detector": node})
    # No 'value' key at all (note: value=None is a *different* case and must be allowed)
    data = {"node_id": "detector", "param_name": "conf_threshold"}
    await _handle_set_param(ws, pipeline, data)
    resp = json.loads(ws.send_str.call_args[0][0])
    assert resp["type"] == "error"
    assert "value" in resp["message"]
    node.set_param.assert_not_called()


@pytest.mark.asyncio
async def test_set_param_node_not_found_returns_error() -> None:
    ws = AsyncMock()
    existing = _fake_node()
    pipeline = _fake_pipeline_with_nodes({"detector": existing})
    data = {"node_id": "nonexistent", "param_name": "conf_threshold", "value": 0.5}
    await _handle_set_param(ws, pipeline, data)
    resp = json.loads(ws.send_str.call_args[0][0])
    assert resp["type"] == "error"
    assert "not found" in resp["message"]
    existing.set_param.assert_not_called()


@pytest.mark.asyncio
async def test_set_param_node_raises_returns_error() -> None:
    ws = AsyncMock()
    node = MagicMock()
    node.set_param = MagicMock(side_effect=RuntimeError("invalid value"))
    pipeline = _fake_pipeline_with_nodes({"detector": node})
    data = {"node_id": "detector", "param_name": "conf_threshold", "value": 0.5}
    await _handle_set_param(ws, pipeline, data)
    resp = json.loads(ws.send_str.call_args[0][0])
    assert resp["type"] == "error"
    assert "raised" in resp["message"]
    node.set_param.assert_called_once_with("conf_threshold", 0.5)


@pytest.mark.asyncio
async def test_set_param_node_returns_false_returns_rejected_error() -> None:
    ws = AsyncMock()
    node = _fake_node(set_param_return=False)
    pipeline = _fake_pipeline_with_nodes({"detector": node})
    data = {"node_id": "detector", "param_name": "unknown_param", "value": 0.5}
    await _handle_set_param(ws, pipeline, data)
    resp = json.loads(ws.send_str.call_args[0][0])
    assert resp["type"] == "error"
    assert "rejected" in resp["message"]
    node.set_param.assert_called_once_with("unknown_param", 0.5)


# ---------------------------------------------------------------------------
# set_param — boundary / invariant cases
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_set_param_value_none_is_forwarded_not_rejected() -> None:
    """JSON null is a legal value; the protocol layer must not reject it."""
    ws = AsyncMock()
    node = _fake_node()
    pipeline = _fake_pipeline_with_nodes({"detector": node})
    data = {"node_id": "detector", "param_name": "optional_arg", "value": None}
    await _handle_set_param(ws, pipeline, data)
    node.set_param.assert_called_once_with("optional_arg", None)
    resp = json.loads(ws.send_str.call_args[0][0])
    assert resp == {"type": "ack", "ref_type": "set_param"}


@pytest.mark.asyncio
async def test_set_param_value_zero_int_is_forwarded() -> None:
    """0 is falsy but legal; must be forwarded."""
    ws = AsyncMock()
    node = _fake_node()
    pipeline = _fake_pipeline_with_nodes({"detector": node})
    data = {"node_id": "detector", "param_name": "max_age", "value": 0}
    await _handle_set_param(ws, pipeline, data)
    node.set_param.assert_called_once_with("max_age", 0)
    resp = json.loads(ws.send_str.call_args[0][0])
    assert resp == {"type": "ack", "ref_type": "set_param"}


@pytest.mark.asyncio
async def test_set_param_value_zero_float_is_forwarded() -> None:
    """0.0 is falsy but legal; must be forwarded."""
    ws = AsyncMock()
    node = _fake_node()
    pipeline = _fake_pipeline_with_nodes({"detector": node})
    data = {"node_id": "detector", "param_name": "conf_threshold", "value": 0.0}
    await _handle_set_param(ws, pipeline, data)
    node.set_param.assert_called_once_with("conf_threshold", 0.0)
    resp = json.loads(ws.send_str.call_args[0][0])
    assert resp == {"type": "ack", "ref_type": "set_param"}


@pytest.mark.asyncio
async def test_set_param_value_empty_string_is_forwarded() -> None:
    """Empty string is falsy but legal; must be forwarded."""
    ws = AsyncMock()
    node = _fake_node()
    pipeline = _fake_pipeline_with_nodes({"sink": node})
    data = {"node_id": "sink", "param_name": "label", "value": ""}
    await _handle_set_param(ws, pipeline, data)
    node.set_param.assert_called_once_with("label", "")
    resp = json.loads(ws.send_str.call_args[0][0])
    assert resp == {"type": "ack", "ref_type": "set_param"}


@pytest.mark.asyncio
async def test_set_param_value_empty_list_is_forwarded() -> None:
    """Empty list is falsy but legal; e.g. clearing a polygon."""
    ws = AsyncMock()
    node = _fake_node()
    pipeline = _fake_pipeline_with_nodes({"detector": node})
    data = {"node_id": "detector", "param_name": "roi_polygon", "value": []}
    await _handle_set_param(ws, pipeline, data)
    node.set_param.assert_called_once_with("roi_polygon", [])
    resp = json.loads(ws.send_str.call_args[0][0])
    assert resp == {"type": "ack", "ref_type": "set_param"}


# ---------------------------------------------------------------------------
# set_param — concurrency
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_set_param_concurrent_20_all_succeed() -> None:
    """20 concurrent set_param calls all succeed; node.set_param called 20 times."""
    node = _fake_node()
    pipeline = _fake_pipeline_with_nodes({"detector": node})

    async def one(i: int) -> None:
        ws = AsyncMock()
        data = {
            "node_id": "detector",
            "param_name": "conf_threshold",
            "value": float(i) / 100.0,
        }
        await _handle_set_param(ws, pipeline, data)
        resp = json.loads(ws.send_str.call_args[0][0])
        assert resp["type"] == "ack"

    await asyncio.gather(*[one(i) for i in range(20)])
    assert node.set_param.call_count == 20


# ---------------------------------------------------------------------------
# set_param — dispatch routing
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_set_param_routed_via_dispatch_calls_node() -> None:
    """_dispatch must route {type: set_param, ...} to _handle_set_param, not unknown."""
    ws = AsyncMock()
    node = _fake_node()
    pipeline = _fake_pipeline_with_nodes({"my_node": node})
    data = {
        "type": "set_param",
        "node_id": "my_node",
        "param_name": "conf_threshold",
        "value": 0.5,
    }
    await _dispatch(ws, pipeline, data)
    # If routing falls through to unknown, set_param wouldn't be invoked.
    node.set_param.assert_called_once_with("conf_threshold", 0.5)
    resp = json.loads(ws.send_str.call_args[0][0])
    assert resp == {"type": "ack", "ref_type": "set_param"}


@pytest.mark.asyncio
async def test_set_param_dispatch_error_path_is_set_param_not_unknown() -> None:
    """An invalid set_param payload must still hit _handle_set_param (not unknown)."""
    ws = AsyncMock()
    node = _fake_node()
    pipeline = _fake_pipeline_with_nodes({"detector": node})
    # Missing node_id → set_param-specific error, not "unknown type" error
    data = {"type": "set_param", "param_name": "x", "value": 1}
    await _dispatch(ws, pipeline, data)
    resp = json.loads(ws.send_str.call_args[0][0])
    assert resp["type"] == "error"
    assert "node_id" in resp["message"]
    # The unknown-type handler would produce a message mentioning the unknown
    # type name verbatim ("no_such_type"-style). A 'node_id' validation message
    # proves _dispatch routed to _handle_set_param rather than the unknown branch.


# ---------------------------------------------------------------------------
# set_param — coexists with existing protocol (ROI)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_set_param_followed_by_roi_both_succeed_in_same_dispatcher() -> None:
    """Interleaving set_param and roi on the same dispatcher both produce ack."""
    ws = AsyncMock()
    det = _fake_detector()
    # The same detector mock also accepts set_param
    det.set_param = MagicMock(return_value=True)
    pipeline = _fake_pipeline_with_nodes({"detector": det})

    # 1) set_param
    sp_data = {
        "type": "set_param",
        "node_id": "detector",
        "param_name": "conf_threshold",
        "value": 0.6,
    }
    await _dispatch(ws, pipeline, sp_data)

    # 2) ROI (patch _find_detector for the ROI flow)
    roi_data = {
        "type": "roi",
        "polygons": [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]],
        "coord": "normalized",
    }
    with patch("visionpipe.server.control_ws._find_detector", return_value=det):
        await _dispatch(ws, pipeline, roi_data)

    calls = ws.send_str.call_args_list
    assert len(calls) == 2
    resp_sp = json.loads(calls[0][0][0])
    resp_roi = json.loads(calls[1][0][0])
    assert resp_sp == {"type": "ack", "ref_type": "set_param"}
    assert resp_roi == {"type": "ack", "ref_type": "roi"}

    det.set_param.assert_called_once_with("conf_threshold", 0.6)
    det.set_roi.assert_called_once()

