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

from visionpipe.server.control_ws import _dispatch, _handle_roi, _handle_roi_clear

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
