"""Unit tests for T4.2 WebRTC Sink.

Tests cover:
  - WebRTCSinkConfig defaults and mutation
  - WebRTCSink stub interface (no GPU / no VISIONPIPE_USE_WEBRTC build needed)
  - signaling.py protocol via mocked sink
  - management_api.py /ws/{id}/webrtc endpoint via TestClient
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from aiohttp.test_utils import TestClient, TestServer

ROOT = Path(__file__).resolve().parents[3]
PYTHON_DIR = ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

# ---------------------------------------------------------------------------
# GPU / extension detection
# ---------------------------------------------------------------------------


def _has_gpu() -> bool:
    try:
        import visionpipe
        visionpipe.PipelineManager()
        return True
    except Exception:
        return False


requires_gpu = pytest.mark.skipif(not _has_gpu(), reason="GPU required")


# ---------------------------------------------------------------------------
# Helper: try to import visionpipe, skip if extension not built
# ---------------------------------------------------------------------------


def _import_visionpipe() -> Any:
    try:
        import visionpipe
        return visionpipe
    except ImportError as exc:
        pytest.skip(f"visionpipe extension not available: {exc}")


# ---------------------------------------------------------------------------
# WebRTCSinkConfig tests
# ---------------------------------------------------------------------------


class TestWebRTCSinkConfig:
    def test_default_values_are_valid(self):
        vp = _import_visionpipe()
        cfg = vp.WebRTCSinkConfig()
        assert cfg.video_bitrate_kbps > 0
        assert cfg.fps > 0
        assert cfg.keyframe_interval > 0
        assert isinstance(cfg.stun_server, str)
        assert len(cfg.stun_server) > 0

    def test_field_assignment(self):
        vp = _import_visionpipe()
        cfg = vp.WebRTCSinkConfig()
        cfg.video_bitrate_kbps = 4000
        cfg.fps = 25
        cfg.keyframe_interval = 50
        cfg.stun_server = "stun:example.com:3478"
        cfg.use_nvenc = False

        assert cfg.video_bitrate_kbps == 4000
        assert cfg.fps == 25
        assert cfg.keyframe_interval == 50
        assert cfg.stun_server == "stun:example.com:3478"
        assert cfg.use_nvenc is False


# ---------------------------------------------------------------------------
# WebRTCSink stub interface tests (no GPU, no WebRTC build required)
# ---------------------------------------------------------------------------


class TestWebRTCSinkStubInterface:
    """These tests work with both the stub (no VISIONPIPE_USE_WEBRTC) and the
    real implementation since they only verify safe behavior."""

    def test_is_sink(self):
        vp = _import_visionpipe()
        sink = vp.WebRTCSink()
        assert sink.is_sink() is True

    def test_is_not_source(self):
        vp = _import_visionpipe()
        sink = vp.WebRTCSink()
        assert sink.is_source() is False

    def test_initial_peer_count_zero(self):
        vp = _import_visionpipe()
        sink = vp.WebRTCSink()
        assert sink.peer_count() == 0

    def test_create_peer_returns_string(self):
        vp = _import_visionpipe()
        sink = vp.WebRTCSink()
        peer_id = sink.create_peer()
        # Stub returns "" and real impl returns a non-empty ID
        assert isinstance(peer_id, str)

    def test_remove_nonexistent_peer_does_not_raise(self):
        vp = _import_visionpipe()
        sink = vp.WebRTCSink()
        # Must not raise KeyError or anything else
        sink.remove_peer("nonexistent-peer-id")

    def test_drain_candidates_nonexistent_peer_returns_empty(self):
        vp = _import_visionpipe()
        sink = vp.WebRTCSink()
        result = sink.drain_candidates("nonexistent-peer-id")
        assert result == []

    def test_add_candidate_nonexistent_peer_does_not_raise(self):
        vp = _import_visionpipe()
        sink = vp.WebRTCSink()
        sink.add_candidate("ghost", "candidate:0 1 UDP ...", "0")

    def test_config_returns_webrtcsinkconfig(self):
        vp = _import_visionpipe()
        cfg = vp.WebRTCSinkConfig()
        cfg.fps = 20
        sink = vp.WebRTCSink(cfg, "ws")
        assert sink.config().fps == 20

    def test_custom_name(self):
        vp = _import_visionpipe()
        sink = vp.WebRTCSink(vp.WebRTCSinkConfig(), "my_webrtc")
        assert sink.name() == "my_webrtc"


# ---------------------------------------------------------------------------
# Signaling protocol tests (mock sink)
# ---------------------------------------------------------------------------


class _FakeSink:
    """Minimal mock of visionpipe.WebRTCSink for signaling tests."""

    def __init__(self, offer_sdp: str = "v=0\r\nm=video ...", raise_on_offer: bool = False):
        self._offer = offer_sdp
        self._raise_on_offer = raise_on_offer
        self._peers: dict[str, dict] = {}
        self.set_answer_called: list[tuple[str, str]] = []
        self.add_candidate_called: list[tuple[str, str, str]] = []
        self.remove_peer_called: list[str] = []

    def create_peer(self) -> str:
        import uuid
        pid = str(uuid.uuid4())
        self._peers[pid] = {}
        return pid

    def get_offer(self, peer_id: str, timeout_ms: int = 10_000) -> str:
        if self._raise_on_offer:
            raise RuntimeError("timeout")
        return self._offer

    def set_answer(self, peer_id: str, sdp: str) -> None:
        self.set_answer_called.append((peer_id, sdp))

    def add_candidate(self, peer_id: str, candidate: str, mid: str) -> None:
        self.add_candidate_called.append((peer_id, candidate, mid))

    def drain_candidates(self, peer_id: str) -> list:
        return []

    def remove_peer(self, peer_id: str) -> None:
        self.remove_peer_called.append(peer_id)
        self._peers.pop(peer_id, None)


@pytest.mark.asyncio
async def test_signaling_sends_offer_on_connect():
    """Server must send {"type": "offer"} immediately after connection."""
    from aiohttp import web
    from visionpipe.server.signaling import handle_webrtc_signaling

    fake_sink = _FakeSink(offer_sdp="v=0\r\nm=video 9 UDP/TLS/RTP/SAVPF 96\r\n")

    async def handler(request: web.Request) -> web.WebSocketResponse:
        return await handle_webrtc_signaling(request, fake_sink)

    app = web.Application()
    app.router.add_get("/ws", handler)

    async with TestClient(TestServer(app)) as client:
        async with client.ws_connect("/ws") as ws:
            msg = await asyncio.wait_for(ws.receive_str(), timeout=5.0)
            data = json.loads(msg)
            assert data["type"] == "offer"
            assert "sdp" in data
            assert data["sdp"] == fake_sink._offer


@pytest.mark.asyncio
async def test_signaling_answer_forwarded_to_sink():
    """When browser sends answer, sink.set_answer() must be called."""
    from aiohttp import web
    from visionpipe.server.signaling import handle_webrtc_signaling

    fake_sink = _FakeSink()

    async def handler(request: web.Request) -> web.WebSocketResponse:
        return await handle_webrtc_signaling(request, fake_sink)

    app = web.Application()
    app.router.add_get("/ws", handler)

    answer_sdp = "v=0\r\nm=video 9 UDP/TLS/RTP/SAVPF 96\r\na=setup:active\r\n"

    async with TestClient(TestServer(app)) as client:
        async with client.ws_connect("/ws") as ws:
            # Consume the offer
            await asyncio.wait_for(ws.receive_str(), timeout=5.0)
            # Send answer
            await ws.send_str(json.dumps({"type": "answer", "sdp": answer_sdp}))
            # Give handler time to process
            await asyncio.sleep(0.2)

    assert len(fake_sink.set_answer_called) == 1
    _, received_sdp = fake_sink.set_answer_called[0]
    assert received_sdp == answer_sdp


@pytest.mark.asyncio
async def test_signaling_candidate_forwarded_to_sink():
    """When browser sends a candidate, sink.add_candidate() must be called."""
    from aiohttp import web
    from visionpipe.server.signaling import handle_webrtc_signaling

    fake_sink = _FakeSink()

    async def handler(request: web.Request) -> web.WebSocketResponse:
        return await handle_webrtc_signaling(request, fake_sink)

    app = web.Application()
    app.router.add_get("/ws", handler)

    cand = "candidate:1 1 UDP 2130706431 192.168.1.1 54400 typ host"
    mid = "0"

    async with TestClient(TestServer(app)) as client:
        async with client.ws_connect("/ws") as ws:
            await asyncio.wait_for(ws.receive_str(), timeout=5.0)
            await ws.send_str(json.dumps({
                "type": "candidate",
                "candidate": cand,
                "sdpMid": mid,
            }))
            await asyncio.sleep(0.2)

    assert len(fake_sink.add_candidate_called) == 1
    _, received_cand, received_mid = fake_sink.add_candidate_called[0]
    assert received_cand == cand
    assert received_mid == mid


@pytest.mark.asyncio
async def test_signaling_remove_peer_on_disconnect():
    """After WebSocket closes, sink.remove_peer() must be called exactly once."""
    from aiohttp import web
    from visionpipe.server.signaling import handle_webrtc_signaling

    fake_sink = _FakeSink()

    async def handler(request: web.Request) -> web.WebSocketResponse:
        return await handle_webrtc_signaling(request, fake_sink)

    app = web.Application()
    app.router.add_get("/ws", handler)

    async with TestClient(TestServer(app)) as client:
        async with client.ws_connect("/ws") as ws:
            await asyncio.wait_for(ws.receive_str(), timeout=5.0)
        # ws is now closed

    await asyncio.sleep(0.2)
    assert len(fake_sink.remove_peer_called) == 1


@pytest.mark.asyncio
async def test_signaling_offer_error_closes_ws():
    """If get_offer raises, the WebSocket should close gracefully (not hang)."""
    from aiohttp import web
    from visionpipe.server.signaling import handle_webrtc_signaling

    fake_sink = _FakeSink(raise_on_offer=True)

    async def handler(request: web.Request) -> web.WebSocketResponse:
        return await handle_webrtc_signaling(request, fake_sink)

    app = web.Application()
    app.router.add_get("/ws", handler)

    async with TestClient(TestServer(app)) as client:
        async with client.ws_connect("/ws") as ws:
            # Server should close the WS after the error without hanging
            msg = await asyncio.wait_for(ws.receive(), timeout=5.0)
            # Either closed or empty; the important thing is we don't hang
            assert msg is not None

    # remove_peer must still be called (cleanup in finally block)
    await asyncio.sleep(0.2)
    assert len(fake_sink.remove_peer_called) == 1


# ---------------------------------------------------------------------------
# management_api /ws/{id}/webrtc endpoint tests
# ---------------------------------------------------------------------------


class TestManagementApiWebRTC:
    """Tests for the /ws/{id}/webrtc endpoint in ManagementServer."""

    @pytest.fixture
    async def client_with_server(self):
        """Create a ManagementServer with a fake manager."""
        from visionpipe.server import ManagementServer

        fake_manager = MagicMock()

        server = ManagementServer(fake_manager, host="127.0.0.1", port=0)
        async with TestClient(TestServer(server._app)) as client:
            yield client, fake_manager

    @pytest.mark.asyncio
    async def test_unknown_pipeline_closes_4004(self, client_with_server):
        """Connecting to an unknown pipeline closes with code 4004."""
        client, fake_manager = client_with_server

        # Make manager.get() raise so the pipeline is "not found"
        fake_manager.get.side_effect = RuntimeError("not found")

        async with client.ws_connect("/ws/unknown-id/webrtc") as ws:
            await asyncio.wait_for(ws.receive(), timeout=5.0)
            assert ws.closed
            assert ws.close_code == 4004

    @pytest.mark.asyncio
    async def test_pipeline_without_webrtc_sink_closes_4004(self, client_with_server):
        """Pipeline that has no WebRTCSink closes the WS with code 4004."""
        import visionpipe
        client, fake_manager = client_with_server

        # Pipeline with only a ByteTrackNode (no WebRTCSink)
        tracker = visionpipe.ByteTrackNode(visionpipe.ByteTrackConfig(), "tracker")
        pipeline_cfg = visionpipe.PipelineConfig()
        pipeline_cfg.name = "test"
        pipeline = visionpipe.Pipeline(pipeline_cfg)
        pipeline.add_node(tracker)

        fake_manager.get.return_value = pipeline
        fake_manager.get.side_effect = None

        async with client.ws_connect("/ws/pid/webrtc") as ws:
            await asyncio.wait_for(ws.receive(), timeout=5.0)
            assert ws.closed
            assert ws.close_code == 4004
