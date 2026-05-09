"""E2E tests for T4.2 WebRTC Sink.

Tests verify:
  1. WebRTC signaling handshake via WebSocket (/ws/{id}/webrtc)
  2. Browser-level video reception via Playwright headless Chrome
  3. End-to-end latency < 300 ms on loopback

Requires:
  - NVIDIA GPU (CUDA)
  - VISIONPIPE_USE_WEBRTC=ON build (libdatachannel + FFmpeg NVENC)
  - playwright installed: `playwright install chromium`
  - pytest-playwright

All tests are skipped when the GPU or WebRTC extension is unavailable.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PYTHON_DIR = ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

# ---------------------------------------------------------------------------
# Feature / GPU detection
# ---------------------------------------------------------------------------


def _has_gpu() -> bool:
    try:
        import visionpipe
        visionpipe.PipelineManager()
        return True
    except Exception:
        return False


def _has_webrtc() -> bool:
    """True only when the extension was built with VISIONPIPE_USE_WEBRTC."""
    try:
        import visionpipe
        sink = visionpipe.WebRTCSink()
        # If the stub is active, create_peer returns "" (no-op)
        peer_id = sink.create_peer()
        return bool(peer_id)
    except Exception:
        return False


requires_gpu = pytest.mark.skipif(not _has_gpu(), reason="GPU required")
requires_webrtc = pytest.mark.skipif(not _has_webrtc(), reason="VISIONPIPE_USE_WEBRTC build required")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def event_loop():
    """Module-scoped event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="module")
def server_and_pipeline():
    """Start a ManagementServer with a WebRTCSink pipeline, yield base_url."""
    import visionpipe
    from visionpipe.server import ManagementServer

    manager = visionpipe.PipelineManager()

    # Build a minimal pipeline: ByteTrack → WebRTCSink
    tracker_cfg = visionpipe.ByteTrackConfig()
    tracker = visionpipe.ByteTrackNode(tracker_cfg, "tracker")

    webrtc_cfg = visionpipe.WebRTCSinkConfig()
    webrtc_cfg.fps = 15
    webrtc_cfg.video_bitrate_kbps = 500
    webrtc_cfg.use_nvenc = True
    sink = visionpipe.WebRTCSink(webrtc_cfg, "webrtc_sink")

    pipeline_cfg = visionpipe.PipelineConfig()
    pipeline_cfg.name = "e2e-webrtc"
    pipeline = visionpipe.Pipeline(pipeline_cfg)
    pipeline.add_node(tracker)
    pipeline.add_node(sink)
    pipeline.connect(tracker, sink)

    pipeline_id: str = manager.create_pipeline(pipeline)

    server = ManagementServer(manager, host="127.0.0.1", port=18765)

    loop = asyncio.new_event_loop()
    loop.run_until_complete(server.start())

    yield "http://127.0.0.1:18765", pipeline_id, sink

    loop.run_until_complete(server.stop())
    try:
        manager.stop(pipeline_id)
    except Exception:
        pass
    manager.destroy(pipeline_id)
    loop.close()


# ---------------------------------------------------------------------------
# Tests – signaling protocol (no browser, pure Python WebSocket client)
# ---------------------------------------------------------------------------


@requires_webrtc
@pytest.mark.asyncio
async def test_signaling_offer_received(server_and_pipeline):
    """Server must send an SDP offer upon WebSocket connection."""
    from aiohttp import ClientSession

    base_url, pipeline_id, _ = server_and_pipeline
    ws_url = base_url.replace("http://", "ws://") + f"/ws/{pipeline_id}/webrtc"

    async with ClientSession() as session:
        async with session.ws_connect(ws_url, timeout=10) as ws:
            msg = await asyncio.wait_for(ws.receive_str(), timeout=15.0)
            data = json.loads(msg)
            assert data["type"] == "offer"
            assert "sdp" in data
            sdp = data["sdp"]
            assert "m=video" in sdp.lower() or "video" in sdp.lower(), \
                f"SDP offer does not contain a video section: {sdp[:200]}"


@requires_webrtc
@pytest.mark.asyncio
async def test_signaling_peer_created_and_removed(server_and_pipeline):
    """Connecting creates a peer; disconnecting removes it."""
    from aiohttp import ClientSession

    base_url, pipeline_id, sink = server_and_pipeline
    ws_url = base_url.replace("http://", "ws://") + f"/ws/{pipeline_id}/webrtc"

    before = sink.peer_count()

    async with ClientSession() as session:
        async with session.ws_connect(ws_url, timeout=10) as ws:
            # Wait for offer so the peer is fully registered
            await asyncio.wait_for(ws.receive_str(), timeout=15.0)
            during = sink.peer_count()
            assert during == before + 1, \
                f"Expected peer_count to increase by 1; before={before} during={during}"

    # Give cleanup a moment
    await asyncio.sleep(0.2)
    after = sink.peer_count()
    assert after == before, \
        f"Expected peer_count to return to {before} after disconnect; got {after}"


@requires_webrtc
@pytest.mark.asyncio
async def test_signaling_unknown_pipeline_closes_with_4004(server_and_pipeline):
    """Connecting to an unknown pipeline ID must close with code 4004."""
    from aiohttp import ClientSession, WSServerHandshakeError

    base_url, _, _ = server_and_pipeline
    ws_url = base_url.replace("http://", "ws://") + "/ws/nonexistent-id/webrtc"

    async with ClientSession() as session:
        try:
            async with session.ws_connect(ws_url, timeout=5) as ws:
                await asyncio.wait_for(ws.receive(), timeout=5.0)
                assert ws.closed
                assert ws.close_code == 4004
        except WSServerHandshakeError:
            pass  # server may refuse upgrade immediately


# ---------------------------------------------------------------------------
# Tests – browser-level E2E with Playwright
# ---------------------------------------------------------------------------

# Inline HTML page that connects to the signaling WebSocket and renders video
_WEBRTC_TEST_PAGE = """\
<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>VisionPipe WebRTC E2E</title></head>
<body>
<video id="v" autoplay playsinline muted width="320" height="240"></video>
<div id="status">connecting</div>
<script>
const sigUrl = window.__VP_WS_URL__;
const pc = new RTCPeerConnection({iceServers:[{urls:"stun:stun.l.google.com:19302"}]});
const ws = new WebSocket(sigUrl);
let firstFrameTs = null;
let connected = false;

pc.ontrack = e => {
  document.getElementById('v').srcObject = e.streams[0];
  connected = true;
  document.getElementById('status').textContent = 'connected';
};

pc.onicecandidate = e => {
  if(e.candidate) {
    ws.send(JSON.stringify({type:'candidate', candidate:e.candidate.candidate,
                            sdpMid:e.candidate.sdpMid}));
  }
};

ws.onmessage = async msg => {
  const d = JSON.parse(msg.data);
  if(d.type==='offer'){
    await pc.setRemoteDescription({type:'offer', sdp:d.sdp});
    const ans = await pc.createAnswer();
    await pc.setLocalDescription(ans);
    ws.send(JSON.stringify({type:'answer', sdp:ans.sdp}));
    document.getElementById('status').textContent = 'answered';
  } else if(d.type==='candidate'){
    try{ await pc.addIceCandidate({candidate:d.candidate,sdpMid:d.sdpMid}); }
    catch(e){}
  }
};

const video = document.getElementById('v');
video.addEventListener('loadeddata', () => {
  firstFrameTs = Date.now();
  document.getElementById('status').textContent = 'playing';
});
</script>
</body>
</html>
"""


@requires_webrtc
@pytest.mark.skipif(True, reason="Playwright browser tests require `playwright install chromium`")
def test_browser_video_playback(server_and_pipeline, page):
    """Browser receives and plays the WebRTC video stream.

    Skip this test until `playwright install chromium` is run in the CI
    environment.  Remove the skipif marker to enable it.
    """
    import re

    from playwright.sync_api import expect

    base_url, pipeline_id, _ = server_and_pipeline
    ws_url = base_url.replace("http://", "ws://") + f"/ws/{pipeline_id}/webrtc"

    # Inject the test page via data: URL so we can set the WS URL
    html = _WEBRTC_TEST_PAGE.replace("window.__VP_WS_URL__", json.dumps(ws_url))
    page.set_content(html)

    # Wait up to 15 s for 'playing' status (first video frame decoded)
    status = page.locator("#status")
    expect(status).to_have_text(re.compile(r"playing"), timeout=15_000)


@requires_webrtc
@pytest.mark.skipif(True, reason="Playwright browser tests require `playwright install chromium`")
def test_browser_latency_under_300ms(server_and_pipeline, page):
    """End-to-end latency (server → browser first frame) must be < 300 ms.

    Measured as: time from WS connection open to first 'loadeddata' event.
    Remove the skipif marker once Playwright is installed in the environment.
    """
    import re

    from playwright.sync_api import expect

    base_url, pipeline_id, _ = server_and_pipeline
    ws_url = base_url.replace("http://", "ws://") + f"/ws/{pipeline_id}/webrtc"

    html = _WEBRTC_TEST_PAGE.replace("window.__VP_WS_URL__", json.dumps(ws_url))

    t0 = time.monotonic()
    page.set_content(html)

    status = page.locator("#status")
    expect(status).to_have_text(re.compile(r"playing"), timeout=5_000)
    latency_ms = (time.monotonic() - t0) * 1000

    assert latency_ms < 300, f"Latency {latency_ms:.1f} ms exceeds 300 ms target"
