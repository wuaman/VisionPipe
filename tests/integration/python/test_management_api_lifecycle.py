"""Integration tests for T4.1 supplemented REST API endpoints.

Tests the new/changed endpoints:
  - POST   /pipelines/{id}/start    (new)
  - POST   /pipelines/{id}/stop     (new)
  - DELETE  /pipelines/{id}          (changed: requires stopped/init state)
  - GET    /pipelines/{id}/nodes    (new)

Requires: aiohttp, pytest-asyncio, visionpipe (with C++ extension built).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from aiohttp.test_utils import TestClient, TestServer

ROOT = Path(__file__).resolve().parents[3]
PYTHON_DIR = ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

import visionpipe
from visionpipe.server import ManagementServer


def _has_gpu() -> bool:
    try:
        visionpipe.PipelineManager()
        return True
    except Exception:
        return False


requires_gpu = pytest.mark.skipif(not _has_gpu(), reason="GPU required")

MINIMAL_SPEC_DICT: dict = {
    "name": "test-pipe",
    "nodes": [{"name": "tracker", "type": "bytetrack", "params": {}}],
    "edges": [],
}

TEST_VIDEO = str(ROOT / "tests" / "data" / "48-3.mp4")

STARTABLE_SPEC_DICT: dict = {
    "name": "test-pipe",
    "nodes": [
        {"name": "src", "type": "file_source", "params": {"uri": TEST_VIDEO, "decode_mode": "CPU", "stream_id": 0}},
        {"name": "tracker", "type": "bytetrack", "params": {}},
    ],
    "edges": [{"from_node": "src", "to_node": "tracker"}],
}


async def _create_startable_pipeline(client: TestClient) -> str:
    resp = await client.post("/pipelines", json={"spec": STARTABLE_SPEC_DICT})
    assert resp.status == 201
    body = await resp.json()
    return body["id"]


@pytest.fixture
def manager():
    m = visionpipe.PipelineManager()
    yield m
    import time
    for pid in list(m.list()):
        try:
            status = m.status(pid)
            if status.name not in ("STOPPED", "ERROR", "INIT"):
                m.stop(pid, False)
        except Exception:
            pass
        try:
            m.destroy(pid)
        except Exception:
            pass
    time.sleep(0.05)


@pytest.fixture
def server_app(manager):
    srv = ManagementServer(manager, host="127.0.0.1", port=0)
    return srv._app


@pytest.fixture
async def client(server_app):
    async with TestClient(TestServer(server_app)) as c:
        yield c


async def _create_pipeline(client: TestClient) -> str:
    resp = await client.post("/pipelines", json={"spec": MINIMAL_SPEC_DICT})
    assert resp.status == 201
    body = await resp.json()
    return body["id"]


# ---------------------------------------------------------------------------
# Tests: POST /pipelines/{id}/start
# ---------------------------------------------------------------------------


@requires_gpu
class TestStartPipeline:
    async def test_start_pipeline_from_init(self, client: TestClient) -> None:
        pid = await _create_startable_pipeline(client)
        resp = await client.post(f"/pipelines/{pid}/start")
        assert resp.status == 200
        body = await resp.json()
        assert body["id"] == pid
        assert body["state"] == "RUNNING"

    async def test_start_nonexistent_pipeline(self, client: TestClient) -> None:
        resp = await client.post("/pipelines/nonexistent-id-999/start")
        assert resp.status == 404
        body = await resp.json()
        assert "error" in body

    async def test_start_already_running(self, client: TestClient) -> None:
        pid = await _create_startable_pipeline(client)
        await client.post(f"/pipelines/{pid}/start")
        resp = await client.post(f"/pipelines/{pid}/start")
        # Should not error — either 200 or idempotent
        assert resp.status in (200, 409)


# ---------------------------------------------------------------------------
# Tests: POST /pipelines/{id}/stop
# ---------------------------------------------------------------------------


@requires_gpu
class TestStopPipeline:
    async def test_stop_running_pipeline(self, client: TestClient) -> None:
        pid = await _create_startable_pipeline(client)
        await client.post(f"/pipelines/{pid}/start")
        resp = await client.post(f"/pipelines/{pid}/stop")
        assert resp.status == 200
        body = await resp.json()
        assert body["id"] == pid
        assert body["state"] == "STOPPED"

    async def test_stop_nonexistent_pipeline(self, client: TestClient) -> None:
        resp = await client.post("/pipelines/nonexistent-id-999/stop")
        assert resp.status == 404
        body = await resp.json()
        assert "error" in body

    async def test_stop_already_stopped(self, client: TestClient) -> None:
        pid = await _create_startable_pipeline(client)
        await client.post(f"/pipelines/{pid}/start")
        await client.post(f"/pipelines/{pid}/stop")
        resp = await client.post(f"/pipelines/{pid}/stop")
        # Should not crash — either 200 or idempotent
        assert resp.status in (200, 409)


# ---------------------------------------------------------------------------
# Tests: DELETE /pipelines/{id} (changed behavior)
# ---------------------------------------------------------------------------


@requires_gpu
class TestDeletePipelineLifecycle:
    async def test_delete_init_state(self, client: TestClient) -> None:
        """Pipeline in INIT state can be deleted directly."""
        pid = await _create_pipeline(client)
        resp = await client.delete(f"/pipelines/{pid}")
        assert resp.status == 204

    async def test_delete_running_returns_409(self, client: TestClient) -> None:
        """Running pipeline cannot be deleted — returns 409."""
        pid = await _create_startable_pipeline(client)
        await client.post(f"/pipelines/{pid}/start")
        resp = await client.delete(f"/pipelines/{pid}")
        assert resp.status == 409
        body = await resp.json()
        assert "error" in body

    async def test_delete_after_stop(self, client: TestClient) -> None:
        """Pipeline can be deleted after being stopped."""
        pid = await _create_startable_pipeline(client)
        await client.post(f"/pipelines/{pid}/start")
        await client.post(f"/pipelines/{pid}/stop")
        resp = await client.delete(f"/pipelines/{pid}")
        assert resp.status == 204


# ---------------------------------------------------------------------------
# Tests: GET /pipelines/{id}/nodes
# ---------------------------------------------------------------------------


@requires_gpu
class TestNodesEndpoint:
    async def test_nodes_returns_list(self, client: TestClient) -> None:
        pid = await _create_pipeline(client)
        resp = await client.get(f"/pipelines/{pid}/nodes")
        assert resp.status == 200
        body = await resp.json()
        assert isinstance(body, list)
        assert len(body) == 1  # single bytetrack node (tracker-only spec)

    async def test_nodes_schema_fields(self, client: TestClient) -> None:
        pid = await _create_pipeline(client)
        resp = await client.get(f"/pipelines/{pid}/nodes")
        assert resp.status == 200
        body = await resp.json()
        node = body[0]
        assert node["name"] == "tracker"
        assert "fps" in node
        assert "latency_ms" in node
        assert "frames_processed" in node
        assert "errors" in node
        assert "state" in node
        assert "input_queue" in node
        assert node["state"] in ("INIT", "RUNNING", "DRAINING", "STOPPED")

    async def test_nodes_state_after_start(self, client: TestClient) -> None:
        pid = await _create_startable_pipeline(client)
        await client.post(f"/pipelines/{pid}/start")
        resp = await client.get(f"/pipelines/{pid}/nodes")
        assert resp.status == 200
        body = await resp.json()
        tracker = next(n for n in body if n["name"] == "tracker")
        assert tracker["state"] == "RUNNING"

    async def test_nodes_nonexistent_pipeline(self, client: TestClient) -> None:
        resp = await client.get("/pipelines/nonexistent-id-999/nodes")
        assert resp.status == 404
        body = await resp.json()
        assert "error" in body

    async def test_nodes_input_queue_fields(self, client: TestClient) -> None:
        pid = await _create_pipeline(client)
        resp = await client.get(f"/pipelines/{pid}/nodes")
        body = await resp.json()
        queue = body[0]["input_queue"]
        assert "capacity" in queue
        assert "current_size" in queue
        assert "total_pushed" in queue
        assert "total_popped" in queue
        assert "dropped_count" in queue


# ---------------------------------------------------------------------------
# Tests: Full lifecycle flow
# ---------------------------------------------------------------------------


@requires_gpu
class TestFullLifecycleFlow:
    async def test_create_start_stop_delete(self, client: TestClient) -> None:
        """Full lifecycle: create → start → stop → delete."""
        # Create
        pid = await _create_startable_pipeline(client)

        # Verify INIT state
        list_resp = await client.get("/pipelines")
        pipelines = await list_resp.json()
        pipe = next(p for p in pipelines if p["id"] == pid)
        assert pipe["state"] == "INIT"

        # Start
        start_resp = await client.post(f"/pipelines/{pid}/start")
        assert start_resp.status == 200

        # Verify RUNNING state
        list_resp = await client.get("/pipelines")
        pipelines = await list_resp.json()
        pipe = next(p for p in pipelines if p["id"] == pid)
        assert pipe["state"] == "RUNNING"

        # Stop
        stop_resp = await client.post(f"/pipelines/{pid}/stop")
        assert stop_resp.status == 200

        # Delete
        del_resp = await client.delete(f"/pipelines/{pid}")
        assert del_resp.status == 204

        # Verify gone
        list_resp = await client.get("/pipelines")
        pipelines = await list_resp.json()
        ids = [p["id"] for p in pipelines]
        assert pid not in ids
