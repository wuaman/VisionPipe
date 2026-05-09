"""Integration tests for the T4.1 Management REST API.

Tests the following endpoints:
  - POST   /pipelines
  - GET    /pipelines
  - DELETE /pipelines/{id}
  - GET    /pipelines/{id}/health
  - POST   /pipelines/{id}/params

Requires: aiohttp, pytest-asyncio, visionpipe (with C++ extension built).
GPU-dependent tests are skipped when no NVIDIA GPU is available.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from aiohttp.test_utils import TestClient, TestServer

# Ensure the Python package is importable from the source tree.
ROOT = Path(__file__).resolve().parents[3]
PYTHON_DIR = ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

import visionpipe
from visionpipe.server import ManagementServer


# ---------------------------------------------------------------------------
# GPU detection helper
# ---------------------------------------------------------------------------


def _has_gpu() -> bool:
    """Check if an NVIDIA GPU is available via the C++ extension."""
    try:
        # PipelineManager construction requires CUDA init; if it fails, no GPU.
        visionpipe.PipelineManager()
        return True
    except Exception:
        return False


requires_gpu = pytest.mark.skipif(not _has_gpu(), reason="GPU required")


# ---------------------------------------------------------------------------
# Minimal pipeline spec fixtures
# ---------------------------------------------------------------------------

MINIMAL_SPEC_DICT: dict = {
    "name": "test-pipe",
    "nodes": [{"name": "tracker", "type": "bytetrack", "params": {}}],
    "edges": [],
}

MINIMAL_SPEC_YAML: str = """\
name: test-pipe
nodes:
  - name: tracker
    type: bytetrack
    params: {}
edges: []
"""

INVALID_NODE_TYPE_SPEC: dict = {
    "name": "bad-pipe",
    "nodes": [{"name": "bogus", "type": "nonexistent_type", "params": {}}],
    "edges": [],
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def manager():
    """Create a fresh PipelineManager for each test, with teardown cleanup."""
    m = visionpipe.PipelineManager()
    yield m
    # Teardown: stop and destroy all remaining pipelines to prevent C++ lifecycle issues.
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
    # Brief pause to let background C++ threads finish before GC.
    time.sleep(0.05)


@pytest.fixture
def server_app(manager):
    """Create the aiohttp Application from ManagementServer without binding a port."""
    srv = ManagementServer(manager, host="127.0.0.1", port=0)
    # Access the internal aiohttp app for use with TestClient.
    return srv._app


@pytest.fixture
async def client(server_app):
    """Provide an aiohttp TestClient that doesn't bind a real port."""
    async with TestClient(TestServer(server_app)) as c:
        yield c


# ---------------------------------------------------------------------------
# Tests: POST /pipelines
# ---------------------------------------------------------------------------


@requires_gpu
class TestCreatePipeline:
    """Tests for POST /pipelines endpoint."""

    async def test_create_pipeline_json(self, client: TestClient) -> None:
        """POST /pipelines with JSON spec returns 201 + {id: str}."""
        resp = await client.post(
            "/pipelines",
            json={"spec": MINIMAL_SPEC_DICT},
        )
        assert resp.status == 201
        body = await resp.json()
        assert "id" in body
        assert isinstance(body["id"], str)
        assert len(body["id"]) > 0

    async def test_create_pipeline_yaml(self, client: TestClient) -> None:
        """POST /pipelines with YAML string in spec field returns 201."""
        resp = await client.post(
            "/pipelines",
            json={"spec": MINIMAL_SPEC_YAML},
        )
        assert resp.status == 201
        body = await resp.json()
        assert "id" in body
        assert isinstance(body["id"], str)

    async def test_create_pipeline_invalid_spec(self, client: TestClient) -> None:
        """POST /pipelines with invalid node type returns 422."""
        resp = await client.post(
            "/pipelines",
            json={"spec": INVALID_NODE_TYPE_SPEC},
        )
        assert resp.status == 422
        body = await resp.json()
        assert "error" in body

    async def test_create_pipeline_bad_json(self, client: TestClient) -> None:
        """POST /pipelines with malformed body returns 400."""
        resp = await client.post(
            "/pipelines",
            data=b"this is not json{{{",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status == 400
        body = await resp.json()
        assert "error" in body


# ---------------------------------------------------------------------------
# Tests: GET /pipelines
# ---------------------------------------------------------------------------


@requires_gpu
class TestListPipelines:
    """Tests for GET /pipelines endpoint."""

    async def test_list_pipelines_empty(self, client: TestClient) -> None:
        """GET /pipelines on a fresh manager returns an empty list."""
        resp = await client.get("/pipelines")
        assert resp.status == 200
        body = await resp.json()
        assert body == []

    async def test_list_pipelines_after_create(self, client: TestClient) -> None:
        """After creating a pipeline, GET /pipelines includes it."""
        # Create
        create_resp = await client.post(
            "/pipelines",
            json={"spec": MINIMAL_SPEC_DICT},
        )
        assert create_resp.status == 201
        created_id = (await create_resp.json())["id"]

        # List
        list_resp = await client.get("/pipelines")
        assert list_resp.status == 200
        pipelines = await list_resp.json()
        assert isinstance(pipelines, list)
        assert len(pipelines) >= 1
        ids = [p["id"] if isinstance(p, dict) else p for p in pipelines]
        assert created_id in ids


# ---------------------------------------------------------------------------
# Tests: DELETE /pipelines/{id}
# ---------------------------------------------------------------------------


@requires_gpu
class TestDeletePipeline:
    """Tests for DELETE /pipelines/{id} endpoint."""

    async def test_delete_pipeline(self, client: TestClient) -> None:
        """POST create, DELETE, then GET list should no longer contain the id."""
        # Create
        create_resp = await client.post(
            "/pipelines",
            json={"spec": MINIMAL_SPEC_DICT},
        )
        assert create_resp.status == 201
        pipe_id = (await create_resp.json())["id"]

        # Delete
        del_resp = await client.delete(f"/pipelines/{pipe_id}")
        assert del_resp.status == 204

        # Verify removed
        list_resp = await client.get("/pipelines")
        assert list_resp.status == 200
        pipelines = await list_resp.json()
        ids = [p["id"] if isinstance(p, dict) else p for p in pipelines]
        assert pipe_id not in ids

    async def test_delete_nonexistent(self, client: TestClient) -> None:
        """DELETE /pipelines/nonexistent returns 404."""
        resp = await client.delete("/pipelines/nonexistent-id-999")
        assert resp.status == 404
        body = await resp.json()
        assert "error" in body


# ---------------------------------------------------------------------------
# Tests: GET /pipelines/{id}/health
# ---------------------------------------------------------------------------


@requires_gpu
class TestHealthEndpoint:
    """Tests for GET /pipelines/{id}/health endpoint."""

    async def test_health_fields_present(self, client: TestClient) -> None:
        """GET /pipelines/{id}/health returns state, total_frames_processed, nodes."""
        # Create a pipeline first
        create_resp = await client.post(
            "/pipelines",
            json={"spec": MINIMAL_SPEC_DICT},
        )
        assert create_resp.status == 201
        pipe_id = (await create_resp.json())["id"]

        # Query health
        health_resp = await client.get(f"/pipelines/{pipe_id}/health")
        assert health_resp.status == 200
        body = await health_resp.json()

        # Required top-level fields
        assert "state" in body
        assert "total_frames_processed" in body
        assert "nodes" in body
        assert isinstance(body["nodes"], list)

        # If nodes are present, verify node health schema fields
        if len(body["nodes"]) > 0:
            node = body["nodes"][0]
            assert "name" in node
            assert "fps" in node
            assert "input_queue" in node
            queue = node["input_queue"]
            assert "capacity" in queue
            assert "current_size" in queue

    async def test_health_nonexistent(self, client: TestClient) -> None:
        """GET /pipelines/nonexistent/health returns 404."""
        resp = await client.get("/pipelines/nonexistent-id-999/health")
        assert resp.status == 404
        body = await resp.json()
        assert "error" in body


# ---------------------------------------------------------------------------
# Tests: POST /pipelines/{id}/params
# ---------------------------------------------------------------------------


@requires_gpu
class TestSetParams:
    """Tests for POST /pipelines/{id}/params endpoint."""

    async def test_set_param_nonexistent_pipeline(self, client: TestClient) -> None:
        """POST /pipelines/nonexistent/params returns 404."""
        resp = await client.post(
            "/pipelines/nonexistent-id-999/params",
            json={"node_id": "tracker", "param_name": "track_thresh", "value": 0.5},
        )
        assert resp.status == 404
        body = await resp.json()
        assert "error" in body

    async def test_set_param_nonexistent_node(self, client: TestClient) -> None:
        """POST /pipelines/{id}/params with bad node_id returns 404."""
        # Create pipeline
        create_resp = await client.post(
            "/pipelines",
            json={"spec": MINIMAL_SPEC_DICT},
        )
        assert create_resp.status == 201
        pipe_id = (await create_resp.json())["id"]

        # Try setting param on non-existent node
        resp = await client.post(
            f"/pipelines/{pipe_id}/params",
            json={
                "node_id": "does_not_exist",
                "param_name": "track_thresh",
                "value": 0.5,
            },
        )
        assert resp.status == 404
        body = await resp.json()
        assert "error" in body


# ---------------------------------------------------------------------------
# Tests: End-to-end flow
# ---------------------------------------------------------------------------


@requires_gpu
class TestE2EFlow:
    """End-to-end test: create -> list -> health -> delete -> list (verify deleted)."""

    async def test_e2e_create_list_health_delete(self, client: TestClient) -> None:
        """Full lifecycle: POST create -> GET list -> GET health -> DELETE -> GET list."""
        # Step 1: Create pipeline
        create_resp = await client.post(
            "/pipelines",
            json={"spec": MINIMAL_SPEC_DICT},
        )
        assert create_resp.status == 201
        pipe_id = (await create_resp.json())["id"]
        assert isinstance(pipe_id, str)
        assert len(pipe_id) > 0

        # Step 2: Verify it appears in the list
        list_resp = await client.get("/pipelines")
        assert list_resp.status == 200
        pipelines = await list_resp.json()
        ids = [p["id"] if isinstance(p, dict) else p for p in pipelines]
        assert pipe_id in ids

        # Step 3: Query health
        health_resp = await client.get(f"/pipelines/{pipe_id}/health")
        assert health_resp.status == 200
        health = await health_resp.json()
        assert health["id"] == pipe_id
        assert "state" in health
        assert "total_frames_processed" in health
        assert isinstance(health["nodes"], list)

        # Step 4: Delete the pipeline
        del_resp = await client.delete(f"/pipelines/{pipe_id}")
        assert del_resp.status == 204

        # Step 5: Verify it's gone from the list
        list_resp2 = await client.get("/pipelines")
        assert list_resp2.status == 200
        pipelines2 = await list_resp2.json()
        ids2 = [p["id"] if isinstance(p, dict) else p for p in pipelines2]
        assert pipe_id not in ids2

        # Step 6: Health on deleted pipeline should return 404
        health_resp2 = await client.get(f"/pipelines/{pipe_id}/health")
        assert health_resp2.status == 404
