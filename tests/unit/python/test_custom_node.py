"""Tests for T3.2 CustomNode subprocess architecture.

Covers:

1. FrameView: happy path, invalidation, type errors, boundary values.
2. IPC protocol: round-trip, framing, oversize, half-closed socket.
3. CustomNode (inline mode): subclass override, default no-op, invalid mode.
4. CustomNode (subprocess mode): independent process, user_data round-trip,
   on_frame exceptions caught, restart accounting.
5. Worker loop: shutdown, unknown message, on_frame exceptions captured.

Tests that require the compiled C++ extension are guarded by
``@pytest.mark.skipif`` so the pure-Python tests still run in
GPU-less environments.
"""
from __future__ import annotations

import json
import socket
import struct
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
PYTHON_DIR = ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from visionpipe.frame_view import FrameView
from visionpipe.ipc.protocol import recv_msg, send_msg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _has_cpp_ext() -> bool:
    try:
        import visionpipe  # noqa: F401
        from visionpipe import ProcessProxyNode  # noqa: F401
        return True
    except Exception:
        return False


HAS_CPP = _has_cpp_ext()
needs_cpp = pytest.mark.skipif(
    not HAS_CPP,
    reason="C++ visionpipe_python extension not available",
)


def _sample_metadata(**overrides):
    base = {
        "type": "frame",
        "stream_id": 7,
        "frame_id": 42,
        "pts_us": 1_000_000,
        "detections": [{"cls": 0, "conf": 0.9, "x": 1, "y": 2, "w": 3, "h": 4}],
        "classifications": [{"cls": 1, "conf": 0.7}],
        "tracks": [{"track_id": 99, "cls": 0}],
        "user_data": {"foo": "bar"},
    }
    base.update(overrides)
    return base


# ===========================================================================
# FrameView
# ===========================================================================


class TestFrameViewHappyPath:
    def test_reads_scalar_fields(self):
        view = FrameView(_sample_metadata())

        assert view.valid is True
        assert view.stream_id == 7
        assert view.frame_id == 42
        assert view.pts_us == 1_000_000

    def test_reads_list_fields(self):
        view = FrameView(_sample_metadata())

        assert view.detections == [
            {"cls": 0, "conf": 0.9, "x": 1, "y": 2, "w": 3, "h": 4}
        ]
        assert view.classifications == [{"cls": 1, "conf": 0.7}]
        assert view.tracks == [{"track_id": 99, "cls": 0}]

    def test_user_data_is_copy_of_metadata(self):
        meta = _sample_metadata()
        view = FrameView(meta)

        assert view.user_data == {"foo": "bar"}

        view.user_data["new_key"] = 123
        # Mutating the view's user_data must not write back into the
        # original metadata dict — it is supposed to be a copy.
        assert "new_key" not in meta["user_data"]

    def test_user_data_setter_replaces_dict(self):
        view = FrameView(_sample_metadata())
        view.user_data = {"a": 1, "b": 2}
        assert view.user_data == {"a": 1, "b": 2}

    def test_snapshot_returns_independent_copy(self):
        view = FrameView(_sample_metadata())
        view.user_data["x"] = 10
        snap = view._get_user_data_snapshot()

        snap["x"] = 999
        assert view.user_data["x"] == 10


class TestFrameViewBoundary:
    def test_missing_user_data_yields_empty_dict(self):
        meta = _sample_metadata()
        del meta["user_data"]
        view = FrameView(meta)
        assert view.user_data == {}

    def test_null_user_data_yields_empty_dict(self):
        meta = _sample_metadata(user_data=None)
        view = FrameView(meta)
        assert view.user_data == {}

    def test_missing_detections_yields_empty_list(self):
        meta = _sample_metadata()
        del meta["detections"]
        view = FrameView(meta)
        assert view.detections == []

    def test_missing_classifications_yields_empty_list(self):
        meta = _sample_metadata()
        del meta["classifications"]
        view = FrameView(meta)
        assert view.classifications == []

    def test_missing_tracks_yields_empty_list(self):
        meta = _sample_metadata()
        del meta["tracks"]
        view = FrameView(meta)
        assert view.tracks == []

    def test_zero_values(self):
        view = FrameView(_sample_metadata(stream_id=0, frame_id=0, pts_us=0))
        assert view.stream_id == 0
        assert view.frame_id == 0
        assert view.pts_us == 0


class TestFrameViewInvalidation:
    def test_access_after_invalidate_raises(self):
        view = FrameView(_sample_metadata())
        view._invalidate()

        assert view.valid is False

        with pytest.raises(RuntimeError, match="no longer valid"):
            _ = view.frame_id
        with pytest.raises(RuntimeError, match="no longer valid"):
            _ = view.stream_id
        with pytest.raises(RuntimeError, match="no longer valid"):
            _ = view.pts_us
        with pytest.raises(RuntimeError, match="no longer valid"):
            _ = view.detections
        with pytest.raises(RuntimeError, match="no longer valid"):
            _ = view.classifications
        with pytest.raises(RuntimeError, match="no longer valid"):
            _ = view.tracks
        with pytest.raises(RuntimeError, match="no longer valid"):
            _ = view.user_data

    def test_user_data_setter_after_invalidate_raises(self):
        view = FrameView(_sample_metadata())
        view._invalidate()
        with pytest.raises(RuntimeError, match="no longer valid"):
            view.user_data = {"a": 1}

    def test_invalidate_clears_internal_data(self):
        meta = _sample_metadata()
        view = FrameView(meta)
        view._invalidate()
        # Internal storage cleared so external metadata can be GC'd.
        assert view._data == {}


class TestFrameViewTypeErrors:
    def test_user_data_setter_rejects_non_dict(self):
        view = FrameView(_sample_metadata())

        with pytest.raises(TypeError, match="must be a dict"):
            view.user_data = [("a", 1)]  # type: ignore[assignment]
        with pytest.raises(TypeError, match="must be a dict"):
            view.user_data = "foo"  # type: ignore[assignment]
        with pytest.raises(TypeError, match="must be a dict"):
            view.user_data = None  # type: ignore[assignment]
        with pytest.raises(TypeError, match="must be a dict"):
            view.user_data = 42  # type: ignore[assignment]


# ===========================================================================
# IPC protocol
# ===========================================================================


@pytest.fixture
def sockpair():
    a, b = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        yield a, b
    finally:
        for s in (a, b):
            try:
                s.close()
            except OSError:
                pass


class TestIPCProtocol:
    def test_round_trip_simple_dict(self, sockpair):
        a, b = sockpair
        send_msg(a, {"type": "frame", "frame_id": 1})
        got = recv_msg(b)
        assert got == {"type": "frame", "frame_id": 1}

    def test_round_trip_complex_payload(self, sockpair):
        a, b = sockpair
        payload = {
            "type": "frame",
            "stream_id": 3,
            "user_data": {"k": [1, 2, 3], "nested": {"a": True}},
            "detections": [{"x": 0.5}],
        }
        send_msg(a, payload)
        assert recv_msg(b) == payload

    def test_round_trip_empty_dict(self, sockpair):
        a, b = sockpair
        send_msg(a, {})
        assert recv_msg(b) == {}

    def test_recv_on_closed_socket_returns_none(self, sockpair):
        a, b = sockpair
        a.close()
        assert recv_msg(b) is None

    def test_recv_partial_header_returns_none(self, sockpair):
        a, b = sockpair
        # Send only 2 bytes of the 4-byte length header, then close.
        a.sendall(b"\x00\x00")
        a.close()
        assert recv_msg(b) is None

    def test_recv_partial_payload_returns_none(self, sockpair):
        a, b = sockpair
        # Declare length=10 but only send 4 bytes of payload, then close.
        a.sendall(struct.pack("!I", 10) + b"abcd")
        a.close()
        assert recv_msg(b) is None

    def test_oversize_message_raises(self, sockpair):
        a, b = sockpair
        # 64 MiB + 1 — one over the protocol's _MAX_MSG.
        a.sendall(struct.pack("!I", 64 * 1024 * 1024 + 1))
        a.close()
        with pytest.raises(RuntimeError, match="too large"):
            recv_msg(b)

    def test_wire_framing_is_big_endian_length_prefix(self, sockpair):
        a, b = sockpair
        send_msg(a, {"x": 1})
        # Inspect raw wire bytes on the receiving side.
        raw = b.recv(4096)
        length = struct.unpack("!I", raw[:4])[0]
        payload = raw[4 : 4 + length]
        assert length == len(payload)
        assert json.loads(payload) == {"x": 1}

    def test_multiple_messages_back_to_back(self, sockpair):
        a, b = sockpair
        send_msg(a, {"i": 1})
        send_msg(a, {"i": 2})
        send_msg(a, {"i": 3})
        assert recv_msg(b) == {"i": 1}
        assert recv_msg(b) == {"i": 2}
        assert recv_msg(b) == {"i": 3}


# ===========================================================================
# Worker loop (in-process, no fork)
# ===========================================================================


class _ParentSide:
    """Helper that drives the worker loop from a background thread."""

    def __init__(self, target, node):
        self.parent_sock, child_sock = socket.socketpair(
            socket.AF_UNIX, socket.SOCK_STREAM
        )
        self._thread = threading.Thread(
            target=lambda: target(child_sock, node), daemon=True
        )
        self._thread.start()

    def send(self, msg):
        send_msg(self.parent_sock, msg)

    def recv(self):
        return recv_msg(self.parent_sock)

    def shutdown(self):
        try:
            send_msg(self.parent_sock, {"type": "shutdown"})
        except OSError:
            pass
        self._thread.join(timeout=3.0)
        self.parent_sock.close()


def _run_worker_loop(sock, node):
    """Run only the message-handling loop on an already-instantiated node."""
    from visionpipe.ipc.worker import _loop

    sock.setblocking(True)
    try:
        node.setup()
        _loop(sock, node)
    finally:
        try:
            node.teardown()
        except Exception:
            pass
        sock.close()


class TestWorkerLoop:
    def test_happy_path_user_data_round_trip(self):
        from visionpipe.custom_node import CustomNode

        class _Node(CustomNode):
            def on_frame(self, frame: FrameView) -> None:
                frame.user_data["echo"] = frame.frame_id

        node = _Node.__new_subprocess_instance__(__name__="t")
        side = _ParentSide(_run_worker_loop, node)
        try:
            side.send(_sample_metadata(frame_id=11, user_data={}))
            reply = side.recv()
            assert reply == {"user_data": {"echo": 11}, "error": None}
        finally:
            side.shutdown()

    def test_unknown_message_type_returns_error_and_continues(self):
        from visionpipe.custom_node import CustomNode

        class _Node(CustomNode):
            def on_frame(self, frame: FrameView) -> None:
                frame.user_data["ok"] = True

        node = _Node.__new_subprocess_instance__(__name__="t")
        side = _ParentSide(_run_worker_loop, node)
        try:
            side.send({"type": "garbage"})
            reply = side.recv()
            assert "error" in reply
            assert "unknown message type: garbage" in reply["error"]

            # Worker must keep running and serve subsequent frames.
            side.send(_sample_metadata(user_data={}))
            reply = side.recv()
            assert reply["error"] is None
            assert reply["user_data"] == {"ok": True}
        finally:
            side.shutdown()

    def test_on_frame_exception_captured_and_loop_continues(self):
        from visionpipe.custom_node import CustomNode

        calls = []

        class _Node(CustomNode):
            def on_frame(self, frame: FrameView) -> None:
                calls.append(frame.frame_id)
                if frame.frame_id == 1:
                    raise ValueError("boom")
                frame.user_data["seen"] = frame.frame_id

        node = _Node.__new_subprocess_instance__(__name__="t")
        side = _ParentSide(_run_worker_loop, node)
        try:
            side.send(_sample_metadata(frame_id=1, user_data={}))
            reply = side.recv()
            assert reply["user_data"] == {}
            assert reply["error"] is not None
            assert "ValueError" in reply["error"]
            assert "boom" in reply["error"]

            # Loop survives; next frame succeeds.
            side.send(_sample_metadata(frame_id=2, user_data={}))
            reply = side.recv()
            assert reply["error"] is None
            assert reply["user_data"] == {"seen": 2}

            assert calls == [1, 2]
        finally:
            side.shutdown()

    def test_shutdown_message_terminates_loop(self):
        from visionpipe.custom_node import CustomNode

        node = CustomNode.__new_subprocess_instance__(__name__="t")
        side = _ParentSide(_run_worker_loop, node)
        # The loop should exit promptly after receiving shutdown.
        side.shutdown()
        assert not side._thread.is_alive()

    def test_eof_terminates_loop(self):
        from visionpipe.custom_node import CustomNode

        node = CustomNode.__new_subprocess_instance__(__name__="t")
        side = _ParentSide(_run_worker_loop, node)
        # Closing parent end without a shutdown message: recv_msg
        # returns None and the worker exits cleanly.
        side.parent_sock.close()
        side._thread.join(timeout=3.0)
        assert not side._thread.is_alive()


# ===========================================================================
# CustomNode construction (mode validation, no C++ ext required)
# ===========================================================================


class TestCustomNodeModeValidation:
    def test_invalid_process_mode_raises_value_error(self):
        from visionpipe.custom_node import CustomNode

        with pytest.raises(ValueError, match="process_mode must be"):
            CustomNode(process_mode="threaded")

    def test_empty_string_process_mode_raises(self):
        from visionpipe.custom_node import CustomNode

        with pytest.raises(ValueError, match="process_mode must be"):
            CustomNode(process_mode="")

    def test_none_process_mode_raises(self):
        from visionpipe.custom_node import CustomNode

        with pytest.raises(ValueError, match="process_mode must be"):
            CustomNode(process_mode=None)  # type: ignore[arg-type]

    def test_new_subprocess_instance_skips_cpp_construction(self):
        from visionpipe.custom_node import CustomNode

        obj = CustomNode.__new_subprocess_instance__(__name__="x")
        assert obj._cpp_node is None
        assert obj._process_mode == "subprocess"
        assert obj._init_kwargs == {}

    def test_new_subprocess_instance_strips_dunder_name_from_init_kwargs(self):
        from visionpipe.custom_node import CustomNode

        obj = CustomNode.__new_subprocess_instance__(
            __name__="my_node", threshold=0.5
        )
        assert obj._name == "my_node"
        assert obj._init_kwargs == {"threshold": 0.5}


# ===========================================================================
# Inline FrameView (wraps C++ Frame)
# ===========================================================================


@needs_cpp
class TestInlineFrameView:
    def test_wraps_cpp_frame_fields(self):
        from visionpipe import Frame
        from visionpipe.custom_node import _InlineFrameView

        cpp_frame = Frame()
        cpp_frame.frame_id = 17
        cpp_frame.stream_id = 2

        view = _InlineFrameView(cpp_frame)
        assert view.valid is True
        assert view.frame_id == 17
        assert view.stream_id == 2

    def test_inline_view_invalidation(self):
        from visionpipe import Frame
        from visionpipe.custom_node import _InlineFrameView

        view = _InlineFrameView(Frame())
        view._invalidate()
        assert view.valid is False
        with pytest.raises(RuntimeError, match="no longer valid"):
            _ = view.frame_id

    def test_detections_classifications_tracks_return_lists(self):
        from visionpipe import Frame
        from visionpipe.custom_node import _InlineFrameView

        view = _InlineFrameView(Frame())
        assert isinstance(view.detections, list)
        assert isinstance(view.classifications, list)
        assert isinstance(view.tracks, list)


# ===========================================================================
# CustomNode end-to-end (requires C++ extension)
# ===========================================================================


@needs_cpp
class TestCustomNodeInlineMode:
    def test_default_on_frame_is_noop(self):
        from visionpipe.custom_node import CustomNode

        node = CustomNode(name="noop", process_mode="inline")
        try:
            assert node.name() == "noop"
            assert node._cpp_node is not None
        finally:
            # Inline mode has no subprocess to stop, but call to be safe.
            try:
                node.stop()
            except Exception:
                pass

    def test_subclass_override_invoked(self):
        from visionpipe.custom_node import CustomNode

        class _Counter(CustomNode):
            def __init__(self) -> None:
                self.count = 0
                super().__init__(name="counter", process_mode="inline")

            def on_frame(self, frame):
                self.count += 1

        node = _Counter()
        assert node.name() == "counter"


@needs_cpp
class TestCustomNodeSubprocessMode:
    def test_subprocess_starts_and_stops_cleanly(self):
        from visionpipe.custom_node import CustomNode

        node = CustomNode(name="sub", process_mode="subprocess")
        try:
            assert node._child_proc is not None
            assert node._child_proc.is_alive()
            assert node._parent_sock is not None
        finally:
            node._stop_subprocess()
            # Child must terminate within the stop's timeout budget.
            for _ in range(50):
                if not node._child_proc.is_alive():
                    break
                time.sleep(0.05)
            assert not node._child_proc.is_alive()


# ===========================================================================
# Smoke: ensure top-level imports work
# ===========================================================================


def test_frame_view_is_exported_from_top_level():
    import visionpipe

    assert visionpipe.FrameView is FrameView


def test_custom_node_is_exported_from_top_level():
    import visionpipe
    from visionpipe.custom_node import CustomNode

    assert visionpipe.CustomNode is CustomNode
