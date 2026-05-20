"""CustomNode: user-facing base class for custom processing nodes.

Supports two execution modes:

* **subprocess** (default) — ``on_frame`` runs in a dedicated child process,
  free from the GIL.  Frame metadata is serialised over a Unix Domain Socket;
  the GPU tensor stays in the main process.
* **inline** — same-process callback via the existing :class:`PyNode`
  mechanism.  Lower overhead, but holds the GIL during execution.
"""

from __future__ import annotations

import logging
import multiprocessing
import os
import signal
import socket
import threading
import time
from importlib import import_module
from typing import TYPE_CHECKING, Any

from visionpipe.frame_view import FrameView

if TYPE_CHECKING:
    from visionpipe import Frame

logger = logging.getLogger("visionpipe.custom_node")


def _get_ext():
    try:
        return import_module("visionpipe.visionpipe_python")
    except ImportError:
        return import_module("visionpipe_python")


class CustomNode:
    """User-facing base class for custom pipeline nodes.

    Subclass and override :meth:`on_frame` to inject arbitrary logic.

    Parameters
    ----------
    name:
        Node name shown in logs and stats.
    process_mode:
        ``"subprocess"`` (default) for out-of-process execution, or
        ``"inline"`` for same-process callback (like :class:`PyNode`).
    restart_limit:
        Maximum automatic restarts when the subprocess crashes
        (subprocess mode only).
    restart_delay:
        Seconds to wait before restarting a crashed subprocess.
    """

    def __init__(
        self,
        name: str = "custom_node",
        process_mode: str = "subprocess",
        restart_limit: int = 3,
        restart_delay: float = 1.0,
    ) -> None:
        if process_mode not in ("subprocess", "inline"):
            raise ValueError(f"process_mode must be 'subprocess' or 'inline', got {process_mode!r}")

        self._name = name
        self._process_mode = process_mode
        self._restart_limit = restart_limit
        self._restart_delay = restart_delay

        self._cpp_node = None
        self._parent_sock: socket.socket | None = None
        self._child_proc: multiprocessing.Process | None = None
        self._monitor_thread: threading.Thread | None = None
        self._stopped = threading.Event()
        self._restart_count = 0
        self._init_kwargs: dict[str, Any] = {}

        self._build_node()

    # ------------------------------------------------------------------ #
    #  User-overridable hooks
    # ------------------------------------------------------------------ #

    def on_frame(self, frame: FrameView) -> None:
        """Override to process each frame.

        *frame* is a :class:`~visionpipe.frame_view.FrameView` — a safe,
        invalidated-after-return view of the C++ Frame metadata.
        Modify ``frame.user_data`` to pass data downstream.
        """

    def setup(self) -> None:
        """Called once in the subprocess before the first ``on_frame``."""

    def teardown(self) -> None:
        """Called once in the subprocess after the last ``on_frame``."""

    # ------------------------------------------------------------------ #
    #  Internal: node construction
    # ------------------------------------------------------------------ #

    @classmethod
    def __new_subprocess_instance__(cls, **kwargs) -> "CustomNode":
        """Construct an instance inside the subprocess (no C++ node)."""
        obj = object.__new__(cls)
        obj._name = kwargs.pop("__name__", "custom_node")
        obj._process_mode = "subprocess"
        obj._cpp_node = None
        obj._parent_sock = None
        obj._child_proc = None
        obj._monitor_thread = None
        obj._stopped = threading.Event()
        obj._restart_count = 0
        obj._restart_limit = 0
        obj._restart_delay = 0
        obj._init_kwargs = kwargs
        return obj

    def _build_node(self) -> None:
        if self._process_mode == "inline":
            self._build_inline()
        else:
            self._build_subprocess()

    def _build_inline(self) -> None:
        ext = _get_ext()

        def callback(cpp_frame: "Frame") -> None:
            view = FrameView._from_cpp_frame(cpp_frame)
            try:
                self.on_frame(view)
            finally:
                if view.valid:
                    _apply_inline_updates(cpp_frame, view)
                view._invalidate()

        self._cpp_node = ext.PyNode(callback, self._name)

    def _build_subprocess(self) -> None:
        ext = _get_ext()

        parent_sock, child_sock = socket.socketpair(
            socket.AF_UNIX, socket.SOCK_STREAM
        )
        self._parent_sock = parent_sock

        self._cpp_node = ext.ProcessProxyNode(self._name, parent_sock.fileno())

        self._start_child(child_sock)

    def _start_child(self, child_sock: socket.socket) -> None:
        ctx = multiprocessing.get_context("fork")
        module_name = type(self).__module__
        class_name = type(self).__qualname__

        kwargs = dict(self._init_kwargs)
        kwargs["__name__"] = self._name

        self._child_proc = ctx.Process(
            target=_child_entry,
            args=(child_sock.fileno(), module_name, class_name, kwargs),
            daemon=True,
            name=f"vp-custom-{self._name}",
        )
        self._child_proc.start()
        child_sock.close()

        self._stopped.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitor_child, daemon=True,
            name=f"vp-monitor-{self._name}",
        )
        self._monitor_thread.start()

    def _monitor_child(self) -> None:
        while not self._stopped.is_set():
            if self._child_proc is None:
                break
            self._child_proc.join(timeout=1.0)
            if self._child_proc.is_alive():
                continue
            exitcode = self._child_proc.exitcode
            if self._stopped.is_set():
                break
            if exitcode == 0:
                break

            logger.warning(
                "CustomNode '%s' subprocess exited with code %s "
                "(restart %d/%d)",
                self._name, exitcode,
                self._restart_count + 1, self._restart_limit,
            )

            if self._restart_count >= self._restart_limit:
                logger.error(
                    "CustomNode '%s': restart limit reached, giving up",
                    self._name,
                )
                break

            self._restart_count += 1
            time.sleep(self._restart_delay)

            try:
                new_parent, child_sock = socket.socketpair(
                    socket.AF_UNIX, socket.SOCK_STREAM,
                )
                old_parent = self._parent_sock
                self._parent_sock = new_parent

                ext = _get_ext()
                self._cpp_node = ext.ProcessProxyNode(
                    self._name, new_parent.fileno(),
                )

                if old_parent is not None:
                    old_parent.close()

                self._start_child(child_sock)
                return
            except Exception:
                logger.exception(
                    "CustomNode '%s': restart failed", self._name
                )
                break

    def _stop_subprocess(self) -> None:
        self._stopped.set()
        if self._child_proc is not None and self._child_proc.is_alive():
            self._child_proc.join(timeout=3.0)
            if self._child_proc.is_alive():
                self._child_proc.kill()
                self._child_proc.join(timeout=2.0)
        if self._monitor_thread is not None:
            self._monitor_thread.join(timeout=2.0)
        if self._parent_sock is not None:
            self._parent_sock.close()
            self._parent_sock = None

    def __del__(self) -> None:
        try:
            self._stop_subprocess()
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    #  NodeBase delegation (same pattern as PyNode)
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        self._cpp_node.start()

    def stop(self, drain: bool = True) -> None:
        self._cpp_node.stop(drain)
        if self._process_mode == "subprocess":
            self._stop_subprocess()

    def wait_stop(self) -> None:
        self._cpp_node.wait_stop()

    def name(self) -> str:
        return self._cpp_node.name()

    def state(self):
        return self._cpp_node.state()

    def stats(self):
        return self._cpp_node.stats()

    def is_source(self) -> bool:
        return False

    def is_sink(self) -> bool:
        return False

    def __rshift__(self, other):
        return self._cpp_node.__rshift__(other)

    def __rrshift__(self, other):
        return self._cpp_node.__rrshift__(other)

    def pop_frame(self, timeout_ms: int = 500):
        return self._cpp_node.pop_frame(timeout_ms)


# ------------------------------------------------------------------ #
#  Module-level helpers
# ------------------------------------------------------------------ #


class _InlineFrameView(FrameView):
    """FrameView backed by a live C++ Frame for inline mode."""

    __slots__ = ("_cpp_frame",)

    def __init__(self, cpp_frame: "Frame") -> None:
        self._cpp_frame = cpp_frame
        self._valid = True
        self._user_data: dict[str, Any] = {}
        self._data = {}

    @property
    def stream_id(self) -> int:
        self._check()
        return self._cpp_frame.stream_id

    @property
    def frame_id(self) -> int:
        self._check()
        return self._cpp_frame.frame_id

    @property
    def pts_us(self) -> int:
        self._check()
        return self._cpp_frame.pts_us

    @property
    def detections(self) -> list:
        self._check()
        return list(self._cpp_frame.detections)

    @property
    def classifications(self) -> list:
        self._check()
        return list(self._cpp_frame.classifications)

    @property
    def tracks(self) -> list:
        self._check()
        return list(self._cpp_frame.tracks)


FrameView._from_cpp_frame = classmethod(  # type: ignore[attr-defined]
    lambda cls, cpp_frame: _InlineFrameView(cpp_frame)
)


def _apply_inline_updates(cpp_frame: "Frame", view: FrameView) -> None:
    for key, val in view.user_data.items():
        cpp_frame.set_user_data(key, val)


def _child_entry(
    sock_fd: int,
    module_name: str,
    class_name: str,
    init_kwargs: dict[str, Any],
) -> None:
    """Target function for the forked child process."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    from visionpipe.ipc.worker import run_worker

    run_worker(sock_fd, module_name, class_name, init_kwargs)
