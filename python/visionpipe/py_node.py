from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from visionpipe import Frame


def _get_ext():
    try:
        return import_module("visionpipe.visionpipe_python")
    except ImportError:
        return import_module("visionpipe_python")


class PyNode:
    """Python-side base class for custom processing nodes.

    Subclass this and override :meth:`process` to inject arbitrary Python
    logic into a VisionPipe pipeline.  The underlying C++ node holds a
    reference to the bound ``_cpp_node`` attribute; do not replace it after
    construction.

    Example::

        class MyNode(PyNode):
            def process(self, frame: Frame) -> None:
                frame.user_data = {"count": len(frame.detections)}

        node = MyNode(name="counter")
        # node._cpp_node is a visionpipe_python.PyNode
    """

    def __init__(self, name: str = "py_node") -> None:
        ext = _get_ext()
        # Wrap self.process in a C++-compatible callback.
        # nanobind releases the GIL before calling into C++ and re-acquires it
        # when invoking the Python callable, so GIL management is automatic.
        self._cpp_node = ext.PyNode(self._safe_process, name)

    def process(self, frame: "Frame") -> None:
        """Override in subclasses to process each frame.

        Modify *frame* in-place (e.g. set ``frame.user_data``).
        Raising an exception here is safe: the C++ layer will catch it,
        log it, and continue processing subsequent frames.
        """

    def _safe_process(self, frame: "Frame") -> None:
        """Internal wrapper — do not override."""
        self.process(frame)

    # ------------------------------------------------------------------ #
    # Delegate NodeBase interface to the underlying C++ node so that
    # PyNode instances can be used wherever a NodeBase is expected.
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        self._cpp_node.start()

    def stop(self, drain: bool = True) -> None:
        self._cpp_node.stop(drain)

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

    def pop_frame(self, timeout_ms: int = 500):
        return self._cpp_node.pop_frame(timeout_ms)
