"""Subprocess worker loop for CustomNode.

Spawned by :class:`~visionpipe.custom_node.CustomNode` in *subprocess*
mode.  Receives frame metadata from the C++ ProcessProxyNode, calls the
user's ``on_frame`` callback, and sends back ``user_data`` modifications.
"""

from __future__ import annotations

import importlib
import logging
import socket
import traceback
from typing import Any

from visionpipe.frame_view import FrameView
from visionpipe.ipc.protocol import recv_msg, send_msg

logger = logging.getLogger("visionpipe.ipc.worker")


def run_worker(
    sock_fd: int,
    module_name: str,
    class_name: str,
    init_kwargs: dict[str, Any] | None = None,
) -> None:
    """Entry-point executed inside the child process.

    Parameters
    ----------
    sock_fd:
        File descriptor of the connected Unix Domain Socket.
    module_name:
        Fully-qualified Python module containing the CustomNode subclass.
    class_name:
        Name of the CustomNode subclass to instantiate.
    init_kwargs:
        Optional keyword arguments forwarded to the subclass constructor.
    """
    sock = socket.socket(fileno=sock_fd)
    sock.setblocking(True)

    node = _instantiate(module_name, class_name, init_kwargs or {})
    try:
        node.setup()
        _loop(sock, node)
    except Exception:
        logger.exception("worker crashed")
    finally:
        try:
            node.teardown()
        except Exception:
            logger.exception("teardown error")
        sock.close()


def _instantiate(module_name: str, class_name: str, kwargs: dict):
    mod = importlib.import_module(module_name)
    cls = getattr(mod, class_name)
    return cls.__new_subprocess_instance__(**kwargs)


def _loop(sock: socket.socket, node) -> None:
    while True:
        msg = recv_msg(sock)
        if msg is None:
            break

        msg_type = msg.get("type")
        if msg_type == "shutdown":
            break

        if msg_type != "frame":
            send_msg(sock, {"error": f"unknown message type: {msg_type}"})
            continue

        view = FrameView(msg)
        error = None
        try:
            node.on_frame(view)
        except Exception:
            error = traceback.format_exc()
            logger.error("on_frame error:\n%s", error)
        finally:
            user_data = view._get_user_data_snapshot() if view.valid else {}
            view._invalidate()

        response: dict[str, Any] = {"user_data": user_data, "error": error}
        send_msg(sock, response)
