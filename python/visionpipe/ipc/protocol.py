"""Length-prefixed JSON protocol over Unix Domain Sockets.

Wire format: ``[4 bytes big-endian length][UTF-8 JSON payload]``

Matches the C++ ProcessProxyNode framing exactly so both sides can
interoperate without an intermediate layer.
"""

from __future__ import annotations

import json
import struct
from typing import Any

_HEADER = struct.Struct("!I")
_MAX_MSG = 64 * 1024 * 1024  # 64 MiB


def send_msg(sock, data: dict[str, Any]) -> None:
    payload = json.dumps(data, separators=(",", ":")).encode()
    sock.sendall(_HEADER.pack(len(payload)) + payload)


def recv_msg(sock) -> dict[str, Any] | None:
    header = _recv_exact(sock, _HEADER.size)
    if header is None:
        return None
    (length,) = _HEADER.unpack(header)
    if length > _MAX_MSG:
        raise RuntimeError(f"IPC message too large: {length} bytes")
    payload = _recv_exact(sock, length)
    if payload is None:
        return None
    return json.loads(payload)


def _recv_exact(sock, n: int) -> bytes | None:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)
