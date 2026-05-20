"""FrameView: safe read/write view over frame metadata for subprocess CustomNodes.

Automatically invalidated after ``on_frame`` returns to prevent dangling
references across IPC boundaries.
"""

from __future__ import annotations

from typing import Any


class FrameView:
    """Safe metadata view passed to :meth:`CustomNode.on_frame`.

    Provides read access to frame scalars (``frame_id``, ``stream_id``,
    ``pts_us``), detection/classification/track lists, and a mutable
    ``user_data`` dict.  The view is invalidated when ``on_frame`` returns;
    accessing it afterwards raises :class:`RuntimeError`.
    """

    __slots__ = (
        "_data",
        "_user_data",
        "_valid",
    )

    def __init__(self, metadata: dict[str, Any]) -> None:
        self._data = metadata
        self._user_data: dict[str, Any] = dict(metadata.get("user_data") or {})
        self._valid = True

    def _check(self) -> None:
        if not self._valid:
            raise RuntimeError(
                "FrameView is no longer valid — do not store or access it "
                "outside of on_frame()"
            )

    def _invalidate(self) -> None:
        self._valid = False
        self._data = {}

    @property
    def valid(self) -> bool:
        return self._valid

    @property
    def stream_id(self) -> int:
        self._check()
        return self._data["stream_id"]

    @property
    def frame_id(self) -> int:
        self._check()
        return self._data["frame_id"]

    @property
    def pts_us(self) -> int:
        self._check()
        return self._data["pts_us"]

    @property
    def detections(self) -> list[dict[str, Any]]:
        self._check()
        return self._data.get("detections", [])

    @property
    def classifications(self) -> list[dict[str, Any]]:
        self._check()
        return self._data.get("classifications", [])

    @property
    def tracks(self) -> list[dict[str, Any]]:
        self._check()
        return self._data.get("tracks", [])

    @property
    def user_data(self) -> dict[str, Any]:
        self._check()
        return self._user_data

    @user_data.setter
    def user_data(self, value: dict[str, Any]) -> None:
        self._check()
        if not isinstance(value, dict):
            raise TypeError("user_data must be a dict")
        self._user_data = value

    def _get_user_data_snapshot(self) -> dict[str, Any]:
        return dict(self._user_data)
