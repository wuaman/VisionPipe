"""WebRTC signaling server for VisionPipe.

Handles WebSocket-based SDP offer/answer exchange and ICE candidate relay
for a single WebRTCSink node.

Protocol (JSON messages)
------------------------
Server → Browser:  {"type": "offer",     "sdp": "<SDP offer string>"}
Browser → Server:  {"type": "answer",    "sdp": "<SDP answer string>"}
Server → Browser:  {"type": "candidate", "candidate": "...", "sdpMid": "..."}
Browser → Server:  {"type": "candidate", "candidate": "...", "sdpMid": "..."}

The server is the offerer (it creates the SDP offer via the C++ WebRTCSink).
The browser is the answerer.
"""

from __future__ import annotations

import asyncio
import json
import logging

from aiohttp import WSMsgType, web

logger = logging.getLogger(__name__)

# How often (seconds) to poll drain_candidates() for new local ICE candidates
_CANDIDATE_POLL_INTERVAL = 0.05


async def handle_webrtc_signaling(
    request: web.Request,
    sink: object,
) -> web.WebSocketResponse:
    """Run the full WebRTC signaling exchange for one browser peer.

    Parameters
    ----------
    request:
        The aiohttp WebSocket upgrade request.
    sink:
        A ``visionpipe.WebRTCSink`` instance bound to the pipeline.

    Returns
    -------
    web.WebSocketResponse
    """
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    peer_id: str | None = None

    try:
        # Allocate a new WebRTC peer connection in C++ and get an opaque ID
        peer_id = sink.create_peer()  # type: ignore[attr-defined]
        logger.info("WebRTC signaling: peer %s connected", peer_id)

        # Wait (in thread-pool) for libdatachannel to generate the SDP offer
        loop = asyncio.get_event_loop()
        offer_sdp: str = await loop.run_in_executor(
            None, lambda: sink.get_offer(peer_id, 10_000)  # type: ignore[attr-defined]
        )
        await ws.send_str(json.dumps({"type": "offer", "sdp": offer_sdp}))
        logger.debug("WebRTC signaling: sent SDP offer to peer %s", peer_id)

        # Background task: forward locally generated ICE candidates to browser
        async def _forward_candidates() -> None:
            while not ws.closed:
                pid = peer_id  # capture in loop-local scope
                candidates = await loop.run_in_executor(
                    None, lambda: sink.drain_candidates(pid)  # type: ignore[attr-defined]
                )
                for candidate, mid in candidates:
                    await ws.send_str(json.dumps({
                        "type": "candidate",
                        "candidate": candidate,
                        "sdpMid": mid,
                    }))
                await asyncio.sleep(_CANDIDATE_POLL_INTERVAL)

        cand_task = asyncio.ensure_future(_forward_candidates())

        # Process incoming messages from the browser
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                data = json.loads(msg.data)
                msg_type = data.get("type")

                if msg_type == "answer":
                    sdp = data.get("sdp", "")
                    await loop.run_in_executor(
                        None, lambda s=sdp: sink.set_answer(peer_id, s)  # type: ignore[attr-defined]
                    )
                    logger.debug("WebRTC signaling: applied SDP answer for peer %s", peer_id)

                elif msg_type == "candidate":
                    cand = data.get("candidate", "")
                    mid = data.get("sdpMid", "")
                    await loop.run_in_executor(
                        None,
                        lambda c=cand, m=mid: sink.add_candidate(peer_id, c, m),  # type: ignore[attr-defined]
                    )

            elif msg.type in (WSMsgType.ERROR, WSMsgType.CLOSE):
                break

        cand_task.cancel()
        try:
            await cand_task
        except asyncio.CancelledError:
            pass

    except Exception:
        logger.exception("WebRTC signaling error for peer %s", peer_id)
    finally:
        if peer_id is not None:
            sink.remove_peer(peer_id)  # type: ignore[attr-defined]
            logger.info("WebRTC signaling: peer %s disconnected", peer_id)

    return ws
