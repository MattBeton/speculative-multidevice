# shared.py
from __future__ import annotations

import asyncio
import functools
import json
from concurrent.futures import ThreadPoolExecutor
from typing import Type

# Try to use orjson for faster JSON serialization
try:
    import orjson
    _USE_ORJSON = True
except ImportError:
    _USE_ORJSON = False

# Pre-computed message bytes for common messages (with orjson)
_RESET_REQUEST_BYTES = None
_RESET_RESPONSE_BYTES = None

try:
    import numpy as _np
except ModuleNotFoundError:  # pragma: no cover
    _np = None

from pydantic import BaseModel, ConfigDict

# ---------- MLX single-thread executor ----------
# All MLX work runs on this single thread to avoid thread-safety gotchas.
_MLX_EXEC = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx")

async def run_mlx(func, *args, **kwargs):
    """Run blocking MLX code on the dedicated single-thread executor."""
    loop = asyncio.get_running_loop()
    bound = functools.partial(func, *args, **kwargs)
    return await loop.run_in_executor(_MLX_EXEC, bound)


# ---------- Wire messages ----------
class Message(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)


class ResetRequest(Message):
    """Reset whole session on the server (clears all stream states)."""
    pass

class ResetResponse(Message):
    """Acknowledgment that reset completed."""
    ok: bool = True

class PrefillRequest(Message):
    prompts: list[list[int]]

class PrefillResponse(Message):
    ok: bool = True

class VerifyRequest(Message):
    draft_toks: list[list[int]]
    draft_topk_vals: list[list[list[float]]]
    draft_topk_idx: list[list[list[int]]]

class VerifyResponse(Message):
    accepted_len: list[int]
    base_token: list[int]
    hit_eos: list[bool]


MessageType = (
    ResetRequest | ResetResponse |
    PrefillRequest | PrefillResponse |
    VerifyRequest | VerifyResponse
)

_TYPE_TO_NAME: dict[Type[Message], str] = {
    ResetRequest: "reset",
    ResetResponse: "reset_response",
    PrefillRequest: "prefill",
    PrefillResponse: "prefill_response",
    VerifyRequest: "verify",
    VerifyResponse: "verify_response",
}
_NAME_TO_TYPE: dict[str, Type[Message]] = {v: k for k, v in _TYPE_TO_NAME.items()}

class MessageChannel:
    """Line-delimited JSON message transport."""
    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        self._reader = reader
        self._writer = writer
        # Enable TCP_NODELAY to disable Nagle's algorithm for lower latency
        sock = writer.get_extra_info('socket')
        if sock is not None:
            import socket
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    async def recv(self) -> MessageType | None:
        line = await self._reader.readline()
        if not line:
            return None
        if _USE_ORJSON:
            envelope = orjson.loads(line)
        else:
            envelope = json.loads(line)
        msg_type = envelope.get("type")
        payload = envelope.get("payload", {})
        cls = _NAME_TO_TYPE.get(msg_type)
        if cls is None:
            raise ValueError(f"unknown message type: {msg_type!r}")

        # Fast path for simple messages without validation overhead
        if msg_type == "reset":
            return ResetRequest()
        elif msg_type == "reset_response":
            return ResetResponse(ok=payload.get("ok", True))

        return cls.model_validate(payload)

    async def send(self, message: MessageType) -> None:
        global _RESET_REQUEST_BYTES, _RESET_RESPONSE_BYTES

        # Fast path for common messages with pre-computed bytes
        if _USE_ORJSON and isinstance(message, ResetRequest):
            if _RESET_REQUEST_BYTES is None:
                _RESET_REQUEST_BYTES = orjson.dumps({"type": "reset", "payload": {}}) + b"\n"
            self._writer.write(_RESET_REQUEST_BYTES)
            await self._writer.drain()
            return
        elif _USE_ORJSON and isinstance(message, ResetResponse) and message.ok:
            if _RESET_RESPONSE_BYTES is None:
                _RESET_RESPONSE_BYTES = orjson.dumps({"type": "reset_response", "payload": {"ok": True}}) + b"\n"
            self._writer.write(_RESET_RESPONSE_BYTES)
            await self._writer.drain()
            return

        msg_name = _TYPE_TO_NAME.get(type(message))
        if msg_name is None:
            raise ValueError(f"unregistered message type: {type(message)!r}")
        envelope = {"type": msg_name, "payload": message.model_dump()}
        if _USE_ORJSON:
            data = orjson.dumps(envelope) + b"\n"
        else:
            data = (json.dumps(envelope) + "\n").encode("utf-8")
        self._writer.write(data)
        await self._writer.drain()

    async def close(self) -> None:
        self._writer.close()
        await self._writer.wait_closed()
