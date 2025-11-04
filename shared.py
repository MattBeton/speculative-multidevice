# shared.py
from __future__ import annotations

import asyncio
import typing as _t
import functools
import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional, Type, Union, List

try:
    import numpy as _np
except ModuleNotFoundError:  # pragma: no cover
    _np = None

from pydantic import BaseModel, ConfigDict, field_validator

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


MessageType = Union[
    ResetRequest,
    PrefillRequest, PrefillResponse,
    VerifyRequest, VerifyResponse,
]

_TYPE_TO_NAME: Dict[Type[Message], str] = {
    ResetRequest: "reset",
    PrefillRequest: "prefill",
    PrefillResponse: "prefill_response",
    VerifyRequest: "verify",
    VerifyResponse: "verify_response",
}
_NAME_TO_TYPE: Dict[str, Type[Message]] = {v: k for k, v in _TYPE_TO_NAME.items()}

class MessageChannel:
    """Line-delimited JSON message transport."""
    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        self._reader = reader
        self._writer = writer

    async def recv(self) -> Optional[MessageType]:
        line = await self._reader.readline()
        if not line:
            return None
        envelope = json.loads(line)
        msg_type = envelope.get("type")
        payload = envelope.get("payload", {})
        cls = _NAME_TO_TYPE.get(msg_type)
        if cls is None:
            raise ValueError(f"unknown message type: {msg_type!r}")
        return cls.model_validate(payload)

    async def send(self, message: MessageType) -> None:
        msg_name = _TYPE_TO_NAME.get(type(message))
        if msg_name is None:
            raise ValueError(f"unregistered message type: {type(message)!r}")
        envelope = {"type": msg_name, "payload": message.model_dump()}
        data = (json.dumps(envelope) + "\n").encode("utf-8")
        self._writer.write(data)
        await self._writer.drain()

    async def close(self) -> None:
        self._writer.close()
        await self._writer.wait_closed()
