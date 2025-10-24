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


# --------- SINGLE (legacy) ---------
class PrefillRequest(Message):
    prompt: list[int]

    @field_validator("prompt", mode="before")
    @classmethod
    def _ensure_list(cls, value: Any) -> list[int]:
        if _np is not None and isinstance(value, _np.ndarray):
            return value.astype(int).tolist()
        return [int(v) for v in list(value)]


class PrefillResponse(Message):
    ok: bool = True


class VerifyRequest(Message):
    draft_toks: list[int]
    draft_topk_vals: list[list[float]]
    draft_topk_idx: list[list[int]]

    @field_validator("draft_toks", mode="before")
    @classmethod
    def _ensure_tokens(cls, value: Any) -> list[int]:
        if _np is not None and isinstance(value, _np.ndarray):
            return value.astype(int).tolist()
        return [int(v) for v in list(value)]

    @field_validator("draft_topk_vals", mode="before")
    @classmethod
    def _ensure_vals(cls, value: Any) -> list[list[float]]:
        rows = value.tolist() if (_np is not None and isinstance(value, _np.ndarray)) else list(value)
        return [[float(v) for v in row] for row in rows]

    @field_validator("draft_topk_idx", mode="before")
    @classmethod
    def _ensure_idx(cls, value: Any) -> list[list[int]]:
        rows = value.tolist() if (_np is not None and isinstance(value, _np.ndarray)) else list(value)
        return [[int(v) for v in row] for row in rows]


class VerifyResponse(Message):
    accepted_len: int            # how many drafted tokens were accepted
    base_token: Optional[int]    # the base fallback token (None if EOS hit in accepted draft)
    hit_eos: bool                # whether an EOS was encountered among accepted tokens


# --------- BATCH (new) ---------
class PrefillItem(Message):
    stream_id: str
    prompt: list[int]

    @field_validator("prompt", mode="before")
    @classmethod
    def _ensure_list(cls, value: Any) -> list[int]:
        if _np is not None and isinstance(value, _np.ndarray):
            return value.astype(int).tolist()
        return [int(v) for v in list(value)]


class PrefillBatchRequest(Message):
    items: List[PrefillItem]


class PrefillBatchResponse(Message):
    ok: bool = True
    count: int = 0


class VerifyItem(Message):
    stream_id: str
    draft_toks: list[int]
    draft_topk_vals: list[list[float]]
    draft_topk_idx: list[list[int]]

    @field_validator("draft_toks", mode="before")
    @classmethod
    def _ensure_tokens(cls, value: Any) -> list[int]:
        if _np is not None and isinstance(value, _np.ndarray):
            return value.astype(int).tolist()
        return [int(v) for v in list(value)]

    @field_validator("draft_topk_vals", mode="before")
    @classmethod
    def _ensure_vals(cls, value: Any) -> list[list[float]]:
        rows = value.tolist() if (_np is not None and isinstance(value, _np.ndarray)) else list(value)
        return [[float(v) for v in row] for row in rows]

    @field_validator("draft_topk_idx", mode="before")
    @classmethod
    def _ensure_idx(cls, value: Any) -> list[list[int]]:
        rows = value.tolist() if (_np is not None and isinstance(value, _np.ndarray)) else list(value)
        return [[int(v) for v in row] for row in rows]


class VerifyResponseItem(Message):
    stream_id: str
    accepted_len: int
    base_token: Optional[int]
    hit_eos: bool


class VerifyBatchRequest(Message):
    items: List[VerifyItem]


class VerifyBatchResponse(Message):
    items: List[VerifyResponseItem]


MessageType = Union[
    ResetRequest,
    PrefillRequest, PrefillResponse,
    VerifyRequest, VerifyResponse,
    PrefillBatchRequest, PrefillBatchResponse,
    VerifyBatchRequest, VerifyBatchResponse,
]

_TYPE_TO_NAME: Dict[Type[Message], str] = {
    ResetRequest: "reset",
    PrefillRequest: "prefill",
    PrefillResponse: "prefill_response",
    VerifyRequest: "verify",
    VerifyResponse: "verify_response",
    PrefillBatchRequest: "prefill_batch",
    PrefillBatchResponse: "prefill_batch_response",
    VerifyBatchRequest: "verify_batch",
    VerifyBatchResponse: "verify_batch_response",
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
