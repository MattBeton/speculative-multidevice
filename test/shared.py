from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, Optional, Type, Union

try:
    import numpy as _np
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    _np = None

from pydantic import BaseModel, ConfigDict, field_validator


class Message(BaseModel):
    """Base class for messages exchanged between client and server."""

    model_config = ConfigDict(arbitrary_types_allowed=True)


class ResetRequest(Message):
    """Clear any per-sequence state on the verifier."""

    pass


class PrefillRequest(Message):
    """Send the initial prompt tokens to warm up caches."""

    prompt: list[int]

    @field_validator("prompt", mode="before")
    @classmethod
    def _ensure_list(cls, value: Any) -> list[int]:
        if _np is not None and isinstance(value, _np.ndarray):
            return value.astype(int).tolist()
        return [int(v) for v in list(value)]


class VerifyRequest(Message):
    """Ask the verifier to accept or reject drafted tokens."""

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
    """Response indicating how many drafted tokens were accepted."""

    accepted_len: int


MessageType = Union[ResetRequest, PrefillRequest, VerifyRequest, VerifyResponse]

_TYPE_TO_NAME: Dict[Type[Message], str] = {
    ResetRequest: "reset",
    PrefillRequest: "prefill",
    VerifyRequest: "verify",
    VerifyResponse: "verify_response",
}

_NAME_TO_TYPE: Dict[str, Type[Message]] = {v: k for k, v in _TYPE_TO_NAME.items()}


def encode_message(message: MessageType) -> bytes:
    """Serialize a message to line-delimited JSON bytes."""

    msg_type = _TYPE_TO_NAME.get(type(message))
    if msg_type is None:
        raise ValueError(f"unregistered message type: {type(message)!r}")
    envelope = {
        "type": msg_type,
        "payload": message.model_dump(),
    }
    return (json.dumps(envelope) + "\n").encode("utf-8")


def decode_message(data: bytes) -> MessageType:
    """Deserialize a single line of JSON into a message instance."""

    if not data:
        raise ValueError("cannot decode empty payload")
    envelope = json.loads(data)
    msg_type = envelope.get("type")
    payload = envelope.get("payload", {})
    cls = _NAME_TO_TYPE.get(msg_type)
    if cls is None:
        raise ValueError(f"unknown message type: {msg_type!r}")
    return cls.model_validate(payload)


class MessageChannel:
    """Line-delimited JSON message transport helper."""

    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        self._reader = reader
        self._writer = writer

    async def recv(self) -> Optional[MessageType]:
        line = await self._reader.readline()
        if not line:
            return None
        return decode_message(line)

    async def send(self, message: MessageType) -> None:
        self._writer.write(encode_message(message))
        await self._writer.drain()

    async def close(self) -> None:
        self._writer.close()
        await self._writer.wait_closed()
