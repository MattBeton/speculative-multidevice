
# speculative_client.py
# speculative_client.py (batched)
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from model import MLXGenerationModel
from timing import TokenTimer
from shared import (
    MessageChannel,
    PrefillBatchRequest,
    PrefillBatchResponse,
    VerifyBatchRequest,
    VerifyBatchResponse,
    VerifyItem,
    VerifyResponseItem,
    run_mlx,
)

# ---- Configure the draft (client) model ----
DRAFT_MODEL_PATH = next(Path(
    "/Users/frank/.cache/huggingface/hub/models--mlx-community--Llama-3.2-1B-Instruct-bf16/snapshots/"
).glob("*"))

# Prompts: list of (stream_id, prompt_text)
PROMPTS: List[Tuple[str, str]] = [
    ("s1", "Why is the sky blue?"),
    ("s2", "Explain speculative decoding in simple terms."),
    ("s3", "Write a terse haiku about Apple MLX."),
]

SPEC_K = 8
MAX_NEW_TOKENS = 64


def _print_phase_summary(name: str, tokens: int, seconds: float) -> None:
    tokps = (tokens / seconds) if seconds > 0 else float("inf")
    print(f"[{name:7}] {tokens} toks in {seconds:.3f}s  → {tokps:.1f} tok/s")


class _StreamDraft:
    """Holds per-stream local state for the draft model."""
    def __init__(self, stream_id: str, model_path: Path):
        self.stream_id = stream_id
        self.model = MLXGenerationModel(model_path)
        self.eos = self.model.eos_token_id()
        self.ids: Optional[np.ndarray] = None   # tokenized full prompt
        self.prompt_last: Optional[int] = None  # last token of prompt

        self.generated: List[int] = []
        self.last_token: Optional[int] = None   # last committed token "last" for next round
        self.finished: bool = False

    async def tokenize(self, text: str) -> None:
        ids = await run_mlx(self.model.tokenize, text)
        self.ids = np.array(ids, dtype=np.int32)

    async def prefill_local(self) -> None:
        """Prefill draft KV with all but last prompt token."""
        if self.ids is None or len(self.ids) < 1:
            raise RuntimeError("empty tokenized prompt")
        prefix = self.ids[:-1]
        await run_mlx(self.model.forward, prefix, False)
        self.prompt_last = int(self.ids[-1])
        self.last_token = self.prompt_last

    def token_budget_used(self) -> int:
        return len(self.ids) - 1 if self.ids is not None else 0

    def decoded_text(self) -> str:
        return self.model.decode(self.generated)


class BatchDraftClient:
    def __init__(self, channel: MessageChannel, stream_prompts: List[Tuple[str, str]]):
        self._channel = channel
        self._streams: Dict[str, _StreamDraft] = {
            sid: _StreamDraft(sid, DRAFT_MODEL_PATH) for sid, _ in stream_prompts
        }
        self._prompts = dict(stream_prompts)

    async def reset_remote(self) -> None:
        from shared import ResetRequest
        await self._channel.send(ResetRequest())

    async def prefill_both(self, timer: TokenTimer) -> None:
        """Tokenize + prefill locally; send prefill_batch remotely."""
        # 1) Tokenize all
        await asyncio.gather(*(st.tokenize(self._prompts[st.stream_id]) for st in self._streams.values()))

        # 2) Prefill local (one by one on MLX thread)
        async def _prefill_local(st: _StreamDraft):
            await st.prefill_local()

        # 3) Prefill remote (batch)
        items = []
        for st in self._streams.values():
            # remote needs the full prompt token ids (not numpy)
            ids_list = st.ids.astype(int).tolist()
            items.append({"stream_id": st.stream_id, "prompt": ids_list})

        with timer.measure("prefill", lambda: sum(st.token_budget_used() for st in self._streams.values())):
            await asyncio.gather(*( _prefill_local(st) for st in self._streams.values()))
            await self._channel.send(PrefillBatchRequest(items=items))
            msg = await self._channel.recv()
            if not isinstance(msg, PrefillBatchResponse):
                raise RuntimeError(f"expected PrefillBatchResponse, got {type(msg)!r}")

    async def decode_batch(self, spec_k: int, max_new_tokens: int, timer: TokenTimer) -> Dict[str, str]:
        """Drive speculative decode for all streams until each finishes or budget hit."""
        active = set(self._streams.keys())

        def _total_committed() -> int:
            return sum(len(st.generated) for st in self._streams.values())

        with timer.measure("decode", _total_committed):
            while active:
                verify_items: List[VerifyItem] = []

                # 1) Local draft for each active stream
                for sid in list(active):
                    st = self._streams[sid]
                    if st.finished:
                        active.discard(sid)
                        continue

                    # Stop if max tokens reached already
                    if len(st.generated) >= max_new_tokens:
                        st.finished = True
                        active.discard(sid)
                        continue

                    # Draft up to K tokens autoregressively
                    draft_toks: List[int] = []
                    draft_topk_idx: List[List[int]] = []
                    draft_topk_vals: List[List[float]] = []

                    cur = st.last_token
                    for _ in range(spec_k):
                        y = np.array([[int(cur)]], dtype=np.int32)
                        tok, topk_idx, topk_vals = await run_mlx(st.model.forward, y)
                        t = int(tok[-1])
                        draft_toks.append(t)
                        draft_topk_idx.append(topk_idx[-1].tolist())
                        draft_topk_vals.append([float(v) for v in topk_vals[-1]])
                        cur = t

                    verify_items.append(VerifyItem(
                        stream_id=sid,
                        draft_toks=draft_toks,
                        draft_topk_idx=draft_topk_idx,
                        draft_topk_vals=draft_topk_vals,
                    ))

                if not verify_items:
                    break

                # 2) Remote batch verify
                await self._channel.send(VerifyBatchRequest(items=verify_items))
                resp = await self._channel.recv()
                if not isinstance(resp, VerifyBatchResponse):
                    raise RuntimeError(f"expected VerifyBatchResponse, got {type(resp)!r}")

                # 3) Apply results per stream: commit + KV alignment
                by_id: Dict[str, VerifyResponseItem] = {it.stream_id: it for it in resp.items}

                for vi in verify_items:
                    sid = vi.stream_id
                    if sid not in by_id:
                        # Should not happen
                        continue
                    st = self._streams[sid]
                    r = by_id[sid]

                    m = int(r.accepted_len)
                    base_tok = r.base_token
                    hit_eos = bool(r.hit_eos)
                    accepted_tokens = vi.draft_toks[:m]
                    base_appended = 0 if base_tok is None else 1

                    # Commit locally: accepted + optional base
                    st.generated.extend(accepted_tokens)
                    if base_tok is not None:
                        st.generated.append(int(base_tok))

                    # Align local draft KV cache
                    draft_trim = SPEC_K - (m + base_appended)
                    if draft_trim > 0:
                        await run_mlx(st.model.trim_cache, draft_trim)
                    elif draft_trim < 0:
                        # Only possible when m==K and base appended -> catch up one step
                        await run_mlx(st.model.forward, np.array([[int(vi.draft_toks[m - 1])]], dtype=np.int32))

                    # Prepare next iteration's "last"
                    if hit_eos or len(st.generated) >= max_new_tokens:
                        st.finished = True
                        active.discard(sid)
                    else:
                        st.last_token = int(base_tok) if base_tok is not None else int(accepted_tokens[-1])

        # Decode final texts
        out: Dict[str, str] = {sid: st.decoded_text() for sid, st in self._streams.items()}
        return out


async def main() -> None:
    # Connect to server
    # IMPORTANT: change IP/port to the verifier host if remote
    ip = '192.168.200.2'
    reader, writer = await asyncio.open_connection(ip, 7070)
    channel = MessageChannel(reader, writer)
    client = BatchDraftClient(channel, PROMPTS)

    timer = TokenTimer()
    try:
        await client.reset_remote()
        await client.prefill_both(timer)
        texts = await client.decode_batch(SPEC_K, MAX_NEW_TOKENS, timer)

        # Stats
        prefill = timer.get("prefill")
        decode = timer.get("decode")
        if prefill:
            _print_phase_summary("prefill", prefill.tokens, prefill.seconds)
        if decode:
            _print_phase_summary("decode", decode.tokens, decode.seconds)

        # Output per stream
        print()
        for sid, txt in texts.items():
            print(f"--- [{sid}] ---")
            print(txt)
            print()
    finally:
        await channel.close()


if __name__ == "__main__":
    asyncio.run(main())
