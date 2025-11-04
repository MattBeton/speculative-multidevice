# speculative_client.py
import asyncio
from pathlib import Path
from typing import List, Optional

import numpy as np

from model import MLXGenerationModel
from timing import TokenTimer
from shared import (
    MessageChannel,
    PrefillRequest,
    PrefillResponse,
    ResetRequest,
    VerifyRequest,
    VerifyResponse,
    run_mlx,
)

# ---- Configure the draft (client) model ----
DRAFT_MODEL_PATH = next(Path(
    "/Users/frank/.cache/huggingface/hub/models--mlx-community--Llama-3.2-1B-Instruct-bf16/snapshots/"
).glob("*"))

# Fixed prompts list
PROMPTS: List[str] = [
    "Why is the sky blue?",
    "Explain speculative decoding in simple terms.",
    "Write a sonnet about the iPhone.",
]

SPEC_K = 8
MAX_NEW_TOKENS = 64


def _print_phase_summary(name: str, tokens: int, seconds: float) -> None:
    tokps = (tokens / seconds) if seconds > 0 else float("inf")
    print(f"[{name:7}] {tokens} toks in {seconds:.3f}s  → {tokps:.1f} tok/s")


class StreamDraft:
    """Holds per-stream local state for the draft model."""
    def __init__(self, model_path: Path):
        self.model = MLXGenerationModel(model_path)
        self.eos = self.model.eos_token_id()
        self.ids: Optional[np.ndarray] = None   # tokenized full prompt
        self.prompt_last: Optional[int] = None  # last token of prompt

        self.generated: List[int] = []
        self.last_token: Optional[int] = None   # last committed token for next round
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
    """Client for batched speculative decoding with fixed batch size."""
    def __init__(self, channel: MessageChannel, prompts: List[str]):
        self._channel = channel
        self._prompts = prompts
        self._streams: List[StreamDraft] = [
            StreamDraft(DRAFT_MODEL_PATH) for _ in prompts
        ]

    async def reset_remote(self) -> None:
        """Send reset request to server."""
        await self._channel.send(ResetRequest())

    async def prefill_both(self, timer: TokenTimer) -> None:
        """Tokenize + prefill locally; send prefill request remotely."""
        # 1) Tokenize all streams
        await asyncio.gather(*(
            stream.tokenize(prompt)
            for stream, prompt in zip(self._streams, self._prompts)
        ))

        # 2) Prepare prompts for remote prefill (list of list of ints)
        prompts: List[List[int]] = []
        for stream in self._streams:
            if stream.ids is not None:
                # Convert numpy array to list of ints
                prompts.append(stream.ids.astype(int).tolist())

        # 3) Prefill local and remote
        with timer.measure("prefill", lambda: sum(st.token_budget_used() for st in self._streams)):
            # Prefill local models
            await asyncio.gather(*(
                stream.prefill_local()
                for stream in self._streams
            ))

            # Prefill remote
            await self._channel.send(PrefillRequest(prompts=prompts))
            msg = await self._channel.recv()
            if not isinstance(msg, PrefillResponse):
                raise RuntimeError(f"expected PrefillResponse, got {type(msg)!r}")

    async def decode_batch(self, spec_k: int, max_new_tokens: int, timer: TokenTimer) -> List[str]:
        """Drive speculative decode for all streams until each finishes or budget hit."""
        def _total_committed() -> int:
            return sum(len(st.generated) for st in self._streams)

        with timer.measure("decode", _total_committed):
            round_count = 0
            max_rounds = 5
            while True:
                # Check if all streams are finished
                if all(st.finished for st in self._streams):
                    break
                
                # Limit to 5 rounds of drafting
                if round_count >= max_rounds:
                    break
                
                round_count += 1

                # Prepare batch verify request
                draft_toks_batch: List[List[int]] = []
                draft_topk_idx_batch: List[List[List[int]]] = []
                draft_topk_vals_batch: List[List[List[float]]] = []

                # 1) Generate draft tokens for each stream
                for stream in self._streams:
                    # For finished streams, add empty drafts
                    if stream.finished or len(stream.generated) >= max_new_tokens:
                        stream.finished = True
                        # Add placeholder empty draft for this stream
                        draft_toks_batch.append([])
                        draft_topk_idx_batch.append([])
                        draft_topk_vals_batch.append([])
                        continue

                    # Draft up to K tokens autoregressively
                    draft_toks: List[int] = []
                    draft_topk_idx: List[List[int]] = []
                    draft_topk_vals: List[List[float]] = []

                    cur = stream.last_token
                    for _ in range(spec_k):
                        y = np.array([[int(cur)]], dtype=np.int32)
                        tok, topk_idx, topk_vals = await run_mlx(stream.model.forward, y)

                        t = int(tok[-1])
                        draft_toks.append(t)
                        # Convert numpy arrays to lists
                        draft_topk_idx.append(topk_idx[-1].astype(int).tolist())
                        draft_topk_vals.append([float(v) for v in topk_vals[-1]])
                        cur = t

                    # Add to batch
                    draft_toks_batch.append(draft_toks)
                    draft_topk_idx_batch.append(draft_topk_idx)
                    draft_topk_vals_batch.append(draft_topk_vals)

                # If no active streams, break
                if all(len(toks) == 0 for toks in draft_toks_batch):
                    break

                # 2) Send batch verify request
                verify_req = VerifyRequest(
                    draft_toks=draft_toks_batch,
                    draft_topk_vals=draft_topk_vals_batch,
                    draft_topk_idx=draft_topk_idx_batch
                )
                await self._channel.send(verify_req)

                resp = await self._channel.recv()
                if not isinstance(resp, VerifyResponse):
                    raise RuntimeError(f"expected VerifyResponse, got {type(resp)!r}")

                # 3) Apply results per stream
                for i, stream in enumerate(self._streams):
                    # Skip finished streams
                    if stream.finished:
                        continue

                    # Extract results for this stream
                    accepted = resp.accepted_len[i]
                    base_tok = resp.base_token[i]
                    hit_eos = resp.hit_eos[i]

                    # Handle -1 as None for base_token
                    if base_tok == -1:
                        base_tok = None

                    # Get accepted tokens from draft
                    accepted_tokens = draft_toks_batch[i][:accepted]
                    base_appended = 0 if base_tok is None else 1

                    # Commit locally: accepted + optional base
                    stream.generated.extend(accepted_tokens)
                    if base_tok is not None:
                        stream.generated.append(base_tok)

                    # Align local draft KV cache
                    draft_trim = spec_k - (accepted + base_appended)
                    if draft_trim > 0:
                        await run_mlx(stream.model.trim_cache, draft_trim)
                    elif draft_trim < 0:
                        # Only possible when m==K and base appended -> catch up one step
                        last_accepted = draft_toks_batch[i][accepted - 1]
                        await run_mlx(
                            stream.model.forward,
                            np.array([[last_accepted]], dtype=np.int32)
                        )

                    # Prepare next iteration's "last"
                    if hit_eos or len(stream.generated) >= max_new_tokens:
                        stream.finished = True
                    else:
                        stream.last_token = base_tok if base_tok is not None else (
                            accepted_tokens[-1] if accepted_tokens else stream.last_token
                        )

        # Decode final texts
        return [stream.decoded_text() for stream in self._streams]


async def main() -> None:
    # Connect to server
    ip = '192.168.200.2'
    # ip = 'localhost'
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
        total_tokens = 0
        total_seconds = 0

        if prefill:
            _print_phase_summary("prefill", prefill.tokens, prefill.seconds)
            total_tokens += prefill.tokens
            total_seconds += prefill.seconds

        if decode:
            _print_phase_summary("decode", decode.tokens, decode.seconds)
            total_tokens += decode.tokens
            total_seconds += decode.seconds

        # Print total TPS
        if total_seconds > 0:
            total_tps = total_tokens / total_seconds
            print(f"[TOTAL  ] {total_tokens} toks in {total_seconds:.3f}s  → {total_tps:.1f} tok/s")

        # Output per stream
        print()
        for i, txt in enumerate(texts):
            print(f"--- [Stream {i}] ---")
            print(txt)
            print()
    finally:
        await channel.close()


if __name__ == "__main__":
    asyncio.run(main())